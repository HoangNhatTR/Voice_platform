"""Process-wide bounded admission; callers include queue time in deadlines."""

import asyncio
from contextlib import asynccontextmanager

from .errors import CapacityExceeded
from .clock import now_ms
from .events import EventType
from ..observability.probe import current_probe


class WorkLimiter:
    def __init__(self, parallel: int, max_queue: int) -> None:
        if parallel < 1 or max_queue < 0:
            raise ValueError("invalid work limits")
        self.semaphore = asyncio.Semaphore(parallel)
        self.capacity = parallel + max_queue
        self.pending = 0
        self.active = 0
        self.parallel = parallel

    @asynccontextmanager
    async def slot(self, *, priority="speech"):
        if self.pending >= self.capacity:
            raise CapacityExceeded("work queue is full")
        self.pending += 1
        probe = current_probe()
        queued_at = now_ms()
        if probe:
            probe.mark(EventType.MODEL_QUEUED, active=self.active, waiting=max(0, self.pending-self.active-1))
        try:
            async with self.semaphore:
                self.active += 1
                try:
                    if probe:
                        probe.mark(EventType.MODEL_SLOT_ACQUIRED, queue_ms=now_ms()-queued_at, active=self.active, waiting=self.pending-self.active)
                    yield
                finally:
                    self.active -= 1
        finally:
            self.pending -= 1

    def snapshot(self):
        return {"active": self.active, "waiting": max(0, self.pending-self.active), "capacity": self.capacity, "parallel": self.parallel}


class PriorityWorkLimiter:
    """Non-preemptive shared LLM admission: speech first, bounded search.

    A native request already running cannot be preempted. Search may occupy
    at most parallel-1 slots (one when parallel=1), and queued speech gets
    the next slot before queued search. Cancellation reclaims a grant even
    when it races with wake-up; counts represent requests, not coroutine age.

    Admission is separate too: speech is admitted against parallel+max_queue
    speech requests, search against search_parallel+search_queue (default
    max_queue). One shared bound let queued searches use up the queue, and the
    next speech turn was refused with "LLM queue is full" while search waited.
    """
    def __init__(self, parallel, max_queue, search_parallel=1, search_queue=None):
        search_queue=max_queue if search_queue is None else search_queue
        if parallel<1 or max_queue<0 or search_parallel<1 or search_queue<0:
            raise ValueError("invalid priority work limits")
        self.parallel=parallel;self.capacity=parallel+max_queue
        self.search_parallel=min(search_parallel,max(1,parallel-1))
        self.search_capacity=self.search_parallel+search_queue
        self.pending=0;self.search_pending=0;self.active=0;self.search_active=0;self.waiters=[]

    def _drain(self):
        for waiter in sorted(self.waiters,key=lambda w:w['priority']!='speech'):
            if self.active>=self.parallel:break
            if waiter['future'].cancelled():continue
            if waiter['priority']=='search' and self.search_active>=self.search_parallel:continue
            self.waiters.remove(waiter);waiter['granted']=True
            self.active+=1;self.search_active+=waiter['priority']=='search'
            waiter['future'].set_result(None)

    @asynccontextmanager
    async def slot(self, *, priority="speech"):
        if priority not in ('speech','search'):raise ValueError('unknown work priority')
        search=priority=='search'
        if search and self.search_pending>=self.search_capacity:raise CapacityExceeded('search queue is full')
        if not search and self.pending-self.search_pending>=self.capacity:raise CapacityExceeded('LLM queue is full')
        self.pending+=1;self.search_pending+=search
        queued_at=now_ms();probe=current_probe()
        waiter={'future':asyncio.get_running_loop().create_future(),'priority':priority,'granted':False}
        self.waiters.append(waiter)
        try:
            if probe:probe.mark(EventType.MODEL_QUEUED,priority=priority,active=self.active,waiting=len(self.waiters)-1)
            self._drain()
            await waiter['future']
            if probe:probe.mark(EventType.MODEL_SLOT_ACQUIRED,priority=priority,queue_ms=now_ms()-queued_at,active=self.active,waiting=len(self.waiters))
            yield
        finally:
            if waiter in self.waiters:self.waiters.remove(waiter)
            if waiter['granted']:
                self.active-=1;self.search_active-=search
            self.pending-=1;self.search_pending-=search
            self._drain()

    def snapshot(self):
        return {'active':self.active,'waiting':len(self.waiters),'capacity':self.capacity,'parallel':self.parallel,
            'speech_active':self.active-self.search_active,'search_active':self.search_active,
            'speech_waiting':sum(w['priority']=='speech' for w in self.waiters),
            'search_waiting':sum(w['priority']=='search' for w in self.waiters),'search_parallel':self.search_parallel,
            'search_capacity':self.search_capacity}
