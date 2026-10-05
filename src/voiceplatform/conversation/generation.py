"""Generation identity, fencing and cancellation.

One rule holds the realtime side together: work is tagged with the generation
it was started for, and anything arriving under a generation that is no longer
current is dropped and counted. Without it, a cancelled turn's LLM tokens and
TTS audio keep arriving — late, out of order, and indistinguishable from the
new turn's — which is the race that makes voice agents talk over themselves.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Coroutine
from typing import Any

from ..core.ids import GenerationKey


class GenerationManager:
    def __init__(self, session_id: str) -> None:
        self.session_id = session_id
        self._turn_id = 0
        self._generation_id = 0
        self._current: GenerationKey | None = None
        self._tasks: dict[int, set[asyncio.Task]] = {}
        self.stale_drops = 0

    # --- identity ----------------------------------------------------------
    @property
    def current(self) -> GenerationKey | None:
        return self._current

    @property
    def turn_id(self) -> int:
        return self._turn_id

    def next_turn(self) -> int:
        self._turn_id += 1
        return self._turn_id

    def begin(self, turn_id: int | None = None) -> GenerationKey:
        """Open a new generation. Every previous one is now stale."""
        self._generation_id += 1
        # Done callbacks empty a generation's bucket but never removed the
        # bucket itself; only cancel() did, so an uninterrupted session leaked
        # one empty set per turn for as long as it ran.
        for stale in [
            generation_id
            for generation_id, tasks in self._tasks.items()
            if generation_id != self._SESSION_BUCKET and not tasks
        ]:
            self._tasks.pop(stale, None)
        key = GenerationKey(
            session_id=self.session_id,
            turn_id=self._turn_id if turn_id is None else turn_id,
            generation_id=self._generation_id,
        )
        self._current = key
        self._tasks.setdefault(key.generation_id, set())
        return key

    def is_current(self, key: GenerationKey | None) -> bool:
        return key is not None and self._current is not None and key == self._current

    def check(self, key: GenerationKey | None) -> bool:
        """is_current, counting the misses. Use on every emit boundary."""
        if self.is_current(key):
            return True
        self.stale_drops += 1
        return False

    # --- task ownership ----------------------------------------------------
    def spawn(
        self,
        coro: Coroutine[Any, Any, Any],
        key: GenerationKey | None = None,
        *,
        name: str | None = None,
    ) -> asyncio.Task:
        target = key or self._current
        if target is None:
            raise RuntimeError("spawn() before begin()")
        task = asyncio.create_task(coro, name=name)
        bucket = self._tasks.setdefault(target.generation_id, set())
        bucket.add(task)
        task.add_done_callback(bucket.discard)
        task.add_done_callback(self._consume_exception)
        return task

    @staticmethod
    def _consume_exception(task: asyncio.Task) -> None:
        if not task.cancelled():
            task.exception()

    _SESSION_BUCKET = 0

    def spawn_detached(self, coro: Coroutine[Any, Any, Any], *, name: str | None = None) -> asyncio.Task:
        """Việc thuộc về PHIÊN, không thuộc lượt nói nào.

        Tra cứu phải sống qua mọi lần ngắt lời và đổi chủ đề: gắn nó vào
        generation thì mỗi lần người dùng chen ngang là mất luôn câu trả lời
        họ đang chờ. Nó chỉ chết khi phiên đóng.
        """
        task = asyncio.create_task(coro, name=name)
        bucket = self._tasks.setdefault(self._SESSION_BUCKET, set())
        bucket.add(task)
        task.add_done_callback(bucket.discard)
        task.add_done_callback(self._consume_exception)
        return task

    def detach(self, key: GenerationKey, *, keep: Callable[[asyncio.Task], bool]) -> list[asyncio.Task]:
        """Move a generation's live tasks matching `keep` to the session bucket.

        They then survive `cancel(key)` and die only with the session (or when
        their owner cancels them). Used for a barge-in that may turn out to be
        a cough: the voice stops at once, the LLM keeps writing until we know.
        """
        bucket = self._tasks.get(key.generation_id, set())
        moved = [t for t in bucket if not t.done() and keep(t)]
        session = self._tasks.setdefault(self._SESSION_BUCKET, set())
        for task in moved:
            bucket.discard(task)
            session.add(task)
            task.add_done_callback(session.discard)
        return moved

    def live_tasks(self, key: GenerationKey | None = None) -> int:
        target = key or self._current
        if target is None:
            return 0
        return len([t for t in self._tasks.get(target.generation_id, set()) if not t.done()])

    async def cancel(self, key: GenerationKey | None = None, *, timeout_s: float = 1.0) -> int:
        """Cancel every task of a generation and wait for them to unwind.

        The fence closes *first*, synchronously, before anything is awaited.
        Cancellation is not instant: a task only stops at its next suspension
        point, and between the request and that point it keeps running. Marking
        the generation stale afterwards leaves that whole window open, and
        audio sent inside it is indistinguishable from audio the user should
        still be hearing.
        """
        target = key or self._current
        if target is None:
            return 0
        if self.is_current(target):
            self._current = None
        tasks = [t for t in self._tasks.get(target.generation_id, set()) if not t.done()]
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.wait(tasks, timeout=timeout_s)
        if not any(not task.done() for task in tasks):
            self._tasks.pop(target.generation_id, None)
        return len(tasks)

    async def cancel_all(self) -> int:
        count = 0
        for generation_id in list(self._tasks):
            tasks = [t for t in self._tasks.get(generation_id, set()) if not t.done()]
            for task in tasks:
                task.cancel()
            count += len(tasks)
            if tasks:
                await asyncio.wait(tasks, timeout=1.0)
            if not any(not task.done() for task in tasks):
                self._tasks.pop(generation_id, None)
        self._current = None
        return count


class Fence:
    """A small callable to hand to code that must not import the manager."""

    __slots__ = ("_manager", "_key")

    def __init__(self, manager: GenerationManager, key: GenerationKey) -> None:
        self._manager = manager
        self._key = key

    @property
    def key(self) -> GenerationKey:
        return self._key

    def __call__(self) -> bool:
        return self._manager.is_current(self._key)

    def check(self) -> bool:
        return self._manager.check(self._key)
