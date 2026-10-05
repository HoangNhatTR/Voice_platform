"""Shared LLM admission: queued searches must not refuse speech."""

from __future__ import annotations

import asyncio
import random

import pytest

from voiceplatform.core.errors import CapacityExceeded
from voiceplatform.core.limits import PriorityWorkLimiter
from voiceplatform.models.llm.openai_compat import OpenAiCompatLlm


async def test_queued_searches_do_not_use_up_speech_admission():
    """Deployed shape: 3 slots + 8 queue. 6 searches (3 sessions x 2 in
    flight) + 5 speech used to refuse the next speech turn as 'LLM queue is
    full' — the fallback reply for a turn that had priority."""
    limiter = PriorityWorkLimiter(3, 8)
    hold = asyncio.Event()

    async def request(priority):
        async with limiter.slot(priority=priority):
            await hold.wait()

    tasks = [asyncio.create_task(request("search")) for _ in range(6)]
    tasks += [asyncio.create_task(request("speech")) for _ in range(5)]
    await asyncio.sleep(0)
    speech = asyncio.create_task(request("speech"))
    await asyncio.sleep(0)
    assert not speech.done()                       # admitted, waiting its turn
    snap = limiter.snapshot()
    assert snap["speech_waiting"] == 4 and snap["search_waiting"] == 5
    hold.set()
    await asyncio.gather(*tasks, speech)
    assert limiter.pending == limiter.search_pending == limiter.active == 0


async def test_search_has_its_own_bound():
    limiter = PriorityWorkLimiter(3, 8, search_queue=2)
    assert limiter.snapshot()["search_capacity"] == 3   # 1 running + 2 queued
    hold = asyncio.Event()

    async def request(priority):
        async with limiter.slot(priority=priority):
            await hold.wait()

    tasks = [asyncio.create_task(request("search")) for _ in range(3)]
    await asyncio.sleep(0)
    with pytest.raises(CapacityExceeded, match="search queue"):
        async with limiter.slot(priority="search"):
            pass
    tasks += [asyncio.create_task(request("speech")) for _ in range(11)]   # speech still has its 11
    await asyncio.sleep(0)
    with pytest.raises(CapacityExceeded, match="LLM queue"):
        async with limiter.slot(priority="speech"):
            pass
    hold.set()
    await asyncio.gather(*tasks)


def test_llm_option_sets_the_search_queue():
    assert OpenAiCompatLlm(max_queue=8, max_search_queue=3).limiter.search_capacity == 4
    assert OpenAiCompatLlm(max_queue=8).limiter.search_capacity == 9


@pytest.mark.parametrize("seed", range(10))
async def test_cancellation_and_errors_never_leak_either_bound(seed):
    rng = random.Random(seed)
    limiter = PriorityWorkLimiter(3, 8, search_queue=4)
    broken = []

    async def user():
        try:
            async with limiter.slot(priority=rng.choice(["speech", "search"])):
                if limiter.active > limiter.parallel or limiter.search_active > limiter.search_parallel:
                    broken.append(limiter.snapshot())
                await asyncio.sleep(rng.random() * 0.005)
                if rng.random() < 0.1:
                    raise RuntimeError("boom")
        except (CapacityExceeded, RuntimeError):
            pass

    tasks = []
    for _ in range(300):
        tasks.append(asyncio.create_task(user()))
        if rng.random() < 0.5:
            await asyncio.sleep(0)
        if rng.random() < 0.3:
            victim = rng.choice(tasks)
            # Cancel now, or after the grant has been set but not yet resumed.
            asyncio.get_running_loop().call_soon(victim.cancel) if rng.random() < 0.5 else victim.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)
    assert not broken
    assert (limiter.pending, limiter.search_pending, limiter.active, limiter.search_active) == (0, 0, 0, 0)
    assert not limiter.waiters
