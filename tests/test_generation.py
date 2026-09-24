"""Generation fencing: the mechanism that keeps cancelled work out."""

from __future__ import annotations

import asyncio

import pytest

from voiceplatform.conversation.generation import Fence, GenerationManager


def test_begin_makes_previous_generations_stale():
    gen = GenerationManager("s1")
    gen.next_turn()
    first = gen.begin()
    second = gen.begin()
    assert gen.is_current(second)
    assert not gen.is_current(first)


def test_check_counts_stale_hits():
    gen = GenerationManager("s1")
    gen.next_turn()
    stale = gen.begin()
    gen.begin()
    assert gen.check(stale) is False
    assert gen.check(stale) is False
    assert gen.stale_drops == 2


async def test_cancel_stops_only_that_generation():
    gen = GenerationManager("s1")
    gen.next_turn()
    key = gen.begin()
    started = asyncio.Event()

    async def work():
        started.set()
        await asyncio.sleep(10)

    task = gen.spawn(work(), key)
    await started.wait()
    cancelled = await gen.cancel(key)
    assert cancelled == 1
    assert task.cancelled()
    assert gen.current is None


async def test_fence_is_false_after_a_new_generation():
    gen = GenerationManager("s1")
    gen.next_turn()
    fence = Fence(gen, gen.begin())
    assert fence() is True
    gen.begin()
    assert fence() is False


def test_spawn_before_begin_is_a_programming_error():
    gen = GenerationManager("s1")

    async def work():
        return None

    coro = work()
    with pytest.raises(RuntimeError):
        gen.spawn(coro)
    coro.close()


async def test_fenced_sink_drops_audio_of_a_dead_generation():
    """The gate sits at the write, not at the top of the loop."""
    from voiceplatform.conversation.engine import FencedSink
    from voiceplatform.conversation.sink import CollectingSink
    from voiceplatform.core.audio import AudioFrame
    import numpy as np

    gen = GenerationManager("s1")
    gen.next_turn()
    inner = CollectingSink()
    sink = FencedSink(inner, gen)
    key = gen.begin()
    frame = AudioFrame(samples=np.zeros(160, dtype=np.float32), sample_rate=16000)

    await sink.send_audio(frame, key)
    assert len(inner.audio) == 1

    gen.begin()  # the turn moved on
    await sink.send_audio(frame, key)
    assert len(inner.audio) == 1
    assert sink.dropped == 1


async def test_fenced_sink_still_delivers_control_after_cancellation():
    from voiceplatform.conversation.engine import FencedSink
    from voiceplatform.conversation.sink import CollectingSink
    from voiceplatform.media.transport.base import ControlMessage

    gen = GenerationManager("s1")
    gen.next_turn()
    inner = CollectingSink()
    sink = FencedSink(inner, gen)
    key = gen.begin()
    await gen.cancel(key)
    # playback_reset exists to be sent after the generation is dead.
    await sink.send_control(ControlMessage("playback_reset", {"generation_id": 1}))
    assert inner.control_types() == ["playback_reset"]


async def test_cancel_closes_the_fence_before_it_waits():
    """A task still unwinding must already be stale, or its output leaks."""
    gen = GenerationManager("s1")
    gen.next_turn()
    key = gen.begin()
    seen: list[bool] = []
    running = asyncio.Event()

    async def work():
        running.set()
        try:
            await asyncio.sleep(10)
        except asyncio.CancelledError:
            # Exactly the window the bug lived in: still running, already
            # cancelled, about to emit.
            seen.append(gen.is_current(key))
            raise

    gen.spawn(work(), key)
    await running.wait()
    await gen.cancel(key)
    assert seen == [False]
