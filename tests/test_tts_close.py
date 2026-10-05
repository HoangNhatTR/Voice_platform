"""Closing ZeroTTS (or its pool) mid-phrase is an error, never a finished phrase."""

from __future__ import annotations

import asyncio
import time

import numpy as np
import pytest

from voiceplatform.core.errors import ModelUnavailable
from voiceplatform.models.tts.pool import TtsPool
from voiceplatform.models.tts.zerotts import ZeroTtsEngine


class _Slow:
    sample_rate = 48000

    def list_voices(self):
        return ["maichi"]

    def synthesize_stream(self, text, voice=None, **kwargs):
        for _ in range(100):
            time.sleep(0.01)
            yield np.zeros((1, 3840), dtype=np.float32)


def _engine() -> ZeroTtsEngine:
    engine = ZeroTtsEngine(output_sample_rate=24000, max_pending_chunks=2)
    engine._tts = _Slow()
    engine.capabilities.voices = ("maichi",)
    engine.capabilities.native_sample_rate = 24000
    return engine


async def _cut_by_close(tts, close):
    got = []
    first = asyncio.Event()

    async def consume():
        async for chunk in tts.synthesize("một câu dài"):
            got.append(chunk)
            first.set()

    task = asyncio.create_task(consume())
    await asyncio.wait_for(first.wait(), 2)
    await close()
    with pytest.raises(ModelUnavailable, match="closing"):
        await asyncio.wait_for(task, 2)
    assert 0 < len(got) < 100


async def test_engine_close_mid_phrase_surfaces_as_unavailable():
    engine = _engine()
    await _cut_by_close(engine, engine.close)
    assert not engine._workers and not engine._lock.locked()


async def test_pool_close_mid_phrase_surfaces_as_unavailable():
    pool = TtsPool(_engine, pool_size=2)
    await pool.start()
    await _cut_by_close(pool, pool.close)
