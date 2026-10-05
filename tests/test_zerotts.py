"""Adapter ZeroTTS: đường ngắt lời là chỗ duy nhất đáng test ở đây.

Không nạp model thật — bài test dựng một generator ĐỒNG BỘ chậm y như
`synthesize_stream`, vì cái hỏng nằm ở chỗ ghép generator đồng bộ vào vòng lặp
sự kiện, không nằm trong model.
"""

from __future__ import annotations

import asyncio
import threading
import time

import numpy as np
import pytest

from voiceplatform.core.errors import ModelUnavailable
from voiceplatform.models.tts.zerotts import ZeroTtsEngine


class _SlowVoice:
    """Mô phỏng ZeroTTS: generator đồng bộ, mỗi chunk tốn thời gian thật."""

    sample_rate = 48000

    def __init__(self, chunks: int = 6, chunk_ms: float = 40.0) -> None:
        self.chunks = chunks
        self.chunk_ms = chunk_ms
        self.closed = threading.Event()
        self.produced = 0
        self.concurrent = 0
        self.max_concurrent = 0
        self._guard = threading.Lock()

    def list_voices(self):
        return ["maichi", "giahuy"]

    def warmup(self):
        return None

    def synthesize_stream(self, text, voice=None, **kwargs):
        with self._guard:
            self.concurrent += 1
            self.max_concurrent = max(self.max_concurrent, self.concurrent)
        try:
            for _ in range(self.chunks):
                time.sleep(self.chunk_ms / 1000.0)
                self.produced += 1
                yield np.full((1, 480), 0.1, dtype=np.float32)
        finally:
            with self._guard:
                self.concurrent -= 1
            self.closed.set()


async def _engine(monkeypatch, voice: _SlowVoice) -> ZeroTtsEngine:
    engine = ZeroTtsEngine(output_sample_rate=24000)
    engine._tts = voice
    engine.source_sample_rate = voice.sample_rate
    engine.capabilities.voices = tuple(voice.list_voices())
    engine.capabilities.native_sample_rate = 24000
    engine.voice = "maichi"
    return engine


async def test_audio_comes_back_resampled_to_the_platform_rate(monkeypatch):
    voice = _SlowVoice(chunks=3, chunk_ms=5)
    engine = await _engine(monkeypatch, voice)
    chunks = [c async for c in engine.synthesize("xin chào")]
    assert len(chunks) == 3
    assert all(c.sample_rate == 24000 for c in chunks)
    # 480 mẫu ở 48 kHz thành 240 ở 24 kHz.
    assert chunks[0].samples.size == 240


async def test_abandoning_mid_stream_does_not_break_the_generator(monkeypatch):
    """Đây là lỗi thật đã gặp: ngắt lời giữa chừng ném
    `ValueError: generator already executing`, vì generator bị đóng trong khi
    một thread vẫn đang ở trong nó."""
    voice = _SlowVoice(chunks=20, chunk_ms=20)
    engine = await _engine(monkeypatch, voice)

    stream = engine.synthesize("một câu dài")
    assert (await stream.__anext__()).samples.size > 0
    await stream.aclose()          # đúng thứ barge-in làm

    # Thread phải tự dừng ở biên chunk và tự đóng generator.
    assert voice.closed.wait(timeout=3.0), "generator không bao giờ được đóng"
    assert voice.produced < voice.chunks, "thread chạy hết dù đã bị bỏ"


async def test_cancelling_while_a_chunk_is_in_flight_is_clean(monkeypatch):
    """Hình dạng THẬT của một lần ngắt lời.

    `gen.cancel()` ném CancelledError vào bất cứ chỗ await nào đang chạy — và
    chỗ đó thường nằm GIỮA một chunk, không phải ở điểm yield. Bản đầu đóng
    generator ngay tại đó trong khi thread vẫn ở trong nó:
    `ValueError: generator already executing`.
    """
    voice = _SlowVoice(chunks=20, chunk_ms=50)
    engine = await _engine(monkeypatch, voice)
    leaked: list[BaseException] = []

    async def consume() -> None:
        try:
            async for _ in engine.synthesize("một câu rất dài"):
                pass
        except asyncio.CancelledError:
            raise
        except BaseException as exc:   # noqa: BLE001 - đúng thứ đang bắt
            leaked.append(exc)

    task = asyncio.create_task(consume())
    await asyncio.sleep(0.12)          # đang ở giữa một chunk, không ở yield
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert leaked == [], f"huỷ giữa chunk làm rò lỗi: {leaked}"
    assert voice.closed.wait(timeout=3.0), "generator không bao giờ được đóng"


async def test_the_next_turn_never_overlaps_the_abandoned_one(monkeypatch):
    """Khoá phải do chính thread nhả, không phải do coroutine thoát.

    Nhả theo coroutine thì lượt kế tiếp bắt đầu trong lúc thread cũ còn sinh
    nốt một chunk, và hai thread cùng chạm vào một model.
    """
    voice = _SlowVoice(chunks=20, chunk_ms=20)
    engine = await _engine(monkeypatch, voice)

    stream = engine.synthesize("câu bị bỏ")
    await stream.__anext__()
    await stream.aclose()

    chunks = [c async for c in engine.synthesize("câu kế tiếp")]
    assert chunks
    assert voice.max_concurrent == 1, "hai lượt chạm model cùng lúc"


async def test_an_error_inside_the_thread_reaches_the_caller(monkeypatch):
    class _Broken(_SlowVoice):
        def synthesize_stream(self, text, voice=None, **kwargs):
            yield np.zeros((1, 480), dtype=np.float32)
            raise RuntimeError("codec hỏng")

    engine = await _engine(monkeypatch, _Broken())
    with pytest.raises(RuntimeError, match="codec hỏng"):
        async for _ in engine.synthesize("xin chào"):
            pass


async def test_an_unknown_voice_falls_back_loudly_not_silently(monkeypatch):
    voice = _SlowVoice(chunks=1, chunk_ms=1)
    engine = await _engine(monkeypatch, voice)
    assert engine._resolve("khong-co") == "maichi"
    assert engine._resolve("giahuy") == "giahuy"


async def test_initialization_failure_never_leaves_the_consumer_waiting(monkeypatch):
    class Broken(_SlowVoice):
        def synthesize_stream(self, *args, **kwargs):
            raise RuntimeError("init failed")
    engine = await _engine(monkeypatch, Broken())
    with pytest.raises(RuntimeError, match="init failed"):
        await asyncio.wait_for(anext(engine.synthesize("hello")), .5)
    await engine.close()
    assert not engine._lock.locked() and not engine._workers


async def test_stream_close_failure_reaches_consumer_and_releases_lock(monkeypatch):
    class Stream:
        def __iter__(self):
            return iter([np.zeros(480, dtype=np.float32)])
        def close(self):
            raise RuntimeError("close failed")
    class Broken(_SlowVoice):
        def synthesize_stream(self, *args, **kwargs):
            return Stream()
    engine = await _engine(monkeypatch, Broken())
    async def consume():
        return [c async for c in engine.synthesize("hello")]
    with pytest.raises(RuntimeError, match="close failed"):
        await asyncio.wait_for(consume(), .5)
    await engine.close()
    assert not engine._workers and not engine._lock.locked()


async def test_close_stops_the_native_worker_and_wakes_active_consumer(monkeypatch):
    voice = _SlowVoice(chunks=100, chunk_ms=10)
    engine = await _engine(monkeypatch, voice)
    first = asyncio.Event()
    async def consume():
        async for _ in engine.synthesize("long"):
            first.set()
    task = asyncio.create_task(consume())
    await asyncio.wait_for(first.wait(), 1)
    await engine.close()
    # Woken with an error: a phrase cut by shutdown is not a complete phrase.
    with pytest.raises(ModelUnavailable, match="closing"):
        await asyncio.wait_for(task, 1)
    assert voice.closed.is_set()
    assert not engine._workers and not engine._lock.locked()
