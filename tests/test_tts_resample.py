"""ZeroTTS 48 kHz -> platform rate: band-limited, continuous across chunks.

`resample_linear` per chunk was plain decimation at 48 -> 24 kHz: a 15 kHz
component came back at 9 kHz at full level.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from voiceplatform.core.audio import BandLimitedResampler
from voiceplatform.models.tts.zerotts import ZeroTtsEngine


def _level_db(samples: np.ndarray) -> float:
    return 20 * math.log10(math.sqrt(2) * float(np.std(samples)) + 1e-12)


def _tone(freq: float, rate: int, seconds: float = 1.0) -> np.ndarray:
    return np.sin(2 * np.pi * freq * np.arange(int(rate * seconds)) / rate).astype(np.float32)


@pytest.mark.parametrize("freq,low,high", [
    (1000, -0.1, 0.1), (5000, -0.1, 0.1), (9000, -0.5, 0.1),   # speech band: untouched
    (12500, -300, -60), (15000, -300, -60), (20000, -300, -60),  # above 12 kHz: gone
])
def test_48k_to_24k_keeps_the_band_and_removes_what_would_alias(freq, low, high):
    out = BandLimitedResampler(48000, 24000).process(_tone(freq, 48000))
    assert low <= _level_db(out[2400:]) <= high


@pytest.mark.parametrize("dst", [24000, 22050, 16000])
def test_chunked_stream_equals_one_shot_so_seams_cannot_click(dst):
    rng = np.random.default_rng(7)
    signal = (0.5 * _tone(440, 48000, 2.0) + 0.1 * rng.standard_normal(96000)).astype(np.float32)
    whole = BandLimitedResampler(48000, dst).process(signal)
    streaming = BandLimitedResampler(48000, dst)
    parts, at = [], 0
    while at < signal.size:
        size = int(rng.integers(1, 5000))
        parts.append(streaming.process(signal[at:at + size]))
        at += size
    chunked = np.concatenate(parts)
    assert chunked.size == whole.size == math.ceil(signal.size * dst / 48000)
    assert np.max(np.abs(chunked - whole)) < 1e-6


class _Tone:
    """A ZeroTTS stand-in: 48 kHz, frame-sized chunks of one continuous signal."""

    sample_rate = 48000

    def __init__(self, signal: np.ndarray, chunk: int = 3840) -> None:
        self.signal, self.chunk = signal, chunk

    def list_voices(self):
        return ["maichi"]

    def synthesize_stream(self, text, voice=None, **kwargs):
        for at in range(0, self.signal.size, self.chunk):
            yield self.signal[None, at:at + self.chunk]


def _engine(voice) -> ZeroTtsEngine:
    engine = ZeroTtsEngine(output_sample_rate=24000)
    engine._tts = voice
    engine.capabilities.voices = ("maichi",)
    engine.capabilities.native_sample_rate = 24000
    return engine


async def test_zerotts_output_has_no_aliased_sibilant():
    engine = _engine(_Tone(_tone(15000, 48000)))
    out = np.concatenate([c.samples async for c in engine.synthesize("xin chào")])
    assert out.size == 24000
    assert _level_db(out[2400:]) < -60          # was 0 dB, at 9 kHz


async def test_zerotts_chunk_boundaries_are_seamless():
    signal = 0.5 * _tone(440, 48000)
    engine = _engine(_Tone(signal, chunk=3840))
    chunks = [c async for c in engine.synthesize("xin chào")]
    out = np.concatenate([c.samples for c in chunks])
    reference = BandLimitedResampler(48000, 24000).process(signal)
    assert len(chunks) > 5 and np.max(np.abs(out - reference)) < 1e-6
    # A click is a jump far above a 440 Hz sine's own step at 24 kHz.
    assert np.max(np.abs(np.diff(out[100:]))) < 0.07
