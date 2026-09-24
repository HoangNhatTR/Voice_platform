"""Deterministic TTS for tests and the no-model demo.

Emits an obviously synthetic but clearly *audible* tone whose length tracks the
text. Audible is the point: the first build hummed at -27 dBFS around 140 Hz,
which laptop speakers reproduce as nothing at all, and "I cannot hear anything"
then looks identical to a broken pipeline. It has a moving pitch so a listener
can tell the assistant is talking, and never sounds like a real voice so nobody
mistakes the mock stack for a working one.

RTF is a knob, not a constant: at or above 1.0 the talker becomes the
bottleneck, which is what makes the scheduler's playout cushion observable.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

import numpy as np

from ..base import SpeechChunk, TtsCapabilities

_CHUNK_MS = 40.0
_MS_PER_CHAR = 55.0


class MockTtsEngine:
    name = "mock"

    def __init__(
        self,
        sample_rate: int = 24000,
        rtf: float = 0.35,
        first_audio_delay_ms: float = 60.0,
        ms_per_char: float = _MS_PER_CHAR,
    ) -> None:
        self.capabilities = TtsCapabilities(
            streaming=True,
            emotion_cues=False,
            voices=("mock-a", "mock-b"),
            native_sample_rate=sample_rate,
            expected_rtf=rtf,
        )
        self.sample_rate = sample_rate
        self.rtf = rtf
        self.first_audio_delay_ms = first_audio_delay_ms
        self.ms_per_char = ms_per_char
        self._phase = 0.0

    async def start(self) -> None:
        return None

    async def close(self) -> None:
        return None

    async def synthesize(self, text: str, *, voice: str | None = None) -> AsyncIterator[SpeechChunk]:
        clean = text.strip()
        if not clean:
            return
        total_ms = max(_CHUNK_MS, len(clean) * self.ms_per_char)
        chunk_samples = int(self.sample_rate * _CHUNK_MS / 1000.0)
        n_chunks = max(1, int(round(total_ms / _CHUNK_MS)))
        await asyncio.sleep(self.first_audio_delay_ms / 1000.0)
        base_hz = 220.0 if (voice or "mock-a").endswith("a") else 300.0
        for i in range(n_chunks):
            if i:
                await asyncio.sleep((_CHUNK_MS / 1000.0) * self.rtf)
            t = (np.arange(chunk_samples, dtype=np.float32) + self._phase) / self.sample_rate
            self._phase += chunk_samples
            # A little pitch movement per chunk, so it reads as speech rhythm
            # rather than a dial tone.
            hz = base_hz * (1.0 + 0.12 * np.sin(2 * np.pi * 1.7 * t[0]))
            wave = 0.30 * np.sin(2 * np.pi * hz * t) + 0.10 * np.sin(
                2 * np.pi * (hz * 2.0) * t
            )
            # Soft edges so the buzz does not click at chunk boundaries.
            ramp = min(64, chunk_samples // 4)
            if ramp:
                wave[:ramp] *= np.linspace(0.0, 1.0, ramp, dtype=np.float32)
                wave[-ramp:] *= np.linspace(1.0, 0.0, ramp, dtype=np.float32)
            yield SpeechChunk(
                samples=wave.astype(np.float32),
                sample_rate=self.sample_rate,
                text=clean if i == 0 else "",
            )
