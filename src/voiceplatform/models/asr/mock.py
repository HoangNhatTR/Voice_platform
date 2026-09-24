"""Deterministic ASR for tests, CI and the no-model demo.

It is also the reference for how a *batch* engine satisfies a streaming
contract: buffer on push, decode on finish, emit coarse partials so the rest of
the pipeline still exercises its asr_first_partial path.
"""

from __future__ import annotations

import asyncio

import numpy as np

from ...core.audio import AudioFrame
from ..base import AsrCapabilities, Transcript


class MockAsrStream:
    def __init__(self, engine: "MockAsrEngine", sample_rate: int) -> None:
        self._engine = engine
        self._sample_rate = sample_rate
        self._chunks: list[np.ndarray] = []
        self._frames = 0
        self._emitted_partial = False

    async def push(self, frame: AudioFrame) -> Transcript | None:
        self._chunks.append(frame.samples)
        self._frames += 1
        if self._engine.partial_every_frames and (
            self._frames % self._engine.partial_every_frames == 0
        ):
            self._emitted_partial = True
            return Transcript(text=self._engine.peek_partial(self._frames), is_final=False)
        return None

    async def finish(self) -> Transcript:
        if self._engine.finish_delay_ms:
            await asyncio.sleep(self._engine.finish_delay_ms / 1000.0)
        samples = np.concatenate(self._chunks) if self._chunks else np.zeros(0, np.float32)
        duration_ms = 1000.0 * samples.size / float(self._sample_rate)
        text = self._engine.next_text(duration_ms)
        return Transcript(text=text, is_final=True, confidence=1.0, language="vi")

    async def close(self) -> None:
        self._chunks.clear()


class MockAsrEngine:
    name = "mock"

    def __init__(
        self,
        script: list[str] | None = None,
        finish_delay_ms: float = 0.0,
        partial_every_frames: int = 10,
    ) -> None:
        self.capabilities = AsrCapabilities(streaming_partials=True, native_sample_rate=16000)
        self._script = list(script or [])
        self._index = 0
        self.finish_delay_ms = finish_delay_ms
        self.partial_every_frames = partial_every_frames

    async def start(self) -> None:
        return None

    async def open_stream(self, *, sample_rate: int, language: str | None = None) -> MockAsrStream:
        return MockAsrStream(self, sample_rate)

    async def close(self) -> None:
        return None

    def peek_partial(self, frames: int) -> str:
        if self._script:
            target = self._script[self._index % len(self._script)]
            words = target.split()
            take = max(1, min(len(words), frames // 10))
            return " ".join(words[:take])
        return "..."

    def next_text(self, duration_ms: float) -> str:
        if self._script:
            text = self._script[self._index % len(self._script)]
            self._index += 1
            return text
        return f"[mock utterance {duration_ms:.0f} ms]"
