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
        # One utterance is one script entry, however many times it is decoded:
        # the first decode claims it, finish() returns the same text.
        self._claimed: str | None = None
        self._finished = False

    async def push(self, frame: AudioFrame) -> Transcript | None:
        self._chunks.append(frame.samples)
        self._frames += 1
        if self._engine.partial_every_frames and (
            self._frames % self._engine.partial_every_frames == 0
        ):
            self._emitted_partial = True
            return Transcript(text=self._engine.peek_partial(self._frames), is_final=False)
        return None

    def _claim(self) -> str:
        if self._claimed is None:
            samples = np.concatenate(self._chunks) if self._chunks else np.zeros(0, np.float32)
            duration_ms = 1000.0 * samples.size / float(self._sample_rate)
            self._claimed = self._engine.next_text(duration_ms)
        return self._claimed

    async def decode_now(self) -> Transcript:
        """Whole-utterance decode at a pause, as the real bridge does."""
        if self._engine.endpoint_delay_ms:
            await asyncio.sleep(self._engine.endpoint_delay_ms / 1000.0)
        if not self._engine.endpoint_decode:
            return Transcript(text="", is_final=False)
        text = self._engine.endpoint_text(self._claim())
        return Transcript(text=text, is_final=False, confidence=1.0, language="vi")

    async def finish(self) -> Transcript:
        if self._engine.finish_delay_ms:
            await asyncio.sleep(self._engine.finish_delay_ms / 1000.0)
        self._engine.finals += 1
        self._finished = True
        return Transcript(text=self._claim(), is_final=True, confidence=1.0, language="vi")

    def accept_final(self, text: str) -> None:
        """The engine took the endpoint decode as this utterance's final."""
        self._finished = True

    async def close(self) -> None:
        # An utterance that never became a turn (a cough, a barge-in cut)
        # does not use up a script line — as before endpoint decodes existed.
        if self._claimed is not None and not self._finished:
            self._engine.unclaim()
            self._claimed = None
        self._chunks.clear()


class MockAsrEngine:
    name = "mock"

    def __init__(
        self,
        script: list[str] | None = None,
        finish_delay_ms: float = 0.0,
        partial_every_frames: int = 10,
        endpoint_decode: bool = True,
        endpoint_delay_ms: float = 0.0,
        endpoint_script: list[str] | None = None,
    ) -> None:
        self.capabilities = AsrCapabilities(streaming_partials=True, native_sample_rate=16000)
        self._script = list(script or [])
        self._index = 0
        self.finish_delay_ms = finish_delay_ms
        self.partial_every_frames = partial_every_frames
        self.endpoint_decode = endpoint_decode
        self.endpoint_delay_ms = endpoint_delay_ms
        # What a decode AT THE PAUSE hears, when a test needs it to differ
        # from the final (the user kept talking after the pause). Consumed in
        # order, one per endpoint decode.
        self._endpoint_script = list(endpoint_script or [])
        self.finals = 0
        self.endpoint_decodes = 0

    def unclaim(self) -> None:
        if self._script and self._index:
            self._index -= 1

    def endpoint_text(self, final: str) -> str:
        self.endpoint_decodes += 1
        if self._endpoint_script:
            return self._endpoint_script.pop(0)
        return final

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
