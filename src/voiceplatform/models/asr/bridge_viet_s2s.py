"""ASR engines borrowed from speech2speech (PhoWhisper / Gipformer / Parakeet).

Those backends are utterance-level: they take a whole clip and return text.
This adapter gives them the streaming contract by buffering frames and, if the
config asks for it, decoding the speech prefix every N frames so the trace
still has a real `asr_first_partial`.
"""

from __future__ import annotations

import asyncio

import numpy as np

from ...core.audio import AudioFrame, resample_linear
from ..base import AsrCapabilities, Transcript
from ..bridge import load_viet_s2s, split_options


class _BridgeAsrStream:
    def __init__(self, engine: "BridgeAsrEngine", sample_rate: int, language: str | None) -> None:
        self._engine = engine
        self._sample_rate = sample_rate
        self._language = language
        self._chunks: list[np.ndarray] = []
        self._frames = 0
        self._partial_task: asyncio.Task | None = None
        self._pending_partial: Transcript | None = None

    def _audio(self) -> np.ndarray:
        if not self._chunks:
            return np.zeros(0, dtype=np.float32)
        audio = np.concatenate(self._chunks)
        native = self._engine.capabilities.native_sample_rate
        if self._sample_rate != native:
            audio = resample_linear(audio, self._sample_rate, native)
        return audio

    async def push(self, frame: AudioFrame) -> Transcript | None:
        self._chunks.append(frame.samples)
        self._frames += 1
        every = self._engine.partial_every_frames
        if not every or self._frames % every:
            return self._take_pending()
        if self._partial_task and not self._partial_task.done():
            # A decode is still running; never queue a second one behind it.
            return self._take_pending()
        audio = self._audio()
        self._partial_task = asyncio.create_task(self._decode(audio, partial=True))
        return self._take_pending()

    def _take_pending(self) -> Transcript | None:
        if self._partial_task is not None and self._partial_task.done():
            try:
                result = self._partial_task.result()
            except Exception:
                result = None
            self._partial_task = None
            return result
        return None

    async def _decode(self, audio: np.ndarray, *, partial: bool) -> Transcript | None:
        native = self._engine.capabilities.native_sample_rate
        out = await self._engine.backend.transcribe(audio, native, partial=partial)
        text = getattr(out, "text", "") or ""
        if not text.strip():
            return None
        return Transcript(
            text=text,
            is_final=not partial,
            confidence=getattr(out, "confidence", None),
            language=self._language,
        )

    async def finish(self) -> Transcript:
        if self._partial_task and not self._partial_task.done():
            self._partial_task.cancel()
        audio = self._audio()
        if audio.size == 0:
            return Transcript(text="", is_final=True, language=self._language)
        result = await self._decode(audio, partial=False)
        return result or Transcript(text="", is_final=True, language=self._language)

    async def close(self) -> None:
        if self._partial_task and not self._partial_task.done():
            self._partial_task.cancel()
        self._chunks.clear()


class BridgeAsrEngine:
    def __init__(self, backend: str = "phowhisper", **options) -> None:
        root, opts = split_options(options)
        self.name = f"s2s:{backend}"
        self.partial_every_frames = int(opts.pop("partial_every_frames", 0))
        module = load_viet_s2s(root)
        from viet_s2s.backends import create_asr
        from viet_s2s.config import ASRConfig

        cfg = ASRConfig(backend=backend, **opts)
        self.backend = create_asr(cfg)
        self.capabilities = AsrCapabilities(
            streaming_partials=self.partial_every_frames > 0,
            languages=(cfg.language,),
            native_sample_rate=16000,
        )
        self._loaded = False
        self._module = module

    async def start(self) -> None:
        if not self._loaded:
            await self.backend.load()
            self._loaded = True

    async def open_stream(self, *, sample_rate: int, language: str | None = None) -> _BridgeAsrStream:
        await self.start()
        return _BridgeAsrStream(self, sample_rate, language)

    async def close(self) -> None:
        close = getattr(self.backend, "close", None)
        if close is not None:
            await close()
        self._loaded = False
