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
from ...core.errors import ModelTimeout, ModelUnavailable
from ...core.limits import WorkLimiter
from ...core.events import EventType
from ...core.clock import now_ms
from ...observability.probe import current_probe, observing
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
        self._workers: set[asyncio.Task] = set()
        self.probe = current_probe()

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

    async def decode_now(self) -> Transcript:
        """Decode everything pushed so far, now, and return it.

        The endpoint decision needs the text of the WHOLE utterance at the
        moment the speaker pauses; the last periodic partial is up to
        `partial_every_frames` old plus its own decode time. The backend is
        an offline transducer, so this is exactly what `finish()` would say
        about the same audio.
        """
        audio = self._audio()
        if audio.size == 0:
            return Transcript(text="", is_final=False, language=self._language)
        result = await self._decode(audio, partial=True, operation="endpoint")
        return result or Transcript(text="", is_final=False, language=self._language)

    async def _decode(self, audio: np.ndarray, *, partial: bool, operation: str | None = None) -> Transcript | None:
        native = self._engine.capabilities.native_sample_rate
        started = False
        parent = current_probe() or self.probe
        probe = parent.child(operation=operation or ("partial" if partial else "final")) if parent else None
        async def decode():
            nonlocal started
            with observing(probe):
                return await run_decode()
        async def run_decode():
            nonlocal started
            async with self._engine.limiter.slot():
                if self._engine._closing:
                    raise ModelUnavailable("ASR is closing")
                started = True
                at = now_ms()
                if probe:
                    probe.mark(EventType.MODEL_INFERENCE_START, audio_ms=audio.size*1000.0/native)
                outcome = "complete"
                try:
                    return await self._engine.backend.transcribe(audio, native, partial=partial)
                except BaseException:
                    outcome = "error"
                    raise
                finally:
                    if probe:
                        probe.mark(EventType.MODEL_INFERENCE_END, compute_ms=now_ms()-at, outcome=outcome)
        # Cancelling an await on to_thread does not stop native inference.
        # Shield its owner and retain it until the backend really returns.
        worker = asyncio.create_task(decode(), name="asr-decode")
        self._workers.add(worker)
        self._engine._workers.add(worker)
        worker.add_done_callback(self._workers.discard)
        worker.add_done_callback(self._engine._workers.discard)
        worker.add_done_callback(lambda task: task.exception() if not task.cancelled() else None)
        try:
            async with asyncio.timeout(self._engine.decode_timeout_s):
                out = await asyncio.shield(worker)
        except BaseException:
            if not started:
                worker.cancel()
                await asyncio.gather(worker, return_exceptions=True)
            raise
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
            await asyncio.gather(self._partial_task, return_exceptions=True)
        audio = self._audio()
        if audio.size == 0:
            return Transcript(text="", is_final=True, language=self._language)
        result = await self._decode(audio, partial=False)
        return result or Transcript(text="", is_final=True, language=self._language)

    async def close(self) -> None:
        if self._partial_task and not self._partial_task.done():
            self._partial_task.cancel()
        if self._partial_task is not None:
            await asyncio.gather(self._partial_task, return_exceptions=True)
        if self._workers:
            _, pending = await asyncio.wait(self._workers, timeout=self._engine.close_timeout_s)
            if pending:
                raise ModelTimeout("ASR stream inference did not stop before close deadline")
        self._chunks.clear()


class BridgeAsrEngine:
    def __init__(self, backend: str = "phowhisper", **options) -> None:
        root, opts = split_options(options)
        self.name = f"s2s:{backend}"
        self.limiter = WorkLimiter(int(opts.pop("max_parallel", 3)), int(opts.pop("max_queue", 8)))
        self.decode_timeout_s = float(opts.pop("decode_timeout_s", 30.0))
        self.close_timeout_s = float(opts.pop("close_timeout_s", 5.0))
        self._workers: set[asyncio.Task] = set()
        self._closing = False
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
        if self._closing:
            raise ModelUnavailable("ASR is closing")
        if not self._loaded:
            await self.backend.load()
            self._loaded = True

    async def open_stream(self, *, sample_rate: int, language: str | None = None) -> _BridgeAsrStream:
        await self.start()
        return _BridgeAsrStream(self, sample_rate, language)

    async def close(self) -> None:
        self._closing = True
        if self._workers:
            _, pending = await asyncio.wait(self._workers, timeout=self.close_timeout_s)
            if pending:
                raise ModelTimeout("ASR inference did not stop before model close deadline")
        close = getattr(self.backend, "close", None)
        if close is not None:
            await close()
        self._loaded = False
