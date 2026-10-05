"""ZeroTTS — talker tiếng Việt zero-shot, ONNX, chạy CPU.

Khác với các engine trong `bridge_viet_s2s.py`: cái này KHÔNG mượn từ
`speech2speech`. Nó là gói `zerotts` trên PyPI, chỉ phụ thuộc numpy và
onnxruntime — hai thứ đã có sẵn — nên không kéo theo xung đột dependency nào
và không cần venv riêng như coqui/viXTTS.

Đo trên máy này (CPU, 2,8 giây tiếng): **tiếng đầu 118 ms**, RTF 0,71. Đổi lại
tốc độ tổng hợp cả cụm chậm hơn VieNeu Nano (RTF 0,45) nhưng tiếng đầu nhanh
hơn bốn lần — mà TTFA mới là con số quyết định cảm giác, không phải tổng thời
gian. Ra 48 kHz.

Tám giọng preset, không nhân bản giọng được ở bản này.
"""

from __future__ import annotations

import asyncio
import threading
from concurrent.futures import TimeoutError as ThreadTimeout
from collections.abc import AsyncIterator
from typing import Any

import numpy as np

from ...core.audio import BandLimitedResampler
from ...core.errors import ModelTimeout, ModelUnavailable
from ...core.limits import WorkLimiter
from ...core.clock import now_ms
from ...core.events import EventType
from ...observability.probe import current_probe
from ...observability.logging import get_logger
from ..base import SpeechChunk, TtsCapabilities

log = get_logger("zerotts")

_DEFAULT_REPO = "zeroweight-ai/ZeroTTS"
_DEFAULT_VOICE = "maichi"
# Đo được 0,708 trên máy này; khai đúng số đo chứ không khai số mong muốn, vì
# scheduler dùng nó để quyết định đệm phát sâu bao nhiêu.
_MEASURED_RTF = 0.71


class ZeroTtsEngine:
    name = "zerotts"
    instrumented = True

    def __init__(
        self,
        model_id: str = _DEFAULT_REPO,
        revision: str | None = None,
        voice: str = _DEFAULT_VOICE,
        expected_rtf: float = _MEASURED_RTF,
        local_files_only: bool = False,
        cache_dir: str | None = None,
        output_sample_rate: int | None = None,
        max_pending_chunks: int = 8,
        close_timeout_s: float = 5.0,
        max_queue: int = 16,
        intra_op_num_threads: int = 4,
        codec_intra_op_num_threads: int | None = None,
        onnx_allow_spinning: bool | None = None,
        onnx_spin_duration_us: int | None = None,
        **generation: Any,
    ) -> None:
        self.model_id = model_id
        self.revision = revision
        self.voice = voice
        # Model sinh 48 kHz. Hạ về `audio.output_sample_rate` để cái nút vặn đó
        # có cùng một nghĩa với MỌI talker — đặt nó thành 48000 thì giữ nguyên
        # bản gốc. Engine tự quyết một mình là cách nút vặn trở thành trang trí.
        self.output_sample_rate = output_sample_rate
        self.source_sample_rate = 48000
        self.local_files_only = local_files_only
        self.cache_dir = cache_dir
        if intra_op_num_threads < 1 or (codec_intra_op_num_threads is not None and codec_intra_op_num_threads < 1):
            raise ValueError("TTS threads must be positive")
        self.runtime = {
            "intra_op_num_threads": intra_op_num_threads,
            "codec_intra_op_num_threads": codec_intra_op_num_threads,
            # The adapter performs one explicit warmup below. The package's
            # constructor otherwise performs the same expensive warmup twice.
            "warmup": False,
        }
        if onnx_allow_spinning is not None and type(onnx_allow_spinning) is not bool:
            raise ValueError("ONNX spinning must be boolean")
        self.onnx_allow_spinning = onnx_allow_spinning
        if onnx_spin_duration_us is not None and (type(onnx_spin_duration_us) is not int or onnx_spin_duration_us < 0):
            raise ValueError("ONNX spin duration must be a non-negative integer")
        self.onnx_spin_duration_us = onnx_spin_duration_us
        # cfg_scale, audio_temperature, max_frames... đi thẳng vào synthesize_stream
        self.generation = generation
        self._tts: Any = None
        # One native generator per wrapper. A checked CPU pool may share the
        # immutable graphs while retaining independent generator/codec state.
        self._lock = asyncio.Lock()
        self._workers: dict[asyncio.Future, tuple] = {}
        self.limiter = WorkLimiter(1, max_queue)
        self._closing = False
        if max_pending_chunks < 1 or close_timeout_s <= 0:
            raise ValueError("invalid TTS worker limits")
        self.max_pending_chunks = max_pending_chunks
        self.close_timeout_s = close_timeout_s
        self.capabilities = TtsCapabilities(
            streaming=True,
            # ZeroTTS không khai token cảm xúc nào, nên cue phải bị BỎ trước khi
            # tới đây. Khai True ở đây là cách "[cười]" bị đọc thành chữ.
            emotion_cues=False,
            voices=(),
            native_sample_rate=48000,
            expected_rtf=expected_rtf,
        )

    async def start(self) -> None:
        if self._closing:
            raise ModelUnavailable("TTS is closing")
        if self._tts is not None:
            return
        try:
            from zerotts import ZeroTTS
        except ImportError as exc:  # pragma: no cover - env specific
            raise ModelUnavailable(
                "cần gói `zerotts` (pip install zerotts). Nó chỉ phụ thuộc numpy "
                "và onnxruntime nên cài chung venv được."
            ) from exc

        def _load() -> Any:
            engine = ZeroTTS.from_pretrained(
                self.model_id,
                revision=self.revision,
                cache_dir=self.cache_dir,
                local_files_only=self.local_files_only,
                **self.runtime,
            )
            if self.onnx_allow_spinning is not None or self.onnx_spin_duration_us is not None:
                _set_session_spinning(engine, self.onnx_allow_spinning, self.onnx_spin_duration_us)
            # Trả giá dựng đồ thị một lần ở đây thay vì trong lượt nói đầu tiên.
            warmup = getattr(engine, "warmup", None)
            if warmup is not None:
                try:
                    warmup()
                except Exception as exc:  # pragma: no cover - backend dependent
                    log.warning("warmup thất bại, bỏ qua: %s", exc)
            return engine

        self._tts = await asyncio.to_thread(_load)
        voices = tuple(self._tts.list_voices())
        self.source_sample_rate = int(getattr(self._tts, "sample_rate", 48000))
        delivered = self.output_sample_rate or self.source_sample_rate
        self.capabilities.voices = voices
        self.capabilities.native_sample_rate = delivered
        if voices and self.voice not in voices:
            log.warning("giọng %r không có trong ZeroTTS; dùng %r", self.voice, voices[0])
            self.voice = voices[0]
        if delivered != self.source_sample_rate:
            log.info(
                "ZeroTTS sinh %d Hz, phát ra %d Hz theo audio.output_sample_rate",
                self.source_sample_rate, delivered,
            )
        log.info("ZeroTTS sẵn sàng, %d giọng: %s", len(voices), ", ".join(voices))

    def _resolve(self, voice: str | None) -> str:
        wanted = voice or self.voice
        available = self.capabilities.voices
        if available and wanted not in available:
            # Cùng lý do bản Nano từ chối tên lạ: im lặng đổi sang giọng khác
            # là một khác biệt người nghe thấy ngay mà log không nói gì.
            log.warning("bỏ qua giọng %r: ZeroTTS không có; dùng %r", wanted, self.voice)
            return self.voice
        return wanted

    async def synthesize(self, text: str, *, voice: str | None = None) -> AsyncIterator[SpeechChunk]:
        from contextlib import aclosing
        async with self.limiter.slot(), aclosing(self._synthesize(text, voice=voice)) as stream:
            async for chunk in stream:
                yield chunk

    async def _synthesize(
        self, text: str, *, voice: str | None = None
    ) -> AsyncIterator[SpeechChunk]:
        """Bơm generator đồng bộ qua MỘT thread, dừng bằng cờ hợp tác.

        Bản đầu gọi `asyncio.to_thread(next, stream)` cho từng chunk. Nó chạy
        đúng cho tới lần ngắt lời đầu tiên: huỷ một `to_thread` KHÔNG dừng được
        thread, nên `finally` đóng generator trong khi thread vẫn đang ở trong
        `next()` — `ValueError: generator already executing`, và thread thì
        tiếp tục sinh audio cho một lượt đã chết.

        Nên generator sống trọn đời trong đúng một thread, thread đó tự đóng nó,
        và việc huỷ chỉ bật một `threading.Event` để nó dừng ở biên chunk. Khoá
        được nhả bằng done-callback của chính thread, không phải khi coroutine
        thoát — nếu không, lượt kế tiếp có thể chạm vào model trong lúc thread
        cũ còn đang sinh nốt một chunk.
        """
        await self.start()
        name = self._resolve(voice)
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[Any] = asyncio.Queue(maxsize=self.max_pending_chunks)
        stop = threading.Event()
        done = object()

        def publish(item: Any) -> bool:
            if stop.is_set():
                return False
            future = asyncio.run_coroutine_threadsafe(queue.put(item), loop)
            while not stop.is_set():
                try:
                    future.result(timeout=0.1)
                    return True
                except ThreadTimeout:
                    continue
            future.cancel()
            return False

        probe = current_probe()
        lock_at = now_ms()
        await self._lock.acquire()
        if probe:
            probe.mark(EventType.TTS_LOCK_ACQUIRED, lock_wait_ms=now_ms()-lock_at)
        owns_lock = True
        first = True
        try:
            if self._closing or self._tts is None:
                raise ModelUnavailable("TTS is closing")
            model = self._tts  # A worker retains its model until it actually exits.

            def pump() -> None:
                stream = None
                error = None
                compute_ms = 0.0
                samples_count = 0
                chunks_count = 0
                if probe:
                    probe.mark(EventType.MODEL_INFERENCE_START)
                try:
                    at = now_ms()
                    stream = model.synthesize_stream(text, voice=name, **self.generation)
                    compute_ms += now_ms()-at
                    iterator = iter(stream)
                    while not stop.is_set():
                        at = now_ms()
                        try:
                            block = next(iterator)
                        except StopIteration:
                            compute_ms += now_ms()-at
                            break
                        compute_ms += now_ms()-at
                        samples_count += np.asarray(block).size
                        if probe and chunks_count == 0:
                            probe.mark(EventType.TTS_CHUNK_READY)
                        chunks_count += 1
                        if not publish(block):
                            break
                except BaseException as exc:
                    error = exc
                finally:
                    if stream is not None:
                        try:
                            close = getattr(stream, "close", None)
                            if close is not None:
                                close()
                        except BaseException as exc:
                            error = error or exc
                    if probe:
                        audio_ms = samples_count*1000.0/self.source_sample_rate
                        probe.mark(EventType.MODEL_INFERENCE_END, compute_ms=compute_ms, audio_ms=audio_ms,
                                   rtf=compute_ms/audio_ms if audio_ms else None, chunks=chunks_count,
                                   outcome="error" if error else "cancelled" if stop.is_set() else "complete")
                    if error is not None:
                        publish(error)
                    publish(done)

            worker = loop.run_in_executor(None, pump)
            self._workers[worker] = (stop, queue, done)

            def finished(future: asyncio.Future) -> None:
                self._workers.pop(future, None)
                self._lock.release()
                if not future.cancelled():
                    future.exception()

            worker.add_done_callback(finished)
            owns_lock = False
            target = self.capabilities.native_sample_rate
            # One filter per utterance, state carried across chunks. Linear
            # per-chunk resampling was plain decimation at 48 -> 24 kHz and
            # folded the 12-24 kHz band (sibilants, breath) back in as noise.
            resampler = (BandLimitedResampler(self.source_sample_rate, target)
                         if target != self.source_sample_rate else None)
            while True:
                item = await queue.get()
                if item is done:
                    break
                if isinstance(item, BaseException):
                    raise item
                samples = np.asarray(item, dtype=np.float32).reshape(-1)
                if resampler is not None:
                    samples = resampler.process(samples)
                if samples.size == 0:
                    continue
                yield SpeechChunk(
                    samples=samples,
                    sample_rate=target,
                    text=text if first else "",
                )
                first = False
        finally:
            stop.set()
            if owns_lock:
                self._lock.release()

    async def close(self) -> None:
        self._closing = True
        workers = list(self._workers)
        for stop, queue, _ in self._workers.values():
            stop.set()
            if queue.full():
                queue.get_nowait()
            # An error, not `done`: a phrase cut by shutdown must not reach
            # the consumer as a complete one (audio_end, TTS_PHRASE_COMPLETE).
            queue.put_nowait(ModelUnavailable("TTS is closing"))
        if workers:
            _, pending = await asyncio.wait(workers, timeout=self.close_timeout_s)
            if pending:
                # Python cannot kill a native inference thread. Keep ownership
                # and surface the failure rather than unloading under its feet.
                raise ModelTimeout("TTS worker did not stop before close deadline")
        self._tts = None


def _set_session_spinning(engine, allow: bool | None, duration_us: int | None = None) -> None:
    """Configure owned ONNX graphs before warmup, without a global patch.

    ZeroTTS's installed API exposes thread counts but not SessionOptions.
    Re-create one graph at a time with its original providers/options and the
    documented spinning setting. The package's five session attributes and
    ORT's retained model path are checked; incompatible versions fail closed.
    No weights, graph math, sampling or shared ASR sessions are changed.
    """
    import onnxruntime as ort
    from pathlib import Path

    if duration_us is not None:
        # SessionOptions accepts arbitrary strings even on older builds. Verify
        # the loaded native binary knows both duration keys before using them.
        keys = [f"session.{kind}_op.spin_duration_us".encode() for kind in ("intra", "inter")]
        libraries = list((Path(ort.__file__).parent / "capi").glob("onnxruntime_pybind11_state*.so"))
        if not libraries or not all(key in libraries[0].read_bytes() for key in keys):
            raise ModelUnavailable("installed ONNX Runtime does not support bounded thread spinning")

    holders = [(engine, name) for name in (
        "prefix_step_sess", "local_frame_decode_sess", "text_encoder_sess",
    )] + [(engine.codec, name) for name in ("_decode_full_sess", "_decode_step_sess")]
    for holder, name in holders:
        session = getattr(holder, name, None)
        path = getattr(session, "_model_path", None)
        if not isinstance(path, str) or not Path(path).is_file():
            raise ModelUnavailable(f"ZeroTTS ONNX session API incompatible: {name}")
        options = session.get_session_options()
        for kind in ("intra", "inter"):
            if allow is not None:
                options.add_session_config_entry(f"session.{kind}_op.allow_spinning", "1" if allow else "0")
            if duration_us is not None:
                options.add_session_config_entry(f"session.{kind}_op.spin_duration_us", str(duration_us))
        providers = session.get_providers()
        provider_options = session.get_provider_options()
        replacement = ort.InferenceSession(path, sess_options=options, providers=providers,
                                           provider_options=[provider_options.get(p, {}) for p in providers])
        setattr(holder, name, replacement)
