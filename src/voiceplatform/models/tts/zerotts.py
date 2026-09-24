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
from collections.abc import AsyncIterator
from typing import Any

import numpy as np

from ...core.audio import resample_linear
from ...core.errors import ModelUnavailable
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

    def __init__(
        self,
        model_id: str = _DEFAULT_REPO,
        voice: str = _DEFAULT_VOICE,
        expected_rtf: float = _MEASURED_RTF,
        local_files_only: bool = False,
        cache_dir: str | None = None,
        output_sample_rate: int | None = None,
        **generation: Any,
    ) -> None:
        self.model_id = model_id
        self.voice = voice
        # Model sinh 48 kHz. Hạ về `audio.output_sample_rate` để cái nút vặn đó
        # có cùng một nghĩa với MỌI talker — đặt nó thành 48000 thì giữ nguyên
        # bản gốc. Engine tự quyết một mình là cách nút vặn trở thành trang trí.
        self.output_sample_rate = output_sample_rate
        self.source_sample_rate = 48000
        self.local_files_only = local_files_only
        self.cache_dir = cache_dir
        # cfg_scale, audio_temperature, max_frames... đi thẳng vào synthesize_stream
        self.generation = generation
        self._tts: Any = None
        # Một model, một luồng sinh. Hai lượt chồng nhau sẽ trộn frame của nhau.
        self._lock = asyncio.Lock()
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
                cache_dir=self.cache_dir,
                local_files_only=self.local_files_only,
            )
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

    async def synthesize(
        self, text: str, *, voice: str | None = None
    ) -> AsyncIterator[SpeechChunk]:
        await self.start()
        name = self._resolve(voice)
        first = True
        async with self._lock:
            stream = self._tts.synthesize_stream(text, voice=name, **self.generation)
            try:
                while True:
                    # Một lần nhảy thread cho mỗi chunk. `synthesize_stream` là
                    # generator ĐỒNG BỘ và onnxruntime giữ GIL trong lúc chạy,
                    # nên kéo nó thẳng trong vòng lặp sự kiện là đóng băng cả
                    # đường audio vào.
                    block = await asyncio.to_thread(next, stream, None)
                    if block is None:
                        break
                    samples = np.asarray(block, dtype=np.float32).reshape(-1)
                    if samples.size == 0:
                        continue
                    target = self.capabilities.native_sample_rate
                    if target != self.source_sample_rate:
                        samples = resample_linear(samples, self.source_sample_rate, target)
                    yield SpeechChunk(
                        samples=samples,
                        sample_rate=target,
                        text=text if first else "",
                    )
                    first = False
            finally:
                stream.close()

    async def close(self) -> None:
        self._tts = None
