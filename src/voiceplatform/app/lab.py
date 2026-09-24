"""Bàn thử model: chọn engine đang chạy, và thử riêng từng cái.

Vì sao cần tách khỏi bàn đo: bàn đo chạy cả pipeline một lượt, nên một câu sai
ở đó có thể do VAD, do bộ đếm lượt, do LLM hay do talker — và không ai đọc được
năm mươi file WAV vào micro từng cái một.

Hai quy tắc giữ cho các con số ở đây có nghĩa:

1. **Dùng engine ĐANG NẠP, không tự dựng bản mới.** Mọi bài thử gọi thẳng
   `platform.models.*`. Dựng một engine mới để thử là đo một thứ mà phiên thật
   không bao giờ chạm tới.
2. **Văn bản đi qua đúng bộ chia cụm của đường nói**
   (`conversation.segmenter.pipeline_segmenter`). Chuẩn hoá và xử lý cue cảm
   xúc là một nửa thứ người nghe nghe được; chép lại logic ở đây là tự đo một
   sản phẩm không tồn tại.

Không từ chối khi đang có phiên chạy — chỉ **đánh dấu** con số đó là đo trong
lúc có người khác dùng máy. Từ chối thì phiền (ai cũng để một tab bàn đo mở),
còn im lặng trả một con số bị nhiễu thì tệ hơn nhiều.

Đổi model thì khác: chỉ đổi được khi KHÔNG có phiên nào, vì tráo engine dưới
chân một lượt đang nói là cách chắc chắn nhất để có một lỗi không tái hiện được.
"""

from __future__ import annotations

import asyncio
import base64
import io
import time
import wave
from typing import Any

import numpy as np
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse

from ..conversation.context import ConversationContext
from ..conversation.segmenter import pipeline_segmenter
from ..core.audio import AudioFrame
from ..core.config import Config, EngineSpec
from ..core.errors import VoicePlatformError
from ..models import registry as model_registry
from ..models.base import Message
from ..observability.logging import get_logger
from ..tasks.search import SearchRequest

log = get_logger("lab")

_KINDS = {
    "asr": model_registry.ASR_BACKENDS,
    "llm": model_registry.LLM_BACKENDS,
    "tts": model_registry.TTS_BACKENDS,
    "search": model_registry.SEARCH_BACKENDS,
}
_MAX_AUDIO_BYTES = 32 * 1024 * 1024
_MAX_TEXT = 4000


# ------------------------------------------------------------------ âm thanh

def read_wav(payload: bytes) -> tuple[np.ndarray, int]:
    """WAV 16-bit PCM về float32 mono. Stdlib, không thêm dependency."""
    with wave.open(io.BytesIO(payload), "rb") as handle:
        width = handle.getsampwidth()
        if width != 2:
            raise VoicePlatformError(
                f"WAV {width * 8}-bit chưa đọc được; cần PCM 16-bit "
                "(sox in.wav -b 16 out.wav, hoặc ffmpeg -i in.wav -c:a pcm_s16le out.wav)"
            )
        rate = handle.getframerate()
        channels = handle.getnchannels()
        raw = handle.readframes(handle.getnframes())
    data = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    if channels > 1:
        data = data.reshape(-1, channels).mean(axis=1)
    return data, rate


def wav_bytes(samples: np.ndarray, rate: int) -> bytes:
    pcm = (np.clip(samples, -1.0, 1.0) * 32767.0).astype("<i2")
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(pcm.tobytes())
    return buffer.getvalue()


def word_errors(reference: str, hypothesis: str) -> tuple[int, int]:
    """Khoảng cách sửa theo TỪ, trả về (số lỗi, số từ tham chiếu).

    Trả về hai số chứ không trả tỉ lệ, để gộp nhiều file thì cộng tử và mẫu
    riêng. Trung bình các tỉ lệ sẽ cho một clip ba từ cùng trọng số với một
    clip ba phút.
    """
    ref = reference.lower().split()
    hyp = hypothesis.lower().split()
    if not ref:
        return (len(hyp), 0)
    previous = list(range(len(hyp) + 1))
    for i, ref_word in enumerate(ref, 1):
        current = [i]
        for j, hyp_word in enumerate(hyp, 1):
            current.append(min(
                previous[j] + 1,
                current[j - 1] + 1,
                previous[j - 1] + (ref_word != hyp_word),
            ))
        previous = current
    return (previous[-1], len(ref))


# ------------------------------------------------------------------ dịch vụ

class LabService:
    def __init__(self, platform: Any, config: Config) -> None:
        self.platform = platform
        self.config = config
        # Tiến trình chỉ có MỘT ASR và MỘT talker. Hai bài thử chồng nhau sẽ
        # báo thời gian của không ai cả.
        self._lock = asyncio.Lock()

    # --- trạng thái ----------------------------------------------------
    def describe(self) -> dict[str, Any]:
        models = self.platform.models
        loaded = models.describe()
        out: dict[str, Any] = {
            "mode": models.mode,
            "live_sessions": len(self.platform.sessions),
            "voice": getattr(self.platform, "voice", None),
        }
        kinds: dict[str, Any] = {}
        for kind, choices in _KINDS.items():
            spec: EngineSpec = getattr(self.config.models, kind)
            kinds[kind] = {
                "backend": spec.backend,
                "options": spec.options,
                "choices": list(choices),
                "loaded": loaded.get(kind),
            }
        # Danh sách giọng đọc từ ENGINE ĐÃ NẠP, không phải từ config: một
        # config khai giọng mà checkpoint không có là cách đổi giọng âm thầm
        # không xảy ra gì cả.
        kinds["tts"]["voices"] = list(models.tts.capabilities.voices)
        out["kinds"] = kinds
        return out

    def set_voice(self, voice: str | None) -> dict[str, Any]:
        """Đổi giọng mà KHÔNG nạp lại model.

        Cả ZeroTTS lẫn các talker mượn đều nhận `voice` theo từng lần gọi, nên
        đổi giọng không cần dựng lại engine — và một lần dựng lại có thể mất
        vài chục giây.
        """
        wanted = (voice or "").strip() or None
        available = self.platform.models.tts.capabilities.voices
        if wanted and available and wanted not in available:
            raise VoicePlatformError(
                f"giọng {wanted!r} không có. Engine khai: {', '.join(available)}"
            )
        self.platform.voice = wanted
        # Phiên đang mở cũng phải đổi theo. Chỉ đặt cho phiên MỚI thì người
        # dùng đổi giọng ngay trên bàn đo, nghe tiếp vẫn giọng cũ, và không có
        # gì nói cho họ biết là phải kết nối lại.
        for engine in list(self.platform.sessions.values()):
            engine.voice = wanted
        options = self.config.models.tts.options
        if wanted:
            options["voice"] = wanted
        else:
            options.pop("voice", None)
        return self.describe()

    # --- đổi engine ----------------------------------------------------
    async def swap(self, kind: str, backend: str, options: dict[str, Any]) -> dict[str, Any]:
        if kind not in _KINDS:
            raise VoicePlatformError(f"không có loại engine '{kind}'")
        if self.platform.sessions:
            raise Busy(
                f"đang có {len(self.platform.sessions)} phiên chạy. Đóng các tab "
                "bàn đo rồi thử lại — tráo engine dưới chân một lượt đang nói là "
                "cách chắc chắn nhất để có một lỗi không tái hiện được."
            )
        spec = EngineSpec(backend=backend, options=dict(options or {}))
        async with self._lock:
            builders = {
                "asr": lambda: model_registry.build_asr(spec),
                "llm": lambda: model_registry.build_llm(spec),
                "tts": lambda: model_registry.build_tts(
                    spec, output_sample_rate=self.config.audio.output_sample_rate
                ),
                "search": lambda: model_registry.build_search_agent(spec),
            }
            started = time.monotonic()
            engine = builders[kind]()
            if engine is not None:
                try:
                    await engine.start()
                except Exception:
                    # Engine cũ vẫn nguyên: chỉ đóng nó SAU khi bản mới đã nạp
                    # được. Đóng trước là mất cả hai khi model mới không lên.
                    close = getattr(engine, "close", None)
                    if close is not None:
                        try:
                            await close()
                        except Exception:
                            pass
                    raise
            previous = getattr(self.platform.models, kind)
            setattr(self.platform.models, kind, engine)
            setattr(self.config.models, kind, spec)
            if kind == "tts":
                # Câu "Để tôi tra cứu nhé." đã tổng hợp sẵn bằng talker CŨ, ở
                # tốc độ lấy mẫu CŨ. Giữ lại là phát một câu sai cao độ ngay
                # lượt tra cứu kế tiếp.
                self.platform.models._speech_cache.clear()
                # Giọng cũ hiếm khi tồn tại ở engine mới. Giữ lại thì mỗi lần
                # tổng hợp là một dòng cảnh báo rồi âm thầm đổi giọng.
                voices = engine.capabilities.voices if engine is not None else ()
                current = getattr(self.platform, "voice", None)
                if current and voices and current not in voices:
                    log.info("giọng %r không có ở engine mới, bỏ về mặc định", current)
                    self.platform.voice = None
                    self.config.models.tts.options.pop("voice", None)
            if previous is not None and previous is not engine:
                try:
                    await previous.close()
                except Exception as exc:  # pragma: no cover - backend dependent
                    log.warning("không đóng được engine cũ: %s", exc)
            log.info("đã đổi %s sang %s trong %.0f ms", kind, backend,
                     (time.monotonic() - started) * 1000)
        return self.describe()

    # --- các bài thử ---------------------------------------------------
    def _contended(self) -> dict[str, Any]:
        live = len(self.platform.sessions)
        return {"contended": live > 0, "live_sessions": live}

    async def try_asr(self, audio: np.ndarray, rate: int, reference: str) -> dict[str, Any]:
        engine = self.platform.models.asr
        target = self.config.audio.sample_rate
        frame = self.config.frame_samples
        async with self._lock:
            started = time.monotonic()
            stream = await engine.open_stream(sample_rate=rate)
            partials = 0
            for index in range(0, audio.size, frame):
                block = audio[index : index + frame]
                result = await stream.push(AudioFrame(samples=block, sample_rate=rate))
                if result is not None and result.text:
                    partials += 1
            transcript = await stream.finish()
            elapsed_ms = (time.monotonic() - started) * 1000
            await stream.close()
        audio_ms = 1000.0 * audio.size / rate
        text = (transcript.text or "").strip()
        out: dict[str, Any] = {
            "engine": getattr(engine, "name", "?"),
            "text": text,
            "decode_ms": round(elapsed_ms, 1),
            "audio_ms": round(audio_ms, 1),
            "rtf": round(elapsed_ms / audio_ms, 4) if audio_ms else None,
            "partials": partials,
            "input_sample_rate": rate,
            "session_sample_rate": target,
            **self._contended(),
        }
        if reference.strip():
            errors, words = word_errors(reference, text)
            out["reference"] = reference.strip()
            out["wer_errors"] = errors
            out["wer_words"] = words
            out["wer"] = round(errors / words, 4) if words else None
        return out

    async def try_tts(self, text: str, voice: str | None) -> dict[str, Any]:
        engine = self.platform.models.tts
        # Giọng thật sự dùng: tham số của bài thử, nếu không có thì giọng phiên.
        used = voice or getattr(self.platform, "voice", None)
        segmenter = pipeline_segmenter(engine.capabilities)
        phrases = segmenter.push(text) + segmenter.flush()
        if not phrases:
            raise VoicePlatformError("sau khi chuẩn hoá không còn chữ nào để đọc")
        chunks: list[np.ndarray] = []
        rows: list[dict[str, Any]] = []
        rate = engine.capabilities.native_sample_rate
        async with self._lock:
            run_started = time.monotonic()
            for phrase in phrases:
                phrase_started = time.monotonic()
                first_ms: float | None = None
                samples = 0
                async for chunk in engine.synthesize(phrase, voice=used):
                    if first_ms is None:
                        first_ms = (time.monotonic() - phrase_started) * 1000
                    rate = chunk.sample_rate
                    chunks.append(chunk.samples)
                    samples += chunk.samples.size
                rows.append({
                    "text": phrase,
                    "first_chunk_ms": None if first_ms is None else round(first_ms, 1),
                    "total_ms": round((time.monotonic() - phrase_started) * 1000, 1),
                    "audio_ms": round(1000.0 * samples / rate, 1) if rate else None,
                })
            total_ms = (time.monotonic() - run_started) * 1000
        audio = np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)
        audio_ms = 1000.0 * audio.size / rate if rate else 0.0
        return {
            "engine": getattr(engine, "name", "?"),
            "voice": used,
            "voices": list(engine.capabilities.voices),
            # Cột này tồn tại để NHÌN THẤY engine không hỗ trợ cue đã bỏ
            # "[cười]" đi, thay vì đọc nó thành chữ mà không ai biết.
            "prepared": phrases,
            "emotion_cues": engine.capabilities.emotion_cues,
            "phrases": rows,
            "first_audio_ms": rows[0]["first_chunk_ms"] if rows else None,
            "total_ms": round(total_ms, 1),
            "audio_ms": round(audio_ms, 1),
            "rtf": round(total_ms / audio_ms, 4) if audio_ms else None,
            "sample_rate": rate,
            "wav_base64": base64.b64encode(wav_bytes(audio, rate)).decode("ascii"),
            **self._contended(),
        }

    async def try_llm(self, prompt: str, with_tools: bool) -> dict[str, Any]:
        engine = self.platform.models.llm
        # Dựng prompt bằng đúng ConversationContext của sản phẩm: thứ tự khối
        # công cụ so với khối giọng nói đã đo được là chênh 0/10 với 10/10 lượt
        # gọi đúng, nên một bài thử dựng prompt kiểu khác là vô nghĩa.
        context = ConversationContext(
            self.config.conversation.system_prompt, self.config.conversation.history_turns
        )
        context.start_turn(1, prompt)
        tool_names = None
        tools = None
        if with_tools:
            from ..tasks.registry import build_registry

            registry = build_registry(self.config.tasks.tools)
            if len(registry):
                tools = registry.openai_tools()
                tool_names = registry.names()
        messages: list[Message] = context.messages(tool_names=tool_names)

        parts: list[str] = []
        calls: list[dict[str, Any]] = []
        finish = None
        started = time.monotonic()
        ttft: float | None = None
        async with self._lock:
            async for delta in engine.stream(messages, tools=tools):
                if delta.text:
                    if ttft is None:
                        ttft = (time.monotonic() - started) * 1000
                    parts.append(delta.text)
                if delta.tool_call is not None:
                    calls.append({"name": delta.tool_call.name,
                                  "arguments": delta.tool_call.arguments})
                if delta.finish_reason:
                    finish = delta.finish_reason
        total_ms = (time.monotonic() - started) * 1000
        text = "".join(parts)
        return {
            "engine": getattr(engine, "name", "?"),
            "system_prompt": messages[0].content,
            "text": text,
            "ttft_ms": None if ttft is None else round(ttft, 1),
            "total_ms": round(total_ms, 1),
            "chars": len(text),
            "chars_per_s": round(len(text) / (total_ms / 1000), 1) if total_ms else None,
            "tool_calls": calls,
            "finish_reason": finish,
            **self._contended(),
        }

    async def try_search(self, query: str) -> dict[str, Any]:
        agent = self.platform.models.search
        if agent is None:
            raise VoicePlatformError(
                "chưa cấu hình tác nhân tra cứu (models.search.backend đang là 'none')"
            )
        started = time.monotonic()
        async with self._lock:
            result = await agent.search(SearchRequest(query=query, turn_id=0))
        return {
            "engine": getattr(agent, "name", "?"),
            "ok": result.ok,
            "content": result.content,
            "source": result.source,
            "error": result.error,
            "latency_ms": round((time.monotonic() - started) * 1000, 1),
            **self._contended(),
        }


class Busy(VoicePlatformError):
    """Việc không làm được lúc này, không phải một cấu hình sai."""


# ------------------------------------------------------------------ định tuyến

def register(app: FastAPI, platform: Any, config: Config) -> None:
    service = LabService(platform, config)
    web_dir = config.server.web_dir

    def _fail(exc: Exception) -> JSONResponse:
        status = 409 if isinstance(exc, Busy) else 400
        return JSONResponse({"detail": str(exc)}, status_code=status)

    @app.get("/lab")
    async def lab_page() -> Any:
        from pathlib import Path

        page = Path(web_dir) / "lab.html"
        if page.exists():
            return FileResponse(str(page))
        return JSONResponse({"detail": "chưa có web/lab.html"}, status_code=404)

    @app.get("/engines")
    async def engines() -> Any:
        return service.describe()

    @app.post("/engines/{kind}")
    async def set_engine(kind: str, request: Request) -> Any:
        body = await request.json()
        try:
            return await service.swap(
                kind, str(body.get("backend", "")), body.get("options") or {}
            )
        except VoicePlatformError as exc:
            return _fail(exc)
        except Exception as exc:
            log.exception("đổi engine %s thất bại", kind)
            return _fail(VoicePlatformError(f"{type(exc).__name__}: {exc}"))

    @app.post("/engines/tts/voice")
    async def set_voice(request: Request) -> Any:
        body = await request.json()
        try:
            return service.set_voice(body.get("voice"))
        except VoicePlatformError as exc:
            return _fail(exc)

    @app.post("/try/asr")
    async def try_asr(request: Request) -> Any:
        payload = await request.body()
        if not payload:
            return _fail(VoicePlatformError("không có dữ liệu âm thanh"))
        if len(payload) > _MAX_AUDIO_BYTES:
            return _fail(VoicePlatformError("file quá lớn (tối đa 32 MB)"))
        reference = request.query_params.get("reference", "")
        try:
            if payload[:4] == b"RIFF":
                audio, rate = read_wav(payload)
            else:
                rate = int(request.query_params.get("rate", config.audio.sample_rate))
                audio = np.frombuffer(payload, dtype="<i2").astype(np.float32) / 32768.0
            if audio.size == 0:
                raise VoicePlatformError("file không có mẫu nào")
            return await service.try_asr(audio, rate, reference)
        except VoicePlatformError as exc:
            return _fail(exc)
        except Exception as exc:
            log.exception("thử ASR thất bại")
            return _fail(VoicePlatformError(f"{type(exc).__name__}: {exc}"))

    @app.post("/try/tts")
    async def try_tts(request: Request) -> Any:
        body = await request.json()
        text = str(body.get("text", "")).strip()[:_MAX_TEXT]
        if not text:
            return _fail(VoicePlatformError("chưa nhập chữ để đọc"))
        try:
            return await service.try_tts(text, body.get("voice") or None)
        except VoicePlatformError as exc:
            return _fail(exc)
        except Exception as exc:
            log.exception("thử TTS thất bại")
            return _fail(VoicePlatformError(f"{type(exc).__name__}: {exc}"))

    @app.post("/try/llm")
    async def try_llm(request: Request) -> Any:
        body = await request.json()
        prompt = str(body.get("prompt", "")).strip()[:_MAX_TEXT]
        if not prompt:
            return _fail(VoicePlatformError("chưa nhập câu hỏi"))
        try:
            return await service.try_llm(prompt, bool(body.get("tools")))
        except VoicePlatformError as exc:
            return _fail(exc)
        except Exception as exc:
            log.exception("thử LLM thất bại")
            return _fail(VoicePlatformError(f"{type(exc).__name__}: {exc}"))

    @app.post("/try/search")
    async def try_search(request: Request) -> Any:
        body = await request.json()
        query = str(body.get("query", "")).strip()[:_MAX_TEXT]
        if not query:
            return _fail(VoicePlatformError("chưa nhập câu cần tra"))
        try:
            return await service.try_search(query)
        except VoicePlatformError as exc:
            return _fail(exc)
        except Exception as exc:
            log.exception("thử tra cứu thất bại")
            return _fail(VoicePlatformError(f"{type(exc).__name__}: {exc}"))
