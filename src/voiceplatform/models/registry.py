"""Config name -> engine instance.

Backends are constructed lazily and imported inside the factory so that a
machine with no torch can still run the mock stack, the tests and the server.
"""

from __future__ import annotations

import asyncio
import time
from collections import OrderedDict
from typing import Any

import numpy as np

from ..core.audio import AudioFrame
from ..core.config import EngineSpec, ModelsConfig
from ..core.errors import ConfigError
from ..observability.logging import get_logger
from .base import AsrEngine, LlmEngine, S2sEngine, TtsEngine

log = get_logger("models")

# Một nguồn duy nhất: factory kiểm tra theo đây, và màn chọn model cũng đọc
# đúng đây. Hai danh sách rời nhau là cách một lựa chọn có trong giao diện mà
# không dựng được.
ASR_BACKENDS = ("mock", "gipformer", "phowhisper", "parakeet", "s2s_bridge")
LLM_BACKENDS = ("mock", "openai_compat", "llama_cpp_server", "vllm", "ollama", "openai")
TTS_BACKENDS = ("mock", "zerotts", "vieneu", "vieneu_nano", "vixtts", "f5", "subprocess", "s2s_bridge")
SEARCH_BACKENDS = ("none", "mock", "tools", "llm", "wikipedia_vi")


def build_asr(spec: EngineSpec) -> AsrEngine:
    backend = spec.backend.lower()
    if backend == "mock":
        from .asr.mock import MockAsrEngine

        return MockAsrEngine(**spec.options)
    if backend in ASR_BACKENDS:
        from .asr.bridge_viet_s2s import BridgeAsrEngine

        inner = spec.options.get("backend", backend if backend != "s2s_bridge" else "phowhisper")
        options = {k: v for k, v in spec.options.items() if k != "backend"}
        return BridgeAsrEngine(backend=inner, **options)
    raise ConfigError(f"unknown asr backend: {spec.backend}")


def build_llm(spec: EngineSpec) -> LlmEngine:
    backend = spec.backend.lower()
    if backend == "mock":
        from .llm.mock import MockLlmEngine

        return MockLlmEngine(**spec.options)
    if backend in LLM_BACKENDS:
        from .llm.openai_compat import OpenAiCompatLlm

        return OpenAiCompatLlm(**spec.options)
    raise ConfigError(f"unknown llm backend: {spec.backend}")


def build_tts(spec: EngineSpec, *, output_sample_rate: int | None = None) -> TtsEngine:
    """`output_sample_rate` is `audio.output_sample_rate`, and it has to be passed.

    The borrowed talkers default to 16 kHz internally, so leaving it unset made
    a 24 kHz voice arrive resampled down to 16 kHz with nothing in the logs
    saying so — while this platform's own `audio.output_sample_rate: 24000` sat
    there being read by nobody. A per-engine option still wins over it.
    """
    backend = spec.backend.lower()
    options = {k: v for k, v in spec.options.items() if k != "backend"}
    if backend == "mock":
        from .tts.mock import MockTtsEngine

        if output_sample_rate is not None:
            options.setdefault("sample_rate", output_sample_rate)
        return MockTtsEngine(**options)
    if backend == "zerotts":
        from .tts.zerotts import ZeroTtsEngine

        # Không đi qua bridge: gói này không thuộc speech2speech.
        if output_sample_rate is not None:
            options.setdefault("output_sample_rate", output_sample_rate)
        pool_size = int(options.pop("pool_size", 1))
        share_model = options.pop("share_model", False)
        if pool_size != 1:
            from .tts.pool import TtsPool

            return TtsPool(lambda: ZeroTtsEngine(**options), pool_size=pool_size,
                           max_queue=int(options.get("max_queue", 8)), share_model=share_model)
        return ZeroTtsEngine(**options)
    if backend in TTS_BACKENDS:
        from .tts.bridge_viet_s2s import BridgeTtsEngine

        inner = spec.options.get("backend", backend if backend != "s2s_bridge" else "vieneu")
        if output_sample_rate is not None:
            options.setdefault("output_sample_rate", output_sample_rate)
        return BridgeTtsEngine(backend=inner, **options)
    raise ConfigError(f"unknown tts backend: {spec.backend}")


def build_s2s(spec: EngineSpec) -> S2sEngine | None:
    backend = spec.backend.lower()
    if backend in {"none", ""}:
        return None
    raise ConfigError(
        f"unknown s2s backend: {spec.backend}. Native speech-to-speech is a "
        "declared seat (models.base.S2sEngine) with no implementation yet."
    )


def build_search_agent(spec: EngineSpec, *, executor: Any = None) -> Any:
    """Tác nhân tìm kiếm — có thể là model khác, service, hay chỉ là tool."""
    backend = spec.backend.lower()
    if backend in {"none", ""}:
        return None
    if backend == "mock":
        from ..tasks.search import MockSearchAgent

        return MockSearchAgent(**spec.options)
    if backend == "wikipedia_vi":
        from ..tasks.search import WikipediaSearchAgent

        return WikipediaSearchAgent(**spec.options)
    if backend == "tools":
        from ..tasks.search import ToolSearchAgent

        # No caller ever had an executor to hand in, so this branch only ever
        # raised. Build one from the tool it is asked for: a lookup agent made
        # of plain functions needs nothing else.
        options = dict(spec.options)
        tool_name = options.pop("tool_name", "kb")
        tool_options = options.pop("tool_options", None)
        if executor is None:
            from ..tasks.executor import TaskExecutor
            from ..tasks.registry import build_registry

            executor = TaskExecutor(build_registry([tool_name], tool_options))
        return ToolSearchAgent(executor, tool_name=tool_name, **options)
    if backend in {"llm", "openai_compat", "llama_cpp_server", "vllm", "ollama"}:
        from ..tasks.search import LlmSearchAgent

        options = dict(spec.options)
        system_prompt = options.pop("system_prompt", None)
        max_tokens = int(options.pop("max_tokens", 160))
        options.setdefault('request_priority','search')
        from .llm.openai_compat import OpenAiCompatLlm

        engine = OpenAiCompatLlm(**options)
        return LlmSearchAgent(engine, system_prompt=system_prompt, max_tokens=max_tokens)
    raise ConfigError(f"unknown search backend: {spec.backend}")


class ModelPlane:
    """The three (or four) engines a session talks to, built once per process."""

    def __init__(self, config: ModelsConfig, *, output_sample_rate: int | None = None) -> None:
        self.config = config
        # Câu cố định ("Để tôi tra cứu nhé.") được tổng hợp MỘT lần cho cả
        # tiến trình. Trên talker CPU, tổng hợp lại mỗi lần tốn 1.6 giây — đúng
        # bằng quãng im lặng mà câu đó sinh ra để lấp.
        self._speech_cache: OrderedDict[tuple[str, str], list[Any]] = OrderedDict()
        self._cache_lock: Any = None
        self._last_touch = float("-inf")
        self._started = False
        self.llm_warmup_input = None
        if config.mode != "cascade":
            raise ConfigError("only cascade mode is implemented")
        self.mode = config.mode
        self.asr = build_asr(config.asr)
        self.llm = build_llm(config.llm)
        self.tts = build_tts(config.tts, output_sample_rate=output_sample_rate)
        self.s2s = build_s2s(config.s2s)
        self.search = build_search_agent(config.search)
        self.bind_llm_admission()

    def bind_llm_admission(self) -> None:
        # Share admission when roles use the same native server. Independent
        # semaphores merely move their queues into llama-server and let search
        # occupy every slot. Distinct endpoints retain independent limiters.
        search_engine=getattr(self.search,'engine',None)
        for engine in (self.llm, search_engine):
            if hasattr(engine, '_private_limiter'):
                engine.limiter = engine._private_limiter
        if (getattr(self.llm,'endpoint',None) and
                getattr(search_engine,'endpoint',None)==self.llm.endpoint):
            search_engine.limiter=self.llm.limiter

    async def start(self) -> None:
        opened = []
        try:
            async with asyncio.timeout(self.config.startup_timeout_s):
                for engine in (self.asr, self.llm, self.tts, self.s2s, self.search):
                    if engine is not None:
                        opened.append(engine)
                        await engine.start()
            self._started = True
        except BaseException:
            for engine in reversed(opened):
                try:
                    await engine.close()
                except Exception:
                    log.exception("startup cleanup failed")
            raise

    async def close(self) -> None:
        self._started = False
        errors = []
        for engine in (self.search, self.s2s, self.tts, self.llm, self.asr):
            if engine is not None:
                try:
                    await engine.close()
                except Exception as exc:
                    errors.append(exc)
                    log.exception("model close failed")
        self._speech_cache.clear()
        if errors:
            raise errors[0]

    async def check_dependencies(self, timeout_s: float = 2.0) -> dict[str, Any]:
        async def one(engine: Any) -> dict[str, Any]:
            if engine is None:
                return {"ok": True, "configured": False}
            checker = getattr(engine, "check_ready", None)
            if checker is None:
                checker = getattr(getattr(engine, "engine", None), "check_ready", None)
            if checker is None:
                return {"ok": True, "remote": False}
            try:
                async with asyncio.timeout(timeout_s):
                    return await checker(timeout_s=timeout_s)
            except Exception as exc:
                return {"ok": False, "reason": type(exc).__name__}
        llm, search = await asyncio.gather(one(self.llm), one(self.search))
        return {"llm": llm, "search": search}

    async def touch(self, *, min_interval_s: float = 30.0, include_llm: bool = True) -> bool:
        """Một lượt ASR + TTS thật nhỏ để kéo trang bộ nhớ của model về lại RAM.

        Bộ nhớ đệm câu (`cached_speech`) KHÔNG làm được việc này: trúng cache
        thì không có gì chạy. Đo 25/09 trên máy dùng chung, swap đầy: lần TTS
        đầu sau lúc server nằm im mất 1537 ms (RTF 1.2), các lần sau 60-80 ms.
        Kết quả bỏ đi; lỗi chỉ ghi log — làm ấm hỏng không được làm hỏng phiên.
        """
        now = time.monotonic()
        if now - self._last_touch < min_interval_s:
            return False
        self._last_touch = now
        warmup = getattr(self.llm, "warmup", None)
        if include_llm and warmup is not None and self.llm_warmup_input is not None:
            try:
                messages, tools = self.llm_warmup_input
                await warmup(messages, tools=tools)
            except Exception as exc:
                log.warning("làm ấm LLM thất bại: %s", exc)
        # Mock không có trang nào để kéo về; chạy nó chỉ tiêu mất một câu
        # trong kịch bản của mock ASR và làm lệch test.
        if getattr(self.tts, "name", "") != "mock":
            await self._touch_tts()
        if getattr(self.asr, "name", "") != "mock":
            await self._touch_asr()
        return True

    async def _touch_tts(self) -> None:
        try:
            async with asyncio.timeout(self.config.operation_timeout_s):
                touch = getattr(self.tts, "touch", None)
                if touch is not None:
                    await touch()
                else:
                    async for _ in self.tts.synthesize("Vâng."):
                        pass
        except Exception as exc:  # pragma: no cover - backend dependent
            log.warning("làm ấm TTS thất bại: %s", exc)

    async def _touch_asr(self) -> None:
        try:
            async with asyncio.timeout(self.config.operation_timeout_s):
                rate = int(getattr(self.asr.capabilities, "native_sample_rate", 16000) or 16000)
                stream = await self.asr.open_stream(sample_rate=rate)
                noise = (0.001 * np.random.default_rng(0).normal(0, 1, rate * 3 // 10)).astype(np.float32)
                try:
                    await stream.push(AudioFrame(samples=noise, sample_rate=rate))
                    await stream.finish()
                finally:
                    await stream.close()
        except Exception as exc:  # pragma: no cover - backend dependent
            log.warning("làm ấm ASR thất bại: %s", exc)

    def warm_speech(self, text: str, voice: str | None = None) -> list[Any] | None:
        """Audio ĐÃ tổng hợp sẵn, hoặc None. Không bao giờ chờ.

        Tách khỏi `cached_speech` vì người gọi nó đang ở trên đường nói: chờ
        tổng hợp ở đó là đúng thứ câu mở sinh ra để tránh.
        """
        return self._speech_cache.get((text, voice or ""))

    async def cached_speech(self, text: str, voice: str | None = None) -> list[Any]:
        """Tổng hợp một lần, dùng lại mãi. Trả về list SpeechChunk."""
        import asyncio

        key = (text, voice or "")
        if key in self._speech_cache:
            return self._speech_cache[key]
        if self._cache_lock is None:
            self._cache_lock = asyncio.Lock()
        async with asyncio.timeout(self.config.operation_timeout_s), self._cache_lock:
            if key in self._speech_cache:
                return self._speech_cache[key]
            chunks = [c async for c in self.tts.synthesize(text, voice=voice)]
            self._speech_cache[key] = chunks
            self._speech_cache.move_to_end(key)
            while len(self._speech_cache) > 32:
                self._speech_cache.popitem(last=False)
            return chunks

    def describe(self) -> dict[str, Any]:
        def one(engine: Any) -> dict[str, Any] | None:
            if engine is None:
                return None
            caps = getattr(engine, "capabilities", None)
            return {
                "name": getattr(engine, "name", type(engine).__name__),
                "capabilities": _caps_dict(caps),
            }

        return {
            "mode": self.mode,
            "asr": one(self.asr),
            "llm": one(self.llm),
            "tts": one(self.tts),
            "s2s": one(self.s2s),
            "search": one(self.search),
        }


def _caps_dict(caps: Any) -> dict[str, Any] | None:
    if caps is None:
        return None
    import dataclasses

    if dataclasses.is_dataclass(caps):
        return {f.name: getattr(caps, f.name) for f in dataclasses.fields(caps)}
    return None
