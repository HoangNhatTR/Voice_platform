"""Config name -> engine instance.

Backends are constructed lazily and imported inside the factory so that a
machine with no torch can still run the mock stack, the tests and the server.
"""

from __future__ import annotations

from typing import Any

from ..core.config import EngineSpec, ModelsConfig
from ..core.errors import ConfigError
from .base import AsrEngine, LlmEngine, S2sEngine, TtsEngine

# Một nguồn duy nhất: factory kiểm tra theo đây, và màn chọn model cũng đọc
# đúng đây. Hai danh sách rời nhau là cách một lựa chọn có trong giao diện mà
# không dựng được.
ASR_BACKENDS = ("mock", "gipformer", "phowhisper", "parakeet", "s2s_bridge")
LLM_BACKENDS = ("mock", "openai_compat", "llama_cpp_server", "vllm", "ollama", "openai")
TTS_BACKENDS = ("mock", "zerotts", "vieneu", "vieneu_nano", "vixtts", "f5", "subprocess", "s2s_bridge")
SEARCH_BACKENDS = ("none", "mock", "tools", "llm")


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
        self._speech_cache: dict[tuple[str, str], list[Any]] = {}
        self._cache_lock: Any = None
        self.mode = config.mode
        self.asr = build_asr(config.asr)
        self.llm = build_llm(config.llm)
        self.tts = build_tts(config.tts, output_sample_rate=output_sample_rate)
        self.s2s = build_s2s(config.s2s)
        self.search = build_search_agent(config.search)

    async def start(self) -> None:
        for engine in (self.asr, self.llm, self.tts, self.s2s, self.search):
            if engine is not None:
                await engine.start()

    async def close(self) -> None:
        for engine in (self.asr, self.llm, self.tts, self.s2s, self.search):
            if engine is not None:
                await engine.close()

    async def cached_speech(self, text: str, voice: str | None = None) -> list[Any]:
        """Tổng hợp một lần, dùng lại mãi. Trả về list SpeechChunk."""
        import asyncio

        key = (text, voice or "")
        if key in self._speech_cache:
            return self._speech_cache[key]
        if self._cache_lock is None:
            self._cache_lock = asyncio.Lock()
        async with self._cache_lock:
            if key in self._speech_cache:
                return self._speech_cache[key]
            chunks = [c async for c in self.tts.synthesize(text, voice=voice)]
            self._speech_cache[key] = chunks
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
