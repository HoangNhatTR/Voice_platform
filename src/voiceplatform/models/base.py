"""Model-plane contracts.

The conversation engine is written against these four protocols and nothing
else. Swapping PhoWhisper for Gipformer, or the whole cascade for a native S2S
model, must not touch a line in conversation/.

Two rules that experience has already charged us for:

1. Capabilities are attributes of the engine, never flags in the config. A
   config that claims a voice supports emotion cues when the loaded backend
   does not makes the backend read "[cười]" out loud.
2. Nothing here returns a whole response. Every contract is a stream, because
   perceived latency is time-to-first-audio, not total time.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import numpy as np

from ..core.audio import AudioFrame

# --------------------------------------------------------------------------- #
# shared payloads
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class Transcript:
    text: str
    is_final: bool = False
    confidence: float | None = None
    language: str | None = None


@dataclass(slots=True)
class Message:
    role: str  # system | user | assistant | tool
    content: str
    name: str | None = None
    tool_call_id: str | None = None

    def as_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"role": self.role, "content": self.content}
        if self.name:
            out["name"] = self.name
        if self.tool_call_id:
            out["tool_call_id"] = self.tool_call_id
        return out


@dataclass(slots=True)
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class LLMDelta:
    text: str = ""
    tool_call: ToolCall | None = None
    finish_reason: str | None = None


@dataclass(slots=True)
class SpeechChunk:
    samples: np.ndarray
    sample_rate: int
    text: str = ""

    def to_frame(self, seq: int = 0) -> AudioFrame:
        return AudioFrame(samples=self.samples, sample_rate=self.sample_rate, seq=seq)


# --------------------------------------------------------------------------- #
# capabilities
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class AsrCapabilities:
    streaming_partials: bool = False
    languages: tuple[str, ...] = ("vi",)
    native_sample_rate: int = 16000


@dataclass(slots=True)
class TtsCapabilities:
    streaming: bool = True
    emotion_cues: bool = False       # can it act on "[cười]" instead of reading it
    voices: tuple[str, ...] = ()
    native_sample_rate: int = 24000
    # Measured, not promised: real-time factor above 1.0 means the talker is the
    # bottleneck and the scheduler must keep a deeper playout cushion.
    expected_rtf: float = 0.5


@dataclass(slots=True)
class LlmCapabilities:
    tools: bool = False
    streaming: bool = True
    context_tokens: int = 8192


# --------------------------------------------------------------------------- #
# engines
# --------------------------------------------------------------------------- #


class AsrStream(Protocol):
    async def push(self, frame: AudioFrame) -> Transcript | None:
        """Feed one frame; return a partial transcript if one is ready."""

    async def finish(self) -> Transcript:
        """Close the utterance and return the final transcript."""

    async def close(self) -> None: ...


@runtime_checkable
class AsrEngine(Protocol):
    name: str
    capabilities: AsrCapabilities

    async def start(self) -> None: ...
    async def open_stream(self, *, sample_rate: int, language: str | None = None) -> AsrStream: ...
    async def close(self) -> None: ...


@runtime_checkable
class LlmEngine(Protocol):
    name: str
    capabilities: LlmCapabilities

    async def start(self) -> None: ...

    def stream(
        self,
        messages: list[Message],
        *,
        tools: list[dict[str, Any]] | None = None,
        max_tokens: int | None = None,
    ) -> AsyncIterator[LLMDelta]: ...

    async def close(self) -> None: ...


@runtime_checkable
class TtsEngine(Protocol):
    name: str
    capabilities: TtsCapabilities

    async def start(self) -> None: ...

    def synthesize(self, text: str, *, voice: str | None = None) -> AsyncIterator[SpeechChunk]: ...

    async def close(self) -> None: ...


class S2sStream(Protocol):
    async def push(self, frame: AudioFrame) -> None: ...
    def output(self) -> AsyncIterator[SpeechChunk | Transcript]: ...
    async def close(self) -> None: ...


@runtime_checkable
class S2sEngine(Protocol):
    """Native speech-to-speech: one model owns ASR+LLM+TTS.

    It replaces the cascade, not the conversation plane: turn-taking, barge-in
    and generation fencing stay where they are.
    """

    name: str

    async def start(self) -> None: ...
    async def open_stream(self, *, sample_rate: int) -> S2sStream: ...
    async def close(self) -> None: ...
