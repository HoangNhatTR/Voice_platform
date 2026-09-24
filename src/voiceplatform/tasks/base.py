"""Information / task plane contracts.

Everything here runs off the audio path. A tool may take five seconds; the
conversation plane must stay responsive, keep taking audio, and remain
interruptible while it waits. That is why a tool returns a value instead of
speaking, and why the executor owns the deadline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from ..core.ids import GenerationKey


@dataclass(slots=True)
class ToolSpec:
    name: str
    description: str
    parameters: dict[str, Any] = field(
        default_factory=lambda: {"type": "object", "properties": {}}
    )
    # Budget for one call. The scheduler starts a filler phrase when a call is
    # still running after conversation.filler.after_ms.
    timeout_ms: int = 8000

    def as_openai_tool(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


@dataclass(slots=True)
class TaskContext:
    key: GenerationKey
    session_id: str
    user_text: str = ""
    locale: str = "vi-VN"


@dataclass(slots=True)
class ToolResult:
    ok: bool
    content: str
    data: dict[str, Any] = field(default_factory=dict)
    latency_ms: float = 0.0
    error: str | None = None


@runtime_checkable
class Tool(Protocol):
    spec: ToolSpec

    async def run(self, arguments: dict[str, Any], ctx: TaskContext) -> ToolResult: ...
