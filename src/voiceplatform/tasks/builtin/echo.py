from __future__ import annotations

import asyncio
from typing import Any

from ..base import TaskContext, ToolResult, ToolSpec


class EchoTool:
    spec = ToolSpec(
        name="echo",
        description="Trả lại nguyên văn tham số text. Dùng để kiểm thử.",
        parameters={
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        },
        timeout_ms=500,
    )

    async def run(self, arguments: dict[str, Any], ctx: TaskContext) -> ToolResult:
        return ToolResult(ok=True, content=str(arguments.get("text", "")))


class SlowTool:
    """A deliberately slow tool: the filler and cancellation paths need one."""

    def __init__(self, delay_ms: float = 2000.0, timeout_ms: int = 8000) -> None:
        self.delay_ms = delay_ms
        self.spec = ToolSpec(
            name="slow",
            description="Công cụ chậm dùng để kiểm thử đường tác vụ.",
            parameters={"type": "object", "properties": {}},
            timeout_ms=timeout_ms,
        )

    async def run(self, arguments: dict[str, Any], ctx: TaskContext) -> ToolResult:
        await asyncio.sleep(self.delay_ms / 1000.0)
        return ToolResult(ok=True, content=f"xong sau {self.delay_ms:.0f} mili giây")
