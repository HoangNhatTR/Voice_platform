from __future__ import annotations

from typing import Any

from ..core.errors import ConfigError
from .base import Tool, ToolSpec


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self._tools[tool.spec.name] = tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def __contains__(self, name: object) -> bool:
        return name in self._tools

    def __len__(self) -> int:
        return len(self._tools)

    def specs(self) -> list[ToolSpec]:
        return [t.spec for t in self._tools.values()]

    def openai_tools(self) -> list[dict[str, Any]]:
        return [t.spec.as_openai_tool() for t in self._tools.values()]

    def names(self) -> list[str]:
        return sorted(self._tools)


def build_registry(names: list[str], options: dict[str, Any] | None = None) -> ToolRegistry:
    options = options or {}
    registry = ToolRegistry()
    for name in names:
        if name == "clock":
            from .builtin.clock import ClockTool

            registry.register(ClockTool())
        elif name == "echo":
            from .builtin.echo import EchoTool

            registry.register(EchoTool())
        elif name == "slow":
            from .builtin.echo import SlowTool

            registry.register(SlowTool(**options.get("slow", {})))
        elif name == "kb":
            from .rag.keyword import KeywordKnowledgeTool

            registry.register(KeywordKnowledgeTool(**options.get("kb", {})))
        else:
            raise ConfigError(f"unknown tool: {name}")
    return registry
