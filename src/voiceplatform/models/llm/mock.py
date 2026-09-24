"""Deterministic LLM for tests and the no-model demo.

Streams word by word with a configurable first-token delay, so TTFT shows up in
traces exactly where a real engine's would, and can emit a tool call so the
task path is exercised without a backend.
"""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import AsyncIterator
from typing import Any

from ..base import LlmCapabilities, LLMDelta, Message, ToolCall


class MockLlmEngine:
    name = "mock"

    def __init__(
        self,
        reply: str | None = None,
        first_token_delay_ms: float = 40.0,
        token_delay_ms: float = 8.0,
        tool_triggers: dict[str, str] | None = None,
    ) -> None:
        self.capabilities = LlmCapabilities(tools=True, streaming=True)
        self._reply = reply
        self.first_token_delay_ms = first_token_delay_ms
        self.token_delay_ms = token_delay_ms
        # substring in the user's text -> tool name
        self.tool_triggers = tool_triggers or {"mấy giờ": "clock", "thời gian": "clock"}

    async def start(self) -> None:
        return None

    async def close(self) -> None:
        return None

    async def stream(
        self,
        messages: list[Message],
        *,
        tools: list[dict[str, Any]] | None = None,
        max_tokens: int | None = None,
    ) -> AsyncIterator[LLMDelta]:
        last_user = next(
            (m.content for m in reversed(messages) if m.role == "user"), ""
        )
        already_ran_tool = any(m.role == "tool" for m in messages)
        if tools and not already_ran_tool:
            for needle, tool_name in self.tool_triggers.items():
                if needle in last_user.lower():
                    await asyncio.sleep(self.first_token_delay_ms / 1000.0)
                    yield LLMDelta(
                        tool_call=ToolCall(id=uuid.uuid4().hex[:8], name=tool_name),
                        finish_reason="tool_calls",
                    )
                    return

        text = self._reply or self._default_reply(last_user, messages)
        await asyncio.sleep(self.first_token_delay_ms / 1000.0)
        words = text.split(" ")
        for i, word in enumerate(words):
            if i:
                await asyncio.sleep(self.token_delay_ms / 1000.0)
            yield LLMDelta(text=word if i == 0 else " " + word)
        yield LLMDelta(finish_reason="stop")

    @staticmethod
    def _default_reply(last_user: str, messages: list[Message]) -> str:
        tool_output = next(
            (m.content for m in reversed(messages) if m.role == "tool"), None
        )
        if tool_output:
            return f"Theo tra cứu thì {tool_output}. Bạn cần gì nữa không?"
        if not last_user:
            return "Tôi đang nghe đây."
        return (
            f"Bạn vừa nói {last_user.strip()}. "
            "Đây là câu trả lời giả lập để đo đường truyền."
        )
