"""Tác nhân tìm kiếm — nửa "Back end - search" của sơ đồ.

Khác biệt cốt lõi so với một tool thường: tool trả lời trong vài trăm mili giây
và lượt nói chờ được; tra cứu thì không. Nếu lượt nói phải chờ, người dùng nghe
thấy im lặng, và đó đúng là thứ sơ đồ này muốn bỏ đi.

Nên hợp đồng ở đây là *bất đồng bộ theo thiết kế*: Speech agent gửi yêu cầu rồi
đi tiếp, kết quả về lúc nào thì phát lúc đó. Tác nhân tìm kiếm có thể là một
model khác, một service, hay chỉ là mấy hàm tra bảng — Speech agent không biết
và không cần biết.
"""

from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from ..core.clock import now_ms


@dataclass(slots=True)
class SearchRequest:
    query: str
    turn_id: int
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    requested_at_ms: float = field(default_factory=now_ms)

    @property
    def age_ms(self) -> float:
        return now_ms() - self.requested_at_ms


@dataclass(slots=True)
class SearchResult:
    request: SearchRequest
    ok: bool
    content: str
    latency_ms: float = 0.0
    source: str = ""
    error: str | None = None


@runtime_checkable
class SearchAgent(Protocol):
    name: str

    async def start(self) -> None: ...
    async def search(self, request: SearchRequest) -> SearchResult: ...
    async def close(self) -> None: ...


class MockSearchAgent:
    """Chậm có kiểm soát, để kiểm thử đúng phần khó: khoảng chờ."""

    name = "mock"

    def __init__(self, delay_ms: float = 1500.0, answer: str | None = None) -> None:
        self.delay_ms = delay_ms
        self.answer = answer
        self.calls: list[str] = []

    async def start(self) -> None:
        return None

    async def close(self) -> None:
        return None

    async def search(self, request: SearchRequest) -> SearchResult:
        self.calls.append(request.query)
        started = now_ms()
        await asyncio.sleep(self.delay_ms / 1000.0)
        content = self.answer or f"kết quả giả lập cho “{request.query}”"
        return SearchResult(
            request=request,
            ok=True,
            content=content,
            latency_ms=now_ms() - started,
            source="mock",
        )


class ToolSearchAgent:
    """Tra cứu bằng các tool sẵn có (kb, API nội bộ...).

    Không có model nào ở đây. Dùng khi việc "tìm kiếm" thật ra là gọi một hàm.
    """

    name = "tools"

    def __init__(self, executor: Any, tool_name: str = "kb") -> None:
        self.executor = executor
        self.tool_name = tool_name

    async def start(self) -> None:
        return None

    async def close(self) -> None:
        return None

    async def search(self, request: SearchRequest) -> SearchResult:
        from .base import TaskContext

        started = now_ms()
        ctx = TaskContext(
            key=None,  # type: ignore[arg-type]
            session_id="search",
            user_text=request.query,
        )
        result = await self.executor.run(self.tool_name, {"query": request.query}, ctx)
        return SearchResult(
            request=request,
            ok=result.ok,
            content=result.content,
            latency_ms=now_ms() - started,
            source=self.tool_name,
            error=result.error,
        )


class LlmSearchAgent:
    """Một model RIÊNG lo việc tra cứu — "Back end - search" trong sơ đồ.

    Nó không phải model hội thoại: prompt của nó tối ưu cho chính xác và ngắn
    gọn, không cho giọng điệu. Tách ra để hai bên tiến hoá độc lập — đổi model
    tra cứu sang bản lớn hơn, hay trỏ sang một service khác, mà Speech agent
    không đổi một dòng.
    """

    name = "llm"

    DEFAULT_PROMPT = (
        "Bạn là bộ phận tra cứu. Trả lời NGẮN và CHÍNH XÁC bằng tiếng Việt, "
        "tối đa hai câu, chỉ nêu dữ kiện. Không chào hỏi, không giải thích "
        "thêm. Nếu không chắc chắn, nói thẳng là không tra được."
    )

    def __init__(
        self,
        engine: Any,
        system_prompt: str | None = None,
        max_tokens: int = 160,
        tools: list[dict[str, Any]] | None = None,
    ) -> None:
        self.engine = engine
        self.system_prompt = system_prompt or self.DEFAULT_PROMPT
        self.max_tokens = max_tokens
        self.tools = tools

    async def start(self) -> None:
        await self.engine.start()

    async def close(self) -> None:
        await self.engine.close()

    async def search(self, request: SearchRequest) -> SearchResult:
        from ..models.base import Message

        started = now_ms()
        messages = [
            Message(role="system", content=self.system_prompt),
            Message(role="user", content=request.query),
        ]
        parts: list[str] = []
        try:
            async for delta in self.engine.stream(
                messages, tools=self.tools, max_tokens=self.max_tokens
            ):
                if delta.text:
                    parts.append(delta.text)
        except Exception as exc:
            return SearchResult(
                request=request,
                ok=False,
                content="Tôi chưa tra cứu được thông tin này.",
                latency_ms=now_ms() - started,
                source=self.name,
                error=repr(exc),
            )
        text = "".join(parts).strip()
        return SearchResult(
            request=request,
            ok=bool(text),
            content=text or "Tôi chưa tra cứu được thông tin này.",
            latency_ms=now_ms() - started,
            source=getattr(self.engine, "name", self.name),
        )
