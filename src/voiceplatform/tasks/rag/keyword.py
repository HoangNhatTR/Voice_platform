"""A small local knowledge tool.

Deliberately not a vector store: it is a working seat in the information plane
that costs no service to run, so the plane can be exercised end to end before
anyone chooses an embedding model. Swap it for a real retriever behind the same
Tool protocol.
"""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path
from typing import Any

from ..base import TaskContext, ToolResult, ToolSpec

_SPLIT = re.compile(r"\n\s*\n", re.MULTILINE)


def _fold(text: str) -> str:
    text = unicodedata.normalize("NFD", text.lower())
    return "".join(c for c in text if unicodedata.category(c) != "Mn")


class KeywordKnowledgeTool:
    def __init__(self, path: str = "knowledge", max_chars: int = 400, top_k: int = 2) -> None:
        self.path = Path(path)
        self.max_chars = max_chars
        self.top_k = top_k
        self.spec = ToolSpec(
            name="kb",
            description=(
                "Tra cứu tài liệu nội bộ theo từ khoá. Dùng khi câu hỏi liên "
                "quan đến quy định, hướng dẫn hoặc thông tin sản phẩm."
            ),
            parameters={
                "type": "object",
                "properties": {"query": {"type": "string", "description": "Câu hỏi"}},
                "required": ["query"],
            },
            timeout_ms=1500,
        )
        self._chunks: list[tuple[str, str]] | None = None

    def _load(self) -> list[tuple[str, str]]:
        if self._chunks is not None:
            return self._chunks
        chunks: list[tuple[str, str]] = []
        if self.path.is_dir():
            for file in sorted(self.path.rglob("*")):
                if file.suffix.lower() not in {".md", ".txt"}:
                    continue
                text = file.read_text(encoding="utf-8", errors="replace")
                for part in _SPLIT.split(text):
                    part = part.strip()
                    if len(part) > 40:
                        chunks.append((file.name, part))
        self._chunks = chunks
        return chunks

    async def run(self, arguments: dict[str, Any], ctx: TaskContext) -> ToolResult:
        query = str(arguments.get("query") or ctx.user_text).strip()
        chunks = self._load()
        if not chunks:
            return ToolResult(
                ok=False,
                content="Chưa có tài liệu nào trong kho tri thức.",
                error="empty_corpus",
            )
        terms = [t for t in _fold(query).split() if len(t) > 2]
        if not terms:
            return ToolResult(ok=False, content="Câu hỏi quá ngắn để tra cứu.", error="short_query")
        scored: list[tuple[float, str, str]] = []
        for source, chunk in chunks:
            folded = _fold(chunk)
            score = sum(folded.count(t) for t in terms) / (1 + len(folded) / 800)
            if score > 0:
                scored.append((score, source, chunk))
        if not scored:
            return ToolResult(ok=False, content="Không tìm thấy nội dung phù hợp.", error="no_match")
        scored.sort(key=lambda row: row[0], reverse=True)
        picked = scored[: self.top_k]
        content = " ".join(chunk[: self.max_chars] for _, _, chunk in picked)
        return ToolResult(
            ok=True,
            content=content,
            data={"sources": [src for _, src, _ in picked]},
        )
