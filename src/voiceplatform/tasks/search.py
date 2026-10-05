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
import re
import unicodedata
import uuid
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from ..core.clock import now_ms
from ..observability.logging import get_logger

log = get_logger("search")


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
    source_title: str = ""
    source_url: str = ""


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


class WikipediaSearchAgent:
    """A bounded public source for general knowledge, with a real page URL.

    Wikipedia is not a live market feed or an account backend. Refuse those
    questions instead of presenting an encyclopedia extract as current data.
    """

    name = "wikipedia_vi"
    API = "https://vi.wikipedia.org/w/api.php"
    _CURRENT = re.compile(r"\b(hôm nay|hiện tại|mới nhất|thời gian thực|giá vàng|tỷ giá|lãi suất)\b", re.I)
    _PRIVATE = re.compile(r"\b(số dư|tài khoản|số thẻ|số điện thoại|giao dịch của tôi|mã otp|otp|căn cước|cccd|mật khẩu|chuyển tiền|iban)\b", re.I)
    # An e-mail, or a run of 8+ digits (phone, account, card, CCCD). The old
    # "6 digits with any separators" refused "dân số năm 2019 2020": years and
    # grouped amounts ("1.250.000") are ordinary facts, not identifiers.
    _EMAIL = re.compile(r"[\w.+-]+@[\w.-]+\.[a-z]{2,}", re.I)
    _DIGIT_RUN = re.compile(r"\d+(?:[\s.,-]\d+)*")
    _SOURCE_WORD = re.compile(r"\b(?:(?:theo|trên|từ)\s+)?(?:wikipedia|wiki)(?:\s+tiếng\s+việt)?\b", re.I)
    _PLACE_SUBJECT = re.compile(
        r"\b(hồ|đường|chùa|công viên|bảo tàng|cầu|núi|sông)\s+(.+?)\s+(?:ở|nằm|thuộc)\b", re.I
    )
    _STOP = frozenset((
        "tra cứu tìm kiểm tra giúp tôi bạn thông tin về từ nguồn ở đâu nằm là gì bao nhiêu theo trên tiếng việt hãy cho biết "
        # Conversational particles: words people say AROUND the subject.
        "có thể được không ạ nhé nhỉ vậy thế à ơi hả mình muốn xin vui lòng làm ơn với nào sao đấy chứ nhờ"
    ).split())

    @classmethod
    def _terms(cls, text: str) -> set[str]:
        # Diacritics kept: they are the meaning in Vietnamese. Folded, "hổ"
        # (tiger) matched "Hồ" and "tự" matched "Tử".
        return {word for word in re.findall(r"\w+", unicodedata.normalize("NFC", text.casefold()))
                if word not in cls._STOP and len(word) > 1}

    @classmethod
    def _has_identifier(cls, text: str) -> bool:
        if cls._EMAIL.search(text):
            return True
        for run in cls._DIGIT_RUN.findall(text):
            if re.fullmatch(r"\d{1,3}(?:[.,]\d{3})+", run):
                continue   # 1.250.000
            groups = re.split(r"[\s.,-]", run)
            if len(groups) <= 3 and all(len(g) == 4 and 1000 <= int(g) <= 2100 for g in groups):
                continue   # years: "2019 2020", "1945-1975"
            if sum(len(g) for g in groups) >= 8:
                return True
        return False

    @staticmethod
    def _has_contact(user_agent: str) -> bool:
        if re.search(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+", user_agent):
            return True
        hosts = [h.lower() for h in re.findall(r"https?://([^/\s;)]+)", user_agent)]
        return any(not h.endswith(("mediawiki.org", "wikipedia.org", "wikimedia.org")) for h in hosts)

    def __init__(self, *, timeout_s: float = 5.0, max_chars: int = 500,
                 user_agent: str = "VoicePlatform/0.1 (local evaluation; https://www.mediawiki.org/wiki/API:Etiquette)") -> None:
        if timeout_s <= 0 or not 100 <= max_chars <= 2000 or not user_agent.strip():
            raise ValueError("invalid Wikipedia search options")
        self.timeout_s = timeout_s
        self.max_chars = max_chars
        self.user_agent = user_agent
        self._client: Any = None
        if not self._has_contact(user_agent):
            # Not invented here: only the operator can say who to contact.
            log.warning(
                "wikipedia_vi user_agent has no operator contact (an e-mail or a URL of your own): %r. "
                "Wikimedia's User-Agent policy requires one and may throttle or block requests without it; "
                "set models.search.options.user_agent.", user_agent,
            )

    async def start(self) -> None:
        import httpx
        self._client = httpx.AsyncClient(timeout=self.timeout_s,
                                         headers={"User-Agent": self.user_agent}, follow_redirects=False)

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def search(self, request: SearchRequest) -> SearchResult:
        import httpx

        started = now_ms()
        query = self._SOURCE_WORD.sub(" ", unicodedata.normalize("NFC", request.query)).strip(" .,:;?!")
        if self._PRIVATE.search(query) or self._has_identifier(query) or self._CURRENT.search(query):
            return SearchResult(request, False, "Nguồn bách khoa không có dữ liệu cá nhân hoặc dữ liệu thời gian thực.",
                                now_ms() - started, source=self.name, error="unsupported_query")
        subject = self._PLACE_SUBJECT.search(query)
        subject_terms = self._terms(subject.group(2)) if subject else set()
        # Search for the subject, not the whole spoken sentence: CirrusSearch
        # wants every word, and "… được không ạ" is not in the article.
        lookup = (f"{subject.group(1)} {subject.group(2)}" if subject else
                  " ".join(w for w in re.findall(r"\w+", query) if w.casefold() not in self._STOP)) or query
        terms = self._terms(lookup)
        if not terms:
            return SearchResult(request, False, "Câu hỏi tra cứu đang trống.",
                                now_ms() - started, source=self.name, error="empty_query")
        if self._client is None:
            await self.start()
        params = {
            "action": "query", "generator": "search", "gsrsearch": lookup[:300],
            "gsrlimit": 3, "gsrnamespace": 0, "prop": "extracts|info",
            "inprop": "url", "exintro": 1, "explaintext": 1,
            "exchars": self.max_chars, "format": "json", "formatversion": 2,
        }
        try:
            # httpx's timeout is per phase (connect, each read...); this is
            # the whole lookup.
            async with asyncio.timeout(self.timeout_s):
                response = await self._client.get(self.API, params=params)
                response.raise_for_status()
                payload = response.json()
            if not isinstance(payload, dict) or "error" in payload:
                raise ValueError("invalid search response")
            query_data = payload.get("query", {})
            pages = query_data.get("pages", []) if isinstance(query_data, dict) else []
            if not isinstance(pages, list):
                raise ValueError("invalid page list")
            candidates = []
            for page in pages:
                if not isinstance(page, dict):
                    continue
                title = str(page.get("title", "")).strip()
                url = str(page.get("fullurl", ""))
                excerpt = " ".join(str(page.get("extract", "")).split())[:self.max_chars].strip()
                title_terms = self._terms(title)
                excerpt_terms = self._terms(excerpt)
                hits = terms & title_terms
                # Common location words can make an unrelated page appear to
                # match. The named subject must appear in the title or excerpt.
                if len(subject_terms) >= 2 and not (
                    subject_terms <= title_terms or subject_terms <= excerpt_terms
                ):
                    continue
                # A search-engine ranking alone is not evidence that the page
                # is about the user's subject. Score how much of the TITLE the
                # question names: dividing by every word the user said let a
                # few polite words reject the exact page. A redirect ("Hồ Gươm"
                # -> "Hồ Hoàn Kiếm") is accepted when the excerpt names the
                # rest. Fail closed if every returned page is unrelated.
                coverage = len(hits) / len(title_terms) if title_terms else 0.0
                named = terms <= (title_terms | excerpt_terms)
                if (title and excerpt and url.startswith("https://vi.wikipedia.org/wiki/") and hits
                        and (named or (coverage >= 0.5 and len(hits) >= min(2, len(terms))))):
                    candidates.append((named, len(hits), coverage,
                                       -int(page.get("index", 999)), title, excerpt, url))
            if candidates:
                *_, title, excerpt, url = max(candidates)
                return SearchResult(request, True, excerpt, now_ms() - started,
                                    source=self.name, source_title=title, source_url=url)
            return SearchResult(request, False, "Không tìm thấy bài phù hợp trên Wikipedia tiếng Việt.",
                                now_ms() - started, source=self.name, error="no_result")
        except (httpx.HTTPError, TimeoutError, ValueError, TypeError, KeyError) as exc:
            return SearchResult(request, False, "Tạm thời không tra được Wikipedia tiếng Việt.",
                                now_ms() - started, source=self.name, error=type(exc).__name__)


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
