"""Theo dõi các yêu cầu tra cứu đang bay.

Chỗ này tồn tại vì một lý do: yêu cầu tra cứu KHÔNG thuộc về lượt nói đã sinh
ra nó. Người dùng ngắt lời, đổi chủ đề, hỏi thêm ba câu nữa — việc tra cứu vẫn
phải chạy tiếp và kết quả vẫn phải được nói ra. Nếu gắn nó vào generation thì
mọi lần ngắt lời đều giết luôn việc tra cứu, và người dùng chờ mãi một câu trả
lời không bao giờ đến.

Nhưng "không bị huỷ" không có nghĩa là "sống mãi": một kết quả về sau hai phút
thì nói ra chỉ làm người nghe bối rối, nên có hạn dùng.
"""

from __future__ import annotations

from collections import deque

from ..core.clock import now_ms
from ..tasks.search import SearchRequest, SearchResult


class PendingSearches:
    def __init__(self, max_inflight: int = 2, ttl_ms: float = 60000.0) -> None:
        self.max_inflight = max_inflight
        self.ttl_ms = ttl_ms
        self._inflight: dict[str, SearchRequest] = {}
        self._ready: deque[SearchResult] = deque()
        self.dropped_full = 0
        self.dropped_expired = 0
        self.dropped_stale = 0

    @property
    def inflight(self) -> int:
        return len(self._inflight)

    @property
    def has_ready(self) -> bool:
        return bool(self._ready)

    def open(self, query: str, turn_id: int) -> SearchRequest | None:
        """None khi đã quá nhiều yêu cầu đang bay — thà từ chối còn hơn xếp hàng."""
        if len(self._inflight) >= self.max_inflight:
            self.dropped_full += 1
            return None
        request = SearchRequest(query=query, turn_id=turn_id)
        self._inflight[request.id] = request
        return request

    def complete(self, result: SearchResult) -> bool:
        """False nếu yêu cầu đã bị bỏ (quá hạn) trong lúc chờ."""
        if self._inflight.pop(result.request.id, None) is None:
            self.dropped_stale += 1
            return False
        if result.request.age_ms > self.ttl_ms:
            self.dropped_expired += 1
            return False
        self._ready.append(result)
        return True

    def pop_ready(self) -> SearchResult | None:
        while self._ready:
            result = self._ready.popleft()
            if result.request.age_ms > self.ttl_ms:
                self.dropped_expired += 1
                continue
            return result
        return None

    def sweep(self) -> list[SearchRequest]:
        """Bỏ các yêu cầu bay quá lâu; trả về danh sách để ghi log."""
        expired = [r for r in self._inflight.values() if r.age_ms > self.ttl_ms]
        for request in expired:
            self._inflight.pop(request.id, None)
            self.dropped_expired += 1
        return expired

    def stats(self) -> dict[str, int]:
        return {
            "inflight": len(self._inflight),
            "ready": len(self._ready),
            "dropped_full": self.dropped_full,
            "dropped_expired": self.dropped_expired,
            "dropped_stale": self.dropped_stale,
        }
