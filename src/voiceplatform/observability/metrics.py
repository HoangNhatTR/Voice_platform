"""Process-wide latency aggregation.

Percentiles over the same per-turn fields the trace computes, so a dashboard
and a single session's log can never disagree about what "TTFA" means.
"""

from __future__ import annotations

import math
from collections import defaultdict, deque
from typing import Any, Hashable

_FIELDS = (
    "endpoint_ms",
    "asr_first_partial_ms",
    "asr_final_ms",
    "llm_ttft_ms",
    "llm_total_ms",
    "tts_ttfa_ms",
    "tool_ms",
    "e2e_ttfa_ms",
    "response_total_ms",
    "barge_in_stop_ms",
)


class MetricsRegistry:
    def __init__(self, window: int = 500, seen_window: int = 4000) -> None:
        self._series: dict[str, deque[float]] = defaultdict(lambda: deque(maxlen=window))
        self.counters: dict[str, int] = defaultdict(int)
        # A live session is re-read on every scrape, so the same turn is
        # offered again and again. Without an identity the percentiles became
        # a function of how often /metrics was polled: one turn, three scrapes,
        # n=3. The ring bounds the memory a long session can cost.
        self._seen: set[Hashable] = set()
        self._seen_order: deque[Hashable] = deque(maxlen=seen_window)

    def observe_turn(self, metrics: dict[str, float | None], *, key: Hashable = None) -> bool:
        """Record one turn. Returns False when this turn was already counted."""
        if key is not None:
            if key in self._seen:
                return False
            if len(self._seen_order) == self._seen_order.maxlen and self._seen_order:
                self._seen.discard(self._seen_order[0])
            self._seen_order.append(key)
            self._seen.add(key)
        for field in _FIELDS:
            value = metrics.get(field)
            if value is not None and math.isfinite(value):
                self._series[field].append(float(value))
        return True

    def incr(self, name: str, amount: int = 1) -> None:
        self.counters[name] += amount

    @staticmethod
    def _pct(values: list[float], q: float) -> float:
        if not values:
            return 0.0
        ordered = sorted(values)
        index = min(len(ordered) - 1, max(0, int(round(q * (len(ordered) - 1)))))
        return round(ordered[index], 1)

    def snapshot(self) -> dict[str, Any]:
        out: dict[str, Any] = {"counters": dict(self.counters), "latency": {}}
        for field, series in self._series.items():
            values = list(series)
            if not values:
                continue
            out["latency"][field] = {
                "n": len(values),
                "p50": self._pct(values, 0.5),
                "p95": self._pct(values, 0.95),
                "max": round(max(values), 1),
            }
        return out
