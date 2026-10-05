"""Process-wide latency aggregation.

Percentiles over the same per-turn fields the trace computes, so a dashboard
and a single session's log can never disagree about what "TTFA" means.
"""

from __future__ import annotations

import math
from collections import defaultdict, deque
from typing import Any, Hashable

_FIELDS = (
    "asr_stream_duration_ms", "first_phrase_ready_ms", "first_any_audio_sent_ms",
    "first_content_audio_sent_ms", "content_playback_start_ms", "content_playback_signal_ms",
    "last_voice_to_content_sent_ms", "content_underruns", "content_gap_ms", "content_phrase_gap_ms",
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
    def __init__(self, window: int = 500, seen_window: int = 40000) -> None:
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
        if metrics.get("outcome", {}).get("success") is False:
            return False
        changed = False
        for field in _FIELDS:
            changed |= self.observe(field, metrics.get(field), key=(key, field) if key is not None else None)
        for row in metrics.get("llm_rounds", []):
            if row.get("outcome") != "complete":
                continue
            for source, field in (("queue_ms","llm_queue_ms"), ("request_ttft_ms","llm_request_ttft_ms"),
                                  ("request_first_tool_ms","llm_request_first_tool_ms"), ("request_total_ms","llm_request_total_ms")):
                field = "search_" + field if row.get("role") == "search" else field
                self.observe(field, row.get(source), key=(key, row["request_id"], field))
        for row in metrics.get("operations", []):
            if row.get("outcome") != "complete":
                continue
            if row.get("stage") == "tts" and row.get("role") != "content":
                continue
            prefix = row.get("stage", "model")
            if prefix == "asr":
                prefix += "_" + (row.get("operation") or "unknown")
            for source in ("queue_ms", "compute_ms", "first_chunk_ms", "lock_wait_ms", "rtf"):
                field = f"{prefix}_{source}"
                self.observe(field, row.get(source), key=(key,row["request_id"],field))
        return changed

    def observe(self, field, value, *, key=None):
        if value is None or not isinstance(value, (int, float)) or not math.isfinite(value):
            return False
        if key is not None:
            if key in self._seen:
                return False
            if len(self._seen_order) == self._seen_order.maxlen and self._seen_order:
                self._seen.discard(self._seen_order[0])
            self._seen_order.append(key)
            self._seen.add(key)
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
        return round(ordered[index], 3)

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
                "max": round(max(values), 3),
            }
        return out
