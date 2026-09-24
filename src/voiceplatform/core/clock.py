"""Monotonic time helpers.

Latency numbers are the product here, so every timestamp comes from one place
and is monotonic: wall-clock jumps must never show up as a negative TTFA.
"""

from __future__ import annotations

import time


def now_ms() -> float:
    """Milliseconds on a monotonic clock."""
    return time.monotonic() * 1000.0


def wall_iso() -> str:
    """UTC wall clock, for logs a human reads."""
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
