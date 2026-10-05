"""Model-backed turn detection seat.

Takes any async probe that scores "this utterance is complete" in [0, 1]. The
optional data-only text model can be trained from human-labeled Vietnamese
clips and evaluated on speakers held out from training.

It is a seat with a working default: the heuristic detector is the fallback
whenever the probe is missing, slow or throws, so wiring a model in later never
becomes a hard dependency.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable

from .heuristic import HeuristicTurnDetector

Probe = Callable[[str], Awaitable[float]]


class SemanticTurnDetector:
    name = "semantic"

    def __init__(
        self,
        probe: Probe | None = None,
        silence_ms: float = 480.0,
        max_silence_ms: float = 1400.0,
        probe_timeout_ms: float = 60.0,
        complete_threshold: float = 0.6,
        fast_silence_ms: float = 0.0,
    ) -> None:
        self.probe = probe
        self.silence_ms = silence_ms
        self.max_silence_ms = max_silence_ms
        self.probe_timeout_ms = probe_timeout_ms
        self.complete_threshold = complete_threshold
        self._fallback = HeuristicTurnDetector(silence_ms, max_silence_ms, fast_silence_ms=fast_silence_ms)
        self.probe_failures = 0

    async def required_silence_ms(self, *, text: str, utterance_ms: float, stable: bool = False) -> float:
        base = self._fallback.evaluate(text, stable=stable)
        if self.probe is None or not (text or "").strip():
            return base
        try:
            score = await asyncio.wait_for(
                self.probe(text), timeout=self.probe_timeout_ms / 1000.0
            )
        except (asyncio.TimeoutError, Exception):
            # A detector that blocks the turn is worse than a dumb detector.
            self.probe_failures += 1
            return base
        score = max(0.0, min(1.0, float(score)))
        if score >= self.complete_threshold:
            return self.silence_ms
        # Confidence maps onto the wait: unsure means wait most of the budget.
        span = self.max_silence_ms - self.silence_ms
        return self.silence_ms + span * (1.0 - score / self.complete_threshold)
