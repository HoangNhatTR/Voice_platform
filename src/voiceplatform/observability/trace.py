"""Per-turn timeline and the latencies derived from it.

The metric that decides whether a session feels alive is time-to-first-audio,
not total response time: a five-second answer that starts in 400 ms feels
immediate, a two-second answer that starts in two seconds feels broken. So
every stage timestamp is kept per turn and TTFA is computed from the moment the
turn was confirmed, which is the first instant the user could expect a reply.
"""

from __future__ import annotations

import json
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..core.events import Event, EventType


@dataclass(slots=True)
class TurnTrace:
    session_id: str
    turn_id: int
    events: list[Event] = field(default_factory=list)
    firsts: dict[str, float] = field(default_factory=dict)
    lasts: dict[str, float] = field(default_factory=dict)

    def add(self, event: Event) -> None:
        self.events.append(event)
        name = event.type.value
        self.firsts.setdefault(name, event.ts_ms)
        self.lasts[name] = event.ts_ms

    @property
    def finished(self) -> bool:
        """True once this turn can no longer gain events."""
        return EventType.TURN_END.value in self.firsts

    def _span(self, start: str, end: str) -> float | None:
        a, b = self.firsts.get(start), self.firsts.get(end)
        if a is None or b is None:
            return None
        return round(b - a, 1)

    def metrics(self) -> dict[str, float | None]:
        e = EventType
        first_audio = self.firsts.get(e.TTS_FIRST_AUDIO.value)
        # From the *last* endpoint candidate, not the first: when speech
        # resumes mid-pause the earlier candidate was correctly abandoned, and
        # measuring from it would report a wait that never happened.
        last_candidate = self.lasts.get(e.ENDPOINT_CANDIDATE.value)
        confirmed_at = self.firsts.get(e.TURN_CONFIRMED.value)
        out: dict[str, float | None] = {
            "endpoint_ms": (
                round(confirmed_at - last_candidate, 1)
                if (last_candidate is not None and confirmed_at is not None)
                else None
            ),
            "asr_final_ms": self._span(e.ASR_START.value, e.ASR_FINAL.value),
            "asr_first_partial_ms": self._span(e.ASR_START.value, e.ASR_FIRST_PARTIAL.value),
            "llm_ttft_ms": self._span(e.LLM_START.value, e.LLM_FIRST_TOKEN.value),
            "llm_total_ms": self._span(e.LLM_START.value, e.LLM_COMPLETE.value),
            "tts_ttfa_ms": self._span(e.TTS_START.value, e.TTS_FIRST_AUDIO.value),
            "tool_ms": self._span(e.TOOL_START.value, e.TOOL_COMPLETE.value),
            # The number that matters: confirmed turn -> user hears something.
            "e2e_ttfa_ms": (
                round(first_audio - confirmed_at, 1)
                if (confirmed_at is not None and first_audio is not None)
                else None
            ),
            "response_total_ms": self._span(e.TURN_CONFIRMED.value, e.TTS_COMPLETE.value),
            # How fast an interruption actually silenced the speaker.
            "barge_in_stop_ms": self._span(e.BARGE_IN.value, e.PLAYBACK_RESET.value),
        }
        return out

    def summary(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "turn_id": self.turn_id,
            "metrics": self.metrics(),
            "events": [ev.as_dict() for ev in self.events],
        }


class SessionTrace:
    def __init__(self, session_id: str, keep_turns: int = 200) -> None:
        self.session_id = session_id
        self.turns: dict[int, TurnTrace] = {}
        self._order: deque[int] = deque(maxlen=keep_turns)
        self.session_events: list[Event] = []

    def record(self, event: Event) -> None:
        if event.turn_id is None:
            self.session_events.append(event)
            return
        trace = self.turns.get(event.turn_id)
        if trace is None:
            trace = TurnTrace(session_id=self.session_id, turn_id=event.turn_id)
            self.turns[event.turn_id] = trace
            self._order.append(event.turn_id)
            while len(self.turns) > self._order.maxlen:
                oldest = min(self.turns)
                if oldest in self._order:
                    break
                self.turns.pop(oldest, None)
        trace.add(event)

    def turn(self, turn_id: int) -> TurnTrace | None:
        return self.turns.get(turn_id)

    def metrics_rows(self) -> list[dict[str, Any]]:
        rows = []
        for turn_id in sorted(self.turns):
            rows.append({"turn_id": turn_id, **self.turns[turn_id].metrics()})
        return rows

    def settled_metrics_rows(self, current_turn_id: int | None = None) -> list[dict[str, Any]]:
        """Rows for turns that can no longer change.

        An aggregate that remembers what it has already counted must never be
        handed a turn still in flight: half its fields are empty, and once the
        turn is marked as counted the real TTFA that arrives a second later is
        dropped for good.
        """
        return [
            row
            for row in self.metrics_rows()
            if not (
                row["turn_id"] == current_turn_id
                and not self.turns[row["turn_id"]].finished
            )
        ]

    def write_jsonl(self, directory: str | Path) -> Path:
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        file = path / f"{self.session_id}.jsonl"
        with file.open("w", encoding="utf-8") as fh:
            for turn_id in sorted(self.turns):
                fh.write(json.dumps(self.turns[turn_id].summary(), ensure_ascii=False) + "\n")
        return file
