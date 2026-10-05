"""Per-turn timeline and the latencies derived from it.

The metric that decides whether a session feels alive is time-to-first-audio,
not total response time: a five-second answer that starts in 400 ms feels
immediate, a two-second answer that starts in two seconds feels broken. So
every stage timestamp is kept per turn and TTFA is computed from the moment the
turn was confirmed, which is the first instant the user could expect a reply.
"""

from __future__ import annotations

import json
import os
import re
import stat
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..core.events import Event, EventType
from .stages import measurements

# Events whose rate the CLIENT decides: playback reports and clock syncs. One
# WebSocket could otherwise append them without end (measured 02/10: 20 000
# messages -> 20 000 events in one turn, 4.5 MB per /turns poll). Each phrase
# keeps its first reports and its latest one; a turn holds a bounded number.
CLIENT_EVENT_TYPES = frozenset(e.value for e in (
    EventType.PLAYBACK_STARTED, EventType.PLAYBACK_SIGNAL_STARTED, EventType.PLAYBACK_STOPPED,
    EventType.PLAYBACK_UNDERRUN, EventType.PLAYBACK_RESUMED, EventType.PLAYBACK_BUFFER,
    EventType.PLAYBACK_GENERATION_END, EventType.CLIENT_CLOCK_SYNC,
))
CLIENT_EVENTS_PER_KEY = 64      # per (phrase, event type): a 30 s phrase sends ~60 buffer reports
CLIENT_EVENTS_PER_TURN = 2048   # all client events of one turn (real turns: < 100 on average)
_GENERATED_TRACE = re.compile(r"s\d{4,}-[0-9a-f]{8}\.jsonl")


@dataclass(slots=True)
class TurnTrace:
    session_id: str
    turn_id: int
    events: list[Event] = field(default_factory=list)
    firsts: dict[str, float] = field(default_factory=dict)
    lasts: dict[str, float] = field(default_factory=dict)

    feedback_dropped: int = 0          # client events folded into a "latest" slot or refused

    _stage_count: int = field(default=-1, init=False, repr=False)
    _stage_cache: Any = field(default=None, init=False, repr=False)
    _version: int = field(default=0, init=False, repr=False)
    _client_count: dict = field(default_factory=dict, init=False, repr=False)
    _client_last: dict = field(default_factory=dict, init=False, repr=False)
    _client_total: int = field(default=0, init=False, repr=False)

    def stages(self):
        # Finished history is scraped repeatedly; deriving it on every poll
        # created event-loop stalls that altered the latency being measured.
        if self._stage_count != self._version:
            self._stage_cache = measurements(self.events)
            self._stage_count = self._version
        return self._stage_cache

    def add(self, event: Event) -> None:
        name = event.type.value
        if name in CLIENT_EVENT_TYPES and not self._admit_client_event(event, name):
            return
        self.events.append(event)
        self._version += 1
        self.firsts.setdefault(name, event.ts_ms)
        self.lasts[name] = event.ts_ms

    def _admit_client_event(self, event: Event, name: str) -> bool:
        """True to append it; otherwise it overwrites this key's latest slot."""
        key = (event.data.get("phrase_id"), name)
        count = self._client_count.get(key, 0)
        if count < CLIENT_EVENTS_PER_KEY and self._client_total < CLIENT_EVENTS_PER_TURN:
            self._client_count[key] = count + 1
            self._client_total += 1
            self._client_last[key] = len(self.events)
            return True
        self.feedback_dropped += 1
        slot = self._client_last.get(key)
        if slot is not None:
            self.events[slot] = event
            self._version += 1
            self.lasts[name] = max(self.lasts.get(name, event.ts_ms), event.ts_ms)
        return False

    @property
    def finished(self) -> bool:
        """True once backend processing ended; client feedback may arrive later."""
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
            "asr_stream_duration_ms": self._span(e.ASR_START.value, e.ASR_FINAL.value),
            "asr_final_ms": self._span(e.ASR_FINALIZE_START.value, e.ASR_FINALIZE_END.value),
            "asr_first_partial_ms": self._span(e.ASR_START.value, e.ASR_FIRST_PARTIAL.value),
            "llm_ttft_ms": self._span(e.LLM_START.value, e.LLM_FIRST_TOKEN.value),
            "llm_total_ms": self._span(e.LLM_START.value, e.LLM_COMPLETE.value),
            "tts_ttfa_ms": self._span(e.TTS_START.value, e.TTS_FIRST_AUDIO.value),
            "tool_ms": self._span(e.TOOL_START.value, e.TOOL_COMPLETE.value),
            # Legacy alias: confirmed turn -> first audio send, including fillers.
            "e2e_ttfa_ms": (
                round(first_audio - confirmed_at, 1)
                if (confirmed_at is not None and first_audio is not None)
                else None
            ),
            "response_total_ms": self._span(e.TURN_CONFIRMED.value, e.TTS_COMPLETE.value),
            # How fast an interruption actually silenced the speaker.
            "barge_in_stop_ms": self._span(e.BARGE_IN.value, e.PLAYBACK_RESET.value),
        }
        rounds, operations, phrases = self.stages()
        content = [p for p in phrases if p["role"] == "content"]
        def first_value(items, name):
            return min((p[name] for p in items if p.get(name) is not None), default=None)
        def since(stamp, origin=confirmed_at):
            return round(stamp-origin,3) if stamp is not None and origin is not None and stamp >= origin else None
        # A discarded shadow-mode guess is not this turn's request.
        speech_rounds = [r for r in rounds if r.get("role") not in ("search", "speculation_discarded")]
        actual = next((r for r in speech_rounds if r["request_ttft_ms"] is not None), None)
        if speech_rounds:
            out["llm_ttft_ms"] = actual["request_ttft_ms"] if actual else None
            out["llm_total_ms"] = round(sum(r["request_total_ms"] or 0 for r in speech_rounds),3)
        out.update({
            "first_phrase_ready_ms": since(first_value(content,"ready_at_ms")),
            "first_any_audio_sent_ms": since(first_value(phrases,"audio_sent_at_ms")),
            "first_content_audio_sent_ms": since(first_value(content,"audio_sent_at_ms")),
            "content_playback_start_ms": since(first_value(content,"playback_started_at_ms")),
            "content_playback_signal_ms": since(first_value(content,"playback_signal_at_ms")),
            "last_voice_to_content_sent_ms": since(first_value(content,"audio_sent_at_ms"), self.firsts.get(e.SPEECH_LAST_FRAME.value)),
            "content_underruns": sum(p["underruns"] for p in content),
            "content_gap_ms": round(sum(p["gap_ms"] for p in content),3),
            "content_phrase_gap_ms": round(sum(p.get("previous_content_gap_ms") or 0 for p in content),3),
            "playback_clock_uncertainty_ms": max((e.data.get("clock_uncertainty_ms",0) for e in self.events if e.type is EventType.PLAYBACK_STARTED), default=None),
        })
        return out

    def outcome(self):
        errors = [e.data.get("stage", e.type.value) for e in self.events if e.type in (EventType.ERROR, EventType.TOOL_FAILED)]
        errors.extend("search" for e in self.events if e.type in (EventType.SEARCH_RESULT, EventType.SEARCH_DELIVERED) and e.data.get("ok") is False)
        fallback = any(e.data.get("role")=="fallback" for e in self.events)
        cancelled = any(e.type is EventType.CANCEL for e in self.events)
        content = any(e.type is EventType.AUDIO_SENT and e.data.get("role")=="content" for e in self.events)
        ended = any(e.type is EventType.TURN_END and e.data.get("answered") for e in self.events)
        return {"success": bool(ended and content and not errors and not fallback and not cancelled),
                "answered": bool(ended), "has_content": content, "fallback": fallback,
                "cancelled": cancelled, "errors": errors}

    def summary(self) -> dict[str, Any]:
        rounds, operations, phrases = self.stages()
        return {
            "schema_version": 2,
            "outcome": self.outcome(),
            "llm_rounds": rounds, "operations": operations, "phrases": phrases,
            "session_id": self.session_id,
            "turn_id": self.turn_id,
            "metrics": self.metrics(),
            "feedback_dropped": self.feedback_dropped,
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
            # Before the first turn only clock syncs arrive in volume; keep the
            # first ones and let the last slot hold the latest event.
            if len(self.session_events) >= CLIENT_EVENTS_PER_TURN:
                self.session_events[-1] = event
            else:
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
            turn = self.turns[turn_id]
            rounds, operations, phrases = turn.stages()
            rows.append({"turn_id": turn_id, **turn.metrics(), "outcome": turn.outcome(),
                         "llm_rounds": rounds, "operations": operations, "phrases": phrases})
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
        path.mkdir(parents=True, exist_ok=True, mode=0o700)
        file = path / f"{self.session_id}.jsonl"
        descriptor = os.open(file, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as fh:
            for turn_id in sorted(self.turns):
                fh.write(json.dumps(self.turns[turn_id].summary(), ensure_ascii=False) + "\n")
        return file


def secure_trace_dir(directory: str | Path) -> int:
    """Owner-only access for traces an older build wrote with the umask.

    New files are created 0600, but files from before that change kept
    0664 (289 of 313 on the deployed host, 02/10). Only generated session
    names are touched and symlinks are skipped: the target is not ours.
    Returns how many files were tightened.
    """
    path = Path(directory)
    if not path.is_dir():
        return 0
    if (path.stat().st_mode & 0o777) != 0o700:
        os.chmod(path, 0o700)
    fixed = 0
    for file in path.iterdir():
        if not _GENERATED_TRACE.fullmatch(file.name):
            continue
        try:
            descriptor = os.open(file, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        except OSError:
            continue                   # a symlink (ELOOP), or gone already
        try:
            info = os.fstat(descriptor)
            if stat.S_ISREG(info.st_mode) and stat.S_IMODE(info.st_mode) != 0o600:
                os.fchmod(descriptor, 0o600)
                fixed += 1
        finally:
            os.close(descriptor)
    return fixed


def prune_session_traces(directory: str | Path, retention_days: int, *, now: float | None = None) -> int:
    """Remove only old completed trace files with generated session IDs."""
    if retention_days <= 0:
        return 0
    path = Path(directory)
    if not path.is_dir():
        return 0
    cutoff = (time.time() if now is None else now) - retention_days * 86400
    removed = 0
    for file in path.glob("s*.jsonl"):
        if not _GENERATED_TRACE.fullmatch(file.name) or file.is_symlink():
            continue
        try:
            if file.stat().st_mtime < cutoff:
                file.unlink()
                removed += 1
        except FileNotFoundError:
            continue
    return removed
