"""Process-wide aggregation: a turn counts once, and only when it is over."""

from __future__ import annotations

from voiceplatform.core.events import Event, EventType
from voiceplatform.observability.metrics import MetricsRegistry
from voiceplatform.observability.trace import SessionTrace


def _row(ttfa: float) -> dict[str, float | None]:
    return {"turn_id": 1, "e2e_ttfa_ms": ttfa}


def test_a_turn_is_counted_once_however_often_it_is_scraped():
    """/metrics re-reads every live session, so it offers the same turn again.

    Without an identity the percentiles became a function of the polling
    interval: one turn, three scrapes, n=3.
    """
    registry = MetricsRegistry()
    assert registry.observe_turn(_row(400.0), key=("s1", 1)) is True
    assert registry.observe_turn(_row(400.0), key=("s1", 1)) is False
    assert registry.observe_turn(_row(400.0), key=("s1", 1)) is False
    assert registry.snapshot()["latency"]["e2e_ttfa_ms"]["n"] == 1


def test_the_same_turn_number_in_another_session_is_another_turn():
    registry = MetricsRegistry()
    registry.observe_turn(_row(400.0), key=("s1", 1))
    registry.observe_turn(_row(600.0), key=("s2", 1))
    assert registry.snapshot()["latency"]["e2e_ttfa_ms"]["n"] == 2


def test_without_a_key_nothing_is_deduplicated():
    registry = MetricsRegistry()
    registry.observe_turn(_row(400.0))
    registry.observe_turn(_row(400.0))
    assert registry.snapshot()["latency"]["e2e_ttfa_ms"]["n"] == 2


def _record(trace: SessionTrace, type: EventType, turn_id: int, ts: float) -> None:
    trace.record(Event(type=type, session_id="s1", turn_id=turn_id, ts_ms=ts))


def test_a_turn_still_in_flight_is_not_offered_to_the_aggregate():
    """Half a turn's fields are empty until it ends.

    Counting it early and then remembering it as counted is how the real TTFA,
    which lands a second later, would be dropped for good.
    """
    trace = SessionTrace("s1")
    _record(trace, EventType.TURN_CONFIRMED, 1, 0.0)
    assert trace.settled_metrics_rows(current_turn_id=1) == []

    _record(trace, EventType.TTS_FIRST_AUDIO, 1, 400.0)
    _record(trace, EventType.TURN_END, 1, 900.0)
    rows = trace.settled_metrics_rows(current_turn_id=1)
    assert [r["turn_id"] for r in rows] == [1]
    assert rows[0]["e2e_ttfa_ms"] == 400.0


def test_an_earlier_turn_is_settled_even_without_its_own_end_event():
    """A barge-in ends a turn without a TURN_END; a newer turn settles it."""
    trace = SessionTrace("s1")
    _record(trace, EventType.TURN_CONFIRMED, 1, 0.0)
    _record(trace, EventType.TTS_FIRST_AUDIO, 1, 300.0)
    _record(trace, EventType.BARGE_IN, 1, 800.0)
    _record(trace, EventType.PLAYBACK_RESET, 1, 850.0)
    _record(trace, EventType.TURN_CONFIRMED, 2, 900.0)

    rows = trace.settled_metrics_rows(current_turn_id=2)
    assert [r["turn_id"] for r in rows] == [1]
    assert rows[0]["barge_in_stop_ms"] == 50.0
