"""The server's model of the client's playback, and what it is told to match.

The live client (measurement_schema 2, playback feedback on) plays through an
AudioWorklet: nothing until `playback_buffer_ms` is buffered (or the phrase's
end marker arrives), then back to back; running dry — an underrun, or a phrase
boundary with nothing queued — re-arms that buffer. The 0.06 s cushion per
frame is the legacy scheduler's, kept for clients without feedback.
"""

from __future__ import annotations

import pytest

from voiceplatform.app.simulate import build_engine, is_idle, wait_until
from voiceplatform.conversation.engine import Phrase, ResponseState
from voiceplatform.core.ids import GenerationKey

KEY = GenerationKey(session_id="s", turn_id=1, generation_id=1)


def _begin(response: ResponseState, phrase_id: str) -> None:
    response.begin_phrase(Phrase("một hai ba bốn", phrase_id=phrase_id))


def test_the_first_frame_waits_for_the_startup_buffer():
    r = ResponseState(key=KEY, startup_s=0.16)
    _begin(r, "p0")
    for at in (10.00, 10.01, 10.02):
        assert r.schedule(0.04, now=at) is None          # buffering, not playing yet
    assert r.current_start is None
    assert r.schedule(0.04, now=10.03) == pytest.approx(10.15)   # this frame fills it
    assert r.current_start == pytest.approx(10.03)
    assert r.play_end == pytest.approx(10.19)
    assert r.schedule(0.04, now=10.04) == pytest.approx(10.19)   # then back to back


def test_a_phrase_shorter_than_the_buffer_starts_on_its_end_marker():
    r = ResponseState(key=KEY, startup_s=0.16)
    _begin(r, "p0")
    assert r.schedule(0.08, now=5.00) is None
    r.end_phrase(now=5.01)
    assert r.current_start == pytest.approx(5.01)
    assert r.play_end == pytest.approx(5.09)


def test_an_underrun_rearms_the_startup_buffer():
    r = ResponseState(key=KEY, startup_s=0.16)
    _begin(r, "p0")
    for _ in range(4):
        r.schedule(0.04, now=10.0)                       # plays 10.00 - 10.16
    # The talker stalls past the end of what was buffered: the worklet stops
    # and buffers 160 ms again — not "now + 60 ms".
    for at in (10.50, 10.54, 10.58):
        assert r.schedule(0.04, now=at) is None
    r.schedule(0.04, now=10.62)
    assert r.play_end == pytest.approx(10.78)
    assert r.current_start == pytest.approx(10.0)        # the phrase itself started earlier


def test_a_drained_phrase_boundary_rearms_the_buffer():
    r = ResponseState(key=KEY, startup_s=0.16)
    _begin(r, "p0")
    for _ in range(4):
        r.schedule(0.04, now=1.0)
    r.end_phrase(now=1.0)                                # p0 plays until 1.16
    _begin(r, "p1")
    assert r.schedule(0.04, now=1.40) is None            # arrived after the client ran dry
    r.end_phrase(now=1.45)
    assert r.current_start == pytest.approx(1.45)


def test_a_phrase_arriving_while_the_last_still_plays_follows_it():
    r = ResponseState(key=KEY, startup_s=0.16)
    _begin(r, "p0")
    for _ in range(8):
        r.schedule(0.04, now=1.0)                        # plays 1.00 - 1.32
    r.end_phrase(now=1.0)
    _begin(r, "p1")
    assert r.schedule(0.04, now=1.10) == pytest.approx(1.32)
    assert r.current_start == pytest.approx(1.32)


def test_without_feedback_the_legacy_cushion_applies():
    r = ResponseState(key=KEY)
    _begin(r, "p0")
    assert r.schedule(0.04, now=1.0) == pytest.approx(1.06)
    assert r.current_start == pytest.approx(1.06)
    assert r.schedule(0.04, now=1.0) == pytest.approx(1.10)


async def test_ready_announces_the_configured_startup_buffer(config):
    config.conversation.barge_in.playback_startup_ms = 240
    eng, sink = build_engine(config)
    await eng.models.start()
    await eng.start()
    try:
        ready = next(m for m in sink.control if m.type == "ready")
        assert ready.data["playback_buffer_ms"] == 240
        eng.playback_feedback_enabled = True
        await eng.push_text("xin chào")
        assert await wait_until(eng, lambda e: e._response is not None, max_ms=2000, feed_silence=False)
        assert eng._response.startup_s == pytest.approx(0.24)
        assert await wait_until(eng, is_idle, max_ms=8000, feed_silence=False)
    finally:
        await eng.close()
        await eng.models.close()


class _Limiter:
    def __init__(self, snapshot: dict) -> None:
        self._snapshot = snapshot

    def snapshot(self) -> dict:
        return dict(self._snapshot)


async def test_a_queued_search_does_not_disable_speculation(config):
    eng, _ = build_engine(config)
    eng.models.llm.limiter = _Limiter({"active": 1, "waiting": 1, "parallel": 3,
                                       "speech_waiting": 0, "search_waiting": 1})
    assert eng._llm_has_room(1)
    eng.models.llm.limiter = _Limiter({"active": 1, "waiting": 1, "parallel": 3,
                                       "speech_waiting": 1, "search_waiting": 0})
    assert not eng._llm_has_room(1)
    # A limiter without priorities still counts every waiter.
    eng.models.llm.limiter = _Limiter({"active": 1, "waiting": 1, "parallel": 3})
    assert not eng._llm_has_room(1)
