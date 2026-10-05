"""A false interruption must not tell the client the answer is over.

The interjection's own turn — "ừ", or a cough — ends in IDLE inside the
engine before the cut answer resumes. Until 02/10/2026 that IDLE also went to
the client as `state: idle`, a flash in the middle of an answer that carried
on: conversation_check took it for the end and closed the session 412 ms into
the resume. Nothing on the wire can tell that flash from "done".
"""

from __future__ import annotations

import pytest

from voiceplatform.app.simulate import Step, feed, is_idle, wait_until

from test_engine_false_barge_in import LONG, _ask_and_hear_one_phrase, _start, _stop, slow_talker  # noqa: F401


def _states_after_barge_in(sink) -> list[str]:
    types = [m.type for m in sink.control]
    start = types.index("playback_reset")
    return [m.data.get("state") for m in sink.control[start:] if m.type == "state"]


@pytest.mark.parametrize("script,interjection", [
    (["kể về Hà Nội"], [Step("speech", 140), Step("silence", 800)]),            # cough: noise path
    (["kể về Hà Nội", "ừ"], [Step("speech", 400), Step("silence", 500)]),       # backchannel turn
])
async def test_the_client_never_sees_idle_between_the_cut_and_the_resume(slow_talker, script, interjection):
    eng, sink = await _start(slow_talker, script, reply=LONG)
    try:
        await _ask_and_hear_one_phrase(eng)
        await feed(eng, interjection)
        assert await wait_until(eng, lambda e: e.counters.get("resumed") == 1, max_ms=5000)
        assert await wait_until(eng, is_idle, max_ms=30000)
        states = _states_after_barge_in(sink)
        if len(script) > 1:
            # The "ừ" turn says why it was not answered: a checker must not
            # count it as a failed turn.
            ends = [e.data for turn in eng.trace.turns.values() for e in turn.events
                    if e.type.value == "turn_end" and not e.data.get("answered")]
            assert {"answered": False, "reason": "backchannel"} in ends
        # Exactly one idle, at the very end, after the resumed answer.
        assert states.count("idle") == 1 and states[-1] == "idle", states
        assert "thinking" in states[:-1]
    finally:
        await _stop(eng)


async def test_a_failed_resume_still_ends_in_idle(slow_talker):
    slow_talker.conversation.barge_in.resume_after_false = False
    eng, sink = await _start(slow_talker, ["kể về Hà Nội"], reply=LONG)
    try:
        await _ask_and_hear_one_phrase(eng)
        await feed(eng, [Step("speech", 140), Step("silence", 800)])
        assert await wait_until(eng, is_idle, max_ms=5000)
        assert not eng.counters.get("resumed")
        assert _states_after_barge_in(sink)[-1] == "idle"
    finally:
        await _stop(eng)
