"""Barge-ins that turn out not to be interruptions, and hearing vs sending.

Three ways speech over the assistant is not the user taking the turn:

* noise — a cough, a door — too short to be a turn at all;
* a backchannel — "ừ", "vâng" — the listener saying they are still listening;
* the user's own sentence continuing after a pause long enough to confirm it.

The first two used to leave the answer cancelled and the assistant silent; the
third answered the second half of a sentence on its own.

Underneath all three: the talker runs faster than real time, so what the
server has SENT is ahead of what the user has HEARD. The engine replays the
client's playback schedule, and these tests hold it to that.
"""

from __future__ import annotations

import time

import pytest

from voiceplatform.app.simulate import Step, build_engine, feed, is_idle, is_speaking, wait_until
from voiceplatform.core.events import EventType


def _events(eng, type_: EventType) -> list:
    return [e for turn in eng.trace.turns.values() for e in turn.events if e.type is type_]


def _heard_content(eng) -> list[str]:
    response = eng._response
    if response is None:
        return []
    return [p.text for p in response.heard_by(time.monotonic()) if not p.filler]


@pytest.fixture
def slow_talker(config):
    # 80 ms of audio per character, synthesised at a quarter of real time: the
    # server is always well ahead of the client, as it is with ZeroTTS.
    config.models.tts.options = {"first_audio_delay_ms": 10, "rtf": 0.25, "ms_per_char": 80}
    return config


async def _start(
    config, script, *, reply: str | None = None, token_delay_ms: float = 1, first_token_ms: float = 10
):
    config.models.asr.options = {"partial_every_frames": 5, "script": script}
    llm = {"first_token_delay_ms": first_token_ms, "token_delay_ms": token_delay_ms}
    if reply is not None:
        llm["reply"] = reply
    config.models.llm.options = llm
    eng, sink = build_engine(config)
    await eng.models.start()
    await eng.start()
    return eng, sink


async def _stop(eng):
    await eng.close()
    await eng.models.close()


LONG = "Hà Nội là thủ đô. Thành phố có nghìn năm lịch sử. Hồ Gươm nằm ở trung tâm. Phố cổ có ba mươi sáu phố."


async def _ask_and_hear_one_phrase(eng) -> None:
    await feed(eng, [Step("silence", 100), Step("speech", 600), Step("silence", 500)])
    assert await wait_until(eng, is_speaking, max_ms=10000)
    assert await wait_until(eng, lambda e: _heard_content(e), max_ms=10000)


async def test_a_cough_over_the_answer_hands_it_back(slow_talker):
    eng, sink = await _start(slow_talker, ["kể về Hà Nội"], reply=LONG)
    try:
        await _ask_and_hear_one_phrase(eng)
        cut = eng.gen.current

        # 140 ms: enough to fire the barge-in (4 frames), too short for a turn.
        # 800 ms of silence: the mock's partial ends in "kể", a connector, so
        # the detector rightly holds for max_silence_ms before deciding.
        await feed(eng, [Step("speech", 140), Step("silence", 800)])

        assert eng.counters.get("barge_ins") == 1
        assert eng.counters.get("resumed") == 1
        assert await wait_until(eng, is_speaking, max_ms=5000)
        assert eng.gen.current != cut
        resumed = _events(eng, EventType.RESUMED)
        assert resumed and resumed[-1].data["why"] == "noise"

        # It finishes, and history holds one answered turn — no noise turn, not
        # marked as interrupted, and the whole answer.
        assert await wait_until(eng, is_idle, max_ms=30000)
        assert [t.user_text for t in eng.context.turns] == ["kể về Hà Nội"]
        assert eng.context.turns[-1].interrupted is False
        assert eng.context.turns[-1].assistant_text == LONG
    finally:
        await _stop(eng)


async def test_the_resume_starts_at_the_phrase_being_heard_not_the_one_being_sent(slow_talker):
    eng, sink = await _start(slow_talker, ["kể về Hà Nội"], reply=LONG)
    try:
        await _ask_and_hear_one_phrase(eng)
        heard = _heard_content(eng)
        sent = [s.phrase.text for s in eng._response.sent]
        assert len(sent) > len(heard), "test needs the server ahead of the client"

        await feed(eng, [Step("speech", 140), Step("silence", 800)])
        assert await wait_until(eng, lambda e: e._response and (e._response.sent or e._response.current), max_ms=5000)
        # The kept audio of the cut phrase goes out at once, so by now it may
        # already be in `sent` with the next phrase current.
        first = eng._response.sent[0].phrase if eng._response.sent else eng._response.current
        # Not a repeat of what was heard, and not a skip over what was only sent.
        assert first.text not in heard
        assert first.text == sent[len(heard)]
    finally:
        await _stop(eng)


async def test_a_cut_answer_keeps_being_written_instead_of_being_asked_again(slow_talker):
    # Slow tokens: the LLM is still writing when the cough lands.
    eng, sink = await _start(slow_talker, ["kể về Hà Nội"], reply=LONG, token_delay_ms=150)
    try:
        await _ask_and_hear_one_phrase(eng)
        assert not eng._response.llm_done, "test needs the LLM still streaming"
        await feed(eng, [Step("speech", 140), Step("silence", 800)])
        assert eng.counters.get("resumed") == 1
        assert await wait_until(eng, is_idle, max_ms=30000)
        assert len(_events(eng, EventType.LLM_START)) == 1
        assert eng.context.turns[-1].assistant_text == LONG
    finally:
        await _stop(eng)


async def test_a_backchannel_is_listened_through_not_answered(slow_talker):
    eng, sink = await _start(slow_talker, ["kể về Hà Nội", "ừ"], reply=LONG)
    try:
        await _ask_and_hear_one_phrase(eng)
        await feed(eng, [Step("speech", 400), Step("silence", 500)])

        assert await wait_until(eng, lambda e: e.counters.get("resumed") == 1, max_ms=5000)
        assert eng.counters.get("backchannels") == 1
        assert await wait_until(eng, is_idle, max_ms=30000)
        # "ừ" never became a user turn the model had to answer.
        assert [t.user_text for t in eng.context.turns] == ["kể về Hà Nội"]
    finally:
        await _stop(eng)


async def test_a_real_interruption_is_answered_and_the_cut_llm_is_stopped(slow_talker):
    eng, sink = await _start(
        slow_talker, ["kể về Hà Nội", "thôi cho tôi hỏi giá vàng"], reply=LONG, token_delay_ms=40
    )
    try:
        await _ask_and_hear_one_phrase(eng)
        await feed(eng, [Step("speech", 400), Step("silence", 500)])

        assert await wait_until(
            eng, lambda e: e.context.turns[-1].user_text == "thôi cho tôi hỏi giá vàng", max_ms=5000
        )
        assert eng.counters.get("resumed") is None
        assert eng._drain_targets == {}
        first = eng.context.turns[0]
        assert first.interrupted is True
        # History holds what was HEARD — not everything that had been sent.
        assert first.spoken_text and len(first.spoken_text) < len(LONG)
    finally:
        await _stop(eng)


async def test_talking_over_the_playback_tail_still_interrupts(slow_talker):
    """Everything sent, the client still playing: that is still the assistant's turn.

    The turn used to end on the last frame SENT. With a talker faster than real
    time the last seconds were still playing, barge-in was already disarmed, and
    talking over them opened a new turn with the old answer playing on top.
    """
    slow_talker.models.tts.options = {"first_audio_delay_ms": 5, "rtf": 0.02, "ms_per_char": 60}
    eng, sink = await _start(slow_talker, ["kể về Hà Nội", "dừng lại đi"], reply=LONG)
    try:
        await _ask_and_hear_one_phrase(eng)
        assert await wait_until(
            eng, lambda e: e._response and e._response.current is None and e._response.queue.empty()
            and len(e._response.sent) >= 4, max_ms=5000,
        )
        assert is_speaking(eng), "the client is still playing; the turn must not be over"
        await feed(eng, [Step("speech", 300)])
        assert eng.counters.get("barge_ins") == 1
        assert "playback_reset" in [m.type for m in sink.control]
    finally:
        await _stop(eng)


async def test_a_sentence_continued_in_the_think_window_is_one_turn(config):
    eng, sink = await _start(
        config, ["chuyển tiền cho", "số tài khoản một hai ba"], first_token_ms=1500
    )
    try:
        # "cho" at the end already makes the detector hold for max_silence_ms;
        # this is the pause that outlasts even that hold.
        await feed(eng, [Step("silence", 100), Step("speech", 600), Step("silence", 800)])
        assert eng.state.state.value == "thinking", "test needs a turn still thinking"

        await feed(eng, [Step("speech", 600), Step("silence", 500)])

        merged = "chuyển tiền cho số tài khoản một hai ba"
        assert await wait_until(eng, lambda e: e.context.turns[-1].user_text == merged, max_ms=5000)
        assert [t.user_text for t in eng.context.turns] == [merged]
        assert eng.counters.get("turns_merged") == 1
        finals = [m.data["text"] for m in sink.control if m.type == "transcript" and m.data.get("final")]
        assert finals[-1] == merged
    finally:
        await _stop(eng)


async def test_nothing_is_merged_once_the_answer_was_heard(slow_talker):
    eng, sink = await _start(slow_talker, ["kể về Hà Nội", "còn Huế thì sao"], reply=LONG)
    try:
        await _ask_and_hear_one_phrase(eng)
        await feed(eng, [Step("speech", 400), Step("silence", 500)])
        assert await wait_until(
            eng, lambda e: e.context.turns[-1].user_text == "còn Huế thì sao", max_ms=5000
        )
        assert eng.counters.get("turns_merged") is None
    finally:
        await _stop(eng)


async def test_the_interrupt_button_never_resumes(slow_talker):
    eng, sink = await _start(slow_talker, ["kể về Hà Nội"], reply=LONG)
    try:
        await _ask_and_hear_one_phrase(eng)
        await eng.interrupt("client button")
        assert await wait_until(eng, is_idle, max_ms=2000)
        await feed(eng, [Step("silence", 600)])
        assert eng.counters.get("resumed") is None
        assert is_idle(eng)
    finally:
        await _stop(eng)


async def test_resume_can_be_turned_off(slow_talker):
    slow_talker.conversation.barge_in.resume_after_false = False
    eng, sink = await _start(slow_talker, ["kể về Hà Nội"], reply=LONG)
    try:
        await _ask_and_hear_one_phrase(eng)
        await feed(eng, [Step("speech", 140), Step("silence", 800)])
        assert await wait_until(eng, is_idle, max_ms=3000)
        assert eng.counters.get("resumed") is None
    finally:
        await _stop(eng)


async def test_a_backchannel_over_the_answer_is_not_held_as_a_hesitation(slow_talker):
    # "ừ" alone is a hesitation to the turn detector (max_silence_ms), which is
    # right when the user is starting a sentence and wrong when they are
    # nodding along to the assistant.
    eng, sink = await _start(slow_talker, ["kể về Hà Nội", "ừ"], reply=LONG)
    try:
        await _ask_and_hear_one_phrase(eng)
        await feed(eng, [Step("speech", 400)])
        assert eng._interrupted is not None
        eng._partial_text = "ừ"
        base = slow_talker.conversation.turn_detection.silence_ms
        assert await eng._required_silence() == base
        eng._partial_text = "ừ nhưng mà"   # not a backchannel: the detector decides
        assert await eng._required_silence() == slow_talker.conversation.turn_detection.max_silence_ms
    finally:
        await _stop(eng)


async def test_a_backchannel_misheard_as_one_word_is_still_a_backchannel(slow_talker):
    # gipformer on a real "ừ": partial "ừ", final "từ".
    eng, sink = await _start(slow_talker, ["kể về Hà Nội", "từ"], reply=LONG)
    eng.models.asr.peek_partial = lambda frames: "ừ"
    try:
        await _ask_and_hear_one_phrase(eng)
        await feed(eng, [Step("speech", 400), Step("silence", 500)])
        assert await wait_until(eng, lambda e: e.counters.get("resumed") == 1, max_ms=5000)
        assert eng.counters.get("backchannels") == 1
        assert [t.user_text for t in eng.context.turns] == ["kể về Hà Nội"]
    finally:
        await _stop(eng)


async def test_a_two_word_mishearing_of_a_backchannel_is_still_a_backchannel(slow_talker):
    # gipformer on a real "vâng ạ": partial "vâng", final "thân ạ".
    eng, sink = await _start(slow_talker, ["kể về Hà Nội", "thân ạ"], reply=LONG)
    eng.models.asr.peek_partial = lambda frames: "vâng"
    try:
        await _ask_and_hear_one_phrase(eng)
        await feed(eng, [Step("speech", 400), Step("silence", 500)])
        assert await wait_until(eng, lambda e: e.counters.get("resumed") == 1, max_ms=5000)
        assert [t.user_text for t in eng.context.turns] == ["kể về Hà Nội"]
    finally:
        await _stop(eng)


async def test_two_words_that_ask_for_something_are_a_real_turn(slow_talker):
    eng, sink = await _start(slow_talker, ["kể về Hà Nội", "ừ dừng"], reply=LONG)
    eng.models.asr.peek_partial = lambda frames: "ừ"
    try:
        await _ask_and_hear_one_phrase(eng)
        await feed(eng, [Step("speech", 400), Step("silence", 500)])
        assert await wait_until(eng, lambda e: len(e.context.turns) == 2, max_ms=5000)
        assert not eng.counters.get("resumed")
        assert eng.context.turns[-1].user_text == "ừ dừng"
    finally:
        await _stop(eng)
