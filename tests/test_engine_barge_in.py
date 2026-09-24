"""Interrupting the assistant: cancel, fence, and tell the client to stop."""

from __future__ import annotations

import pytest

from voiceplatform.app.simulate import (
    Step,
    build_engine,
    feed,
    is_idle,
    is_speaking,
    wait_until,
)
from voiceplatform.core.events import EventType


@pytest.fixture
async def engine(config):
    # Slow the talker down so there is an answer in progress to interrupt.
    config.models.tts.options = {"first_audio_delay_ms": 10, "rtf": 0.25, "ms_per_char": 80}
    eng, sink = build_engine(config)
    await eng.models.start()
    await eng.start()
    yield eng, sink
    await eng.close()
    await eng.models.close()


async def _speak_and_wait_for_answer(eng) -> bool:
    await feed(eng, [Step("silence", 100), Step("speech", 600), Step("silence", 500)])
    # Generous budget on purpose: the mock engines sleep on a wall clock, so a
    # loaded machine (another model loading next door, for instance) can push
    # first audio well past a tight deadline without anything being wrong.
    return await wait_until(eng, is_speaking, max_ms=10000)


async def test_user_speech_over_the_assistant_stops_it(engine):
    eng, sink = engine
    assert await _speak_and_wait_for_answer(eng)
    generation = eng.gen.current.generation_id

    await feed(eng, [Step("speech", 300)])

    types = [m.type for m in sink.control]
    assert "playback_reset" in types
    reset = next(m for m in sink.control if m.type == "playback_reset")
    assert reset.data["generation_id"] == generation

    # The contract is about what happens *after* the reset. Audio sent between
    # the first interrupting frame and the moment the interruption was
    # recognised is the assistant still legitimately speaking.
    after = [k for k in sink.audio_after_control("playback_reset") if k.generation_id == generation]
    assert after == []
    assert eng.counters.get("barge_ins") == 1


async def test_barge_in_opens_a_new_turn_that_keeps_the_interrupting_words(engine):
    eng, _ = engine
    assert await _speak_and_wait_for_answer(eng)
    turn_before = eng.gen.turn_id
    await feed(eng, [Step("speech", 300)])
    assert eng.gen.turn_id == turn_before + 1
    assert eng.state.state.value == "listening"
    # The new turn already has audio in it: the pre-roll carried the words
    # spoken before the interruption was recognised.
    assert eng._utterance_ms > 0


async def test_history_records_what_was_heard_not_what_was_generated(engine):
    eng, _ = engine
    assert await _speak_and_wait_for_answer(eng)
    await feed(eng, [Step("speech", 300)])
    turn = eng.context.turns[-1]
    assert turn.interrupted is True
    assert len(turn.spoken_text) <= len(turn.assistant_text)
    messages = eng.context.messages()
    assistant = [m for m in messages if m.role == "assistant"]
    assert assistant and "bị người dùng ngắt lời" in assistant[-1].content


async def test_explicit_interrupt_button_also_stops_playback(engine):
    eng, sink = engine
    assert await _speak_and_wait_for_answer(eng)
    await eng.interrupt("client button")
    assert "playback_reset" in [m.type for m in sink.control]
    assert await wait_until(eng, is_idle, max_ms=2000)


async def test_the_assistants_own_first_frames_do_not_interrupt_it(engine):
    eng, sink = engine
    assert await _speak_and_wait_for_answer(eng)
    # Silence from the microphone while the assistant talks must never cancel
    # it. (The guard window itself is measured in test_barge_in_detector.py.)
    await feed(eng, [Step("silence", 200)])
    assert eng.barge_in.stats.fired == 0
    assert "playback_reset" not in [m.type for m in sink.control]


async def test_stale_audio_is_dropped_and_counted(engine):
    eng, _ = engine
    assert await _speak_and_wait_for_answer(eng)
    stale_key = eng.gen.current
    await feed(eng, [Step("speech", 300)])
    assert not eng.gen.is_current(stale_key)
    events = [e for e in eng.trace.turns[stale_key.turn_id].events]
    assert any(e.type is EventType.CANCEL for e in events)


async def test_user_speech_in_the_think_window_interrupts(config):
    """Before the first audio frame, not after it.

    Barge-in used to be armed only when the assistant's first frame went out,
    so the whole think window — measured at 1.5-1.9 s on the CPU stack — was
    deaf: speech there fired nothing, was never handed to ASR, and was lost.
    """
    config.models.llm.options = {"first_token_delay_ms": 1500, "token_delay_ms": 1}
    eng, sink = build_engine(config)
    await eng.models.start()
    await eng.start()
    try:
        await feed(eng, [Step("silence", 100), Step("speech", 600), Step("silence", 400)])
        assert eng.state.state.value == "thinking", "test needs a turn still thinking"
        turn_before = eng.gen.turn_id

        await feed(eng, [Step("speech", 400)])

        assert eng.counters.get("barge_ins") == 1
        assert eng.state.state.value == "listening"
        assert eng.gen.turn_id == turn_before + 1
        # The interrupting words go somewhere: a stream is open and the
        # pre-roll has already been pushed into it.
        assert eng._asr_stream is not None
        assert eng._utterance_ms > 0
        assert "playback_reset" in [m.type for m in sink.control]
    finally:
        await eng.close()
        await eng.models.close()


async def test_the_max_utterance_valve_is_not_barged_by_its_own_speaker(config):
    """`max utterance` confirms while the user is mid-sentence.

    Arming there would cancel the turn that safety valve just created, every
    time, and the valve would never produce an answer.
    """
    config.conversation.turn_detection.max_utterance_ms = 400
    config.models.llm.options = {"first_token_delay_ms": 1500, "token_delay_ms": 1}
    eng, _ = build_engine(config)
    await eng.models.start()
    await eng.start()
    try:
        await feed(eng, [Step("silence", 100), Step("speech", 1000)])
        assert eng.counters.get("barge_ins") is None
    finally:
        await eng.close()
        await eng.models.close()
