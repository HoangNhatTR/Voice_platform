"""G3: stop fast, decide later — and when it was nothing, carry on from where
playback stopped, not from the top of the phrase.

Also the bookkeeping underneath: what the user HEARD comes from the client's
own playback reports where they exist, an answer cut twice keeps what the
first part said, and the echo guard is counted from when the client really
started playing.
"""

from __future__ import annotations

import asyncio
import time

import numpy as np
import pytest

from voiceplatform.app.simulate import Step, build_engine, feed, is_idle, is_speaking, wait_until
from voiceplatform.conversation.barge_in import BargeInDetector
from voiceplatform.conversation.engine import Phrase, ResponseState, _Sent, _quiet_point
from voiceplatform.core.audio import AudioFrame
from voiceplatform.core.config import BargeInConfig
from voiceplatform.core.events import EventType
from voiceplatform.core.ids import GenerationKey

LONG = "Hà Nội là thủ đô. Thành phố có nghìn năm lịch sử. Hồ Gươm nằm ở trung tâm. Phố cổ có ba mươi sáu phố."


def _events(eng, kind):
    return [e for turn in eng.trace.turns.values() for e in turn.events if e.type is kind]


@pytest.fixture
def slow_talker(config):
    config.models.tts.options = {"first_audio_delay_ms": 10, "rtf": 0.25, "ms_per_char": 80}
    return config


async def _start(config, script, reply=LONG):
    config.models.asr.options = {"partial_every_frames": 5, "script": script}
    config.models.llm.options = {"first_token_delay_ms": 10, "token_delay_ms": 1, "reply": reply}
    eng, sink = build_engine(config)
    await eng.models.start()
    await eng.start()
    return eng, sink


async def _stop(eng):
    await eng.close()
    await eng.models.close()


def _heard(eng) -> list[str]:
    r = eng._response
    return [] if r is None else [p.text for p in r.heard_by(time.monotonic()) if not p.filler]


async def _ask_and_hear_one_phrase(eng):
    await feed(eng, [Step("silence", 100), Step("speech", 600), Step("silence", 500)])
    assert await wait_until(eng, is_speaking, max_ms=10000)
    assert await wait_until(eng, lambda e: _heard(e), max_ms=10000)


async def _wait_wall(eng, seconds: float) -> None:
    until = time.monotonic() + seconds
    await wait_until(eng, lambda e: time.monotonic() >= until, max_ms=1000 * seconds + 2000)


# ---------------------------------------------------------------- resume

async def test_a_false_interruption_continues_from_where_playback_stopped(slow_talker):
    eng, sink = await _start(slow_talker, ["kể về Hà Nội"])
    try:
        await _ask_and_hear_one_phrase(eng)
        await _wait_wall(eng, 1.0)          # well into the next phrase
        response = eng._response
        now = time.monotonic()
        playing = next((s for s in response.sent if response._end(s) > now), None)
        cut_id = playing.phrase.phrase_id if playing else response.current.phrase_id
        cut_text = playing.phrase.text if playing else response.current.text

        await feed(eng, [Step("speech", 140), Step("silence", 800)])
        assert eng.counters.get("resumed") == 1
        resumed = _events(eng, EventType.RESUMED)[-1]
        cut = resumed.data["cut"]
        assert cut["phrase_id"] == cut_id
        assert 0 < cut["from_ms"] < cut["played_ms"]
        # The rest of that phrase, from the kept audio — not the whole phrase again.
        assert await wait_until(eng, lambda e: e._response and e._response.sent, max_ms=5000)
        first = eng._response.sent[0]
        assert first.phrase.text == cut_text
        full = playing.frames if playing else response.current_frames
        assert 0 < sum(f.samples.size for f in first.frames) < sum(f.samples.size for f in full)
        # All of it had been SENT before the cough (only the client was still
        # playing): the resumed turn used to wait forever on the old queue.
        assert await wait_until(eng, is_idle, max_ms=10000)
        assert not eng.counters.get("orphan_turns")
        assert eng.context.turns[-1].assistant_text == LONG
        assert eng.context.turns[-1].spoken_text == LONG
    finally:
        await _stop(eng)


async def test_a_phrase_cut_mid_synthesis_is_finished_into_memory_not_sent(slow_talker):
    # A talker slower than real time: the phrase is still being made when the
    # cough lands, so the talker is left to finish it — silently.
    slow_talker.models.tts.options = {"first_audio_delay_ms": 10, "rtf": 1.2, "ms_per_char": 60}
    eng, sink = await _start(slow_talker, ["kể về Hà Nội"])
    try:
        await feed(eng, [Step("silence", 100), Step("speech", 600), Step("silence", 500)])
        assert await wait_until(eng, is_speaking, max_ms=10000)
        await _wait_wall(eng, 0.3)
        response = eng._response
        assert response.current is not None, "test needs a phrase in synthesis"
        cut_gen = eng.gen.current.generation_id
        sent_before = sink.audio_samples_for(cut_gen)
        await feed(eng, [Step("speech", 140)])
        assert response.holding
        await asyncio.wait_for(response.hold_done.wait(), 10)
        # Everything the talker finished after the cut stayed on the server.
        assert sink.audio_samples_for(cut_gen) == sent_before
        assert sum(f.samples.size for f in response.current_frames) > sent_before
        await feed(eng, [Step("silence", 800)])
        assert eng.counters.get("resumed") == 1
        assert await wait_until(eng, is_idle, max_ms=30000)
        assert eng.context.turns[-1].spoken_text == LONG
    finally:
        await _stop(eng)


async def test_a_real_interruption_stops_the_held_talker(slow_talker):
    slow_talker.models.tts.options = {"first_audio_delay_ms": 10, "rtf": 1.2, "ms_per_char": 60}
    eng, sink = await _start(slow_talker, ["kể về Hà Nội", "thôi dừng lại"], reply=LONG)
    try:
        await feed(eng, [Step("silence", 100), Step("speech", 600), Step("silence", 500)])
        assert await wait_until(eng, is_speaking, max_ms=10000)
        await _wait_wall(eng, 0.3)
        await feed(eng, [Step("speech", 600), Step("silence", 700)])
        carried_hold = None
        assert await wait_until(eng, lambda e: e.counters.get("resumed") is None and e.gen.turn_id >= 2
                                and e.state.state.value in ("thinking", "speaking", "idle"), max_ms=5000)
        assert await wait_until(eng, is_idle, max_ms=30000)
        assert not eng.counters.get("resumed")
        assert [t.user_text for t in eng.context.turns] == ["kể về Hà Nội", "thôi dừng lại"]
        # Nothing of the held phrase was sent after the cut. The talker ends the
        # turn one loop step before `respond` wakes from `await speak_task`, so
        # an immediate count still sees that task about 1 run in 12; a real leak
        # stays above zero for the whole window.
        assert await wait_until(eng, lambda e: e.gen.live_tasks() == 0, max_ms=200, feed_silence=False)
        del carried_hold
    finally:
        await _stop(eng)


async def test_an_answer_cut_twice_keeps_what_was_heard_before_the_first_cut(slow_talker):
    eng, sink = await _start(slow_talker, ["kể về Hà Nội", "thôi dừng lại"])
    try:
        await _ask_and_hear_one_phrase(eng)
        first_heard = _heard(eng)
        await feed(eng, [Step("speech", 140), Step("silence", 800)])       # cough
        assert eng.counters.get("resumed") == 1
        assert await wait_until(eng, lambda e: e._response and _heard(e), max_ms=10000)
        await feed(eng, [Step("speech", 600), Step("silence", 700)])       # the real one
        assert await wait_until(eng, is_idle, max_ms=30000)
        answered = eng.context.turns[0]
        assert answered.interrupted
        for text in first_heard:
            assert text in answered.spoken_text, (first_heard, answered.spoken_text)
    finally:
        await _stop(eng)


def test_the_quiet_point_backs_off_to_silence_not_mid_syllable():
    rate = 24000
    loud = np.full(rate, 0.3, np.float32)
    gap = np.zeros(int(0.05 * rate), np.float32)
    pcm = np.concatenate([loud, gap, loud])
    frames = [AudioFrame(samples=pcm[i:i + 960], sample_rate=rate) for i in range(0, pcm.size, 960)]
    # Aim 120 ms after the gap; the quietest point within 200 ms before is the gap.
    target = int((1.05 + 0.12) * rate)
    at = _quiet_point(frames, target)
    assert rate <= at <= int(1.05 * rate)


# ---------------------------------------------------------------- heard, from feedback

def _response_with(sent: list[tuple[str, float, float]]) -> ResponseState:
    key = GenerationKey(session_id="s", turn_id=1, generation_id=1)
    response = ResponseState(key=key)
    for index, (text, start, end) in enumerate(sent):
        phrase = Phrase(text, phrase_id=f"p{index}")
        response.sent.append(_Sent(phrase, start, end))
    return response


def test_heard_follows_the_clients_report_not_the_schedule():
    # Scheduled to end at 1.0 s; the client buffered first and reports 1.3 s.
    r = _response_with([("một", 0.0, 1.0), ("hai", 1.0, 2.0)])
    r.observe("p0", "playback_started", 0.3)
    assert [p.text for p in r.heard_by(1.2)] == []          # offset applies to the estimate
    r.observe("p0", "playback_stopped", 1.25)
    assert [p.text for p in r.heard_by(1.26)] == ["một"]
    assert [p.text for p in r.unheard_by(1.26)] == ["hai"]


def test_without_feedback_the_schedule_still_decides():
    r = _response_with([("một", 0.0, 1.0), ("hai", 1.0, 2.0)])
    assert [p.text for p in r.heard_by(1.2)] == ["một"]
    assert r.content_heard_by(0.1)


# ---------------------------------------------------------------- echo guard

def _frame(rms: float) -> AudioFrame:
    n = 320
    return AudioFrame(samples=np.full(n, rms, np.float32), sample_rate=16000)


def test_the_guard_can_be_moved_to_the_reported_onset():
    d = BargeInDetector(BargeInConfig(speech_frames=2, guard_ms=100, min_rms=0.02))
    d.arm(0.0, guard_ms=400)
    d.set_guard_until(250.0)
    assert d.update(1.0, _frame(0.2), 240.0) is False      # still guarded
    assert d.update(1.0, _frame(0.2), 260.0) is False      # run 1
    assert d.update(1.0, _frame(0.2), 280.0) is True


def test_inside_the_guard_a_voice_louder_than_echo_still_counts():
    d = BargeInDetector(BargeInConfig(speech_frames=2, guard_ms=300, min_rms=0.02, guard_min_rms=0.08))
    d.arm(0.0)
    assert d.update(1.0, _frame(0.04), 20.0) is False       # echo-level: reset
    assert d.update(1.0, _frame(0.2), 40.0) is False
    assert d.update(1.0, _frame(0.2), 60.0) is True        # fired inside the guard
    assert d.stats.suppressed_guard == 1


async def test_the_first_audio_guards_the_clients_startup_buffer(config):
    config.conversation.barge_in.guard_ms = 60
    config.conversation.barge_in.playback_startup_ms = 160
    eng, _ = await _start(config, ["kể về Hà Nội"], reply="Xin chào bạn.")
    try:
        await feed(eng, [Step("silence", 100), Step("speech", 600), Step("silence", 500)])
        assert await wait_until(eng, is_speaking, max_ms=5000, feed_silence=False)
        armed = eng.barge_in._armed_at_ms
        assert eng.barge_in._guard_until_ms == pytest.approx(armed + 220)
    finally:
        await _stop(eng)


async def test_a_playback_report_re_anchors_the_guard(config):
    config.conversation.barge_in.guard_ms = 60
    eng, _ = await _start(config, ["kể về Hà Nội"], reply="Xin chào bạn. Hôm nay trời đẹp lắm.")
    eng.playback_feedback_enabled = False
    try:
        eng.set_playback_clock(0.0, 1.0)
        await feed(eng, [Step("silence", 100), Step("speech", 600), Step("silence", 500)])
        assert await wait_until(eng, lambda e: e._response and e._response.current_start is not None,
                                max_ms=5000, feed_silence=False)
        response = eng._response
        phrase = response.current or response.sent[0].phrase
        from voiceplatform.core.clock import now_ms
        eng.playback_feedback({"event": "playback_started", "phrase_id": phrase.phrase_id,
                               "generation_id": response.key.generation_id, "client_ms": now_ms()})
        guard = [e for e in _events(eng, EventType.BARGE_IN_GUARD)]
        assert guard and guard[0].data["anchor"] == "playback"
        left = eng.barge_in._guard_until_ms - eng._audio_clock_ms
        assert 60 <= left <= 60 + 2 * config.audio.frame_ms + 5
        assert response.observed_start[phrase.phrase_id] > 0
    finally:
        await _stop(eng)


# ---------------------------------------------------------------- speech model for interjections

import importlib.util

needs_silero = pytest.mark.skipif(
    importlib.util.find_spec("silero_vad") is None, reason="silero-vad not installed in this interpreter"
)


@needs_silero
async def test_a_noise_the_speech_model_rejects_resumes_even_if_the_gate_took_it(slow_talker):
    # The mock "speech" is noise-like sine+hiss: loud enough for the energy
    # gate to make a turn of it, not a voice to Silero. With verification on,
    # the interjection is noise however long it is.
    slow_talker.conversation.barge_in.vad = "silero"
    slow_talker.conversation.barge_in.verify_speech_ms = 200
    eng, _ = await _start(slow_talker, ["kể về Hà Nội", "đây"])
    try:
        await _ask_and_hear_one_phrase(eng)
        # Silero calls none of this speech, so the barge-in cannot fire from
        # it; force the cut the energy gate would have made, then feed the
        # "cough" long enough to pass min_utterance_ms.
        await eng._handle_barge_in()
        await feed(eng, [Step("speech", 500), Step("silence", 900)])
        assert eng.counters.get("interjections_rejected") == 1
        assert eng.counters.get("resumed") == 1
        rejected = _events(eng, EventType.INTERJECTION_REJECTED)
        assert rejected and rejected[0].data["speech_ms"] < 200
        assert await wait_until(eng, is_idle, max_ms=30000)
        assert [t.user_text for t in eng.context.turns] == ["kể về Hà Nội"]
    finally:
        await _stop(eng)


def test_verification_needs_the_speech_model():
    from voiceplatform.core.config import Config
    from voiceplatform.core.errors import ConfigError

    config = Config()
    config.conversation.barge_in.verify_speech_ms = 200
    with pytest.raises(ConfigError):
        config.validate()


def test_speech_already_running_when_the_first_audio_goes_out_is_not_guarded():
    d = BargeInDetector(BargeInConfig(speech_frames=4, guard_ms=150, min_rms=0.02))
    d.arm(0.0, guard_ms=0)                 # armed at the confirm: nothing playing
    assert d.update(1.0, _frame(0.2), 20.0) is False
    assert d.update(1.0, _frame(0.2), 40.0) is False
    d.arm(50.0, guard_ms=310)              # first assistant audio: re-armed
    d.set_guard_until(400.0)               # and the playback report would extend it
    assert d.update(1.0, _frame(0.2), 60.0) is False
    assert d.update(1.0, _frame(0.2), 80.0) is True    # the run continued: 4th frame


def test_a_fresh_arm_still_guards_the_onset():
    d = BargeInDetector(BargeInConfig(speech_frames=2, guard_ms=150, min_rms=0.02))
    d.arm(0.0)
    assert d.update(1.0, _frame(0.2), 20.0) is False
    assert d.update(1.0, _frame(0.2), 40.0) is False
    assert d.stats.suppressed_guard == 2


async def test_a_late_barge_in_still_hands_asr_the_first_word(slow_talker):
    # "Thôi, dừng lại" at playback onset: the guard swallows "Thôi" and the
    # comma breaks the run, so the barge-in fires on "dừng lại". The ASR must
    # still get audio back to where the burst began, not a fixed 320 ms.
    slow_talker.media.vad.pre_roll_ms = 200
    slow_talker.conversation.barge_in.guard_ms = 60
    eng, _ = await _start(slow_talker, ["kể về Hà Nội", "thôi dừng lại"])
    try:
        await _ask_and_hear_one_phrase(eng)
        eng.barge_in.arm(eng._audio_clock_ms, guard_ms=300)   # a fresh onset guard
        await feed(eng, [Step("speech", 260), Step("silence", 60), Step("speech", 300)])
        assert eng.counters.get("barge_ins") == 1
        start = [e for e in _events(eng, EventType.ASR_START) if "preroll_ms" in e.data][-1]
        # Everything since the burst began (620 ms), not the fixed 200 ms.
        assert start.data["preroll_ms"] >= 500
    finally:
        await _stop(eng)


@needs_silero
async def test_a_short_interjection_is_a_backchannel_whatever_asr_spelled(slow_talker):
    slow_talker.conversation.barge_in.vad = "silero"
    slow_talker.conversation.barge_in.backchannel_max_speech_ms = 500
    eng, _ = await _start(slow_talker, ["kể về Hà Nội", "từ từ"])
    try:
        await _ask_and_hear_one_phrase(eng)
        await eng._handle_barge_in()
        eng._interjection_speech_ms = 300.0      # what Silero heard of an "ừ ừ"
        await feed(eng, [Step("speech", 400), Step("silence", 900)])
        assert eng.counters.get("backchannels") == 1
        assert eng.counters.get("resumed") == 1
    finally:
        await _stop(eng)


def test_a_short_request_word_is_never_a_backchannel():
    from voiceplatform.conversation.engine import _commands
    for text in ("dừng", "thôi", "hả", "sao cơ", "khoan đã"):
        assert _commands(text), text
    for text in ("từ từ", "thân ạ", "ồ"):
        assert not _commands(text), text


@needs_silero
async def test_the_energy_trigger_stops_and_the_speech_model_decides(slow_talker):
    # The default split: energy fires the stop (no more sensitive to echo than
    # before), Silero then says the "cough" had no voice in it.
    slow_talker.conversation.barge_in.speech_model = "silero"
    slow_talker.conversation.barge_in.verify_speech_ms = 200
    eng, _ = await _start(slow_talker, ["kể về Hà Nội", "đây"])
    try:
        await _ask_and_hear_one_phrase(eng)
        await feed(eng, [Step("speech", 500), Step("silence", 900)])
        assert eng.counters.get("barge_ins") == 1                 # energy fired it
        assert eng.counters.get("interjections_rejected") == 1    # Silero heard no voice
        assert eng.counters.get("resumed") == 1
        assert await wait_until(eng, is_idle, max_ms=30000)
        assert [t.user_text for t in eng.context.turns] == ["kể về Hà Nội"]
    finally:
        await _stop(eng)
