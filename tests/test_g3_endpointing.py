"""G3: the end of a turn is judged on the whole utterance, and the model may
start early — but nothing it produces leaves before the turn is confirmed.

Three things that could not be true before:

* the text judged at a pause was the last periodic partial, up to 500 ms old,
  so a dangling "đến" said in that window could never hold the turn;
* the final decode re-read audio the endpoint decode had already read in full;
* the LLM request left only after the confirm, although the transcript was
  known ~200 ms earlier.
"""

from __future__ import annotations

import pytest

from voiceplatform.app.simulate import Step, build_engine, feed, is_idle, wait_until
from voiceplatform.core.events import EventType
from voiceplatform.models.llm.mock import MockLlmEngine


def _all(eng, kind: EventType) -> list:
    return [e for turn in eng.trace.turns.values() for e in turn.events if e.type is kind]


def _counting(eng) -> list:
    """Wrap the mock LLM so every request is recorded with its last user text."""
    calls: list[str] = []
    llm = eng.models.llm
    original = llm.stream

    def stream(messages, *, tools=None, max_tokens=None):
        calls.append(next((m.content for m in reversed(messages) if m.role == "user"), ""))
        return original(messages, tools=tools, max_tokens=max_tokens)

    llm.stream = stream
    return calls


async def _engine(config, script, *, endpoint_script=None, **asr):
    config.models.asr.options = {"partial_every_frames": 5, "script": script, **asr}
    if endpoint_script is not None:
        config.models.asr.options["endpoint_script"] = endpoint_script
    eng, sink = build_engine(config)
    await eng.models.start()
    await eng.start()
    return eng, sink


async def _stop(eng):
    await eng.close()
    await eng.models.close()


# ---------------------------------------------------------------- endpoint decode

@pytest.mark.parametrize("endpoint_decode, turns", [(True, 1), (False, 2)])
async def test_a_dangling_word_said_after_the_last_partial_still_holds_the_turn(config, endpoint_decode, turns):
    # The mock's periodic partial knows only the first words; what the user
    # said last — "đến" — is only in a decode of the whole utterance.
    config.conversation.turn_detection.endpoint_decode = endpoint_decode
    eng, _ = await _engine(
        config, ["chuyển năm trăm nghìn đến số tài khoản", "số tài khoản"],
        endpoint_script=["chuyển năm trăm nghìn đến", "chuyển năm trăm nghìn đến số tài khoản"],
    )
    eng.models.asr.peek_partial = lambda frames: "chuyển năm trăm"
    try:
        await feed(eng, [Step("speech", 600), Step("silence", 400), Step("speech", 500), Step("silence", 800)])
        assert await wait_until(eng, is_idle, max_ms=8000)
        confirmed = _all(eng, EventType.TURN_CONFIRMED)
        assert len(confirmed) == turns, [e.data for e in confirmed]
        if endpoint_decode:
            seen = [e.data["text"] for e in _all(eng, EventType.ASR_ENDPOINT_TRANSCRIPT)]
            assert "chuyển năm trăm nghìn đến" in seen
    finally:
        await _stop(eng)


async def test_the_endpoint_decode_stands_in_for_the_final_decode(config):
    eng, _ = await _engine(config, ["kể về Hà Nội"])
    try:
        await feed(eng, [Step("speech", 600), Step("silence", 600)])
        assert await wait_until(eng, is_idle, max_ms=8000)
        assert eng.models.asr.finals == 0
        start = _all(eng, EventType.ASR_FINALIZE_START)
        assert start and start[0].data.get("reused_endpoint_decode") is True
        assert [e.data["text"] for e in _all(eng, EventType.ASR_FINAL)] == ["kể về Hà Nội"]
        assert eng.context.turns[-1].user_text == "kể về Hà Nội"
    finally:
        await _stop(eng)


async def test_a_decode_that_has_not_covered_the_speech_is_not_reused(config):
    # The endpoint decode is still running when the silence is long enough to
    # confirm: the real final decode must run.
    eng, _ = await _engine(config, ["kể về Hà Nội"], endpoint_delay_ms=2000)
    try:
        await feed(eng, [Step("speech", 600), Step("silence", 600)])
        assert await wait_until(eng, is_idle, max_ms=8000)
        assert eng.models.asr.finals == 1
        assert not _all(eng, EventType.ASR_ENDPOINT_TRANSCRIPT)
    finally:
        await _stop(eng)


async def test_reuse_can_be_turned_off(config):
    config.conversation.turn_detection.reuse_endpoint_transcript = False
    eng, _ = await _engine(config, ["kể về Hà Nội"])
    try:
        await feed(eng, [Step("speech", 600), Step("silence", 600)])
        assert await wait_until(eng, is_idle, max_ms=8000)
        assert eng.models.asr.finals == 1
    finally:
        await _stop(eng)


async def test_a_confident_question_ends_sooner_only_with_the_fast_tier(config):
    waits = {}
    for fast in (0, 60):
        config.conversation.turn_detection.fast_silence_ms = fast
        eng, _ = await _engine(config, ["bây giờ là mấy giờ rồi"])
        try:
            await feed(eng, [Step("speech", 600), Step("silence", 700)])
            assert await wait_until(eng, is_idle, max_ms=8000)
            # Audio clock, not wall time: this feed is faster than real time.
            reason = _all(eng, EventType.TURN_CONFIRMED)[0].data["reason"]
            waits[fast] = float(reason.removeprefix("silence ").removesuffix("ms"))
        finally:
            await _stop(eng)
    # Audio clock: the gate consumed 120 ms; base wants 240, fast 60.
    assert waits[60] < waits[0], waits


async def test_an_unstable_transcript_never_takes_the_fast_path(config):
    config.conversation.turn_detection.fast_silence_ms = 60
    config.conversation.turn_detection.endpoint_decode = False
    eng, _ = await _engine(config, ["bây giờ là mấy giờ rồi"])
    eng.models.asr.peek_partial = lambda frames: "bây giờ là mấy giờ rồi"
    try:
        await feed(eng, [Step("speech", 600), Step("silence", 700)])
        assert await wait_until(eng, is_idle, max_ms=8000)
        candidates = _all(eng, EventType.ENDPOINT_CANDIDATE)
        assert all(c.data["required_silence_ms"] == 240 for c in candidates)
    finally:
        await _stop(eng)


# ---------------------------------------------------------------- shadow mode

@pytest.fixture
def speculative(config):
    config.conversation.speculation.enabled = True
    # A slow first token, so a speculation started at the pause is visibly
    # ahead of a request started at the confirm.
    config.models.llm.options = {"first_token_delay_ms": 150, "token_delay_ms": 1}
    return config


async def test_a_speculation_is_adopted_when_the_transcript_holds(speculative):
    eng, sink = await _engine(speculative, ["kể về Hà Nội"])
    calls = _counting(eng)
    try:
        await feed(eng, [Step("speech", 600), Step("silence", 600)])
        assert await wait_until(eng, is_idle, max_ms=8000)
        assert calls == ["kể về Hà Nội"], calls        # one request, not two
        assert eng.counters.get("speculations_adopted") == 1
        assert sink.audio
        started = _all(eng, EventType.SPECULATION_STARTED)[0].ts_ms
        confirmed = _all(eng, EventType.TURN_CONFIRMED)[0].ts_ms
        assert started < confirmed
        # Nothing it produced reached the client before the confirm.
        first_delta = next(i for i, m in enumerate(sink.control) if m.type == "assistant_delta")
        thinking = next(i for i, m in enumerate(sink.control) if m.type == "state" and m.data["state"] == "thinking")
        assert thinking < first_delta
        sent = _all(eng, EventType.AUDIO_SENT)
        assert sent and min(e.ts_ms for e in sent) > confirmed
    finally:
        await _stop(eng)


async def test_the_user_talking_on_discards_the_speculation(speculative):
    eng, sink = await _engine(
        speculative, ["kể về Hà Nội ngày xưa"],
        endpoint_script=["kể về Hà Nội", "kể về Hà Nội ngày xưa"],
    )
    calls = _counting(eng)
    try:
        # A pause long enough for the endpoint decode, short enough to hold.
        speculative.conversation.turn_detection.silence_ms = 400
        eng.turn_detector.silence_ms = 400
        await feed(eng, [Step("speech", 600), Step("silence", 200), Step("speech", 400), Step("silence", 800)])
        assert await wait_until(eng, is_idle, max_ms=8000)
        discarded = _all(eng, EventType.SPECULATION_DISCARDED)
        assert discarded and discarded[0].data["reason"] == "speech resumed"
        assert calls[-1] == "kể về Hà Nội ngày xưa"
        assert eng.context.turns[-1].user_text == "kể về Hà Nội ngày xưa"
        # The discarded guess never spoke.
        deltas = "".join(m.data["text"] for m in sink.control if m.type == "assistant_delta")
        assert "kể về Hà Nội ngày xưa" in deltas
        assert "Bạn vừa nói kể về Hà Nội." not in deltas
    finally:
        await _stop(eng)


async def test_a_speculated_tool_call_runs_only_after_the_confirm(speculative):
    eng, _ = await _engine(speculative, ["bây giờ là mấy giờ"])
    try:
        await feed(eng, [Step("speech", 600), Step("silence", 600)])
        assert await wait_until(eng, is_idle, max_ms=8000)
        assert eng.counters.get("speculations_adopted") == 1
        confirmed = _all(eng, EventType.TURN_CONFIRMED)[0].ts_ms
        tools = _all(eng, EventType.TOOL_START)
        assert tools and all(t.ts_ms > confirmed for t in tools)
    finally:
        await _stop(eng)


async def test_a_discarded_speculation_never_runs_its_tool(speculative):
    eng, _ = await _engine(
        speculative, ["thôi không cần nữa"],
        endpoint_script=["bây giờ là mấy giờ", "thôi không cần nữa"],
    )
    try:
        speculative.conversation.turn_detection.silence_ms = 400
        eng.turn_detector.silence_ms = 400
        await feed(eng, [Step("speech", 600), Step("silence", 200), Step("speech", 400), Step("silence", 800)])
        assert await wait_until(eng, is_idle, max_ms=8000)
        assert _all(eng, EventType.SPECULATION_DISCARDED)
        assert not _all(eng, EventType.TOOL_START)
    finally:
        await _stop(eng)


async def test_speculation_never_takes_the_last_free_slot(speculative):
    eng, _ = await _engine(speculative, ["kể về Hà Nội"])

    class Busy:
        def snapshot(self):
            return {"active": 2, "waiting": 0, "parallel": 3}

    eng.models.llm.limiter = Busy()
    try:
        await feed(eng, [Step("speech", 600), Step("silence", 600)])
        assert await wait_until(eng, is_idle, max_ms=8000)
        assert eng.counters.get("speculation_skipped_busy") == 1
        assert not eng.counters.get("speculations")
    finally:
        await _stop(eng)


async def test_a_discarded_guess_is_not_the_turns_ttft(speculative):
    eng, _ = await _engine(
        speculative, ["kể về Hà Nội ngày xưa"],
        endpoint_script=["kể về Hà Nội", "kể về Hà Nội ngày xưa"],
    )
    try:
        speculative.conversation.turn_detection.silence_ms = 400
        eng.turn_detector.silence_ms = 400
        await feed(eng, [Step("speech", 600), Step("silence", 200), Step("speech", 400), Step("silence", 800)])
        assert await wait_until(eng, is_idle, max_ms=8000)
        rounds, _, _ = eng.trace.turn(1).stages()
        roles = [r.get("role") for r in rounds]
        assert "speculation_discarded" in roles
        metrics = eng.trace.turn(1).metrics()
        kept = [r for r in rounds if r.get("role") != "speculation_discarded" and r["request_ttft_ms"] is not None]
        assert metrics["llm_ttft_ms"] == kept[0]["request_ttft_ms"]
    finally:
        await _stop(eng)


async def test_closing_the_session_reaps_a_live_speculation(speculative):
    speculative.models.llm.options = {"first_token_delay_ms": 5000, "token_delay_ms": 1}
    eng, _ = await _engine(speculative, ["kể về Hà Nội"])
    try:
        speculative.conversation.turn_detection.silence_ms = 400
        eng.turn_detector.silence_ms = 400
        await feed(eng, [Step("speech", 600), Step("silence", 200)])
        assert await wait_until(eng, lambda e: e._speculation is not None, max_ms=2000, feed_silence=False)
        spec = eng._speculation
    finally:
        await _stop(eng)
    assert spec._task.done()


# ---------------------------------------------------------------- start working, then talk

async def test_a_neutral_confirm_waits_for_the_commit_before_it_speaks(config):
    # Neutral text (no cue either way) confirms at silence_ms; the first sound
    # waits for commit_silence_ms of silence, the LLM does not.
    config.conversation.turn_detection.commit_silence_ms = 600
    config.conversation.opener.enabled = False
    eng, sink = await _engine(config, ["kể về Hà Nội"])
    try:
        await feed(eng, [Step("speech", 600), Step("silence", 240)])
        # Real-time silence from here on, so the audio clock moves with the wall.
        assert await wait_until(eng, is_idle, max_ms=8000)
        confirmed = _all(eng, EventType.TURN_CONFIRMED)[0]
        commit = _all(eng, EventType.AUDIO_COMMIT)
        assert commit and commit[0].data["gated"], "the first audio was not gated"
        assert commit[0].data["waited_ms"] > 100
        first_audio = min(e.ts_ms for e in _all(eng, EventType.AUDIO_SENT))
        assert first_audio >= commit[0].ts_ms >= confirmed.ts_ms
    finally:
        await _stop(eng)


async def test_talking_on_before_the_commit_is_merged_and_never_heard(config):
    config.conversation.turn_detection.commit_silence_ms = 1500
    config.conversation.opener.enabled = False
    eng, sink = await _engine(config, ["chuyển tiền", "cho mẹ tôi"])
    try:
        await feed(eng, [Step("speech", 600), Step("silence", 500)])
        assert await wait_until(eng, lambda e: e.state.state.value == "thinking", max_ms=2000, feed_silence=False)
        await feed(eng, [Step("speech", 500), Step("silence", 2000)])
        assert await wait_until(eng, is_idle, max_ms=10000)
        merged = _all(eng, EventType.TURN_MERGED)
        assert merged and merged[0].data["text"] == "chuyển tiền cho mẹ tôi"
        # The half-question's generation never made a sound.
        first_gen = _all(eng, EventType.TURN_CONFIRMED)[0].generation_id
        assert sink.audio_samples_for(first_gen) == 0
    finally:
        await _stop(eng)


async def test_a_confident_question_is_not_gated(config):
    config.conversation.turn_detection.commit_silence_ms = 1500
    config.conversation.turn_detection.fast_silence_ms = 60
    eng, _ = await _engine(config, ["bây giờ là mấy giờ rồi"])
    try:
        await feed(eng, [Step("speech", 600), Step("silence", 400)])
        assert await wait_until(eng, is_idle, max_ms=8000)
        assert eng.counters.get("fallbacks") is None
        assert not _all(eng, EventType.AUDIO_COMMIT)      # no gate was set at all
    finally:
        await _stop(eng)
