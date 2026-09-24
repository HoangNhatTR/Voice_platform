"""One complete turn, end to end, on mock engines."""

from __future__ import annotations

import pytest

from voiceplatform.app.simulate import Step, build_engine, feed, is_idle, wait_until
from voiceplatform.core.events import EventType


@pytest.fixture
async def engine(config):
    eng, sink = build_engine(config)
    await eng.models.start()
    await eng.start()
    yield eng, sink
    await eng.close()
    await eng.models.close()


async def test_a_turn_runs_asr_llm_tts_and_returns_to_idle(engine):
    eng, sink = engine
    await feed(eng, [Step("silence", 100), Step("speech", 600), Step("silence", 500)])
    assert await wait_until(eng, is_idle, max_ms=5000)

    trace = eng.trace.turn(1)
    seen = {ev.type for ev in trace.events}
    for required in (
        EventType.TURN_START,
        EventType.ASR_START,
        EventType.ENDPOINT_CANDIDATE,
        EventType.TURN_CONFIRMED,
        EventType.ASR_FINAL,
        EventType.LLM_FIRST_TOKEN,
        EventType.TTS_FIRST_AUDIO,
        EventType.TTS_COMPLETE,
        EventType.TURN_END,
    ):
        assert required in seen, f"missing {required.value}"

    assert sink.audio, "no assistant audio was produced"
    metrics = trace.metrics()
    assert metrics["e2e_ttfa_ms"] is not None and metrics["e2e_ttfa_ms"] >= 0
    assert metrics["llm_ttft_ms"] is not None


async def test_the_pause_inside_a_sentence_does_not_end_the_turn(engine, config):
    eng, _ = engine
    # 160 ms of silence: past the gate's 120 ms, short of the 240 ms endpoint.
    await feed(
        eng,
        [
            Step("speech", 400),
            Step("silence", 160),
            Step("speech", 400),
            Step("silence", 500),
        ],
    )
    assert await wait_until(eng, is_idle, max_ms=5000)
    # One turn, not two: the mid-sentence pause was not an ending.
    assert eng.gen.turn_id == 1
    assert eng.counters.get("endpoint_cancelled", 0) >= 1


async def test_a_cough_is_not_a_turn(engine):
    eng, sink = engine
    await feed(eng, [Step("speech", 80), Step("silence", 500)])
    assert await wait_until(eng, is_idle, max_ms=2000)
    assert eng.counters.get("utterance_discarded_short", 0) == 1
    assert not sink.audio


async def test_text_turn_uses_the_same_engine_and_history(engine):
    eng, sink = engine
    await eng.push_text("xin chào")
    assert await wait_until(eng, is_idle, max_ms=5000, feed_silence=False)
    assert sink.audio
    assert eng.context.turns[-1].user_text == "xin chào"
    assert eng.context.turns[-1].spoken_text


async def test_a_typed_turn_tells_the_client_it_is_thinking(engine):
    """A spoken turn sends this; a typed one used not to.

    The badge then sat on the previous state for the whole think window —
    1.48 s on the real CPU stack — which reads as a dead UI.
    """
    eng, sink = engine
    await eng.push_text("xin chào")
    states = [m.data["state"] for m in sink.control if m.type == "state"]
    assert states and states[0] == "thinking"
    assert await wait_until(eng, is_idle, max_ms=5000, feed_silence=False)
