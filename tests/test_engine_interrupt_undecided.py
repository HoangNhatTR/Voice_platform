"""Speech over the answer that is short — and meant.

The barge-in stops the voice at once and keeps the answer around until the
interrupting speech is decided: a cough hands the turn back. Two ways a real
"stop" used to be decided as a cough.
"""

from __future__ import annotations

import time

from voiceplatform.app.simulate import Step, build_engine, feed, is_speaking, wait_until

LONG = "Hà Nội là thủ đô. Thành phố có nghìn năm lịch sử. Hồ Gươm nằm ở trung tâm. Phố cổ có ba mươi sáu phố."


def _heard(eng) -> list[str]:
    response = eng._response
    if response is None:
        return []
    return [p.text for p in response.heard_by(time.monotonic()) if not p.filler]


async def _speaking(config, script):
    config.models.tts.options = {"first_audio_delay_ms": 10, "rtf": 0.25, "ms_per_char": 80}
    config.models.asr.options = {"partial_every_frames": 5, "script": script}
    config.models.llm.options = {"first_token_delay_ms": 10, "token_delay_ms": 1, "reply": LONG}
    eng, sink = build_engine(config)
    await eng.models.start()
    await eng.start()
    await feed(eng, [Step("silence", 100), Step("speech", 600), Step("silence", 500)])
    assert await wait_until(eng, is_speaking, max_ms=10000)
    assert await wait_until(eng, lambda e: _heard(e), max_ms=10000)
    return eng, sink


async def test_the_stop_button_decides_an_undecided_interruption(config):
    eng, sink = await _speaking(config, ["kể về Hà Nội"])
    try:
        await feed(eng, [Step("speech", 140)])        # "dừng..." fires the barge-in
        assert eng.counters.get("barge_ins") == 1
        assert eng.state.state.value == "listening"
        await eng.interrupt("client button")           # ...and the user clicks Stop
        await feed(eng, [Step("silence", 800)])
        assert not eng.counters.get("resumed"), "the answer came back after Stop"
        assert eng.state.state.value == "idle"
        assert eng._interrupted is None and not eng._drain_targets
    finally:
        await eng.close()
        await eng.models.close()


async def test_a_short_command_over_the_answer_is_a_turn_as_it_is_from_silence(config):
    # The frames that fired the barge-in are speech too. Not counting them made
    # a 280 ms "dừng" over the answer "too short" (and the answer resumed),
    # while the same word from silence was a turn.
    eng, sink = await _speaking(config, ["kể về Hà Nội", "dừng"])
    try:
        await feed(eng, [Step("speech", 280), Step("silence", 800)])
        assert not eng.counters.get("utterance_discarded_short")
        assert not eng.counters.get("resumed")
        assert await wait_until(eng, lambda e: e.context.turns[-1].user_text == "dừng", max_ms=5000)
    finally:
        await eng.close()
        await eng.models.close()
