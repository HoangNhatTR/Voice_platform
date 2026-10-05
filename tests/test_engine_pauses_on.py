"""The engine as deployed: `conversation.pauses` ON.

Every other engine test runs with pauses off (the default), while
configs/local-cpu.yaml turns them on — and the pause they add after each
phrase is audio the client plays. Two bugs lived in that gap (found
05/10/2026): the pause after a sentence counted as part of the sentence, so
a reply right after "Bạn muốn chuyển bao nhiêu tiền?" interrupted that
question and the history lost it; and a Stop landing while a resume was
writing to the client left the cut answer's LLM streaming.
"""

from __future__ import annotations

import asyncio
import time

import pytest

from voiceplatform.app.simulate import Step, build_engine, feed, is_speaking, wait_until
from voiceplatform.conversation.sink import CollectingSink

QUESTION = "Bạn muốn chuyển bao nhiêu tiền?"
LONG = "Hà Nội là thủ đô. Thành phố có nghìn năm lịch sử. Hồ Gươm nằm ở trung tâm. Phố cổ có ba mươi sáu phố."


@pytest.fixture
def paused(config):
    config.conversation.pauses.enabled = True
    # Ahead of the client by a lot, as ZeroTTS is: the pause is still playing
    # long after the server sent it.
    config.models.tts.options = {"first_audio_delay_ms": 10, "rtf": 0.25, "ms_per_char": 80}
    return config


async def _start(config, script, *, reply, token_delay_ms: float = 1, sink=None):
    config.models.asr.options = {"partial_every_frames": 5, "script": script}
    config.models.llm.options = {"first_token_delay_ms": 10, "token_delay_ms": token_delay_ms, "reply": reply}
    eng, sink = build_engine(config, sink=sink)
    await eng.models.start()
    await eng.start()
    return eng, sink


async def _stop(eng):
    await eng.close()
    await eng.models.close()


def _in_pause_after(eng, text: str) -> bool:
    """Its last word has played; the pause after it has not.

    Measured from the scheduled end minus the configured pause, not from
    anything the fix added, so this fails on the code it guards against.
    """
    response = eng._response
    if response is None:
        return False
    sent = next((s for s in response.sent if s.phrase.text == text), None)
    if sent is None:
        return False
    pause = eng.config.conversation.pauses.sentence_ms / 1000.0
    now = time.monotonic()
    return response._end(sent) - pause + 0.05 <= now <= response._end(sent) - 0.08


async def _ask(eng) -> None:
    await feed(eng, [Step("silence", 100), Step("speech", 600), Step("silence", 500)])
    assert await wait_until(eng, is_speaking, max_ms=10000)


async def test_a_reply_in_the_pause_after_a_question_keeps_the_question(paused):
    eng, _ = await _start(paused, ["chuyển tiền cho mẹ", "năm trăm nghìn"], reply=QUESTION)
    try:
        await _ask(eng)
        assert await wait_until(eng, lambda e: _in_pause_after(e, QUESTION), max_ms=10000)
        # The user answers at once: speech, not a cough.
        await feed(eng, [Step("speech", 200)])
        asked = next(t for t in eng.context.turns if t.assistant_text == QUESTION)
        assert asked.spoken_text == QUESTION
        # Nothing was left to say: the model must not read its question as cut off.
        assert asked.interrupted is False
    finally:
        await _stop(eng)


async def test_an_interjection_in_the_pause_resumes_from_the_next_sentence(paused):
    first = "Hà Nội là thủ đô."
    eng, _ = await _start(paused, ["kể về Hà Nội", "ừ"], reply=LONG)
    try:
        await _ask(eng)
        assert await wait_until(eng, lambda e: _in_pause_after(e, first), max_ms=10000)
        await feed(eng, [Step("speech", 140)])
        assert eng._interrupted is not None
        replay = [p.text for p in eng._interrupted.replay]
        assert replay and replay[0] != first, replay
        assert await wait_until(eng, lambda e: e.counters.get("resumed") == 1, max_ms=5000)
        assert await wait_until(eng, lambda e: e.state.state.value == "idle", max_ms=30000)
        spoken = " ".join(t.spoken_text for t in eng.context.turns if t.spoken_text)
        assert spoken.count(first) == 1, spoken
    finally:
        await _stop(eng)


class _StopDuringResume(CollectingSink):
    """The resume's first write yields to the receive loop, and a Stop lands in it."""

    def __init__(self) -> None:
        super().__init__()
        self.engine = None
        self.pressed = False

    async def send_control(self, message) -> None:
        await super().send_control(message)
        if not self.pressed and message.type == "state" and message.data.get("source") == "resume":
            self.pressed = True
            asyncio.get_running_loop().create_task(self.engine.interrupt("client button"))
            await asyncio.sleep(0.02)


async def test_a_stop_while_a_resume_is_writing_stops_the_cut_answer(paused):
    sink = _StopDuringResume()
    # A slow LLM: it is still writing (draining) when the resume starts.
    eng, sink = await _start(paused, ["kể về Hà Nội", "ừ"], reply=LONG + " " + LONG, token_delay_ms=150, sink=sink)
    sink.engine = eng
    try:
        await _ask(eng)
        assert await wait_until(eng, lambda e: e._response is not None and e._response.sent, max_ms=10000)
        await feed(eng, [Step("speech", 400), Step("silence", 500)])
        assert await wait_until(eng, lambda e: sink.pressed, max_ms=5000)
        assert await wait_until(eng, lambda e: e.state.state.value == "idle", max_ms=3000)
        live = [t for tasks in eng._drain_tasks.values() for t in tasks if not t.done()]
        assert live == [] and eng._drain_targets == {}
        resumed = eng.gen.current
        written = len(eng._response.generated) if eng._response else 0
        await asyncio.sleep(0.6)
        assert (len(eng._response.generated) if eng._response else 0) == written
        # Nothing of the stopped generation reaches the client after its reset.
        control = [(m.type, m.data.get("generation_id")) for m in sink.control]
        reset = max(i for i, (kind, _) in enumerate(control) if kind == "playback_reset")
        stopped = control[reset][1]
        assert ("assistant_delta", stopped) not in control[reset:]
        assert eng.gen.current == resumed
    finally:
        await _stop(eng)
