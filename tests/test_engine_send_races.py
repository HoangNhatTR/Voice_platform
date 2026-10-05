"""Awaits on the way out: a send can yield, and the world moves meanwhile.

A websocket write yields under backpressure (or while another task holds the
drain lock). Code that checked "am I still current" before such an await and
acted after it did so on a turn that had already moved on.
"""

from __future__ import annotations

import asyncio
import time

from voiceplatform.app.simulate import Step, build_engine, feed, is_idle, is_speaking, wait_until
from voiceplatform.conversation.sink import CollectingSink

LONG = ("Hà Nội là thủ đô. Thành phố có nghìn năm lịch sử. Hồ Gươm nằm ở trung tâm. "
        "Phố cổ có ba mươi sáu phố. Mùa thu Hà Nội rất đẹp.")


def _heard(eng) -> list[str]:
    response = eng._response
    if response is None:
        return []
    return [p.text for p in response.heard_by(time.monotonic()) if not p.filler]


class _AudioEndTrap(CollectingSink):
    """Holds the next `audio_end` write until released."""

    def __init__(self) -> None:
        super().__init__()
        self.armed = False
        self.blocked = asyncio.Event()
        self.release = asyncio.Event()

    async def send_control(self, message) -> None:
        await super().send_control(message)
        if message.type == "audio_end" and self.armed:
            self.armed = False
            self.blocked.set()
            await self.release.wait()


async def test_a_cough_during_a_phrase_end_write_loses_and_repeats_nothing(config):
    # Cut while the talker was writing a phrase's `audio_end`: it then took the
    # next phrase off the queue the resume reads (that phrase was never said)
    # and recorded the cut phrase as heard before the ones it replayed.
    config.models.tts.options = {"first_audio_delay_ms": 10, "rtf": 0.25, "ms_per_char": 80}
    config.models.asr.options = {"partial_every_frames": 5, "script": ["kể về Hà Nội"]}
    config.models.llm.options = {"first_token_delay_ms": 10, "token_delay_ms": 60, "reply": LONG}
    sink = _AudioEndTrap()
    eng, _ = build_engine(config, sink=sink)
    await eng.models.start()
    await eng.start()
    try:
        await feed(eng, [Step("silence", 100), Step("speech", 600), Step("silence", 500)])
        assert await wait_until(eng, is_speaking, max_ms=10000)
        assert await wait_until(eng, lambda e: _heard(e), max_ms=10000)
        sink.armed = True
        assert await wait_until(eng, lambda e: sink.blocked.is_set(), max_ms=10000)
        cut = eng._response
        await feed(eng, [Step("speech", 140)])
        assert eng.counters.get("barge_ins") == 1 and cut.holding
        sink.release.set()
        await feed(eng, [Step("silence", 800)])
        assert eng.counters.get("resumed") == 1
        assert await wait_until(eng, is_idle, max_ms=30000)
        turn = eng.context.turns[-1]
        assert turn.spoken_text == turn.assistant_text == LONG
    finally:
        sink.release.set()
        await eng.close()
        await eng.models.close()


class _SlowEndpointTranscript(CollectingSink):
    """The endpoint decode's transcript write yields until released."""

    def __init__(self) -> None:
        super().__init__()
        self.blocked = asyncio.Event()
        self.release = asyncio.Event()

    async def send_control(self, message) -> None:
        await super().send_control(message)
        task = asyncio.current_task()
        if (message.type == "transcript" and not message.data.get("final") and task is not None
                and task.get_name() == "asr-endpoint-decode" and not self.release.is_set()):
            self.blocked.set()
            await self.release.wait()


async def test_no_speculation_starts_after_the_turn_is_confirmed(config):
    config.conversation.turn_detection.backend = "vad_only"
    config.conversation.speculation.enabled = True
    config.models.asr.options = {"partial_every_frames": 5, "script": ["xin chào bạn"]}
    sink = _SlowEndpointTranscript()
    eng, _ = build_engine(config, sink=sink)
    await eng.models.start()
    await eng.start()
    try:
        await feed(eng, [Step("silence", 100), Step("speech", 600), Step("silence", 60)])
        assert await wait_until(eng, lambda e: sink.blocked.is_set(), max_ms=2000)
        await feed(eng, [Step("silence", 300)])          # the pause confirms the turn
        assert eng.state.state.value in ("thinking", "speaking")
        sink.release.set()
        await asyncio.sleep(0.05)
        assert eng._speculation is None, "a second LLM request opened for a turn already answered"
        assert not eng.counters.get("speculations")
        assert await wait_until(eng, is_idle, max_ms=5000)
    finally:
        sink.release.set()
        await eng.close()
        await eng.models.close()
