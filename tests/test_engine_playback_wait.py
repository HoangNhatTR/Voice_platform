"""A turn lasts until the client has PLAYED it — and not a moment past that.

With playback feedback on, the end of a turn waits for the client's
`playback_generation_end`. Two kinds of turn waited for a report that could
never match (a fallback reply, which was not `_response`; a reply with no
audio at all), and every missing report cost the whole operation timeout and
a false `synthesis_failed`.
"""

from __future__ import annotations

import asyncio
import time

from voiceplatform.app.simulate import Step, build_engine, feed, is_idle, wait_until
from voiceplatform.conversation.sink import CollectingSink
from voiceplatform.core.clock import now_ms
from voiceplatform.core.events import EventType


class ClientLikeSink(CollectingSink):
    """Reports `audio_generation_end` back as web/playback.js does today:
    with the meta of the LAST `audio_segment` it saw."""

    def __init__(self) -> None:
        super().__init__()
        self.engine = None
        self.meta: dict | None = None

    async def send_control(self, message) -> None:
        await super().send_control(message)
        if message.type == "audio_segment":
            self.meta = dict(message.data)
        elif message.type == "audio_generation_end" and self.engine is not None:
            meta, eng = dict(self.meta or {}), self.engine
            asyncio.get_running_loop().call_later(0.05, lambda: eng.playback_feedback(
                {"event": "playback_generation_end", "client_ms": now_ms(), **meta}))


def _events(eng, type_: EventType) -> list:
    return [e for turn in eng.trace.turns.values() for e in turn.events if e.type is type_]


async def _build(config, *, reply=None, sink=None, timeout_s: float = 10.0):
    config.models.operation_timeout_s = timeout_s
    config.conversation.orphan_turn_timeout_ms = 60000
    llm = {"first_token_delay_ms": 10, "token_delay_ms": 1}
    if reply is not None:
        llm["reply"] = reply
    config.models.llm.options = llm
    eng, sink = build_engine(config, sink=sink)
    if isinstance(sink, ClientLikeSink):
        sink.engine = eng
    eng.playback_feedback_enabled = True
    eng.playback_clock = (0.0, 1.0)
    await eng.models.start()
    await eng.start()
    return eng, sink


async def _stop(eng) -> None:
    await eng.close()
    await eng.models.close()


async def _seconds_to_idle(eng, max_ms: float = 8000) -> float:
    started = time.monotonic()
    await wait_until(eng, is_idle, max_ms=max_ms, feed_silence=False)
    return time.monotonic() - started


async def test_a_reply_with_no_audio_does_not_wait_for_playback(config):
    eng, sink = await _build(config, reply=" ", sink=ClientLikeSink())   # nothing speakable
    try:
        await eng.push_text("xin chào")
        took = await _seconds_to_idle(eng)
        assert took < 1.5, f"stuck {took:.1f}s waiting for a report of audio never sent"
        assert not [m for m in sink.control if m.type == "error"]
    finally:
        await _stop(eng)


class _FailingFinish:
    def __init__(self, inner) -> None:
        self._inner = inner

    def __getattr__(self, name):
        return getattr(self._inner, name)

    async def finish(self):
        raise RuntimeError("asr final failed")


async def test_the_fallback_reply_ends_when_the_client_has_played_it(config):
    config.conversation.turn_detection.reuse_endpoint_transcript = False
    config.models.asr.options = {"partial_every_frames": 5, "script": ["xin chào bạn"]}
    eng, sink = await _build(config, sink=ClientLikeSink())
    original = eng.models.asr.open_stream

    async def open_stream(**kwargs):
        return _FailingFinish(await original(**kwargs))

    eng.models.asr.open_stream = open_stream
    try:
        await feed(eng, [Step("silence", 100), Step("speech", 600), Step("silence", 500)])
        assert await wait_until(eng, lambda e: any(m.type == "audio_generation_end" for m in sink.control),
                                max_ms=5000, feed_silence=False)
        took = await _seconds_to_idle(eng)
        assert took < 1.5, f"fallback turn stuck {took:.1f}s"
        stages = [e.data.get("stage") for e in _events(eng, EventType.ERROR)]
        assert stages == ["respond"], stages          # no false "tts" failure on top
    finally:
        await _stop(eng)


async def test_a_missing_report_costs_a_margin_not_the_operation_timeout(config):
    # A client that never reports: the turn still ends shortly after the
    # schedule says playback ended, and it is not a synthesis failure.
    eng, sink = await _build(config, sink=CollectingSink(), timeout_s=20.0)
    try:
        await eng.push_text("xin chào")
        took = await _seconds_to_idle(eng, max_ms=10000)
        assert took < 6.0, f"waited {took:.1f}s for a report that never came"
        assert not [m for m in sink.control if m.type == "error"]
        assert eng.counters.get("playback_done_timeouts") == 1
    finally:
        await _stop(eng)
