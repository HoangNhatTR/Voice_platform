"""A cut answer that keeps being written (the drain), when tools are involved.

A barge-in detaches the cut answer's producer instead of cancelling it, so a
cough can hand the turn back without asking the model again. That promise has
to hold when a tool call is in flight, and a real interruption has to end the
drain — all of it, including the producer a resumed answer was still fed by.
"""

from __future__ import annotations

import asyncio
import time

from voiceplatform.app.simulate import Step, build_engine, feed, is_idle, is_speaking, wait_until
from voiceplatform.core.events import EventType
from voiceplatform.tasks.builtin.echo import SlowTool
from voiceplatform.tasks.executor import TaskExecutor
from voiceplatform.tasks.registry import ToolRegistry

LONG = ("Hà Nội là thủ đô. Thành phố có nghìn năm lịch sử. Hồ Gươm nằm ở trung tâm. "
        "Phố cổ có ba mươi sáu phố. Mùa thu Hà Nội rất đẹp. Người Hà Nội thanh lịch.")


def _events(eng, type_: EventType) -> list:
    return [e for turn in eng.trace.turns.values() for e in turn.events if e.type is type_]


def _heard(eng) -> list[str]:
    response = eng._response
    if response is None:
        return []
    return [p.text for p in response.heard_by(time.monotonic()) if not p.filler]


def _live(name_part: str) -> list[asyncio.Task]:
    return [t for t in asyncio.all_tasks() if name_part in (t.get_name() or "") and not t.done()]


async def _tool_engine(config, *, delay_ms: float, first_token_ms: float = 10):
    config.models.tts.options = {"first_audio_delay_ms": 10, "rtf": 0.25, "ms_per_char": 80}
    config.models.asr.options = {"partial_every_frames": 5, "script": ["chờ tôi tra cứu nhé"]}
    config.models.llm.options = {"first_token_delay_ms": first_token_ms, "token_delay_ms": 1,
                                 "tool_triggers": {"chờ": "slow"}}
    config.conversation.filler.after_ms = 80
    config.conversation.opener.enabled = False
    eng, sink = build_engine(config, tools=[])
    registry = ToolRegistry()
    registry.register(SlowTool(delay_ms=delay_ms))
    eng.executor = TaskExecutor(registry)
    await eng.models.start()
    await eng.start()
    return eng, sink


async def _stop(eng) -> None:
    await eng.close()
    await eng.models.close()


async def test_a_cough_over_the_tool_filler_keeps_the_tool_answer(config):
    # The tool is still running when the cough lands on the filler. Its result
    # used to come back "stale" (the cut generation is never current again) and
    # the tool task finishing first ended the whole drain: the resumed turn said
    # nothing at all.
    eng, _ = await _tool_engine(config, delay_ms=2500)
    try:
        await feed(eng, [Step("silence", 100), Step("speech", 600), Step("silence", 500)])
        assert await wait_until(eng, is_speaking, max_ms=5000)     # the filler
        await wait_until(eng, lambda e: False, max_ms=400)          # past the echo guard
        assert eng.state.state.value == "speaking"
        await feed(eng, [Step("speech", 140), Step("silence", 800)])
        assert eng.counters.get("barge_ins") == 1
        assert eng.counters.get("resumed") == 1
        assert await wait_until(eng, is_idle, max_ms=15000)
        stale = [e.data for e in _events(eng, EventType.STALE_DROPPED)]
        assert not stale, stale
        assert "Theo tra cứu" in eng.context.turns[-1].assistant_text
    finally:
        await _stop(eng)


async def test_a_tool_call_waits_until_the_interruption_is_decided(config):
    # The model asks for a tool while the user's interjection is still being
    # heard. Until it is known to be a cough, the cut turn must not act: "khoan,
    # đừng chuyển" is exactly what such an interjection often says.
    eng, _ = await _tool_engine(config, delay_ms=50, first_token_ms=1200)
    try:
        await feed(eng, [Step("silence", 100), Step("speech", 600), Step("silence", 500)])
        assert eng.state.state.value == "thinking"
        assert await wait_until(eng, lambda e: e._response is not None, max_ms=2000, feed_silence=False)
        await feed(eng, [Step("speech", 140)])            # a cough in the think window
        assert eng.counters.get("barge_ins") == 1
        await asyncio.sleep(1.5)                           # the tool call arrives meanwhile
        assert not _events(eng, EventType.TOOL_START), "a tool ran for a turn the user had just cut"
        await feed(eng, [Step("silence", 800)])            # it was a cough
        assert eng.counters.get("resumed") == 1
        assert await wait_until(eng, is_idle, max_ms=10000)
        assert _events(eng, EventType.TOOL_START)
        assert "Theo tra cứu" in eng.context.turns[-1].assistant_text
    finally:
        await _stop(eng)


async def test_a_real_interruption_after_a_resume_stops_the_first_llm(config):
    # cough -> resume -> a real interruption. The resumed answer was fed by the
    # first turn's producer through a pipe; cancelling only the pipe left that
    # producer streaming (and holding an LLM slot) for a turn nobody wanted.
    config.models.tts.options = {"first_audio_delay_ms": 10, "rtf": 0.25, "ms_per_char": 80}
    config.models.asr.options = {"partial_every_frames": 5,
                                 "script": ["kể về Hà Nội", "thôi kể về Huế đi"]}
    config.models.llm.options = {"first_token_delay_ms": 10, "token_delay_ms": 150, "reply": LONG}
    eng, _ = build_engine(config)
    await eng.models.start()
    await eng.start()
    try:
        await feed(eng, [Step("silence", 100), Step("speech", 600), Step("silence", 500)])
        assert await wait_until(eng, is_speaking, max_ms=10000)
        assert await wait_until(eng, lambda e: _heard(e), max_ms=10000)
        first = eng.gen.current
        assert not eng._response.llm_done

        await feed(eng, [Step("speech", 140), Step("silence", 800)])   # a cough
        assert eng.counters.get("resumed") == 1
        assert await wait_until(eng, is_speaking, max_ms=5000)
        await wait_until(eng, lambda e: False, max_ms=400)

        await feed(eng, [Step("speech", 900), Step("silence", 700)])   # a real turn
        assert eng.counters.get("barge_ins") == 2
        assert await wait_until(eng, lambda e: e.context.turns[-1].user_text.startswith("thôi"),
                                max_ms=5000)
        await asyncio.sleep(0.2)
        assert not _live(f"respond-{first}"), "the first turn's LLM is still streaming"
        assert first not in eng._drain_targets
    finally:
        await eng.close()
        await eng.models.close()
