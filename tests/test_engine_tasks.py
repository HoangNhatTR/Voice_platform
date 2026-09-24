"""The information plane: slow work must not stall the conversation."""

from __future__ import annotations

import pytest

from voiceplatform.app.simulate import build_engine, is_idle, wait_until
from voiceplatform.core.events import EventType
from voiceplatform.tasks.builtin.echo import SlowTool
from voiceplatform.tasks.executor import TaskExecutor
from voiceplatform.tasks.registry import ToolRegistry


@pytest.fixture
async def engine(config):
    config.models.llm.options = {
        "first_token_delay_ms": 10,
        "token_delay_ms": 1,
        "tool_triggers": {"mấy giờ": "clock", "chờ": "slow"},
    }
    eng, sink = build_engine(config, tools=["clock"])
    await eng.models.start()
    await eng.start()
    yield eng, sink
    await eng.close()
    await eng.models.close()


async def test_a_tool_call_feeds_the_answer(engine):
    eng, sink = engine
    await eng.push_text("mấy giờ rồi")
    assert await wait_until(eng, is_idle, max_ms=5000, feed_silence=False)

    trace = eng.trace.turn(eng.gen.turn_id)
    types = [ev.type for ev in trace.events]
    assert EventType.TOOL_START in types
    assert EventType.TOOL_COMPLETE in types
    assert "Theo tra cứu" in eng.context.turns[-1].assistant_text
    assert trace.metrics()["tool_ms"] is not None


async def test_a_slow_tool_gets_a_filler_and_audio_starts_before_it_returns(config):
    config.models.llm.options = {
        "first_token_delay_ms": 10,
        "token_delay_ms": 1,
        "tool_triggers": {"chờ": "slow"},
    }
    config.conversation.filler.after_ms = 80
    eng, sink = build_engine(config, tools=[])
    registry = ToolRegistry()
    registry.register(SlowTool(delay_ms=500))
    eng.executor = TaskExecutor(registry)
    await eng.models.start()
    await eng.start()
    try:
        await eng.push_text("chờ tôi tra cứu nhé")
        assert await wait_until(eng, is_idle, max_ms=8000, feed_silence=False)

        trace = eng.trace.turn(eng.gen.turn_id)
        firsts = trace.firsts
        assert EventType.FILLER.value in firsts, "no filler was spoken over the wait"
        # The user hears something well before the slow path finishes.
        assert firsts[EventType.TTS_FIRST_AUDIO.value] < firsts[EventType.TOOL_COMPLETE.value]
    finally:
        await eng.close()
        await eng.models.close()


async def test_an_unknown_tool_does_not_take_the_turn_down(engine):
    eng, _ = engine
    eng.executor.registry._tools.pop("clock")
    await eng.push_text("mấy giờ rồi")
    assert await wait_until(eng, is_idle, max_ms=5000, feed_silence=False)
    assert eng.context.turns[-1].assistant_text  # it still answered something


def test_the_system_prompt_announces_the_tools():
    """A tools array alone does not make a speech-tuned model call them."""
    from voiceplatform.conversation.context import ConversationContext

    ctx = ConversationContext("Trả lời ngắn gọn như lời nói.")
    ctx.start_turn(1, "mấy giờ rồi")

    plain = ctx.messages()[0].content
    assert "công cụ" not in plain

    announced = ctx.messages(tool_names=["clock", "kb"])[0].content
    assert "Trả lời ngắn gọn như lời nói." in announced
    assert "clock" in announced and "kb" in announced
    assert "không đoán" in announced.lower()


class _ParallelToolLlm:
    """Two tool calls in one round, then an answer built from both results."""

    name = "parallel-tools"

    def __init__(self) -> None:
        from voiceplatform.models.base import LlmCapabilities

        self.capabilities = LlmCapabilities(tools=True, streaming=True)
        self.rounds = 0

    async def start(self) -> None:
        return None

    async def close(self) -> None:
        return None

    async def stream(self, messages, *, tools=None, max_tokens=None):
        from voiceplatform.models.base import LLMDelta, ToolCall

        self.rounds += 1
        if self.rounds == 1:
            yield LLMDelta(tool_call=ToolCall(id="a", name="clock"))
            yield LLMDelta(
                tool_call=ToolCall(id="b", name="echo", arguments={"text": "hai"})
            )
            yield LLMDelta(finish_reason="tool_calls")
            return
        yield LLMDelta(text="Xong rồi.")
        yield LLMDelta(finish_reason="stop")


async def test_every_tool_call_in_a_round_runs(config):
    """Only the last one used to survive.

    The engine kept a single `tool_call` and overwrote it on each delta, so a
    model calling two tools in parallel had one of them silently dropped and
    the history carried one tool message for two calls.
    """
    eng, _ = build_engine(config, tools=["clock", "echo"])
    eng.models.llm = _ParallelToolLlm()
    await eng.models.start()
    await eng.start()
    try:
        await eng.push_text("hỏi hai thứ cùng lúc")
        assert await wait_until(eng, is_idle, max_ms=5000, feed_silence=False)

        trace = eng.trace.turn(eng.gen.turn_id)
        started = [
            ev.data["tool"] for ev in trace.events if ev.type is EventType.TOOL_START
        ]
        assert sorted(started) == ["clock", "echo"]
        assert eng.context.turns[-1].tool_calls == ["clock", "echo"]
    finally:
        await eng.close()
        await eng.models.close()


async def test_one_filler_per_round_not_one_per_tool(config):
    """Two slow tools are one silence, and one silence takes one filler."""
    config.conversation.filler.after_ms = 40
    eng, _ = build_engine(config, tools=[])
    registry = ToolRegistry()
    registry.register(SlowTool(delay_ms=200))
    from voiceplatform.tasks.builtin.clock import ClockTool

    registry.register(ClockTool())
    eng.executor = TaskExecutor(registry)

    class _TwoSlow(_ParallelToolLlm):
        async def stream(self, messages, *, tools=None, max_tokens=None):
            from voiceplatform.models.base import LLMDelta, ToolCall

            self.rounds += 1
            if self.rounds == 1:
                yield LLMDelta(tool_call=ToolCall(id="a", name="slow"))
                yield LLMDelta(tool_call=ToolCall(id="b", name="slow"))
                yield LLMDelta(finish_reason="tool_calls")
                return
            yield LLMDelta(text="Xong rồi.")
            yield LLMDelta(finish_reason="stop")

    eng.models.llm = _TwoSlow()
    await eng.models.start()
    await eng.start()
    try:
        await eng.push_text("chờ tôi tra cứu nhé")
        assert await wait_until(eng, is_idle, max_ms=8000, feed_silence=False)
        trace = eng.trace.turn(eng.gen.turn_id)
        fillers = [ev for ev in trace.events if ev.type is EventType.FILLER]
        assert len(fillers) == 1
    finally:
        await eng.close()
        await eng.models.close()
