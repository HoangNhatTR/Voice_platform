"""OpenAI-compatible stream: deadlines, in-band errors, truncation, tool calls.

No network: `_stream` is replaced by a fake for the deadline cases, and
httpx.MockTransport serves canned SSE for the parsing cases.
"""

from __future__ import annotations

import asyncio
import json
from contextlib import aclosing

import httpx
import pytest

from voiceplatform.app.simulate import build_engine, is_idle, wait_until
from voiceplatform.conversation.first_phrase import with_first_phrase_deadline
from voiceplatform.core.errors import ModelTimeout, ModelUnavailable
from voiceplatform.core.events import EventType
from voiceplatform.models.base import LLMDelta, Message
from voiceplatform.models.llm.openai_compat import OpenAiCompatLlm

USER = [Message(role="user", content="x")]
TOOLS = [{"type": "function", "function": {"name": "search"}}]


def _endless(timeout_s: float = 0.3) -> OpenAiCompatLlm:
    llm = OpenAiCompatLlm(timeout_s=timeout_s)

    async def fake(messages, *, tools=None, max_tokens=None):
        for i in range(2000):
            await asyncio.sleep(0.005)
            yield LLMDelta(text=f"t{i} ")
        yield LLMDelta(finish_reason="stop")

    llm._stream = fake
    return llm


class _Segmenter:
    first_pending = False


# --------------------------------------------------------------------------- #
# the total deadline
# --------------------------------------------------------------------------- #
async def test_deadline_while_consumer_holds_a_delta_is_a_model_timeout():
    """The deadline used to fire inside the CONSUMER's await: a bare
    CancelledError that killed the turn without a fallback reply."""
    llm = _endless()
    with pytest.raises(ModelTimeout):
        async with aclosing(llm.stream(USER)) as stream:
            async for _ in stream:
                await asyncio.sleep(0.1)    # e.g. a slow websocket send
    assert llm.limiter.snapshot()["active"] == 0


async def test_deadline_behind_the_first_phrase_reader_never_hangs():
    """Measured: the reader died parked in queue.put and the consumer waited
    in queue.get() forever, until the orphan sweep."""
    llm = _endless()

    async def consume():
        stream = with_first_phrase_deadline(llm.stream(USER), 250, _Segmenter())
        async with aclosing(stream):
            async for _ in stream:
                await asyncio.sleep(0.1)

    with pytest.raises(ModelTimeout):
        await asyncio.wait_for(consume(), 3.0)
    assert llm.limiter.snapshot()["active"] == 0


async def test_first_phrase_reader_killed_by_base_exception_wakes_the_consumer():
    async def dies():
        yield LLMDelta(text="một ")
        raise asyncio.CancelledError   # not an Exception: used to be swallowed

    async def consume():
        out = []
        stream = with_first_phrase_deadline(dies(), 250, _Segmenter())
        async with aclosing(stream):
            async for delta in stream:
                out.append(delta)
        return out

    with pytest.raises(ModelUnavailable, match="reader stopped"):
        await asyncio.wait_for(consume(), 2.0)


async def test_engine_speaks_the_fallback_when_the_deadline_fires_mid_answer(config):
    config.conversation.streaming_first_phrase_wait_ms = 250
    config.conversation.orphan_turn_timeout_ms = 5000
    engine, sink = build_engine(config, tools=[])
    engine.models.llm = _endless(timeout_s=0.6)
    send = sink.send_control

    async def slow_client(message):
        if message.type == "assistant_delta":
            await asyncio.sleep(0.05)
        return await send(message)

    sink.send_control = slow_client
    await engine.models.start()
    await engine.start()
    try:
        await engine.push_text("kể một câu chuyện dài")
        assert await wait_until(engine, is_idle, max_ms=4000, feed_silence=False)
        errors = [e.data.get("stage") for e in engine.trace.turn(1).events if e.type is EventType.ERROR]
        assert "llm" in errors and "orphan_turn" not in errors
    finally:
        await engine.close()
        await engine.models.close()


# --------------------------------------------------------------------------- #
# SSE parsing
# --------------------------------------------------------------------------- #
def _sse(chunks, *, done=True, raw=b"") -> bytes:
    body = "".join(f"data: {json.dumps(c, ensure_ascii=False)}\n\n" for c in chunks).encode() + raw
    return body + (b"data: [DONE]\n\n" if done else b"")


def _llm(body: bytes, **options) -> tuple[OpenAiCompatLlm, list]:
    llm = OpenAiCompatLlm(endpoint="http://fake/v1", **options)
    seen = []

    def handler(request):
        seen.append(json.loads(request.content))
        return httpx.Response(200, content=body, headers={"content-type": "text/event-stream"})

    llm._client = httpx.AsyncClient(base_url=llm.endpoint, transport=httpx.MockTransport(handler))
    return llm, seen


async def _collect(llm):
    return [d async for d in llm.stream(USER, tools=TOOLS)]


def _text(content, finish=None):
    return {"choices": [{"delta": {"content": content}, "finish_reason": finish}]}


@pytest.mark.parametrize("raw", [
    b'error: {"code": 500, "message": "the request exceeds the available context size"}\n\n',
    b'data: {"error": {"code": 500, "message": "slot crashed"}}\n\n',
])
async def test_in_band_error_is_unavailable_not_a_silent_stop(raw):
    llm, _ = _llm(_sse([_text("Xin ")], done=False, raw=raw))
    with pytest.raises(ModelUnavailable, match="stream error"):
        await _collect(llm)


async def test_error_after_a_complete_answer_does_not_add_a_fallback():
    raw = b'error: {"message": "late"}\n\n'
    llm, _ = _llm(_sse([_text("Xong.", "stop")], done=False, raw=raw))
    out = await _collect(llm)
    assert [d.finish_reason for d in out if d.finish_reason] == ["stop"]


async def test_truncated_stream_raises_instead_of_dropping_the_tool_call():
    frag = {"choices": [{"delta": {"tool_calls": [
        {"index": 0, "id": "a", "function": {"name": "search", "arguments": '{"query": "hồ'}}]}}]}
    llm, _ = _llm(_sse([frag], done=False))
    with pytest.raises(ModelUnavailable, match="before finish_reason"):
        await _collect(llm)


async def test_done_without_finish_reason_still_delivers_the_tool_call():
    frag = {"choices": [{"delta": {"tool_calls": [
        {"index": 0, "id": "a", "function": {"name": "search", "arguments": '{"query": "x"}'}}]}}]}
    llm, _ = _llm(_sse([frag]))
    out = await _collect(llm)
    calls = [d.tool_call for d in out if d.tool_call]
    assert [(c.name, c.arguments) for c in calls] == [("search", {"query": "x"})]
    assert out[-1].finish_reason == "stop"


async def test_tool_calls_without_index_are_not_merged():
    chunks = [
        {"choices": [{"delta": {"tool_calls": [
            {"id": "a", "function": {"name": "search", "arguments": '{"query": "hồ Gươm"}'}},
            {"id": "b", "function": {"name": "clock", "arguments": "{}"}}]}}]},
        # an index-less call streamed in pieces: id once, then bare fragments
        {"choices": [{"delta": {"tool_calls": [{"id": "c", "function": {"name": "search", "arguments": '{"qu'}}]}}]},
        {"choices": [{"delta": {"tool_calls": [{"function": {"arguments": 'ery": "Huế"}'}}]}, "finish_reason": "tool_calls"}]},
    ]
    llm, _ = _llm(_sse(chunks))
    calls = [d.tool_call for d in await _collect(llm) if d.tool_call]
    assert [(c.id, c.name, c.arguments) for c in calls] == [
        ("a", "search", {"query": "hồ Gươm"}), ("b", "clock", {}), ("c", "search", {"query": "Huế"})]


async def test_indexed_parallel_calls_and_split_arguments_still_assemble():
    chunks = [
        {"choices": [{"delta": {"role": "assistant", "content": ""}}]},
        {"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "a", "function": {"name": "search", "arguments": '{"que'}}]}}]},
        {"choices": [{"delta": {"tool_calls": [{"index": 1, "id": "b", "function": {"name": "clock", "arguments": ""}}]}}]},
        {"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": 'ry": "x"}'}},
                                               {"index": 1, "function": {"arguments": "{}"}}]},
                      "finish_reason": "tool_calls"}]},
        {"choices": [], "usage": {"prompt_tokens": 3, "completion_tokens": 4}},
    ]
    llm, _ = _llm(_sse(chunks))
    out = await _collect(llm)
    assert [(d.tool_call.name, d.tool_call.arguments) for d in out if d.tool_call] == [
        ("search", {"query": "x"}), ("clock", {})]
    # Old code treated a missing index as 0; the bare-index check above is the
    # regression, this one guards the indexed path and the empty first delta.
    assert not any(d.text for d in out)


# --------------------------------------------------------------------------- #
# request body traps
# --------------------------------------------------------------------------- #
def test_extra_body_template_kwargs_merge_and_keep_thinking_off():
    llm = OpenAiCompatLlm(extra_body={"seed": 1, "chat_template_kwargs": {"add_vision_id": False}})
    body = llm._body(USER, None, None)
    assert body["chat_template_kwargs"] == {"add_vision_id": False, "enable_thinking": False}
    assert body["seed"] == 1
    with pytest.raises(ValueError, match="enable_thinking"):
        OpenAiCompatLlm(extra_body={"chat_template_kwargs": {"enable_thinking": True}})


def test_literal_policy_never_overwrites_the_users_own_message():
    llm = OpenAiCompatLlm(literal_request_policy=True, require_system_message=False)
    text = "Đọc đúng câu: bây giờ là mấy giờ"
    body = llm._body([Message("user", text)], TOOLS, None)
    assert [m["role"] for m in body["messages"]] == ["system", "user"]
    assert body["messages"][1]["content"] == text and "tools" not in body
