"""Regressions against misleading latency and cross-round pairing."""
import asyncio

import pytest

from voiceplatform.core.events import Event, EventType as E
from voiceplatform.core.limits import WorkLimiter
from voiceplatform.observability.metrics import MetricsRegistry
from voiceplatform.observability.probe import Probe, observing
from voiceplatform.observability.trace import TurnTrace


def event(trace, kind, at, **data):
    trace.add(Event(kind, "s", ts_ms=at, turn_id=1, data=data))


def completed():
    t = TurnTrace("s", 1)
    event(t, E.TURN_CONFIRMED, 100)
    event(t, E.PHRASE_READY, 101, phrase_id="f", role="filler")
    event(t, E.AUDIO_SENT, 110, phrase_id="f", role="filler")
    event(t, E.PHRASE_READY, 130, phrase_id="c", role="content")
    event(t, E.AUDIO_SENT, 500, phrase_id="c", role="content")
    event(t, E.TURN_END, 600, answered=True)
    return t


def test_filler_does_not_lower_content_latency_and_fallback_is_not_success():
    t = completed()
    assert t.metrics()["first_any_audio_sent_ms"] == 10
    assert t.metrics()["first_content_audio_sent_ms"] == 400
    assert t.outcome()["success"]
    event(t, E.PHRASE_READY, 650, phrase_id="err", role="fallback")
    assert not t.outcome()["success"]
    registry = MetricsRegistry()
    assert not registry.observe_turn({**t.metrics(), "outcome":t.outcome()},key="s")
    assert not registry.snapshot()["latency"]


def test_each_llm_round_pairs_its_own_sent_first_content_tool_and_end():
    t = completed()
    for kind, at, req, fields in (
        (E.LLM_REQUEST_SENT, 150,"a",{}),
        (E.LLM_FIRST_TOOL_DELTA, 190,"a",{}),
        (E.LLM_TERMINATED, 250,"a",{"outcome":"complete"}),
        (E.LLM_REQUEST_SENT, 300,"b",{}),
        (E.LLM_FIRST_TOKEN, 330,"b",{}),
        (E.LLM_TERMINATED, 400,"b",{"outcome":"complete"}),
    ):
        event(t, kind, at, request_id=req, stage="llm",round=1 if req=="a" else 2,**fields)
    rows = t.summary()["llm_rounds"]
    assert rows[0]["request_ttft_ms"] is None
    assert rows[0]["request_first_tool_ms"] == 40
    assert rows[1]["request_ttft_ms"] == 30
    assert t.metrics()["llm_ttft_ms"] == 30
    assert t.metrics()["llm_total_ms"] == 200


def test_asr_final_is_finalize_only_not_whole_utterance():
    t = completed()
    event(t, E.ASR_START, 0)
    event(t, E.ASR_FINALIZE_START, 100)
    event(t, E.ASR_FINALIZE_END, 120)
    event(t, E.ASR_FINAL, 125)
    assert t.metrics()["asr_final_ms"] == 20
    assert t.metrics()["asr_stream_duration_ms"] == 125


def test_search_failure_is_not_success_even_if_a_delivery_was_spoken():
    t=completed()
    event(t,E.SEARCH_DELIVERED,700,ok=False,error="timeout")
    assert not t.outcome()["success"]
    assert "search" in t.outcome()["errors"]


def test_repeated_scrapes_reuse_derivation_but_late_render_updates_it(monkeypatch):
    from voiceplatform.observability import trace as module
    real=module.measurements
    calls=[]
    def counted(events):
        calls.append(len(events))
        return real(events)
    monkeypatch.setattr(module,"measurements",counted)
    t=completed()
    for _ in range(20):
        t.summary()
    assert len(calls)==1
    event(t,E.PLAYBACK_STARTED,550,phrase_id="c",source="browser_audio_render",clock_uncertainty_ms=1)
    assert t.summary()["metrics"]["content_playback_start_ms"]==450
    assert len(calls)==2


def test_late_render_feedback_is_counted_once_and_estimates_are_excluded():
    t = completed()
    registry = MetricsRegistry()
    def observe():
        registry.observe_turn({**t.metrics(),"outcome":t.outcome()},key="s")
    observe()
    event(t,E.PLAYBACK_STARTED,530,phrase_id="c",source="legacy_scheduler_estimate",clock_uncertainty_ms=1)
    assert t.metrics()["content_playback_start_ms"] is None
    event(t,E.PLAYBACK_STARTED,540,phrase_id="c",source="browser_audio_render",clock_uncertainty_ms=40)
    assert t.metrics()["content_playback_start_ms"] is None
    event(t,E.PLAYBACK_STARTED,550,phrase_id="c",source="browser_audio_render",clock_uncertainty_ms=1)
    observe(); observe()
    stats = registry.snapshot()["latency"]
    assert stats["first_content_audio_sent_ms"]["n"] == 1
    assert stats["content_playback_start_ms"]["n"] == 1
    assert stats["content_playback_start_ms"]["p50"] == 450


@pytest.mark.asyncio
async def test_queue_wait_and_native_thread_marks_preserve_identity():
    marks = []
    probe = Probe("tts", lambda *args: marks.append(args), {"phrase_id":"c"})
    limiter = WorkLimiter(1, 2)
    entered = asyncio.Event()
    release = asyncio.Event()
    async def holder():
        async with limiter.slot():
            entered.set()
            await release.wait()
    held = asyncio.create_task(holder())
    await entered.wait()
    async def waiter():
        with observing(probe):
            async with limiter.slot():
                await asyncio.to_thread(probe.mark,E.MODEL_INFERENCE_START)
    waiting = asyncio.create_task(waiter())
    await asyncio.sleep(.025)
    assert limiter.snapshot()["waiting"] == 1
    release.set()
    await asyncio.gather(held,waiting)
    await asyncio.sleep(0)
    acquired = next(data for kind,_,data in marks if kind is E.MODEL_SLOT_ACQUIRED)
    assert acquired["queue_ms"] >= 20
    assert all(data["request_id"] == probe.request_id for _,_,data in marks)
    assert limiter.snapshot()["active"] == limiter.snapshot()["waiting"] == 0


@pytest.mark.asyncio
async def test_usage_after_finish_reason_is_retained_without_duplicate_finish():
    import httpx
    from voiceplatform.models.llm.openai_compat import OpenAiCompatLlm
    from voiceplatform.models.base import Message
    import json
    chunks = [
        {"choices":[{"delta":{"content":"Chào"},"finish_reason":None}]},
        {"choices":[{"delta":{},"finish_reason":"stop"}]},
        {"choices":[],"usage":{"prompt_tokens":20,"completion_tokens":2}},
    ]
    payload = ''.join('data: '+json.dumps(c)+'\n\n' for c in chunks)+'data: [DONE]\n\n'
    model = OpenAiCompatLlm(endpoint="http://test/v1",model="test")
    model._client = httpx.AsyncClient(base_url="http://test/v1",transport=httpx.MockTransport(lambda _:httpx.Response(200,text=payload)))
    events=[]
    probe=Probe("llm",lambda *args:events.append(args),{"round":0})
    try:
        with observing(probe):
            deltas=[d async for d in model.stream([Message("user","chào")])]
    finally:
        await model.close()
    assert sum(d.finish_reason is not None for d in deltas)==1
    usage=next(data for kind,_,data in events if kind is E.LLM_USAGE)
    assert usage["prompt_tokens"]==20
    assert events[-1][0] is E.LLM_TERMINATED
