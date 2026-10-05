"""Regression cases for the failure probes from the runtime audit."""
import asyncio
import importlib.util
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from voiceplatform.app.lab import Busy, LabService, redact_options, restore_secrets
from voiceplatform.app.server import Platform, create_app
from voiceplatform.app.simulate import build_engine
from voiceplatform.conversation.context import ConversationContext
from voiceplatform.core.config import Config
from voiceplatform.core.errors import ConfigError, ModelTimeout
from voiceplatform.core.ids import GenerationKey
from voiceplatform.core.limits import WorkLimiter
from voiceplatform.models import registry as model_registry
from voiceplatform.models.base import Message, ToolCall
from voiceplatform.models.llm.openai_compat import OpenAiCompatLlm
from voiceplatform.tasks.base import TaskContext, ToolResult, ToolSpec
from voiceplatform.tasks.executor import TaskExecutor
from voiceplatform.tasks.registry import ToolRegistry


@pytest.mark.parametrize("section,key,value", [
    ("audio", "frame_ms", 0), ("audio", "channels", 2),
    ("tasks", "max_parallel", 0), ("tasks", "max_parallel", True),
    ("server", "max_sessions", "3"), ("server", "port", 70000),
    ("server", "idle_timeout_s", float("nan")), ("models", "mode", "s2s"),
    ("conversation", "history_turns", 0),
])
def test_invalid_config_is_rejected_before_start(section, key, value):
    config = Config()
    setattr(getattr(config, section), key, value)
    with pytest.raises(ConfigError):
        config.validate()


def test_history_storage_is_bounded_including_delivery():
    context = ConversationContext("test", history_turns=2)
    for index in range(1000):
        context.start_turn(index, "hello")
    context.start_delivery_turn(1000)
    assert [t.turn_id for t in context.turns] == [999, 1000]


async def test_cancel_during_filler_wait_cancels_tool_and_close_joins_tasks(config):
    config.conversation.filler.after_ms = 1000
    engine, _ = build_engine(config, tools=[])
    started, cancelled = asyncio.Event(), asyncio.Event()

    class Tool:
        spec = ToolSpec(name="probe", description="regression", timeout_ms=2000)
        async def run(self, arguments, ctx):
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()

    registry = ToolRegistry()
    registry.register(Tool())
    engine.executor = TaskExecutor(registry)
    engine.gen.next_turn()
    key = engine.gen.begin()
    parent = asyncio.create_task(engine._run_tool(key, ToolCall(id="p", name="probe"), asyncio.Queue(), "test"))
    await asyncio.wait_for(started.wait(), 1)
    parent.cancel()
    await asyncio.gather(parent, return_exceptions=True)
    await engine.close()
    assert cancelled.is_set()
    assert not any(not t.done() for tasks in engine.gen._tasks.values() for t in tasks)
    await engine.models.close()


async def test_tool_deadline_includes_waiting_for_slot():
    class Tool:
        spec = ToolSpec(name="probe", description="regression", timeout_ms=30)
        async def run(self, arguments, ctx):
            pytest.fail("timed out work must not start")

    registry = ToolRegistry()
    registry.register(Tool())
    executor = TaskExecutor(registry, max_parallel=1)
    await executor._sem.acquire()
    try:
        result = await asyncio.wait_for(executor.run("probe", {}, TaskContext(key=GenerationKey("p", 1, 1), session_id="p")), .2)
        assert result.error == "timeout"
        assert executor.limiter.pending == 0
    finally:
        executor._sem.release()


async def test_tool_limits_are_shared_by_sessions(config):
    platform = Platform(config)
    assert platform.executor().limiter is platform.executor().limiter
    await platform.close()


async def test_swap_race_preserves_old_engine_and_closes_candidate(config, monkeypatch):
    platform = Platform(config)
    await platform.start()
    service = LabService(platform, config)
    old = platform.models.llm
    new = model_registry.build_llm(config.models.llm)
    entered, proceed, closed = asyncio.Event(), asyncio.Event(), asyncio.Event()
    async def start():
        entered.set()
        await proceed.wait()
    async def close():
        closed.set()
    new.start, new.close = start, close
    monkeypatch.setattr(model_registry, "build_llm", lambda _: new)
    swapping = asyncio.create_task(service.swap("llm", "mock", {}))
    try:
        await asyncio.wait_for(entered.wait(), 1)
        assert platform.maintenance
        # Simulate a noncooperating caller inserting a session during loading.
        platform.sessions["race"] = object()
        proceed.set()
        with pytest.raises(Busy):
            await swapping
        assert platform.models.llm is old
        assert closed.is_set()
        assert not platform.maintenance
    finally:
        platform.sessions.clear()
        await platform.close()


async def test_swap_and_admission_use_the_same_lock(config):
    platform = Platform(config)
    assert LabService(platform, config)._lock is platform.model_lock
    await platform.close()


async def test_readiness_is_cached_and_detects_dependency_failure(config):
    platform = Platform(config)
    await platform.start()
    calls = 0
    async def dependencies(timeout):
        nonlocal calls
        calls += 1
        return {"llm": {"ok": False, "reason": "ConnectionError"}}
    platform.models.check_dependencies = dependencies
    try:
        assert not (await platform.readiness())["ok"]
        assert not (await platform.readiness())["ok"]
        assert calls == 1
        platform.invalidate_readiness()
        await platform.readiness()
        assert calls == 2
    finally:
        await platform.close()


def test_liveness_readiness_and_admission_disagree_when_llm_down(config):
    app = create_app(config)
    with TestClient(app) as client:
        async def dependencies(timeout):
            return {"llm": {"ok": False}}
        app.state.platform.models.check_dependencies = dependencies
        assert client.get("/healthz").status_code == 200
        assert client.get("/readyz").status_code == 503
        with client.websocket_connect("/v1/realtime") as ws:
            assert ws.receive_json()["stage"] == "admission"
        assert not app.state.platform.sessions


def test_capacity_trace_token_and_invalid_audio(config):
    config.server.private_introspection = True
    config.server.max_sessions = 1
    app = create_app(config)
    with TestClient(app) as client:
        with client.websocket_connect("/v1/realtime") as ws:
            ready = ws.receive_json()
            sid = ready["session_id"]
            path = f"/sessions/{sid}/turns"
            assert client.get(path).status_code == 403
            assert client.get(path, headers={"Authorization": "Bearer wrong"}).status_code == 403
            assert client.get(path, headers={b"Authorization": b"Bearer \xff"}).status_code == 403
            assert client.get(path, headers={"Authorization": "Bearer " + ready["session_token"]}).status_code == 200
            with client.websocket_connect("/v1/realtime") as extra:
                assert extra.receive_json()["stage"] == "admission"
            ws.send_bytes(b"\x00")
        assert not app.state.platform.sessions
        assert not app.state.platform.session_tokens


def test_engine_credentials_are_redacted_and_roundtrip_is_safe(config):
    platform = Platform(config)
    config.models.llm.options["api_key"] = "test-secret"
    description = LabService(platform, config).describe()
    assert "test-secret" not in str(description)
    raw = {"api_key": "secret", "headers": {"Authorization": "Bearer token"}, "endpoint": "https://u:p@local/v1?key=secret", "max_tokens": 40}
    safe = redact_options(raw)
    assert "secret" not in str(safe) and "Bearer token" not in str(safe)
    assert safe["max_tokens"] == 40
    assert restore_secrets(safe, raw) == raw


async def test_llm_readiness_checks_actual_model_and_queue_deadline():
    llm = OpenAiCompatLlm(model="wanted", timeout_s=.03, max_parallel=1)
    llm._client = httpx.AsyncClient(transport=httpx.MockTransport(lambda request: httpx.Response(200, json={"data": [{"id": "other"}]})), base_url="http://test/v1")
    assert (await llm.check_ready())["reason"] == "configured_model_not_available"
    occupied = llm.limiter.slot()
    await occupied.__aenter__()
    try:
        with pytest.raises(ModelTimeout):
            async for _ in llm.stream([Message(role="user", content="hi")]):
                pass
        assert llm.limiter.pending == 1  # only the occupied slot survives timeout
    finally:
        await occupied.__aexit__(None, None, None)
        await llm.close()


async def test_stream_timeout_is_total_even_if_tokens_keep_arriving():
    llm = OpenAiCompatLlm(timeout_s=.03)
    async def slow(*args, **kwargs):
        from voiceplatform.models.base import LLMDelta
        while True:
            await asyncio.sleep(.005)
            yield LLMDelta(text="a")
    llm._stream = slow
    with pytest.raises(ModelTimeout):
        async for _ in llm.stream([]):
            pass
    assert llm.limiter.pending == 0


def test_conversation_smoke_rejects_fallback_and_unfinished_turn():
    path = Path(__file__).resolve().parents[1] / "scripts/conversation_check.py"
    spec = importlib.util.spec_from_file_location("conversation_check", path)
    import sys
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    events = [{"type": t, "data": {}} for t in ("turn_confirmed", "llm_first_token", "tts_first_audio")]
    assert not module.validate_turns({"turns": [{"events": events}]}, require_answer=True)[0]
    events.append({"type": "turn_end", "data": {"answered": True}})
    assert module.validate_turns({"turns": [{"events": events}]}, require_answer=True)[0]
    events.append({"type": "error", "data": {"stage": "llm"}})
    assert not module.validate_turns({"turns": [{"events": events}]}, require_answer=True)[0]


def test_conversation_smoke_accepts_an_interjection_left_unanswered_on_purpose():
    path = Path(__file__).resolve().parents[1] / "scripts/conversation_check.py"
    spec = importlib.util.spec_from_file_location("conversation_check_bc", path)
    import sys
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    answered = [{"type": t, "data": {}} for t in ("turn_confirmed", "llm_first_token", "tts_first_audio")]
    answered.append({"type": "turn_end", "data": {"answered": True}})
    def heard(reason):
        return {"events": [{"type": "turn_confirmed", "data": {}},
                           {"type": "turn_end", "data": {"answered": False, **({"reason": reason} if reason else {})}}]}
    # "ừ" over the answer: confirmed, deliberately not answered, the cut answer resumes.
    assert module.validate_turns({"turns": [{"events": answered}, heard("backchannel")]}, require_answer=True)[0]
    # The same turn without a reason is still a turn that failed.
    assert not module.validate_turns({"turns": [{"events": answered}, heard(None)]}, require_answer=True)[0]


async def test_asr_close_retains_native_work_until_it_finishes():
    from types import SimpleNamespace
    import numpy as np
    from voiceplatform.models.asr.bridge_viet_s2s import BridgeAsrEngine, _BridgeAsrStream
    from voiceplatform.models.base import AsrCapabilities
    entered, release, model_closed = asyncio.Event(), asyncio.Event(), asyncio.Event()
    async def transcribe(*args, **kwargs):
        entered.set()
        await release.wait()
        return SimpleNamespace(text="hi")
    async def close():
        model_closed.set()
    engine = object.__new__(BridgeAsrEngine)
    engine.backend = SimpleNamespace(transcribe=transcribe, close=close)
    engine.capabilities = AsrCapabilities()
    engine.limiter = WorkLimiter(1, 1)
    engine._workers = set()
    engine._closing = False
    engine._loaded = True
    engine.decode_timeout_s = 1
    engine.close_timeout_s = 1
    stream = _BridgeAsrStream(engine, 16000, None)
    task = asyncio.create_task(stream._decode(np.zeros(160), partial=True))
    await entered.wait()
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)
    closing = asyncio.create_task(engine.close())
    await asyncio.sleep(.01)
    assert not model_closed.is_set() and engine._workers
    release.set()
    await closing
    assert model_closed.is_set() and not engine._workers


async def test_old_asr_cleanup_does_not_clear_a_new_turn_stream(config):
    from voiceplatform.models.base import Transcript
    engine, _ = build_engine(config, tools=[])
    entered, release = asyncio.Event(), asyncio.Event()
    class OldStream:
        async def finish(self):
            entered.set()
            await release.wait()
            return Transcript(text="old", is_final=True)
        async def close(self):
            pass
    class NewStream:
        async def close(self):
            pass
    old, new = OldStream(), NewStream()
    engine.gen.next_turn()
    key = engine.gen.begin()
    engine._asr_stream = old
    responding = asyncio.create_task(engine._respond(key))
    await entered.wait()
    engine.gen.next_turn()
    engine.gen.begin()
    engine._asr_stream = new
    release.set()
    await responding
    assert engine._asr_stream is new
    await engine.close()
    await engine.models.close()


async def test_tool_returning_error_is_visible_to_the_smoke_trace():
    from voiceplatform.core.events import EventType
    class Tool:
        spec = ToolSpec(name="probe", description="regression")
        async def run(self, arguments, ctx):
            return ToolResult(ok=False, content="failed", error="backend_down")
    registry = ToolRegistry()
    registry.register(Tool())
    events = []
    executor = TaskExecutor(registry, emit=lambda kind, key, data: events.append((kind, data)))
    result = await executor.run("probe", {}, TaskContext(key=GenerationKey("p", 1, 1), session_id="p"))
    assert not result.ok
    assert any(kind is EventType.TOOL_FAILED and data["error"] == "backend_down" for kind, data in events)
