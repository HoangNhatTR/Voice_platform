import asyncio
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path.cwd() / "src"))

import httpx
import numpy as np

from voiceplatform.app.lab import LabService
from voiceplatform.app.server import Platform, create_app
from voiceplatform.app.simulate import build_engine
from voiceplatform.conversation.context import ConversationContext
from voiceplatform.core.config import Config, EngineSpec
from voiceplatform.core.events import Event, EventType
from voiceplatform.core.ids import GenerationKey
from voiceplatform.models import registry as model_registry
from voiceplatform.models.tts.zerotts import ZeroTtsEngine
from voiceplatform.tasks.base import TaskContext, ToolResult, ToolSpec
from voiceplatform.tasks.executor import TaskExecutor
from voiceplatform.tasks.registry import ToolRegistry


def config():
    c = Config()
    c.observability.write_traces = False
    c.observability.log_events = False
    return c


async def tts_initialization_error():
    class Broken:
        def synthesize_stream(self, *args, **kwargs):
            raise RuntimeError("synthetic init failure")
    engine = ZeroTtsEngine()
    engine._tts = Broken()
    stream = engine.synthesize("test")
    try:
        await asyncio.wait_for(stream.__anext__(), 0.3)
        outcome = "unexpected audio"
    except asyncio.TimeoutError:
        outcome = "consumer hung; external timeout required"
    except RuntimeError:
        outcome = "error propagated correctly"
    return {"outcome": outcome, "lock_still_held": engine._lock.locked()}


async def swap_race():
    cfg = config()
    platform = Platform(cfg)
    await platform.start()
    service = LabService(platform, cfg)
    old = platform.models.llm
    entered, proceed = asyncio.Event(), asyncio.Event()
    new = model_registry.build_llm(EngineSpec())
    original_builder = model_registry.build_llm
    old_closed = False
    original_close = old.close
    async def close_old():
        nonlocal old_closed
        old_closed = True
        await original_close()
    async def slow_start():
        entered.set()
        await proceed.wait()
    old.close = close_old
    new.start = slow_start
    model_registry.build_llm = lambda _: new
    try:
        task = asyncio.create_task(service.swap("llm", "mock", {}))
        await entered.wait()
        platform.sessions["connected-during-load"] = SimpleNamespace(voice=None)
        proceed.set()
        await task
        return {"swap_succeeded_with_live_sessions": len(platform.sessions), "old_model_closed": old_closed}
    finally:
        model_registry.build_llm = original_builder
        platform.sessions.clear()
        await platform.close()


async def remote_reads():
    cfg = config()
    cfg.server.private_introspection = True
    cfg.models.llm = EngineSpec(backend="openai_compat", options={"api_key": "audit-fake-secret"})
    app = create_app(cfg)
    platform = app.state.platform
    from voiceplatform.conversation.engine import ConversationEngine
    from voiceplatform.conversation.sink import CollectingSink
    engine = ConversationEngine(cfg, platform.models, CollectingSink(), session_id="audit-known-id")
    engine.trace.record(Event(type=EventType.ASR_FINAL, session_id=engine.session_id, turn_id=1, data={"text": "synthetic private transcript"}))
    platform.sessions[engine.session_id] = engine
    transport = httpx.ASGITransport(app=app, client=("198.51.100.50", 5000))
    async with httpx.AsyncClient(transport=transport, base_url="http://audit") as client:
        sessions = await client.get("/sessions")
        turns = await client.get("/sessions/audit-known-id/turns")
        engines = await client.get("/engines")
    platform.sessions.clear()
    await platform.close()
    return {"remote_sessions_status": sessions.status_code, "remote_turns_status": turns.status_code, "dummy_transcript_visible": "synthetic private transcript" in turns.text, "remote_engines_status": engines.status_code, "dummy_api_key_visible": "audit-fake-secret" in engines.text}


async def orphan_tool():
    cfg = config()
    cfg.conversation.filler.after_ms = 1000
    engine, _ = build_engine(cfg, tools=[])
    started, release = asyncio.Event(), asyncio.Event()
    completed = False
    cancelled = False
    class Tool:
        spec = ToolSpec(name="probe", description="audit probe", timeout_ms=2000)
        async def run(self, arguments, ctx):
            nonlocal completed, cancelled
            started.set()
            try:
                await release.wait()
                completed = True
                return ToolResult(ok=True, content="done")
            except asyncio.CancelledError:
                cancelled = True
                raise
    registry = ToolRegistry()
    registry.register(Tool())
    engine.executor = TaskExecutor(registry)
    engine.gen.next_turn()
    key = engine.gen.begin()
    from voiceplatform.models.base import ToolCall
    parent = asyncio.create_task(engine._run_tool(key, ToolCall(id="probe", name="probe"), asyncio.Queue(), "test"))
    await started.wait()
    parent.cancel()
    await asyncio.gather(parent, return_exceptions=True)
    await engine.close()
    still_running_after_close = not cancelled and not completed
    release.set()
    await asyncio.sleep(0.03)
    await engine.models.close()
    return {"tool_survived_turn_and_session_close": still_running_after_close, "tool_completed_after_session_close": completed, "tool_was_cancelled": cancelled}


async def semaphore_deadline():
    started, release = asyncio.Event(), asyncio.Event()
    class Tool:
        spec = ToolSpec(name="probe", description="audit probe", timeout_ms=50)
        async def run(self, arguments, ctx):
            return ToolResult(ok=True, content="done")
    registry = ToolRegistry()
    registry.register(Tool())
    executor = TaskExecutor(registry, max_parallel=1)
    await executor._sem.acquire()
    ctx = TaskContext(key=GenerationKey("audit", 1, 1), session_id="audit")
    before = time.monotonic()
    task = asyncio.create_task(executor.run("probe", {}, ctx))
    await asyncio.sleep(0.12)
    exceeded_deadline = not task.done()
    executor._sem.release()
    result = await task
    return {"still_waiting_after_50ms_budget_at_120ms": exceeded_deadline, "returned_ok": result.ok, "wall_ms": round((time.monotonic()-before)*1000, 1)}


async def config_validation_and_history():
    from voiceplatform.core.config import _build
    from voiceplatform.media.framer import Framer
    cfg = _build(Config, {"audio": {"frame_ms": 0}, "tasks": {"max_parallel": 0}, "models": {"mode": "imaginary"}}, "audit")
    context = ConversationContext("test", history_turns=2)
    for i in range(1000):
        context.start_turn(i, "test")
    return {"zero_frame_ms_accepted": cfg.audio.frame_ms == 0, "zero_parallel_accepted": cfg.tasks.max_parallel == 0, "unknown_mode_accepted": cfg.models.mode, "history_turns_configured": 2, "history_turns_stored": len(context.turns)}


async def main():
    errors = []
    loop = asyncio.get_running_loop()
    loop.set_exception_handler(lambda _loop, context: errors.append(str(context.get("exception") or context.get("message"))))
    results = {}
    for name, probe in (("tts_initialization_error", tts_initialization_error), ("swap_race", swap_race), ("remote_reads", remote_reads), ("orphan_tool", orphan_tool), ("semaphore_deadline", semaphore_deadline), ("config_and_history", config_validation_and_history)):
        results[name] = await asyncio.wait_for(probe(), 5)
    await asyncio.sleep(0.05)
    results["unhandled_background_errors"] = errors
    path = Path("/tmp/voice-platform-failure-probes.json")
    path.write_text(json.dumps(results, ensure_ascii=False, indent=2))
    print(json.dumps(results, ensure_ascii=False, indent=2))


asyncio.run(main())
