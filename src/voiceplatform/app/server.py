"""HTTP + WebSocket entry point (the user plane's server side).

Endpoints:

    GET  /healthz          liveness plus which engines are loaded
    GET  /metrics          latency percentiles across live sessions
    GET  /sessions         per-session state and per-turn timings
    WS   /v1/realtime      the conversation itself
    GET  /                 the built-in test client

The server owns one model plane for the whole process and one engine per
connection: models stay warm between sessions, conversation state never leaks
between them.
"""

from __future__ import annotations

import asyncio
import json
import hmac
import secrets
import time
from collections import deque
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from ..conversation.engine import ConversationEngine
from ..conversation.state import TurnState
from ..core.config import Config
from ..core.clock import now_ms
from ..core.ids import new_session_id
from ..core.limits import WorkLimiter
from ..core.runtime import identity
from ..media.transport.base import ControlMessage
from ..models.registry import ModelPlane
from ..observability.logging import get_logger, setup_logging
from ..observability.metrics import MetricsRegistry
from ..tasks.executor import TaskExecutor
from ..tasks.registry import build_registry
from .access import local_only, must_be_private
from .collect import install as install_collect
from .lab import register as register_lab
from .transport_ws import WebSocketTransport

log = get_logger("server")

# How the server ends a session. 4000-4999 is the application range of RFC
# 6455; the refusals of a malformed frame keep their protocol codes.
_CLOSE_CODES = {"idle_timeout": 4000, "max_session_age": 4001, "audio_timeout": 1011}
_QUIET = frozenset({TurnState.IDLE, TurnState.CLOSED})
_HARD_GRACE_S = 30.0
# Control (text) messages per session: the test page sends a few a second
# (playback reports, a flush of queued ones after the output clock starts).
_CONTROL_RATE = 60.0
_CONTROL_BURST = 240
_CLIENT_ERROR_LOGS = 5


class _ControlBucket:
    """Token bucket: `rate` messages a second, saving up at most `burst`."""

    def __init__(self, rate: float, burst: int) -> None:
        self.rate = rate
        self.burst = float(burst)
        self.tokens = float(burst)
        self.at = time.monotonic()

    def take(self) -> bool:
        now = time.monotonic()
        self.tokens = min(self.burst, self.tokens + (now - self.at) * self.rate)
        self.at = now
        if self.tokens < 1:
            return False
        self.tokens -= 1
        return True


class Platform:
    """Process-wide state: models, tools, metrics, live sessions."""

    def __init__(self, config: Config) -> None:
        config.validate()
        self.config = config
        self.runtime = identity(config)
        self.loop_lag = deque(maxlen=600)
        self.model_lock = asyncio.Lock()
        self.maintenance = False
        self.tool_limiter = WorkLimiter(config.tasks.max_parallel, config.tasks.max_queue)
        self.session_tokens: dict[str, str] = {}
        self._ready_lock = asyncio.Lock()
        self._ready_at = float("-inf")
        self._dependencies: dict[str, Any] = {}
        self.models = ModelPlane(config.models, output_sample_rate=config.audio.output_sample_rate)
        if config.models.prewarm_before_sessions:
            from ..conversation.context import ConversationContext
            from ..models.base import Message
            from ..tasks.builtin.search_tool import SearchTool

            registry = build_registry(config.tasks.tools if config.tasks.enabled else [])
            tools = registry.openai_tools()
            if config.conversation.search.enabled and self.models.search is not None:
                tools.append(SearchTool(lambda _: None).spec.as_openai_tool())
            names = [t['function']['name'] for t in tools]
            context = ConversationContext(config.conversation.system_prompt,
                                          tool_instruction=config.conversation.tool_instruction)
            # Only a fixed system prefix and greeting are retained here.
            self.models.llm_warmup_input = (
                context.messages(tool_names=names) + [Message(role="user", content="Chào.")], tools or None,
            )
        # Danh sách tên thôi. Registry được dựng lại cho TỪNG phiên, vì công
        # cụ search gắn với hàng đợi tra cứu của chính phiên đó — dùng chung
        # một registry là nối chéo kết quả giữa các phiên.
        self.tool_names = list(config.tasks.tools) if config.tasks.enabled else []
        self.metrics = MetricsRegistry()
        self.sessions: dict[str, ConversationEngine] = {}
        # Giọng dùng cho mọi phiên mới. Đổi được lúc đang chạy mà KHÔNG nạp lại
        # model, vì cả ZeroTTS lẫn các talker mượn đều nhận `voice` theo từng
        # lần gọi chứ không khoá lúc dựng.
        self.voice: str | None = config.models.tts.options.get("voice")
        self._started = False

    def executor(self) -> TaskExecutor:
        return TaskExecutor(
            build_registry(self.tool_names),
            default_timeout_ms=self.config.tasks.default_timeout_ms,
            max_parallel=self.config.tasks.max_parallel,
            limiter=self.tool_limiter,
            emit=lambda *_: None,
        )

    async def start(self) -> None:
        if self._started:
            return
        if self.config.observability.write_traces:
            # Transcripts written by an older build kept the umask (0664).
            from ..observability.trace import secure_trace_dir
            try:
                tightened = secure_trace_dir(self.config.observability.trace_dir)
                if tightened:
                    log.info("trace files made owner-only: %d", tightened)
            except OSError as exc:
                log.warning("could not tighten trace permissions: %s", exc)
        if self.config.observability.trace_retention_days:
            from ..observability.trace import prune_session_traces
            removed = prune_session_traces(
                self.config.observability.trace_dir,
                self.config.observability.trace_retention_days,
            )
            log.info("pruned %d expired session traces", removed)
        await self.models.start()
        if self.config.models.prewarm_before_sessions:
            try:
                async with asyncio.timeout(self.config.models.startup_timeout_s):
                    messages, tools = self.models.llm_warmup_input
                    warmup = getattr(self.models.llm, "warmup", None)
                    if warmup is not None:
                        await warmup(messages, tools=tools)
                    await self.models.touch(min_interval_s=0, include_llm=False)
                    fixed = []
                    if self.config.conversation.opener.enabled:
                        fixed.append(self.config.conversation.opener.text)
                    if self.config.conversation.search.enabled:
                        fixed.append(self.config.conversation.search.instant_ack)
                    if self.config.conversation.filler.enabled:
                        fixed.extend(self.config.conversation.filler.phrases)
                    for text in dict.fromkeys(fixed):
                        if text:
                            await self.models.cached_speech(text, self.voice)
            except BaseException:
                await self.models.close()
                raise
        self._started = True
        log.info("model plane ready: %s", json.dumps(self.models.describe(), ensure_ascii=False))

    def invalidate_readiness(self) -> None:
        self._ready_at = float("-inf")

    async def readiness(self) -> dict[str, Any]:
        if self._started and not self.maintenance:
            async with self._ready_lock:
                if time.monotonic() - self._ready_at >= self.config.server.readiness_ttl_s:
                    self._dependencies = await self.models.check_dependencies(self.config.server.readiness_timeout_s)
                    self._ready_at = time.monotonic()
        ok = self._started and not self.maintenance and all(v["ok"] for v in self._dependencies.values())
        return {"ok": ok, "started": self._started, "maintenance": self.maintenance,
                "dependencies": self._dependencies, "runtime": self.runtime,
                "sessions": len(self.sessions), "max_sessions": self.config.server.max_sessions}

    def start_keep_warm(self) -> asyncio.Task | None:
        every = self.config.models.keep_warm_s
        if every <= 0:
            return None

        async def loop() -> None:
            while True:
                await asyncio.sleep(every)
                # Only while idle: a live session keeps the engines warm by
                # itself, and a touch would queue behind (or ahead of) its TTS.
                if not self.sessions and not self.model_lock.locked():
                    async with self.model_lock:
                        await self.models.touch(min_interval_s=every / 2)

        return asyncio.create_task(loop(), name="keep-warm")

    async def close(self) -> None:
        self._started = False
        results = await asyncio.gather(*(engine.close() for engine in list(self.sessions.values())), return_exceptions=True)
        for result in results:
            if isinstance(result, BaseException):
                log.error("session shutdown failed: %s", result)
        self.sessions.clear()
        self.session_tokens.clear()
        await self.models.close()

    def start_monitor(self):
        async def monitor():
            next_prune = time.monotonic() + 3600
            while True:
                expected = time.monotonic()+0.1
                await asyncio.sleep(0.1)
                self.loop_lag.append(max(0.0,(time.monotonic()-expected)*1000))
                if self.config.observability.trace_retention_days and time.monotonic() >= next_prune:
                    from ..observability.trace import prune_session_traces
                    try:
                        await asyncio.to_thread(
                            prune_session_traces,
                            self.config.observability.trace_dir,
                            self.config.observability.trace_retention_days,
                        )
                    except OSError as exc:
                        log.warning("could not prune session traces: %s", exc)
                    next_prune = time.monotonic() + 3600
        return asyncio.create_task(monitor(), name="event-loop-monitor")

    def gauges(self):
        def limiter(engine):
            value = getattr(engine, "limiter", None)
            return value.snapshot() if value else None
        process = {}
        try:
            status = Path("/proc/self/status").read_text()
            for label, name, scale in (("VmRSS", "process_rss_mb", 1024),
                                       ("VmSwap", "process_swap_mb", 1024),
                                       ("Threads", "process_threads", 1)):
                match = next((line.split()[1] for line in status.splitlines()
                              if line.startswith(label + ":")), None)
                if match is not None:
                    process[name] = round(int(match) / scale, 3)
        except OSError:
            pass
        return {"asr": limiter(self.models.asr), "llm": limiter(self.models.llm),
                "tts": limiter(self.models.tts), "search": limiter(getattr(self.models.search,"engine",None)),
                "tool": self.tool_limiter.snapshot(), "sessions": len(self.sessions),
                "tts_workers": len(getattr(self.models.tts,"_workers",{})),
                "asr_workers": len(getattr(self.models.asr,"_workers",{})),
                **process,
                "event_loop_lag_ms": {"n": len(self.loop_lag),
                    "p95": self.metrics._pct(list(self.loop_lag),.95), "max": round(max(self.loop_lag,default=0),3)}}

    def collect(self, engine: ConversationEngine) -> None:
        """Fold a session's turns into the process aggregate, once each.

        Called both from /metrics while the session is live and again when it
        closes, so every turn is offered several times; the registry keys on
        (session, turn) and ignores the repeats.
        """
        for row in engine.trace.settled_metrics_rows(engine.gen.turn_id):
            self.metrics.observe_turn(row, key=(engine.session_id, row["turn_id"]))
        for name, value in engine.counters.items():
            self.metrics.incr(name, value)


def create_app(config: Config) -> FastAPI:
    setup_logging()
    if must_be_private(config.server) and not config.server.private_introspection:
        # Reachable from other machines (a non-loopback bind, or TLS, which
        # exists for them): /config, /sessions and the engine switches must
        # not be theirs, --lan or not.
        config.server.private_introspection = True
        log.warning("server.host=%s%s: private_introspection forced on", config.server.host,
                    " with TLS" if config.server.ssl_certfile else "")
    platform = Platform(config)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        # Models load once per process and stay warm between connections; a
        # per-session load would put a cold start in front of every caller.
        warm = monitor = None
        try:
            await platform.start()
            warm = platform.start_keep_warm()
            monitor = platform.start_monitor()
            yield
        finally:
            background = [t for t in (warm, monitor) if t is not None]
            for task in background:
                task.cancel()
            await asyncio.gather(*background, return_exceptions=True)
            await platform.close()

    app = FastAPI(title="Realtime Voice Platform", version="0.1.0", lifespan=lifespan)
    app.state.platform = platform

    if config.server.cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=config.server.cors_origins,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    @app.get("/healthz")
    async def healthz() -> dict[str, Any]:
        return {
            "ok": True,
            "runtime": platform.runtime,
            "sessions": len(platform.sessions),
            "models": platform.models.describe(),
            "tools": platform.tool_names,
        }

    @app.get("/readyz")
    async def readyz() -> Any:
        status = await platform.readiness()
        return JSONResponse(status, status_code=200 if status["ok"] else 503)

    @app.get("/metrics")
    async def metrics() -> dict[str, Any]:
        for engine in list(platform.sessions.values()):
            for row in engine.trace.settled_metrics_rows(engine.gen.turn_id):
                platform.metrics.observe_turn(
                    row, key=(engine.session_id, row["turn_id"])
                )
        return {"snapshot": platform.metrics.snapshot(), "gauges": platform.gauges()}

    def _local_only(request: Request) -> JSONResponse | None:
        return local_only(request, config)

    @app.get("/sessions")
    async def sessions(request: Request) -> Any:
        # A session id is the key to that session's transcripts, and this is
        # the only place the ids are listed.
        return _local_only(request) or {
            sid: engine.stats() for sid, engine in platform.sessions.items()
        }

    @app.get("/sessions/{session_id}/turns")
    async def session_turns(session_id: str, request: Request, limit: int = 8) -> Any:
        """Raw per-turn timelines, the only view where stages overlap visibly.

        /sessions returns derived numbers, and a derived number cannot show
        that TTS started before the LLM finished — which is what most latency
        problems actually look like. The test bench draws one lane per engine
        from these stamps instead of guessing a sequence that does not exist.
        """
        if _local_only(request) is not None:
            supplied = request.headers.get("authorization", "").removeprefix("Bearer ")
            expected = platform.session_tokens.get(session_id)
            if expected is None or not hmac.compare_digest(supplied.encode(), expected.encode()):
                return JSONResponse({"detail": "session authorization required"}, status_code=403)
        engine = platform.sessions.get(session_id)
        if engine is None:
            return JSONResponse({"detail": "no such session"}, status_code=404)
        keep = max(1, min(limit, 50))
        turn_ids = sorted(engine.trace.turns)[-keep:]
        return {
            "session_id": session_id,
            "state": engine.state.state.value,
            "counters": dict(engine.counters),
            "stale_drops": engine.gen.stale_drops,
            "turns": [engine.trace.turns[i].summary() for i in turn_ids],
        }

    @app.get("/config")
    async def show_config(request: Request) -> Any:
        return _local_only(request) or config.to_dict()

    # Chọn model và thử riêng từng engine. Đặt sau /metrics và /sessions để
    # thứ tự khai báo route khớp với thứ tự đọc trong docstring ở đầu file.
    register_lab(app, platform, config)
    # Trang thu giọng người thật cho G3; tắt mặc định (collect.enabled).
    install_collect(app, config)

    web_dir = Path(config.server.web_dir)
    if web_dir.is_dir():
        app.mount("/static", StaticFiles(directory=str(web_dir)), name="static")

        @app.get("/")
        async def index() -> Any:
            page = web_dir / "index.html"
            if page.exists():
                return FileResponse(str(page))
            return JSONResponse({"detail": "no web client built"}, status_code=404)

    @app.websocket("/v1/realtime")
    async def realtime(ws: WebSocket) -> None:
        await ws.accept()
        status = await platform.readiness()
        async with platform.model_lock:
            if (not status["ok"] or not platform._started or platform.maintenance
                    or len(platform.sessions) >= config.server.max_sessions):
                await ws.send_json({"type": "error", "stage": "admission", "error": "unavailable_or_at_capacity"})
                await ws.close(code=1013, reason="unavailable_or_at_capacity")
                return
            session_id = new_session_id()
            transport = WebSocketTransport(ws, session_id)
            engine = ConversationEngine(
                config,
                platform.models,
                transport,
                session_id=session_id,
                executor=platform.executor(),
                search_agent=platform.models.search,
                voice=platform.voice,
            )
            platform.sessions[session_id] = engine
            token = secrets.token_urlsafe(32)
            platform.session_tokens[session_id] = token
            engine.session_token = token
        client_rate = config.audio.sample_rate
        server = config.server
        closing: str | None = None          # why the server ends it, if it does
        refusal: int | None = None          # close code for a refused frame
        hard = None
        bucket = _ControlBucket(_CONTROL_RATE, _CONTROL_BURST)
        limited = client_errors = 0

        async def close_with(reason: str, code: int) -> None:
            log.info("session %s closed by server: %s (%d)", session_id, reason, code)
            await transport.send_control(ControlMessage("error", {"stage": "session", "error": reason}))
            await transport.close(reason, code=code)

        try:
            await engine.start()
            opened = last_activity = time.monotonic()
            age_deadline = opened + server.max_session_s
            # Backstop only: the loop checks both deadlines itself, but not
            # while an await inside it (a text turn, a send) is stuck.
            async with asyncio.timeout(server.max_session_s + _HARD_GRACE_S) as hard:
                while True:
                    now = time.monotonic()
                    # Idle means nothing real happened, not "no frame arrived":
                    # a muted tab still streams zeros at 125 frames a second.
                    if engine.state.state not in _QUIET:
                        last_activity = now
                    if now >= age_deadline:
                        closing = "max_session_age"
                        break
                    if now - last_activity >= server.idle_timeout_s:
                        closing = "idle_timeout"
                        break
                    wait = min(age_deadline, last_activity + server.idle_timeout_s) - now
                    try:
                        message = await asyncio.wait_for(ws.receive(), timeout=max(wait, 0.01))
                    except TimeoutError:
                        continue
                    if message["type"] == "websocket.disconnect":
                        break
                    if (payload := message.get("bytes")) is not None:
                        if len(payload) > server.max_audio_bytes or len(payload) % 2:
                            closing, refusal = "invalid_audio_frame", 1009
                            break
                        pcm = np.frombuffer(payload, dtype="<i2").astype(np.float32) / 32768.0
                        if pcm.size and float(np.sqrt(np.mean(pcm * pcm))) >= config.media.vad.energy_threshold:
                            last_activity = time.monotonic()
                        try:
                            async with asyncio.timeout(config.models.operation_timeout_s) as budget:
                                await engine.push_audio(pcm, src_rate=client_rate)
                        except TimeoutError:
                            if not budget.expired():
                                raise
                            closing = "audio_timeout"
                            break
                    elif (text := message.get("text")) is not None:
                        if len(text) > server.max_text_chars + 1024:
                            closing, refusal = "message_too_big", 1009
                            break
                        try:
                            data = json.loads(text)
                        except (json.JSONDecodeError, RecursionError):
                            continue
                        if not isinstance(data, dict):
                            continue
                        kind = data.get("type")
                        # Telemetry arrives at whatever rate the client picks;
                        # stopping and leaving always get through.
                        if kind not in ("bye", "interrupt") and not bucket.take():
                            limited += 1
                            platform.metrics.incr("ws_control_rate_limited")
                            if limited == 1:
                                log.warning("session %s: over %d control messages/s, dropping",
                                            session_id, int(_CONTROL_RATE))
                            continue
                        if kind in ("text", "interrupt", "voice"):
                            last_activity = time.monotonic()
                        if kind == "clock_sync":
                            received_at = now_ms()
                            await transport.send_control(ControlMessage("clock_sync", {
                                "id": data.get("id"), "client_send_ms": data.get("client_send_ms"),
                                "server_receive_ms": received_at, "server_send_ms": now_ms(),
                            }))
                        elif kind == "clock_sync_result":
                            engine.set_playback_clock(data.get("offset_ms"), data.get("uncertainty_ms"))
                        elif kind == "playback":
                            engine.playback_feedback(data)
                        elif kind == "hello":
                            rate = data.get("sample_rate", client_rate)
                            if type(rate) is not int or not 8000 <= rate <= 192000:
                                closing, refusal = "invalid_sample_rate", 1008
                                break
                            client_rate = rate
                            engine.playback_feedback_enabled = data.get("playback_feedback") is True
                            log.info("session %s: client rate %s Hz", session_id, client_rate)
                        elif kind == "text":
                            user_text = data.get("text", "")
                            if not isinstance(user_text, str) or len(user_text) > server.max_text_chars:
                                closing, refusal = "message_too_big", 1009
                                break
                            await engine.push_text(user_text)
                        elif kind == "interrupt":
                            await engine.interrupt("client button")
                        elif kind == "voice":
                            # Giọng theo PHIÊN: người test này đổi giọng không được
                            # đổi luôn giọng của người đang nói ở máy bên cạnh.
                            try:
                                chosen = engine.set_voice(data.get("voice"))
                                await transport.send_control(
                                    ControlMessage("voice", {"voice": chosen, "ok": True})
                                )
                            except ValueError as exc:
                                await transport.send_control(
                                    ControlMessage(
                                        "voice", {"voice": engine.voice, "ok": False, "error": str(exc)}
                                    )
                                )
                        elif kind == "client_error":
                            # The browser's own failures (worklet, AudioContext):
                            # otherwise invisible from here. A broken player
                            # reports one per frame, hence the cap.
                            client_errors += 1
                            if client_errors <= _CLIENT_ERROR_LOGS:
                                log.warning("session %s: client error: %.300s", session_id, data.get("error"))
                        elif kind == "bye":
                            break
        except WebSocketDisconnect:
            pass
        except TimeoutError:
            if hard is not None and hard.expired():
                closing = "max_session_age"
            else:
                log.exception("session %s failed", session_id)
        except Exception:
            log.exception("session %s failed", session_id)
        finally:
            try:
                if closing is not None:
                    await close_with(closing, refusal or _CLOSE_CODES[closing])
            finally:
                try:
                    await engine.close()
                finally:
                    # Release the slot before anything else can fail: a leaked
                    # entry here refuses every later session until a restart.
                    platform.sessions.pop(session_id, None)
                    platform.session_tokens.pop(session_id, None)
                    if client_errors > _CLIENT_ERROR_LOGS:
                        log.warning("session %s: %d more client errors not logged",
                                    session_id, client_errors - _CLIENT_ERROR_LOGS)
                    try:
                        platform.collect(engine)
                    except Exception:
                        log.exception("session %s: could not fold its metrics", session_id)
                    await transport.close()

    return app


def _uvicorn_kwargs(config: Config) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "host": config.server.host,
        "port": config.server.port,
        "log_level": "info",
        "ws_max_size": max(config.server.max_audio_bytes, config.server.max_text_chars * 4 + 1024),
        # No ws_max_queue: uvicorn's default websockets-sansio protocol never
        # reads it (measured 02/10: ~10k frames queued while the app stalled
        # 1.5 s), and the legacy protocol that does is deprecated. Inbound
        # volume is bounded by ws_max_size, the per-session control-message
        # rate, and the idle/age limits instead (docs/OPERATIONS.md).
    }
    if config.server.ssl_certfile and config.server.ssl_keyfile:
        kwargs["ssl_certfile"] = config.server.ssl_certfile
        kwargs["ssl_keyfile"] = config.server.ssl_keyfile
    return kwargs


def run(config: Config) -> None:
    import uvicorn

    uvicorn.run(create_app(config), **_uvicorn_kwargs(config))


async def run_async(config: Config) -> None:  # pragma: no cover - used by scripts
    import uvicorn

    server = uvicorn.Server(uvicorn.Config(create_app(config), **_uvicorn_kwargs(config)))
    await server.serve()


if __name__ == "__main__":  # pragma: no cover
    asyncio.run(run_async(Config()))
