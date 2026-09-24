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
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from ..conversation.engine import ConversationEngine
from ..core.config import Config
from ..core.ids import new_session_id
from ..models.registry import ModelPlane
from ..observability.logging import get_logger, setup_logging
from ..observability.metrics import MetricsRegistry
from ..tasks.executor import TaskExecutor
from ..tasks.registry import build_registry
from .lab import register as register_lab
from .transport_ws import WebSocketTransport

log = get_logger("server")


class Platform:
    """Process-wide state: models, tools, metrics, live sessions."""

    def __init__(self, config: Config) -> None:
        self.config = config
        self.models = ModelPlane(config.models, output_sample_rate=config.audio.output_sample_rate)
        # Danh sách tên thôi. Registry được dựng lại cho TỪNG phiên, vì công
        # cụ search gắn với hàng đợi tra cứu của chính phiên đó — dùng chung
        # một registry là nối chéo kết quả giữa các phiên.
        self.tool_names = list(config.tasks.tools) if config.tasks.enabled else []
        self.metrics = MetricsRegistry()
        self.sessions: dict[str, ConversationEngine] = {}
        self._started = False

    def executor(self) -> TaskExecutor:
        return TaskExecutor(
            build_registry(self.tool_names),
            default_timeout_ms=self.config.tasks.default_timeout_ms,
            max_parallel=self.config.tasks.max_parallel,
            emit=lambda *_: None,
        )

    async def start(self) -> None:
        if self._started:
            return
        await self.models.start()
        self._started = True
        log.info("model plane ready: %s", json.dumps(self.models.describe(), ensure_ascii=False))

    async def close(self) -> None:
        for engine in list(self.sessions.values()):
            await engine.close()
        self.sessions.clear()
        await self.models.close()

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
    platform = Platform(config)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        # Models load once per process and stay warm between connections; a
        # per-session load would put a cold start in front of every caller.
        await platform.start()
        yield
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
            "sessions": len(platform.sessions),
            "models": platform.models.describe(),
            "tools": platform.tool_names,
        }

    @app.get("/metrics")
    async def metrics() -> dict[str, Any]:
        for engine in list(platform.sessions.values()):
            for row in engine.trace.settled_metrics_rows(engine.gen.turn_id):
                platform.metrics.observe_turn(
                    row, key=(engine.session_id, row["turn_id"])
                )
        return {"snapshot": platform.metrics.snapshot()}

    def _local_only(request: Request) -> JSONResponse | None:
        """Refuse an introspection endpoint to anyone off this machine.

        Only meaningful because nothing sits in front of this server: behind a
        reverse proxy every request would look local and this would wave the
        whole LAN through.
        """
        if not config.server.private_introspection:
            return None
        host = request.client.host if request.client else ""
        if host in {"127.0.0.1", "::1", "localhost"}:
            return None
        return JSONResponse(
            {"detail": "chỉ xem được từ chính máy chạy server"}, status_code=403
        )

    @app.get("/sessions")
    async def sessions(request: Request) -> Any:
        # A session id is the key to that session's transcripts, and this is
        # the only place the ids are listed.
        return _local_only(request) or {
            sid: engine.stats() for sid, engine in platform.sessions.items()
        }

    @app.get("/sessions/{session_id}/turns")
    async def session_turns(session_id: str, limit: int = 8) -> Any:
        """Raw per-turn timelines, the only view where stages overlap visibly.

        /sessions returns derived numbers, and a derived number cannot show
        that TTS started before the LLM finished — which is what most latency
        problems actually look like. The test bench draws one lane per engine
        from these stamps instead of guessing a sequence that does not exist.
        """
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
        session_id = new_session_id()
        transport = WebSocketTransport(ws, session_id)
        engine = ConversationEngine(
            config,
            platform.models,
            transport,
            session_id=session_id,
            executor=platform.executor(),
            search_agent=platform.models.search,
        )
        platform.sessions[session_id] = engine
        client_rate = config.audio.sample_rate
        try:
            await engine.start()
            while True:
                message = await ws.receive()
                if message["type"] == "websocket.disconnect":
                    break
                if (payload := message.get("bytes")) is not None:
                    pcm = np.frombuffer(payload, dtype="<i2").astype(np.float32) / 32768.0
                    await engine.push_audio(pcm, src_rate=client_rate)
                elif (text := message.get("text")) is not None:
                    try:
                        data = json.loads(text)
                    except json.JSONDecodeError:
                        continue
                    kind = data.get("type")
                    if kind == "hello":
                        client_rate = int(data.get("sample_rate", client_rate))
                        log.info("session %s: client rate %s Hz", session_id, client_rate)
                    elif kind == "text":
                        await engine.push_text(str(data.get("text", "")))
                    elif kind == "interrupt":
                        await engine.interrupt("client button")
                    elif kind == "bye":
                        break
        except WebSocketDisconnect:
            pass
        except Exception:
            log.exception("session %s failed", session_id)
        finally:
            await engine.close()
            platform.collect(engine)
            platform.sessions.pop(session_id, None)
            await transport.close()

    return app


def _uvicorn_kwargs(config: Config) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "host": config.server.host,
        "port": config.server.port,
        "log_level": "info",
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
