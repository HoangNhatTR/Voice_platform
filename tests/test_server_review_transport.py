"""Transport limits found in the 02/10 review, each reproduced end to end.

A browser tab streams PCM forever (zeros when muted), so "any message" is not
activity; a telemetry stream is client-controlled, so it needs a rate and a
cap; and every way the server ends a session has to say why.
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
import time

import numpy as np
import pytest
from fastapi.testclient import TestClient

from voiceplatform.app.server import _uvicorn_kwargs, create_app
from voiceplatform.conversation.engine import ConversationEngine
from voiceplatform.core.clock import now_ms

SILENCE = b"\x00\x00" * 320          # 20 ms at 16 kHz: what a muted tab sends


def _drain_until_close(ws, limit: int = 100000) -> tuple[dict, list[dict]]:
    controls: list[dict] = []
    for _ in range(limit):
        message = ws.receive()
        if message["type"] == "websocket.close":
            return message, controls
        if message.get("text") is not None:
            controls.append(json.loads(message["text"]))
    raise AssertionError("session never closed")


class _Streamer:
    """Send one frame every 20 ms from another thread, like a capture worklet."""

    def __init__(self, ws, frame: bytes, seconds: float) -> None:
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, args=(ws, frame, seconds), daemon=True)

    def _run(self, ws, frame, seconds):
        until = time.monotonic() + seconds
        while not self._stop.is_set() and time.monotonic() < until:
            try:
                ws.send_bytes(frame)
            except Exception:
                return
            self._stop.wait(0.02)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join(2)


def test_a_muted_tab_streaming_zeros_is_closed_for_idleness(config):
    config.server.idle_timeout_s = 0.5
    with TestClient(create_app(config)) as client:
        with client.websocket_connect("/v1/realtime") as ws:
            assert ws.receive_json()["type"] == "ready"
            started = time.monotonic()
            with _Streamer(ws, SILENCE, seconds=4.0):
                closed, controls = _drain_until_close(ws)
            elapsed = time.monotonic() - started
    assert closed["code"] == 4000 and closed["reason"] == "idle_timeout"
    assert {"type": "error", "stage": "session", "error": "idle_timeout"} in controls
    assert elapsed < 2.5, f"silent PCM kept the session open for {elapsed:.1f}s"


def test_real_activity_keeps_the_session_open(config):
    config.server.idle_timeout_s = 0.6
    loud = (np.full(320, 0.2 * 32767)).astype("<i2").tobytes()
    with TestClient(create_app(config)) as client:
        platform = client.app.state.platform
        with client.websocket_connect("/v1/realtime") as ws:
            assert ws.receive_json()["type"] == "ready"
            with _Streamer(ws, loud, seconds=1.5):
                time.sleep(1.5)
            assert len(platform.sessions) == 1        # 2.5x the idle limit, still open
            for _ in range(4):                        # typed turns count too
                ws.send_json({"type": "text", "text": "xin chào"})
                time.sleep(0.25)
            assert len(platform.sessions) == 1
            closed, _ = _drain_until_close(ws)
    assert closed["code"] == 4000


def test_max_session_age_says_why(config):
    config.server.max_session_s = 0.5
    with TestClient(create_app(config)) as client:
        with client.websocket_connect("/v1/realtime") as ws:
            assert ws.receive_json()["type"] == "ready"
            closed, controls = _drain_until_close(ws)
    assert closed["code"] == 4001 and closed["reason"] == "max_session_age"
    assert {"type": "error", "stage": "session", "error": "max_session_age"} in controls


def test_audio_budget_overrun_says_why(config, monkeypatch):
    config.models.operation_timeout_s = 0.3

    async def stuck(self, samples, src_rate=None):
        await asyncio.sleep(5)

    monkeypatch.setattr(ConversationEngine, "push_audio", stuck)
    with TestClient(create_app(config)) as client:
        platform = client.app.state.platform
        with client.websocket_connect("/v1/realtime") as ws:
            assert ws.receive_json()["type"] == "ready"
            ws.send_bytes(SILENCE)
            closed, controls = _drain_until_close(ws)
        time.sleep(0.1)
        assert not platform.sessions
    assert closed["code"] == 1011 and closed["reason"] == "audio_timeout"
    assert {"type": "error", "stage": "session", "error": "audio_timeout"} in controls


def test_a_feedback_flood_is_rate_limited_and_cannot_grow_the_trace(config):
    with TestClient(create_app(config)) as client:
        platform = client.app.state.platform
        with client.websocket_connect("/v1/realtime") as ws:
            assert ws.receive_json()["type"] == "ready"
            ws.send_json({"type": "hello", "sample_rate": 16000, "playback_feedback": True})
            ws.send_json({"type": "clock_sync_result", "offset_ms": 0, "uncertainty_ms": 1})
            ws.send_json({"type": "text", "text": "xin chào"})
            while True:
                message = ws.receive()
                if message.get("text") and json.loads(message["text"])["type"] == "audio_segment":
                    segment = json.loads(message["text"])
                    break
            engine = next(iter(platform.sessions.values()))
            turn = engine.trace.turns[segment["turn_id"]]
            for _ in range(20000):
                ws.send_text(json.dumps({
                    "type": "playback", "event": "playback_buffer", "phrase_id": segment["phrase_id"],
                    "generation_id": segment["generation_id"], "client_ms": now_ms(), "buffer_ms": 1,
                }))
            ws.send_json({"type": "bye"})            # still honoured after the flood
            _drain_until_close(ws)
        assert len(turn.events) < 600, len(turn.events)
        assert platform.metrics.counters["ws_control_rate_limited"] > 15000


def test_client_errors_reach_the_log_but_cannot_flood_it(config):
    records: list[logging.LogRecord] = []

    class Keep(logging.Handler):
        def emit(self, record):
            records.append(record)

    handler = Keep()
    logger = logging.getLogger("voiceplatform.server")
    logger.addHandler(handler)
    try:
        with TestClient(create_app(config)) as client:
            with client.websocket_connect("/v1/realtime") as ws:
                assert ws.receive_json()["type"] == "ready"
                for i in range(40):
                    ws.send_json({"type": "client_error", "error": f"AudioWorklet failed {i}"})
                ws.send_json({"type": "bye"})
                _drain_until_close(ws)
    finally:
        logger.removeHandler(handler)
    reported = [r.getMessage() for r in records if "client error" in r.getMessage()]
    assert reported and "AudioWorklet failed 0" in reported[0]
    assert len(reported) <= 6, reported


def test_a_failure_while_folding_metrics_does_not_leak_the_admission_slot(config):
    config.server.max_sessions = 1
    app = create_app(config)

    def broken(engine):
        raise RuntimeError("metrics derivation failed")

    app.state.platform.collect = broken
    with TestClient(app) as client:
        with client.websocket_connect("/v1/realtime") as ws:
            assert ws.receive_json()["type"] == "ready"
            ws.send_json({"type": "bye"})
            _drain_until_close(ws)
        time.sleep(0.1)
        assert not app.state.platform.sessions and not app.state.platform.session_tokens
        with client.websocket_connect("/v1/realtime") as ws:
            assert ws.receive_json()["type"] == "ready"


def test_no_uvicorn_option_pretends_to_bound_the_inbound_queue(config):
    # uvicorn 0.51 picks websockets-sansio, which never reads ws_max_queue
    # (measured: ~10k frames queued during a 1.5 s stall). Passing it only
    # made the documented "8 message" limit look real.
    assert "ws_max_queue" not in _uvicorn_kwargs(config)
