"""Opened to a LAN, one tester must not be able to change the server for all.

Starlette's TestClient connects from host "testclient", which is exactly what
an off-machine caller looks like to `local_only`.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from voiceplatform.app.server import create_app


@pytest.fixture
def lan_client(config):
    config.server.private_introspection = True
    config.server.web_dir = "does-not-exist"
    with TestClient(create_app(config)) as client:
        yield client


@pytest.mark.parametrize(
    ("path", "body"),
    [
        ("/engines/tts", {"backend": "mock"}),
        ("/engines/tts/voice", {"voice": "mock-b"}),
        ("/try/tts", {"text": "xin chào"}),
        ("/try/llm", {"prompt": "xin chào"}),
        ("/try/search", {"query": "xin chào"}),
    ],
)
def test_state_changing_and_cpu_heavy_routes_are_local_only(lan_client, path, body):
    response = lan_client.post(path, json=body)
    assert response.status_code == 403


def test_asr_trial_is_local_only(lan_client):
    assert lan_client.post("/try/asr", content=b"\x00\x00" * 1600).status_code == 403


def test_reading_the_engine_list_stays_open(lan_client):
    # The test page needs the voice list to draw its picker.
    response = lan_client.get("/engines")
    assert response.status_code == 200
    assert response.json()["kinds"]["tts"]["voices"]


def test_a_session_picks_its_own_voice_without_touching_the_others(lan_client):
    with lan_client.websocket_connect("/v1/realtime") as first, lan_client.websocket_connect(
        "/v1/realtime"
    ) as second:
        assert first.receive_json()["type"] == "ready"
        assert second.receive_json()["type"] == "ready"
        first.send_json({"type": "voice", "voice": "mock-b"})
        reply = _next_of(first, "voice")
        assert reply == {"type": "voice", "voice": "mock-b", "ok": True}
        sessions = lan_client.app.state.platform.sessions
        voices = sorted((engine.voice or "") for engine in sessions.values())
        assert voices == ["", "mock-b"]
        assert lan_client.app.state.platform.voice is None


def test_an_unknown_voice_is_refused_and_says_why(lan_client):
    with lan_client.websocket_connect("/v1/realtime") as ws:
        assert ws.receive_json()["type"] == "ready"
        ws.send_json({"type": "voice", "voice": "không-có"})
        reply = _next_of(ws, "voice")
        assert reply["ok"] is False
        assert "không có" in reply["error"]


def _next_of(ws, kind: str) -> dict:
    for _ in range(50):
        message = ws.receive_json()
        if message.get("type") == kind:
            return message
    raise AssertionError(f"no {kind!r} message")
