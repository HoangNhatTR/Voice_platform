"""Access and retention rules found in the 02/10 review.

`local_only` trusts the peer address, and a browser on the server host IS on
loopback: any page it opens could read /config and swap engines unless the
request's Origin is checked too.
"""

from __future__ import annotations

import os
import stat

from fastapi.testclient import TestClient

from voiceplatform.app.server import create_app
from voiceplatform.core.config import Config
from voiceplatform.core.events import Event, EventType
from voiceplatform.observability.trace import (
    CLIENT_EVENTS_PER_KEY, CLIENT_EVENTS_PER_TURN, SessionTrace, TurnTrace,
)

EVIL = {"Origin": "https://evil.example"}


def _loopback(config):
    return TestClient(create_app(config), client=("127.0.0.1", 40000))


def test_the_default_config_grants_no_cross_origin_reads(config):
    assert Config().server.cors_origins == []
    with _loopback(config) as client:
        response = client.get("/engines", headers=EVIL)
    assert "access-control-allow-origin" not in response.headers


def test_a_foreign_page_on_the_server_host_cannot_use_loopback_routes(config):
    config.server.private_introspection = True
    with _loopback(config) as client:
        platform = client.app.state.platform
        r = client.get("/config", headers=EVIL)
        assert r.status_code == 403 and "access-control-allow-origin" not in r.headers
        assert client.get("/sessions", headers=EVIL).status_code == 403
        # A "simple" request needs no preflight, so the Origin is the only gate.
        r = client.post("/engines/tts/voice", content='{"voice": "mock-b"}',
                        headers={**EVIL, "Content-Type": "text/plain"})
        assert r.status_code == 403 and platform.voice is None
        assert client.post("/try/asr", content=b"\x00\x00" * 1600, headers=EVIL).status_code == 403
        # Same-origin pages (the lab) and scripts without an Origin keep working.
        assert client.post("/engines/tts/voice", json={"voice": "mock-b"},
                           headers={"Origin": "http://testserver"}).status_code == 200
        assert client.post("/engines/tts/voice", json={"voice": "mock-a"}).status_code == 200
        assert platform.voice == "mock-a"
        # A JSON route only takes JSON: text/plain could only come from a form or no-cors fetch.
        assert client.post("/engines/tts/voice", content='{"voice": "mock-b"}',
                           headers={"Content-Type": "text/plain"}).status_code == 415
        assert client.get("/config").status_code == 200


def test_the_origin_rule_holds_with_introspection_off_and_with_a_wildcard(config):
    config.server.private_introspection = False
    config.server.cors_origins = ["*"]
    with _loopback(config) as client:
        r = client.get("/config", headers=EVIL)
        assert r.status_code == 403 and "models" not in r.text
        assert client.post("/try/tts", json={"text": "xin chào"}, headers=EVIL).status_code == 403
        listed = {"Origin": "https://bench.local"}
    config.server.cors_origins = ["https://bench.local"]
    with _loopback(config) as client:
        assert client.get("/config", headers=listed).status_code == 200


def test_binding_off_loopback_forces_private_introspection(config):
    config.server.private_introspection = False
    config.server.host = "0.0.0.0"
    with TestClient(create_app(config)) as client:          # peer "testclient" = off-machine
        assert client.get("/config").status_code == 403
        assert client.get("/sessions").status_code == 403
    assert config.server.private_introspection is True


def test_tls_on_loopback_also_forces_private_introspection(config):
    config.server.private_introspection = False
    config.server.ssl_certfile = "server.crt"
    config.server.ssl_keyfile = "server.key"
    with TestClient(create_app(config)) as client:
        assert client.get("/config").status_code == 403


def test_startup_tightens_traces_written_by_older_builds(config, tmp_path):
    traces = tmp_path / "traces"
    traces.mkdir()
    os.chmod(traces, 0o775)
    old = traces / "s0001-0315f079.jsonl"
    old.write_text("{}\n")
    os.chmod(old, 0o664)
    other = traces / "notes.jsonl"
    other.write_text("keep")
    os.chmod(other, 0o664)
    outside = tmp_path / "outside.jsonl"
    outside.write_text("not ours")
    os.chmod(outside, 0o644)
    (traces / "s0002-cafebabe.jsonl").symlink_to(outside)
    config.observability.trace_dir = str(traces)
    config.observability.write_traces = True
    with TestClient(create_app(config)):
        pass
    mode = lambda p: stat.S_IMODE(os.lstat(p).st_mode)
    assert mode(traces) == 0o700
    assert mode(old) == 0o600
    assert mode(other) == 0o664          # not a generated session name
    assert mode(outside) == 0o644        # symlink target untouched


def _feedback(turn, kind, ts, phrase="g1-p1"):
    turn.add(Event(type=kind, session_id="s", turn_id=1, generation_id=1, ts_ms=ts,
                   data={"phrase_id": phrase, "source": "browser_audio_render", "clock_uncertainty_ms": 1}))


def test_client_feedback_keeps_first_and_last_per_phrase():
    turn = TurnTrace(session_id="s", turn_id=1)
    turn.add(Event(type=EventType.PHRASE_READY, session_id="s", turn_id=1, generation_id=1, ts_ms=0,
                   data={"phrase_id": "g1-p1", "role": "content"}))
    for i in range(10000):
        _feedback(turn, EventType.PLAYBACK_BUFFER, 1000 + i)
    _feedback(turn, EventType.PLAYBACK_STARTED, 900)
    buffers = [e.ts_ms for e in turn.events if e.type is EventType.PLAYBACK_BUFFER]
    assert len(buffers) == CLIENT_EVENTS_PER_KEY
    assert buffers[0] == 1000 and buffers[-1] == 1000 + 9999     # first kept, latest kept
    assert turn.summary()["feedback_dropped"] == 10000 - CLIENT_EVENTS_PER_KEY
    assert any(e.type is EventType.PLAYBACK_STARTED for e in turn.events)   # other kinds unaffected
    # Many phrases cannot multiply the cap.
    for p in range(200):
        for i in range(CLIENT_EVENTS_PER_KEY):
            _feedback(turn, EventType.PLAYBACK_BUFFER, 20000 + i, phrase=f"g1-p{p + 2}")
    client = [e for e in turn.events if e.type is EventType.PLAYBACK_BUFFER]
    assert len(client) <= CLIENT_EVENTS_PER_TURN


def test_session_level_client_events_are_capped():
    trace = SessionTrace("s0001-deadbeef")
    for i in range(20000):
        trace.record(Event(type=EventType.CLIENT_CLOCK_SYNC, session_id="s", turn_id=None,
                           ts_ms=i, data={"uncertainty_ms": 1000 - i / 100}))
    assert len(trace.session_events) <= CLIENT_EVENTS_PER_TURN
    assert trace.session_events[-1].ts_ms == 19999
