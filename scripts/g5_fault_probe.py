"""Live WebSocket fault smoke on an isolated candidate server.

Checks admission, per-session voice changes, explicit interruption fencing,
disconnect cleanup and reconnect. Dependency failures remain a separate
mock/fault-injection test; this probe never stops the shared LLM service.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import ssl
import struct
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import websockets


def http(base: str, path: str) -> dict:
    context = ssl.create_default_context()
    if base.startswith("https://127.0.0.1:") or base.startswith("https://localhost:"):
        context.check_hostname = False
        context.verify_mode = ssl.CERT_NONE
    with urllib.request.urlopen(base + path, context=context, timeout=10) as response:
        return json.load(response)


async def receive(ws, wanted: str, timeout: float = 20) -> tuple[dict, list[dict]]:
    seen: list[dict] = []
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        raw = await asyncio.wait_for(ws.recv(), max(0.1, deadline - time.monotonic()))
        if isinstance(raw, bytes):
            if len(raw) >= 12:
                seen.append({"type": "audio", "generation_id": struct.unpack_from("<I", raw, 4)[0]})
            continue
        message = json.loads(raw)
        seen.append(message)
        if message.get("type") == wanted:
            return message, seen
    raise TimeoutError(f"waiting for {wanted}; recent={seen[-5:]}")


async def released(base: str, seconds: float = 15) -> bool:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if not await asyncio.to_thread(http, base, "/sessions"):
            return True
        await asyncio.sleep(0.2)
    return False


async def probe(base: str) -> dict:
    ready = await asyncio.to_thread(http, base, "/readyz")
    if not ready.get("ok") or await asyncio.to_thread(http, base, "/sessions"):
        raise RuntimeError("probe needs a ready isolated server with no sessions")
    url = base.replace("https://", "wss://").replace("http://", "ws://") + "/v1/realtime"
    context = None
    if url.startswith("wss://127.0.0.1:") or url.startswith("wss://localhost:"):
        context = ssl._create_unverified_context()
    result = {"schema": 1, "started_at": datetime.now(timezone.utc).isoformat(),
              "source_sha256": ready.get("runtime", {}).get("source_sha256"),
              "config_sha256": ready.get("runtime", {}).get("config_sha256"),
              "checks": {}, "observed": {}}
    sockets = []
    try:
        for _ in range(3):
            ws = await websockets.connect(url, ssl=context, max_size=None)
            sockets.append(ws)
            message, _ = await receive(ws, "ready")
            if not message.get("session_id"):
                raise RuntimeError("missing ready session id")
        fourth = await websockets.connect(url, ssl=context, max_size=None)
        try:
            refusal, _ = await receive(fourth, "error", timeout=5)
            result["checks"]["fourth_session_refused"] = refusal.get("stage") == "admission"
            result["observed"]["admission"] = refusal
        finally:
            await fourth.close()

        first = sockets[0]
        await first.send(json.dumps({"type": "voice", "voice": "quangminh"}))
        voice, _ = await receive(first, "voice")
        result["checks"]["session_voice_changed"] = voice.get("ok") is True and voice.get("voice") == "quangminh"
        result["observed"]["voice"] = voice
        await first.send(json.dumps({"type": "text", "text": "Giải thích vì sao bầu trời màu xanh bằng vài câu ngắn."}))
        speaking, seen = await receive(first, "speaking", timeout=40)
        generation = speaking.get("generation_id")
        result["observed"]["speaking_generation"] = generation
        await first.send(json.dumps({"type": "interrupt"}))
        reset, _ = await receive(first, "playback_reset", timeout=15)
        result["observed"]["reset_generation"] = reset.get("generation_id")
        old = reset.get("generation_id", generation)
        stale = []
        until = time.monotonic() + 1.0
        while time.monotonic() < until:
            try:
                raw = await asyncio.wait_for(first.recv(), until - time.monotonic())
            except asyncio.TimeoutError:
                break
            if isinstance(raw, bytes) and len(raw) >= 12 and struct.unpack_from("<I", raw, 4)[0] == old:
                stale.append(len(raw))
        result["checks"]["explicit_interrupt_reset"] = reset.get("generation_id") == generation
        result["checks"]["no_stale_audio_after_reset"] = not stale
        # A disconnected browser must release its generation and admission
        # slot even when a response was in flight.
        for ws in sockets:
            await ws.close()
        sockets.clear()
        result["checks"]["disconnect_released"] = await released(base)
        reconnect = await websockets.connect(url, ssl=context, max_size=None)
        sockets.append(reconnect)
        reopened, _ = await receive(reconnect, "ready")
        result["checks"]["reconnect_admitted"] = bool(reopened.get("session_id"))
    except Exception as exc:
        result["observed"]["error"] = str(exc)
    finally:
        for ws in sockets:
            await ws.close()
        result["checks"]["final_sessions_released"] = await released(base)
        metrics = await asyncio.to_thread(http, base, "/metrics")
        gauges = metrics.get("gauges", {})
        result["checks"]["final_queues_empty"] = all(
            gauge is None or (gauge.get("active", 0) == 0 and gauge.get("waiting", 0) == 0)
            for gauge in (gauges.get(name) for name in ("asr", "llm", "tts", "search", "tool")))
        result["finished_at"] = datetime.now(timezone.utc).isoformat()
        result["passed"] = len(result["checks"]) >= 8 and all(result["checks"].values())
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = asyncio.run(probe(args.base))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
