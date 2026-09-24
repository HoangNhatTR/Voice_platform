"""Start the server, hold one WebSocket conversation, print what came back.

Deliberately not a unit test: it exercises the transport, the binary framing
and the audio header, which is where a protocol change breaks without any test
noticing.
"""

from __future__ import annotations

import asyncio
import json
import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import uvicorn

from voiceplatform.app.server import create_app
from voiceplatform.core.config import Config

HEADER = struct.Struct("<III")
HOST, PORT = "127.0.0.1", 18199


def pcm16(samples: np.ndarray) -> bytes:
    return (np.clip(samples, -1, 1) * 32767).astype("<i2").tobytes()


def speech(ms: int, rate: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = int(rate * ms / 1000)
    t = np.arange(n) / rate
    return (0.2 * (np.sin(2 * np.pi * 180 * t) + rng.normal(0, 0.3, n))).astype(np.float32)


async def main() -> int:
    import websockets

    config = Config.load("configs/dev-mock.yaml")
    config.server.host, config.server.port = HOST, PORT
    config.observability.write_traces = False
    server = uvicorn.Server(
        uvicorn.Config(create_app(config), host=HOST, port=PORT, log_level="warning")
    )
    task = asyncio.create_task(server.serve())
    for _ in range(100):
        await asyncio.sleep(0.05)
        if server.started:
            break

    rate = config.audio.sample_rate
    frame = config.frame_samples
    audio_frames = 0
    controls: list[str] = []
    generations: set[int] = set()
    try:
        async with websockets.connect(f"ws://{HOST}:{PORT}/v1/realtime") as ws:
            await ws.send(json.dumps({"type": "hello", "sample_rate": rate}))

            async def reader() -> None:
                nonlocal audio_frames
                async for message in ws:
                    if isinstance(message, bytes):
                        turn, generation, _seq = HEADER.unpack_from(message, 0)
                        generations.add(generation)
                        audio_frames += 1
                    else:
                        controls.append(json.loads(message)["type"])

            reading = asyncio.create_task(reader())
            # 600 ms of speech, then a pause long enough to end the turn.
            for i in range(30):
                await ws.send(pcm16(speech(20, rate, i)))
                await asyncio.sleep(0.02)
            for _ in range(60):
                await ws.send(pcm16(np.zeros(frame, dtype=np.float32)))
                await asyncio.sleep(0.02)
            await asyncio.sleep(2.0)
            reading.cancel()
    finally:
        server.should_exit = True
        await task

    summary = {
        "control_messages": sorted(set(controls)),
        "audio_frames": audio_frames,
        "generations": sorted(generations),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    ok = audio_frames > 0 and "transcript" in controls and "speaking" in controls
    print("SMOKE OK" if ok else "SMOKE FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
