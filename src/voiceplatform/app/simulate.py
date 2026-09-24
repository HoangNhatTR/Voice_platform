"""Drive the engine from synthetic audio, with no browser and no models.

This is how turn-taking gets tested and demonstrated: speech and silence are
generated as plain arrays and pushed through the same entry point the transport
uses, so every stage — gate, endpointing, barge-in, fencing, trace — runs
exactly as it does live.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

import numpy as np

from ..conversation.engine import ConversationEngine
from ..conversation.sink import CollectingSink
from ..core.config import Config
from ..models.registry import ModelPlane
from ..tasks.executor import TaskExecutor
from ..tasks.registry import build_registry


def speech(duration_ms: float, sample_rate: int, level: float = 0.2, seed: int = 0) -> np.ndarray:
    """Loud, noisy audio: any VAD calls this speech."""
    rng = np.random.default_rng(seed)
    n = int(sample_rate * duration_ms / 1000)
    t = np.arange(n) / sample_rate
    carrier = np.sin(2 * np.pi * 180 * t) + 0.5 * np.sin(2 * np.pi * 320 * t)
    noise = rng.normal(0, 0.3, n)
    return (level * (carrier + noise)).astype(np.float32)


def silence(duration_ms: float, sample_rate: int, level: float = 0.0005) -> np.ndarray:
    n = int(sample_rate * duration_ms / 1000)
    rng = np.random.default_rng(1)
    return (level * rng.normal(0, 1, n)).astype(np.float32)


@dataclass(slots=True)
class Step:
    kind: str       # "speech" | "silence"
    duration_ms: float


async def feed(engine: ConversationEngine, steps: list[Step], *, realtime: bool = False) -> None:
    rate = engine.config.audio.sample_rate
    frame_ms = engine.config.audio.frame_ms
    seed = 0
    for step in steps:
        remaining = step.duration_ms
        while remaining > 0:
            chunk_ms = min(frame_ms, remaining)
            remaining -= chunk_ms
            seed += 1
            block = (
                speech(chunk_ms, rate, seed=seed)
                if step.kind == "speech"
                else silence(chunk_ms, rate)
            )
            await engine.push_audio(block, src_rate=rate)
            # Let the engine's own tasks run between frames.
            await asyncio.sleep(frame_ms / 1000.0 if realtime else 0)


def build_engine(
    config: Config | None = None,
    *,
    sink: CollectingSink | None = None,
    tools: list[str] | None = None,
) -> tuple[ConversationEngine, CollectingSink]:
    config = config or Config()
    sink = sink or CollectingSink()
    models = ModelPlane(config.models, output_sample_rate=config.audio.output_sample_rate)
    executor = TaskExecutor(build_registry(tools if tools is not None else config.tasks.tools))
    engine = ConversationEngine(
        config, models, sink, executor=executor, search_agent=models.search
    )
    return engine, sink


async def wait_until(
    engine: ConversationEngine,
    predicate,
    *,
    max_ms: float = 4000,
    feed_silence: bool = True,
) -> bool:
    """Wait on engine state while the microphone keeps streaming.

    A real session never stops sending audio just because the assistant is
    talking, and several bugs only appear in those frames (the barge-in guard,
    the echo path, the gate's counters), so the simulator does not stop either.
    """
    rate = engine.config.audio.sample_rate
    frame_ms = engine.config.audio.frame_ms
    waited = 0.0
    while waited < max_ms:
        if predicate(engine):
            return True
        if feed_silence:
            await engine.push_audio(silence(frame_ms, rate), src_rate=rate)
        await asyncio.sleep(frame_ms / 1000.0)
        waited += frame_ms
    return predicate(engine)


def is_idle(engine: ConversationEngine) -> bool:
    return engine.state.state.value == "idle"


def is_speaking(engine: ConversationEngine) -> bool:
    return engine.state.state.value == "speaking"


async def demo(config: Config | None = None, realtime: bool = True) -> dict:
    """One scripted session: a clean turn, then a turn interrupted mid-answer."""
    config = config or Config()
    config.observability.write_traces = False
    engine, sink = build_engine(config)
    await engine.models.start()
    await engine.start()

    # Turn 1: speak, pause long enough to end the turn, let the answer finish.
    await feed(
        engine,
        [Step("silence", 200), Step("speech", 900), Step("silence", 800)],
        realtime=realtime,
    )
    await wait_until(engine, is_idle, max_ms=6000)

    # Turn 2: speak, then cut the assistant off once it is actually speaking.
    await feed(engine, [Step("speech", 700), Step("silence", 800)], realtime=realtime)
    reached = await wait_until(engine, is_speaking, max_ms=4000)
    if reached:
        await feed(engine, [Step("speech", 400)], realtime=realtime)
    await wait_until(engine, is_idle, max_ms=2000, feed_silence=True)

    stats = engine.stats()
    stats["control_messages"] = _summarise(sink.control_types())
    stats["barge_in_reached_speaking"] = reached
    stats["audio_frames_out"] = len(sink.audio)
    await engine.close()
    await engine.models.close()
    return stats


def _summarise(types: list[str]) -> dict[str, int]:
    out: dict[str, int] = {}
    for name in types:
        out[name] = out.get(name, 0) + 1
    return out
