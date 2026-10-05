"""A/B energy VAD vs Silero for the barge-in decision, offline, on the G3 stimuli.

Runs the engine's own BargeInDetector frame by frame over each clip — no
server, no model plane — under a few synthetic room conditions:

  clean      the clip alone
  hiss20     + white noise at 20 dB below the clip's speech level
  hiss10     + white noise at 10 dB below
  babble20   + other voices mixed 20 dB below (a TV, a room)
  echo34     + a continuous other-voice track at -34 dBFS speech level (0.02
             RMS, the barge-in floor) — residual loudspeaker echo after AEC
  echo40     the same at -40 dBFS (0.01 RMS)

It answers two questions per condition: how often a NON-request (noise,
backchannel) fires a barge-in, and how fast a real interruption does. It is
not a substitute for microphones, rooms and real AEC — the conditions are
proxies and say so in the output.

  PYTHONPATH=src python scripts/vad_ab_g3.py \
      --stimuli docs/audits/2026-09-29/g3/stimuli.json --output /tmp/vad-ab.json
"""

from __future__ import annotations

import argparse
import base64
import json
import math
from pathlib import Path

import numpy as np

from voiceplatform.conversation.barge_in import BargeInDetector
from voiceplatform.core.audio import AudioFrame
from voiceplatform.core.config import BargeInConfig
from voiceplatform.media.vad.energy import EnergyVad
from voiceplatform.media.vad.silero import SileroVad

RATE = 16000
FRAME = 320  # 20 ms


def decode(b64: str) -> np.ndarray:
    return np.frombuffer(base64.b64decode(b64), "<i2").astype(np.float32) / 32768.0


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x.astype(np.float64) ** 2))) if x.size else 0.0


def speech_level(x: np.ndarray) -> float:
    frames = x[: x.size // FRAME * FRAME].reshape(-1, FRAME)
    levels = np.sqrt((frames ** 2).mean(1))
    loud = levels[levels >= 0.01]
    return float(np.median(loud)) if loud.size else rms(x)


def wilson(k: int, n: int) -> list[float] | None:
    if not n:
        return None
    z = 1.96
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [round(max(0, c - h), 4), round(min(1, c + h), 4)]


def simulate(clip: np.ndarray, vad_kind: str, condition: str, bed: np.ndarray, rng, frames: int = 6) -> dict:
    lead = int(0.6 * RATE)
    tail = int(0.4 * RATE)
    signal = np.concatenate([np.zeros(lead, np.float32), clip, np.zeros(tail, np.float32)])
    level = speech_level(clip)
    if condition.startswith("hiss"):
        snr = float(condition[4:])
        signal = signal + rng.normal(0, level * 10 ** (-snr / 20), signal.size).astype(np.float32)
    elif condition.startswith("babble"):
        snr = float(condition[6:])
        other = np.resize(bed, signal.size)
        signal = signal + other * (level / (speech_level(other) + 1e-9)) * 10 ** (-snr / 20)
    elif condition.startswith("echo"):
        db = float(condition[4:])
        other = np.resize(bed[::-1], signal.size)
        signal = signal + other / (speech_level(other) + 1e-9) * 10 ** (-db / 20)
        # (dB of full scale for the echo's speech level, not relative to the user)
    vad = EnergyVad(threshold=0.012) if vad_kind == "energy" else SileroVad(threshold=0.5)
    detector = BargeInDetector(BargeInConfig(speech_frames=frames, guard_ms=0, min_rms=0.02))
    detector.arm(0.0)
    onset = None
    fired_at = None
    silero_ms = 0.0     # what a Silero verifier would call speech inside the clip
    verifier = SileroVad(threshold=0.5) if vad_kind == "energy" else None
    for index in range(signal.size // FRAME):
        chunk = signal[index * FRAME:(index + 1) * FRAME]
        t = index * 20.0
        clean_part = index * FRAME - lead
        if onset is None and 0 <= clean_part < clip.size and rms(clip[clean_part:clean_part + FRAME]) >= 0.01:
            onset = t
        frame = AudioFrame(samples=chunk.astype(np.float32), sample_rate=RATE)
        probability = vad.probability(frame)
        speechy = (verifier.probability(frame) if verifier else probability) >= 0.5
        if speechy and 0 <= clean_part < clip.size + int(0.2 * RATE):
            silero_ms += 20.0
        if detector.update(probability, frame, t) and fired_at is None:
            fired_at = t
    return {"fired": fired_at is not None, "silero_speech_ms": silero_ms,
            "latency_ms": None if fired_at is None or onset is None else fired_at - onset}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--stimuli", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--conditions", default="clean,hiss20,hiss10,babble20,echo34,echo40")
    parser.add_argument("--variants", default="energy:6,silero:6,silero:4,silero:3")
    args = parser.parse_args()
    stimuli = json.loads(Path(args.stimuli).read_text())
    bed = np.concatenate([decode(c["pcm"]) for c in stimuli["cases"]["complete"][:40]])
    rng = np.random.default_rng(7)
    results = {"basis": "offline BargeInDetector over G3 stimuli; synthetic noise/babble/echo beds; "
               "variants energy|silero:speech_frames, min_rms=0.02, no guard", "stimuli_sha256": stimuli.get("sha256"), "rows": {}}
    for condition in args.conditions.split(","):
        for variant in args.variants.split(","):
            vad_kind, frames = variant.split(":")
            for family in ("noise", "backchannel", "interrupt"):
                rows = [simulate(decode(c["pcm"]), vad_kind, condition, bed, rng, int(frames)) for c in stimuli["cases"][family]]
                fired = sum(r["fired"] for r in rows)
                lat = sorted(r["latency_ms"] for r in rows if r["latency_ms"] is not None)
                key = f"{condition}/{variant}/{family}"
                results["rows"][key] = {
                    "fired": fired, "n": len(rows), "rate": round(fired / len(rows), 4),
                    "wilson95": wilson(fired, len(rows)),
                    "latency_ms": {"p50": lat[len(lat) // 2], "p95": lat[min(len(lat) - 1, math.ceil(0.95 * len(lat)) - 1)],
                                   "max": lat[-1]} if lat else None,
                    # Hybrid: energy fires the stop, Silero then decides "speech or not".
                    "silero_speech_ms": sorted(r["silero_speech_ms"] for r in rows),
                }
                sil = results["rows"][key]["silero_speech_ms"]
                print(f"{key:34s} fired {fired:3d}/{len(rows)}  latency {results['rows'][key]['latency_ms']}"
                      f"  silero_ms min {sil[0]:.0f} p10 {sil[len(sil)//10]:.0f} max {sil[-1]:.0f}", flush=True)
    Path(args.output).write_text(json.dumps(results, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
