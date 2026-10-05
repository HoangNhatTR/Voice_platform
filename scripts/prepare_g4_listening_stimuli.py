"""Freeze 12 synthetic spoken questions for paired G3/G4 listening clips.

The user voice is synthetic. This prepares a reproducible listening proxy and
does not satisfy G3's human microphone gate.
"""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import io
import json
import wave
from pathlib import Path

from conversation_check import _http


def prepare(cases: Path, output: Path, base: str, voice: str) -> None:
    with cases.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) < 12 or len({row["item_id"] for row in rows}) != len(rows):
        raise ValueError("at least 12 unique listening items are required")
    output.mkdir(parents=True, exist_ok=False)
    stimuli = {"direct": []}
    manifest = []
    for row in rows:
        reply = _http(base, "/try/tts", {"text": row["user_text"], "voice": voice})
        blob = base64.b64decode(reply["wav_base64"])
        with wave.open(io.BytesIO(blob)) as handle:
            if handle.getnchannels() != 1 or handle.getsampwidth() != 2:
                raise ValueError("expected mono 16-bit PCM WAV")
            sample_rate = handle.getframerate()
            pcm = handle.readframes(handle.getnframes())
        (output / f"input-{row['item_id']}.wav").write_bytes(blob)
        stimuli["direct"].append({
            "reference": row["user_text"], "sample_rate": sample_rate,
            "pcm_base64": base64.b64encode(pcm).decode(),
            "search": row["category"].startswith("search"),
        })
        manifest.append({**row, "voice": voice, "input_sha256": hashlib.sha256(blob).hexdigest()})
    (output / "stimuli.json").write_text(json.dumps(stimuli, ensure_ascii=False), encoding="utf-8")
    (output / "cases.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, default=Path("docs/audits/2026-09-30/g4/LISTENING_DIRECT_CASES.csv"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base", default="https://127.0.0.1:18100")
    parser.add_argument("--voice", default="quangminh")
    args = parser.parse_args()
    prepare(args.cases, args.output, args.base, args.voice)


if __name__ == "__main__":
    main()
