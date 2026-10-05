"""Convert human-recorded, human-labeled WAVs to the G3 benchmark schema.

CSV columns: id,family,split,speaker_id,session_id,labeler_id,human_verified,
wav,text,part1,part2,pause_at_ms,pause_ms,keys. Optional acoustic columns:
offset_s,kind,onset_window. Family is hold, continue, complete, backchannel,
noise or interrupt. Only heldout clips enter the benchmark.
All WAVs must be mono 16-bit PCM at 16 kHz. No audio leaves this machine.
"""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import json
import wave
from pathlib import Path

FAMILIES = ("hold", "continue", "complete", "backchannel", "noise", "interrupt")
FIELDS = ("id", "family", "split", "speaker_id", "session_id", "labeler_id",
          "human_verified", "wav", "text", "part1", "part2", "pause_at_ms", "pause_ms", "keys")


def prepare(labels: Path, output: Path, assistant_stimuli: Path | None = None) -> dict:
    with labels.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not set(FIELDS).issubset(reader.fieldnames or []):
            raise ValueError("missing CSV columns: " + ", ".join(sorted(set(FIELDS) - set(reader.fieldnames or []))))
        rows = list(reader)
    if not rows:
        raise ValueError("no labeled rows")
    ids: set[str] = set()
    speakers: dict[str, str] = {}
    sessions: dict[str, str] = {}
    cases: dict[str, list[dict]] = {family: [] for family in FAMILIES}
    for row in rows:
        item_id = row["id"].strip()
        family = row["family"].strip()
        split = row["split"].strip()
        if not item_id or item_id in ids or family not in FAMILIES or split not in ("train", "heldout"):
            raise ValueError(f"invalid or duplicate case id/family/split: {item_id}")
        ids.add(item_id)
        if row["human_verified"].strip().lower() != "yes":
            raise ValueError(f"case {item_id} lacks human_verified=yes")
        for field, groups in (("speaker_id", speakers), ("session_id", sessions)):
            value = row[field].strip()
            if not value:
                raise ValueError(f"case {item_id} lacks {field}")
            if value in groups and groups[value] != split:
                raise ValueError(f"{field} {value} leaks across train and heldout")
            groups[value] = split
        if not row["labeler_id"].strip():
            raise ValueError(f"case {item_id} lacks labeler_id")
        source = labels.parent / row["wav"]
        with wave.open(str(source), "rb") as handle:
            if (handle.getnchannels(), handle.getsampwidth(), handle.getframerate()) != (1, 2, 16000):
                raise ValueError(f"case {item_id} must be mono 16-bit 16 kHz WAV")
            pcm = handle.readframes(handle.getnframes())
        duration_ms = len(pcm) / 32
        if duration_ms < 250:
            raise ValueError(f"case {item_id} is too short")
        case = {"id": item_id, "voice": row["speaker_id"].strip(),
                "session_id": row["session_id"].strip(), "labeler_id": row["labeler_id"].strip(),
                "text": row["text"].strip(), "expect": "respond" if family == "complete" else "one_turn",
                "audio_sha256": hashlib.sha256(pcm).hexdigest(), "pcm": base64.b64encode(pcm).decode()}
        if not case["text"] and family != "noise":
            raise ValueError(f"case {item_id} needs a transcript")
        if family in ("hold", "continue"):
            parts = [row["part1"].strip(), row["part2"].strip()]
            keys = [key.strip() for key in row["keys"].split("|") if key.strip()]
            if not all(parts) or not keys:
                raise ValueError(f"case {item_id} needs two transcript parts and keywords")
            try:
                at, pause = float(row["pause_at_ms"]), float(row["pause_ms"])
            except ValueError as exc:
                raise ValueError(f"case {item_id} needs pause timestamps") from exc
            if at <= 0 or pause <= 0 or at + pause >= duration_ms:
                raise ValueError(f"case {item_id} has pause outside WAV")
            case.update({"parts": parts, "keys": keys, "pause_at_ms": at, "pause_ms": pause})
        elif family in ("backchannel", "noise", "interrupt"):
            try:
                offset = float(row.get("offset_s") or 1.0)
            except ValueError as exc:
                raise ValueError(f"case {item_id} needs a valid offset_s") from exc
            if not 0 <= offset <= 3:
                raise ValueError(f"case {item_id} offset_s must be in 0..3")
            case.update({"offset_s": offset, "expect": "stop_and_answer" if family == "interrupt" else "resume"})
            if family == "interrupt":
                keys = [key.strip() for key in row["keys"].split("|") if key.strip()]
                if not keys:
                    raise ValueError(f"case {item_id} needs interruption keywords")
                case.update({"keys": keys, "onset_window": (row.get("onset_window") or "").strip().lower() == "yes"
                             or offset <= 0.25})
            if family == "noise":
                case["kind"] = (row.get("kind") or "microphone_noise").strip()
                case["text"] = case["text"] or f"<{case['kind']}>"
        if split == "heldout":
            cases[family].append(case)
    if not any(cases.values()):
        raise ValueError("no heldout cases")
    payload = {"schema": 1, "sample_rate": 16000,
               "basis": "human microphone WAVs; pauses, interruption and turn labels checked by a named human labeler; synthetic assistant speech, emulated playback, no real AEC or LAN",
               "source_labels_sha256": hashlib.sha256(labels.read_bytes()).hexdigest(),
               "speakers_heldout": sorted({c["voice"] for group in cases.values() for c in group}),
               "cases": cases}
    if any(cases[family] for family in ("backchannel", "noise", "interrupt")):
        if assistant_stimuli is None:
            raise ValueError("acoustic families require --assistant-stimuli with a long_question")
        source = json.loads(assistant_stimuli.read_text(encoding="utf-8"))
        if not source.get("long_question", {}).get("pcm_by_voice"):
            raise ValueError("assistant stimuli lack long_question audio")
        payload["long_question"] = source["long_question"]
        payload["assistant_stimuli_sha256"] = hashlib.sha256(assistant_stimuli.read_bytes()).hexdigest()
    payload["sha256"] = hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False))
    return {family: len(group) for family, group in cases.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("labels", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--assistant-stimuli", type=Path)
    args = parser.parse_args()
    print(json.dumps(prepare(args.labels, args.output, args.assistant_stimuli), ensure_ascii=False))


if __name__ == "__main__":
    main()
