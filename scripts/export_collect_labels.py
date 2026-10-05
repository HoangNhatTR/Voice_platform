"""Xuất clip thu ở /collect thành labels.csv + WAV cho prepare_g3_human_stimuli.py.

Đọc manifest.jsonl của trang /collect (collect.dir, mặc định runtime/collect),
bỏ clip đã rút, chia train/heldout THEO NGƯỜI NÓI một cách tất định
(--heldout-speakers a,b hoặc --heldout-fraction 0.3 --seed N), nên không người
nói hay phiên nào nằm ở cả hai phía. Họ lời đệm/tiếng động/ngắt lời được gán
offset_s (đều trong 0,5–2,5 s, gieo theo seed), khoảng 20% lời ngắt mỗi phía
được đánh dấu onset_window=yes với offset_s ≤ 0,6 s, và khoảng lặng đầu của
ba họ này được cắt còn 50 ms: benchmark đưa clip vào ở offset_s, nên một giây
im ở đầu clip làm "chen ngay lúc trợ lý cất tiếng" thành chen muộn.

Thư mục đầu ra phải mới (hoặc rỗng): xuất đè lên bản cũ sẽ để sót WAV của clip
đã rút. Không bao giờ xuất vào docs/ — đây là giọng người thật.

    PYTHONPATH=src .venv/bin/python scripts/export_collect_labels.py \
      runtime/collect /tmp/g3-human --heldout-fraction 0.3 --seed 7
    PYTHONPATH=src .venv/bin/python scripts/prepare_g3_human_stimuli.py \
      /tmp/g3-human/labels.csv /tmp/g3-human-stimuli.json \
      --assistant-stimuli docs/audits/2026-09-29/g3/stimuli-confirm.json
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
try:
    import voiceplatform  # noqa: F401
except ImportError:  # chạy không có PYTHONPATH=src
    sys.path.insert(0, str(REPO / "src"))

from voiceplatform.app.collect import (  # noqa: E402
    TARGETS, ManifestState, analyze_pcm, read_wav_pcm16, wav_from_pcm16,
)
from voiceplatform.app.collect_bank import ACOUSTIC_FAMILIES, FAMILIES, PAUSE_FAMILIES  # noqa: E402

# Cột mà prepare_g3_human_stimuli.py đọc (HUMAN_LABELS_TEMPLATE.csv), rồi các
# cột metadata riêng: script kia bỏ qua chúng, người đọc CSV thì cần.
FIELDS = ("id", "family", "split", "speaker_id", "session_id", "labeler_id", "human_verified",
          "wav", "text", "part1", "part2", "pause_at_ms", "pause_ms", "keys", "offset_s", "kind",
          "onset_window")
EXTRA = ("prompt_id", "bank_version", "digits", "verified_by", "device", "playback", "room",
         "accent", "duration_ms", "trim_start_ms")
SAMPLE_RATE = 16000
LEAD_MS = 50
TAIL_MS = 300
MIN_TRIMMED_MS = 400
OFFSET_RANGE = (0.5, 2.5)
ONSET_SHARE = 0.2
ONSET_RANGE = (0.1, 0.6)
DEFAULT_SEED = 20261005


class ExportError(ValueError):
    pass


def split_speakers(speakers: list[str], heldout: list[str] | None, fraction: float | None,
                   seed: int) -> dict[str, str]:
    """speaker → train|heldout. Tất định: cùng đầu vào, cùng kết quả."""
    pool = sorted(set(speakers))
    if not pool:
        raise ExportError("không có clip nào để xuất")
    if heldout:
        unknown = sorted(set(heldout) - set(pool))
        if unknown:
            raise ExportError("người nói heldout không có clip: " + ", ".join(unknown))
        chosen = set(heldout)
    elif fraction is not None:
        if not 0 < fraction < 1:
            raise ExportError("--heldout-fraction phải trong (0, 1)")
        count = round(fraction * len(pool))
        count = max(1, min(count, len(pool) - 1)) if len(pool) > 1 else 1
        order = list(pool)
        random.Random(seed).shuffle(order)
        chosen = set(order[:count])
    else:
        raise ExportError("chọn --heldout-speakers hoặc --heldout-fraction")
    return {speaker: "heldout" if speaker in chosen else "train" for speaker in pool}


def _hash(*parts: Any) -> str:
    return hashlib.sha256(":".join(str(p) for p in parts).encode()).hexdigest()


def _private_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(path, 0o700)


def _write_private(path: Path, data: bytes) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(data)


def _trim(pcm, analysis: dict[str, Any]) -> tuple[Any, int]:
    start_ms, end_ms = analysis.get("speech_start_ms"), analysis.get("speech_end_ms")
    if start_ms is None:
        return pcm, 0
    total_ms = 1000 * pcm.size // SAMPLE_RATE
    start = max(0, start_ms - LEAD_MS)
    end = min(total_ms, end_ms + TAIL_MS)
    if end - start < MIN_TRIMMED_MS:
        end = min(total_ms, start + MIN_TRIMMED_MS)
    per_ms = SAMPLE_RATE // 1000
    return pcm[start * per_ms:end * per_ms], start


def export(collect_dir: Path, output: Path, *, heldout_speakers: list[str] | None = None,
           heldout_fraction: float | None = None, seed: int = DEFAULT_SEED, trim: bool = True) -> dict[str, Any]:
    collect_dir, output = Path(collect_dir), Path(output)
    docs = (REPO / "docs").resolve()
    target = output.resolve()
    if target == docs or docs in target.parents:
        raise ExportError("không xuất giọng người thật vào docs/")
    if output.exists() and any(output.iterdir()):
        raise ExportError(f"{output} đã có nội dung — chọn thư mục mới (xuất đè sẽ để sót WAV của clip đã rút)")
    state = ManifestState.load(collect_dir / "manifest.jsonl")
    clips = sorted(state.clips.values(), key=lambda c: (c["at"], c["id"]))
    sessions: dict[str, set[str]] = {}
    for clip in clips:
        sessions.setdefault(clip["session_id"], set()).add(clip["speaker_id"])
    shared = sorted(s for s, owners in sessions.items() if len(owners) > 1)
    if shared:
        raise ExportError("phiên thuộc nhiều người nói: " + ", ".join(shared))
    splits = split_speakers([c["speaker_id"] for c in clips], heldout_speakers, heldout_fraction, seed)

    onset: set[str] = set()
    for split in ("train", "heldout"):
        interrupts = sorted((c["id"] for c in clips if c["family"] == "interrupt" and splits[c["speaker_id"]] == split),
                            key=lambda cid: _hash(seed, "onset", cid))
        onset.update(interrupts[:round(ONSET_SHARE * len(interrupts))])

    _private_dir(output)
    _private_dir(output / "wavs")
    rows = []
    for clip in clips:
        source = collect_dir / clip["wav"]
        try:
            pcm = read_wav_pcm16(source.read_bytes())
        except (OSError, ValueError) as exc:
            raise ExportError(f"clip {clip['id']}: không đọc được {source}: {exc}") from exc
        if hashlib.sha256(pcm.tobytes()).hexdigest() != clip["audio_sha256"]:
            raise ExportError(f"clip {clip['id']}: audio khác hash trong manifest")
        family = clip["family"]
        trim_start = 0
        if trim and family in ACOUSTIC_FAMILIES:
            pcm, trim_start = _trim(pcm, analyze_pcm(pcm))
        name = f"wavs/{clip['id']}.wav"
        _write_private(output / name, wav_from_pcm16(pcm))
        offset = onset_window = ""
        if family in ACOUSTIC_FAMILIES:
            rng = random.Random(_hash(seed, "offset", clip["id"]))
            low, high = ONSET_RANGE if clip["id"] in onset else OFFSET_RANGE
            offset = f"{rng.uniform(low, high):.3f}"
            onset_window = "yes" if clip["id"] in onset else ""
        conditions = clip["conditions"]
        rows.append({
            "id": clip["id"], "family": family, "split": splits[clip["speaker_id"]],
            "speaker_id": clip["speaker_id"], "session_id": clip["session_id"],
            "labeler_id": clip["labeler_id"], "human_verified": clip["human_verified"], "wav": name,
            "text": "" if family == "noise" else clip["text"],
            "part1": clip["part1"] if family in PAUSE_FAMILIES else "",
            "part2": clip["part2"] if family in PAUSE_FAMILIES else "",
            "pause_at_ms": clip["pause_at_ms"] if family in PAUSE_FAMILIES else "",
            "pause_ms": clip["pause_ms"] if family in PAUSE_FAMILIES else "",
            "keys": "|".join(clip["keys"]) if family in (*PAUSE_FAMILIES, "interrupt") else "",
            "offset_s": offset, "kind": clip["kind"] if family == "noise" else "",
            "onset_window": onset_window,
            "prompt_id": clip["prompt_id"], "bank_version": clip["bank_version"],
            "digits": "yes" if clip["digits"] else "", "verified_by": clip["verified_by"],
            "device": conditions["device"], "playback": conditions["playback"], "room": conditions["room"],
            "accent": conditions["accent"], "duration_ms": round(1000 * pcm.size / SAMPLE_RATE, 1),
            "trim_start_ms": trim_start,
        })
    labels = output / "labels.csv"
    descriptor = os.open(labels, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(descriptor, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS + EXTRA)
        writer.writeheader()
        writer.writerows(rows)
    return summarize(rows, splits, len(state.withdrawn), labels)


def summarize(rows: list[dict[str, Any]], splits: dict[str, str], withdrawn: int, labels: Path) -> dict[str, Any]:
    by_split = {split: Counter(r["family"] for r in rows if r["split"] == split) for split in ("train", "heldout")}

    def have(counter: Counter, speakers: int, total: int) -> dict[str, int]:
        return {"clips": total, "speakers": speakers, "pause": counter["hold"] + counter["continue"],
                "backchannel_noise": counter["backchannel"] + counter["noise"], "interrupt": counter["interrupt"]}

    everything = by_split["train"] + by_split["heldout"]
    total = have(everything, len(splits), len(rows))
    held = have(by_split["heldout"], sum(1 for s in splits.values() if s == "heldout"),
                sum(by_split["heldout"].values()))
    train_prompts = {r["prompt_id"] for r in rows if r["split"] == "train"}
    return {
        "labels": str(labels),
        "exported": len(rows),
        "withdrawn_skipped": withdrawn,
        "speakers": {split: sorted(s for s, v in splits.items() if v == split) for split in ("train", "heldout")},
        "by_split": {split: {f: by_split[split][f] for f in FAMILIES} for split in by_split},
        "digit_sequence": {split: sum(1 for r in rows if r["split"] == split and r["digits"]
                                      and r["family"] in PAUSE_FAMILIES) for split in ("train", "heldout")},
        "onset_window": sum(1 for r in rows if r["onset_window"] == "yes"),
        "targets_total": [{"name": n, "label": label, "need": need, "have": total[n], "ok": total[n] >= need}
                          for n, label, need in TARGETS],
        # Benchmark chỉ dùng heldout (prepare_g3_human_stimuli.py).
        "targets_heldout": [{"name": n, "label": label, "need": need, "have": held[n], "ok": held[n] >= need}
                            for n, label, need in TARGETS if n in ("pause", "backchannel_noise", "interrupt")],
        # Cùng câu mẫu (khác số giả) có thể nằm ở cả hai phía: không làm hỏng
        # phép đo âm học, nhưng làm số heldout của bộ phân loại CHỮ lạc quan.
        "heldout_prompt_also_in_train": sum(1 for r in rows if r["split"] == "heldout"
                                            and r["prompt_id"] in train_prompts),
        "verified_by": dict(Counter(r["verified_by"] for r in rows)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("collect_dir", type=Path, help="collect.dir của server (có manifest.jsonl)")
    parser.add_argument("output", type=Path, help="thư mục MỚI để ghi labels.csv và wavs/")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--heldout-speakers", help="speaker_id ngăn bằng dấu phẩy")
    group.add_argument("--heldout-fraction", type=float, help="tỉ lệ người nói đưa vào heldout")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--no-trim", action="store_true", help="giữ nguyên khoảng lặng đầu của lời đệm/tiếng động/ngắt lời")
    args = parser.parse_args()
    speakers = [s.strip() for s in args.heldout_speakers.split(",") if s.strip()] if args.heldout_speakers else None
    try:
        summary = export(args.collect_dir, args.output, heldout_speakers=speakers,
                         heldout_fraction=args.heldout_fraction, seed=args.seed, trim=not args.no_trim)
    except ExportError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    summary["next"] = (f"PYTHONPATH=src .venv/bin/python scripts/prepare_g3_human_stimuli.py {summary['labels']} "
                       "/tmp/g3-human-stimuli.json --assistant-stimuli docs/audits/2026-09-29/g3/stimuli-confirm.json")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
