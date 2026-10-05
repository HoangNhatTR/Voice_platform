"""Prepare a blind, randomized G4 listening pack and summarize five+ ratings.

Input CSV: item_id,baseline_wav,candidate_wav,category. Each WAV should contain
the user's utterance and the assistant's audible reply, including any wait.
Share only an individual rater directory, never private-key.json.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import shutil
from collections import defaultdict
from pathlib import Path

FIELDS = ("turn_timing", "listenability", "answer_fit")


def prepare(items_csv: Path, output: Path, raters: int, seed: int) -> None:
    if raters < 5:
        raise ValueError("G4 requires at least five independent listeners")
    if output.exists():
        raise FileExistsError(output)
    with items_csv.open(newline="", encoding="utf-8") as handle:
        items = list(csv.DictReader(handle))
    if not items or len({row["item_id"] for row in items}) != len(items):
        raise ValueError("items need unique item_id values")
    for row in items:
        for field in ("baseline_wav", "candidate_wav"):
            path = Path(row[field])
            if not path.is_absolute():
                path = items_csv.parent / path
            if not path.is_file() or path.suffix.lower() != ".wav":
                raise ValueError(f"missing WAV: {path}")
            row[field] = str(path)
    rng = random.Random(seed)
    output.mkdir(parents=True)
    key = []
    for rater in range(1, raters + 1):
        rater_dir = output / f"listener-{rater:02d}"
        rater_dir.mkdir()
        order = list(items)
        rng.shuffle(order)
        sheet = []
        for index, row in enumerate(order, 1):
            swapped = bool(rng.getrandbits(1))
            for position, variant in enumerate(("candidate", "baseline") if swapped else ("baseline", "candidate"), 1):
                clip = f"clip-{index:03d}-{position}.wav"
                shutil.copyfile(row[f"{variant}_wav"], rater_dir / clip)
                sheet.append({"item_id": row["item_id"], "category": row.get("category", ""),
                              "clip_id": clip, **{field: "" for field in FIELDS}, "notes": ""})
                key.append({"rater": rater, "item_id": row["item_id"], "clip_id": clip,
                            "variant": variant})
        with (rater_dir / "scores.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(sheet[0]))
            writer.writeheader()
            writer.writerows(sheet)
        (rater_dir / "README.txt").write_text(
            "Nghe từng clip theo thứ tự trong scores.csv và ghi ba điểm nguyên 1..5.\n"
            "turn_timing: máy có chờ đúng lượt, ngắt hoặc nói chen hợp lý không?\n"
            "listenability: giọng, phát âm và nhịp nghỉ có dễ nghe không?\n"
            "answer_fit: nội dung có đúng và phù hợp câu hỏi không?\n"
            "1 = rất kém; 3 = tạm được; 5 = rất tốt. Ghi lỗi cụ thể ở notes.\n"
            "Chấm độc lập; không xem private-key.json hoặc gói của người khác.\n",
            encoding="utf-8",
        )
    private_key = output / "private-key.json"
    with os.fdopen(os.open(private_key, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600),
                   "w", encoding="utf-8") as handle:
        json.dump(key, handle, ensure_ascii=False, indent=2)


def summarize(output: Path) -> dict:
    key = json.loads((output / "private-key.json").read_text(encoding="utf-8"))
    lookup = {(row["rater"], row["clip_id"]): row for row in key}
    values: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
    listeners = set()
    for sheet in sorted(output.glob("listener-*/scores.csv")):
        rater = int(sheet.parent.name.split("-")[-1])
        with sheet.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        if len(rows) != sum(row["rater"] == rater for row in key):
            raise ValueError(f"incomplete sheet: {sheet}")
        seen = set()
        for row in rows:
            assignment = lookup.get((rater, row["clip_id"]))
            if (assignment is None or row["item_id"] != assignment["item_id"]
                    or row["clip_id"] in seen):
                raise ValueError(f"unrecognized clip: {sheet}: {row['clip_id']}")
            seen.add(row["clip_id"])
            for field in FIELDS:
                try:
                    score = int(row[field])
                except (ValueError, TypeError) as exc:
                    raise ValueError(f"missing score: {sheet}: {row['clip_id']}: {field}") from exc
                if not 1 <= score <= 5:
                    raise ValueError(f"score outside 1..5: {sheet}: {row['clip_id']}: {field}")
                values[assignment["variant"]][field].append(score)
        listeners.add(rater)
    if len(listeners) < 5:
        raise ValueError("G4 requires at least five completed listener sheets")
    result = {"listeners": len(listeners), "clips_per_variant": len(values["candidate"][FIELDS[0]]),
              "means": {variant: {field: round(sum(scores) / len(scores), 3)
                                  for field, scores in dimensions.items()}
                        for variant, dimensions in values.items()}}
    result["listening_threshold_met"] = all(result["means"]["candidate"][field] >= 4 for field in FIELDS)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("items", type=Path)
    prep.add_argument("output", type=Path)
    prep.add_argument("--raters", type=int, default=5)
    prep.add_argument("--seed", type=int, default=20260930)
    report = sub.add_parser("summarize")
    report.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.items, args.output, args.raters, args.seed)
    else:
        print(json.dumps(summarize(args.output), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
