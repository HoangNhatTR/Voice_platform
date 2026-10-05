"""Train an optional text turn detector from human-labeled G3 clips.

Only `train` speakers fit the model. `heldout` speakers are scored once with a
fixed threshold; the report is diagnostic, not a substitute for the realtime
WAV benchmark or microphone/AEC evaluation. Requires scikit-learn offline.

  python scripts/train_g3_turn_model.py labels.csv model.json report.json
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from collections import Counter
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from voiceplatform.conversation.turn_detector.text_model import TextTurnModel


def examples(labels: Path) -> tuple[list[tuple[str, int, str]], list[tuple[str, int, str]], dict]:
    with labels.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("no human labels")
    train, heldout = [], []
    speakers: dict[str, str] = {}
    sessions: dict[str, str] = {}
    speech_speakers: set[str] = set()
    ids: set[str] = set()
    for row in rows:
        case_id, split, family = (row.get(field, "").strip() for field in ("id", "split", "family"))
        if not case_id or case_id in ids or split not in ("train", "heldout") or family not in (
            "hold", "continue", "complete", "backchannel", "noise", "interrupt"
        ):
            raise ValueError(f"invalid id, split or family: {case_id}")
        ids.add(case_id)
        if row.get("human_verified", "").strip().lower() != "yes" or not row.get("labeler_id", "").strip():
            raise ValueError(f"case {case_id} requires human verification and labeler")
        for field, groups in (("speaker_id", speakers), ("session_id", sessions)):
            value = row.get(field, "").strip()
            if not value or (value in groups and groups[value] != split):
                raise ValueError(f"{field} missing or leaks between train and heldout: {value}")
            groups[value] = split
        if family not in ("hold", "continue", "complete"):
            continue
        speech_speakers.add(row["speaker_id"].strip())
        full = row.get("text", "").strip()
        if not full:
            raise ValueError(f"case {case_id} has no final transcript")
        target = train if split == "train" else heldout
        target.append((full[:512], 1, family + "_complete"))
        if family != "complete":
            partial = row.get("part1", "").strip()
            if not partial or not row.get("part2", "").strip():
                raise ValueError(f"case {case_id} needs both transcript parts")
            target.append((partial[:512], 0, family + "_pause"))
    if len({s for s, split in speakers.items() if split == "train"}) < 2 or len({s for s, split in speakers.items() if split == "heldout"}) < 2:
        raise ValueError("need at least two speakers in each split")
    if {label for _, label, _ in train} != {0, 1} or {label for _, label, _ in heldout} != {0, 1}:
        raise ValueError("both splits need complete and incomplete examples")
    speech_cases = sum(row.get("family") in ("hold", "continue", "complete") for row in rows)
    meta = {"label_sha256": hashlib.sha256(labels.read_bytes()).hexdigest(),
            "cases": speech_cases, "all_labeled_clips": len(rows), "speakers": dict(Counter(speakers.values())),
            "sessions": dict(Counter(sessions.values())),
            "coverage_target_met": speech_cases >= 500 and len(speech_speakers) >= 10}
    return train, heldout, meta


def train_model(labels: Path, model_path: Path, report_path: Path, threshold: float = 0.6) -> dict:
    if not 0 < threshold <= 1:
        raise ValueError("threshold must be in (0, 1]")
    train, heldout, meta = examples(labels)
    vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(2, 4), max_features=8000,
                                 min_df=1, lowercase=True, norm="l2", sublinear_tf=False)
    features = vectorizer.fit_transform([text for text, _, _ in train])
    classifier = LogisticRegression(max_iter=500, random_state=20260930, class_weight="balanced")
    classifier.fit(features, [label for _, label, _ in train])
    names = vectorizer.get_feature_names_out()
    payload = {"schema": 1, "analyzer": "char_tfidf_logistic", "ngram_range": [2, 4],
               "intercept": float(classifier.intercept_[0]),
               "features": {name: [float(vectorizer.idf_[index]), float(classifier.coef_[0, index])]
                            for name, index in vectorizer.vocabulary_.items()},
               "training_labels_sha256": meta["label_sha256"]}
    model = TextTurnModel(payload)
    # Verify the dependency-free runtime produces the same scores as sklearn.
    comparison = classifier.predict_proba(vectorizer.transform([text for text, _, _ in heldout]))[:, 1]
    if max(abs(model.score(text) - float(value)) for (text, _, _), value in zip(heldout, comparison)) > 1e-6:
        raise RuntimeError("exported model differs from training runtime")
    by_kind: dict[str, dict[str, int]] = {}
    for text, label, kind in heldout:
        bucket = by_kind.setdefault(kind, {"n": 0, "wrong": 0})
        bucket["n"] += 1
        bucket["wrong"] += int((model.score(text) >= threshold) != bool(label))
    report = {"basis": "human transcript text only; realtime audio and ASR still require heldout WAV benchmark",
              "threshold": threshold, "train_examples": len(train), "heldout_examples": len(heldout),
              "labels": meta, "heldout_by_kind": by_kind,
              "heldout_incomplete_false_complete": {
                  "wrong": sum(v["wrong"] for k, v in by_kind.items() if k.endswith("_pause")),
                  "n": sum(v["n"] for k, v in by_kind.items() if k.endswith("_pause"))}}
    model_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(model_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    os.fchmod(descriptor, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("labels", type=Path)
    parser.add_argument("model", type=Path)
    parser.add_argument("report", type=Path)
    parser.add_argument("--threshold", type=float, default=0.6)
    args = parser.parse_args()
    print(json.dumps(train_model(args.labels, args.model, args.report, args.threshold), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
