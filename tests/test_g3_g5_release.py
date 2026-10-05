"""G3 model wiring and G5 fail-closed release evidence."""

from __future__ import annotations

import csv
import json
import os
import time

import pytest

from voiceplatform.conversation.turn_detector import build_turn_detector
from voiceplatform.conversation.turn_detector.text_model import TextTurnModel
from voiceplatform.core.config import Config
from voiceplatform.core.errors import ConfigError
from voiceplatform.observability.trace import SessionTrace, prune_session_traces


def test_semantic_backend_requires_a_real_model():
    config = Config()
    config.conversation.turn_detection.backend = "semantic"
    with pytest.raises(ConfigError, match="semantic_model_path"):
        config.validate()


async def test_exported_text_model_is_used_by_turn_detector(tmp_path):
    payload = {"schema": 1, "analyzer": "char_tfidf_logistic", "ngram_range": [2, 4],
               "intercept": 0.0, "features": {"xong": [1.0, 5.0]}}
    path = tmp_path / "model.json"
    path.write_text(json.dumps(payload))
    detector = build_turn_detector("semantic", silence_ms=400, max_silence_ms=1400,
                                   semantic_model_path=str(path))
    assert await detector.required_silence_ms(text="xong", utterance_ms=500) == 400
    assert await detector.required_silence_ms(text="đang nói", utterance_ms=500) > 400
    with pytest.raises(ValueError, match="unsupported"):
        TextTurnModel({"schema": 2})


def test_training_uses_disjoint_speakers_and_exports_runtime_equivalent_scores(tmp_path):
    pytest.importorskip("sklearn")
    from scripts.train_g3_turn_model import train_model

    labels = tmp_path / "labels.csv"
    fields = ("id", "family", "split", "speaker_id", "session_id", "labeler_id",
              "human_verified", "text", "part1", "part2")
    with labels.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for i, split in enumerate(("train", "train", "heldout", "heldout")):
            for j, family in enumerate(("hold", "complete")):
                writer.writerow({"id": f"{i}-{j}", "family": family, "split": split,
                                 "speaker_id": f"speaker-{i}", "session_id": f"session-{i}",
                                 "labeler_id": "rater-1", "human_verified": "yes",
                                 "text": f"tôi muốn chuyển tiền cho bạn {i}" if j == 0 else f"bây giờ là mấy giờ rồi {i}",
                                 "part1": "tôi muốn chuyển tiền", "part2": f"cho bạn {i}"})
    report = train_model(labels, tmp_path / "model.json", tmp_path / "report.json")
    assert report["train_examples"] == 6
    assert report["heldout_examples"] == 6
    assert report["labels"]["coverage_target_met"] is False


def test_soak_cleanup_audit_detects_leaked_slot():
    from scripts.soak_g5 import idle_checks

    idle = {"gauges": {"sessions": 0, "llm": {"active": 0, "waiting": 0},
                       "tts": {"active": 0, "waiting": 0}, "tts_workers": 1}}
    assert all(idle_checks({}, idle, {"tts_workers": 1}).values())
    idle["gauges"]["llm"]["active"] = 1
    assert idle_checks({}, idle, {"tts_workers": 1})["llm_queue_empty"] is False


def test_release_gate_blocks_missing_human_and_soak_evidence():
    from scripts.g5_release_gate import evaluate

    result = evaluate()
    assert result["decision"] == "blocked"
    assert result["checks"]["g3_human_corpus"]["status"] == "blocked"
    assert result["checks"]["soak_24h"]["status"] == "blocked"


def test_trace_retention_only_prunes_expired_generated_sessions(tmp_path):
    trace = SessionTrace("s0001-deadbeef")
    old = trace.write_jsonl(tmp_path)
    assert old.stat().st_mode & 0o777 == 0o600
    recent = tmp_path / "s0002-cafebabe.jsonl"
    recent.write_text("keep")
    unrelated = tmp_path / "summary.jsonl"
    unrelated.write_text("keep")
    os.utime(old, (time.time() - 9 * 86400, time.time() - 9 * 86400))
    os.utime(unrelated, (time.time() - 9 * 86400, time.time() - 9 * 86400))
    assert prune_session_traces(tmp_path, 7) == 1
    assert not old.exists() and recent.exists() and unrelated.exists()
