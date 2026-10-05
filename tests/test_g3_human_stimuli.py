import csv
import base64
import json
import wave

import pytest

from scripts.prepare_g3_human_stimuli import FIELDS, prepare


def _wav(path):
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        handle.writeframes(b"\0\0" * 32000)


def _csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _row(item_id, split, speaker, family="hold"):
    return {"id": item_id, "family": family, "split": split,
            "speaker_id": speaker, "session_id": f"session-{speaker}",
            "labeler_id": "reviewer-1", "human_verified": "yes", "wav": "clip.wav",
            "text": "Tôi muốn hỏi về thẻ", "part1": "Tôi muốn", "part2": "hỏi về thẻ",
            "pause_at_ms": "500", "pause_ms": "400", "keys": "muốn|thẻ"}


def test_only_human_verified_heldout_audio_enters_benchmark(tmp_path):
    _wav(tmp_path / "clip.wav")
    labels = tmp_path / "labels.csv"
    _csv(labels, [_row("train-1", "train", "speaker-a"),
                  _row("test-1", "heldout", "speaker-b")])
    out = tmp_path / "stimuli.json"
    counts = prepare(labels, out)
    assert counts["hold"] == 1 and sum(counts.values()) == 1
    payload = json.loads(out.read_text())
    assert payload["speakers_heldout"] == ["speaker-b"]
    assert payload["cases"]["hold"][0]["pause_at_ms"] == 500
    assert len(payload["cases"]["hold"][0]["pcm"]) > 0


def test_speaker_leakage_and_unverified_labels_are_rejected(tmp_path):
    _wav(tmp_path / "clip.wav")
    labels = tmp_path / "labels.csv"
    _csv(labels, [_row("train-1", "train", "speaker-a"),
                  _row("test-1", "heldout", "speaker-a")])
    with pytest.raises(ValueError, match="leaks"):
        prepare(labels, tmp_path / "out.json")
    row = _row("test-1", "heldout", "speaker-b")
    row["human_verified"] = "no"
    _csv(labels, [row])
    with pytest.raises(ValueError, match="human_verified"):
        prepare(labels, tmp_path / "out.json")


def test_labeled_noise_and_interrupts_use_frozen_assistant_audio(tmp_path):
    _wav(tmp_path / "clip.wav")
    labels = tmp_path / "labels.csv"
    noise = _row("noise-1", "heldout", "speaker-b", "noise")
    noise["text"] = ""
    interrupt = _row("interrupt-1", "heldout", "speaker-c", "interrupt")
    _csv(labels, [_row("train-1", "train", "speaker-a"), noise, interrupt])
    source = tmp_path / "assistant.json"
    source.write_text(json.dumps({"long_question": {"text": "kể chuyện", "pcm_by_voice": {
        "voice": base64.b64encode(b"\0\0" * 32000).decode()}}}))
    out = tmp_path / "stimuli.json"
    counts = prepare(labels, out, source)
    assert counts["noise"] == counts["interrupt"] == 1
    payload = json.loads(out.read_text())
    assert payload["cases"]["noise"][0]["expect"] == "resume"
    assert payload["cases"]["interrupt"][0]["keys"] == ["muốn", "thẻ"]
    assert "long_question" in payload
