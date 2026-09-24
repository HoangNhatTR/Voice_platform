from __future__ import annotations

import pytest

from voiceplatform.core.config import Config
from voiceplatform.core.errors import ConfigError


def test_defaults_are_coherent():
    c = Config()
    assert c.frame_samples == c.audio.sample_rate * c.audio.frame_ms // 1000
    assert c.frames_for_ms(c.audio.frame_ms) == 1


def test_nested_sections_load(tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text(
        "audio:\n  sample_rate: 24000\n"
        "conversation:\n  barge_in:\n    speech_frames: 9\n",
        encoding="utf-8",
    )
    c = Config.load(path)
    assert c.audio.sample_rate == 24000
    assert c.conversation.barge_in.speech_frames == 9
    assert c.conversation.turn_detection.silence_ms == 480  # untouched default


def test_a_typo_fails_closed(tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text("audio:\n  sample_ratee: 24000\n", encoding="utf-8")
    with pytest.raises(ConfigError) as exc:
        Config.load(path)
    assert "sample_ratee" in str(exc.value)


def test_missing_file_is_an_error(tmp_path):
    with pytest.raises(ConfigError):
        Config.load(tmp_path / "nope.yaml")
