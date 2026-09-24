"""Shared fixtures.

Every test runs on the mock engines: no GPU, no network, no model download.
Anything that needs a real engine belongs in a separate, opt-in suite.
"""

from __future__ import annotations

import pytest

from voiceplatform.core.config import Config


@pytest.fixture
def config() -> Config:
    cfg = Config()
    cfg.observability.write_traces = False
    cfg.observability.log_events = False
    # Tight but realistic: 20 ms frames, a 120 ms gate, a 240 ms endpoint.
    cfg.media.vad.start_frames = 2
    cfg.media.vad.end_frames = 6
    cfg.conversation.turn_detection.silence_ms = 240
    cfg.conversation.turn_detection.max_silence_ms = 600
    cfg.conversation.barge_in.speech_frames = 4
    cfg.conversation.barge_in.guard_ms = 60
    cfg.models.llm.options = {"first_token_delay_ms": 10, "token_delay_ms": 1}
    cfg.models.tts.options = {"first_audio_delay_ms": 10, "rtf": 0.02}
    cfg.models.asr.options = {"partial_every_frames": 5}
    return cfg
