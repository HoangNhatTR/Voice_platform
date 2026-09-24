"""Config name -> engine, including the two seats that used to be unreachable."""

from __future__ import annotations

import pytest

from voiceplatform.core.config import Config, EngineSpec
from voiceplatform.core.errors import ConfigError
from voiceplatform.models.registry import ModelPlane, build_search_agent, build_tts


def test_the_talker_is_asked_for_the_configured_output_rate():
    """The borrowed talkers default to 16 kHz on their own.

    Left unset, a 24 kHz voice arrived resampled down with nothing saying so,
    while `audio.output_sample_rate` sat in the config being read by nobody.
    """
    engine = build_tts(EngineSpec(backend="mock"), output_sample_rate=16000)
    assert engine.capabilities.native_sample_rate == 16000


def test_a_per_engine_option_still_wins_over_the_audio_section():
    engine = build_tts(
        EngineSpec(backend="mock", options={"sample_rate": 22050}),
        output_sample_rate=16000,
    )
    assert engine.capabilities.native_sample_rate == 22050


def test_the_model_plane_hands_the_rate_down():
    config = Config()
    config.audio.output_sample_rate = 16000
    plane = ModelPlane(config.models, output_sample_rate=config.audio.output_sample_rate)
    assert plane.describe()["tts"]["capabilities"]["native_sample_rate"] == 16000


def test_the_tools_search_backend_can_actually_be_built():
    """It had no caller able to supply an executor, so it only ever raised."""
    agent = build_search_agent(EngineSpec(backend="tools", options={"tool_name": "kb"}))
    assert agent.name == "tools"
    assert "kb" in agent.executor.registry


def test_the_tools_search_backend_defaults_to_the_knowledge_tool():
    agent = build_search_agent(EngineSpec(backend="tools"))
    assert agent.tool_name == "kb"


def test_an_unknown_search_backend_still_fails_closed():
    with pytest.raises(ConfigError):
        build_search_agent(EngineSpec(backend="nope"))
