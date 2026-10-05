"""A turn that never finishes must not end the session."""

from __future__ import annotations

import asyncio

import pytest

from voiceplatform.app.simulate import build_engine, is_idle, wait_until
from voiceplatform.models.base import LlmCapabilities


class HangingLlm:
    """Connects, accepts the request, and then never answers."""

    name = "hanging"

    def __init__(self) -> None:
        self.capabilities = LlmCapabilities(tools=False, streaming=True)

    async def start(self) -> None:
        return None

    async def close(self) -> None:
        return None

    async def stream(self, messages, *, tools=None, max_tokens=None):
        await asyncio.sleep(60)
        yield None  # pragma: no cover


async def test_orphan_turn_is_swept_and_the_session_recovers(config):
    config.conversation.orphan_turn_timeout_ms = 300
    eng, sink = build_engine(config)
    eng.models.llm = HangingLlm()
    await eng.models.start()
    await eng.start()
    try:
        await eng.push_text("xin chào")
        assert await wait_until(eng, is_idle, max_ms=4000, feed_silence=False)
        assert eng.counters.get("orphan_turns") == 1
        assert "playback_reset" in [m.type for m in sink.control]

        # And the next turn still works.
        eng.models.llm = build_engine(config)[0].models.llm
        await eng.push_text("còn đây thì sao")
        assert await wait_until(eng, is_idle, max_ms=5000, feed_silence=False)
        assert sink.audio
    finally:
        await eng.close()
        await eng.models.close()


async def test_a_long_answer_still_playing_is_not_an_orphan(config):
    """A turn lasts until its audio has played; long is not the same as stuck."""
    config.conversation.orphan_turn_timeout_ms = 500
    # ~4 s of audio, synthesised almost instantly: the turn sits in SPEAKING
    # for seconds after the last frame went out.
    config.models.tts.options = {"first_audio_delay_ms": 5, "rtf": 0.02, "ms_per_char": 60}
    config.models.llm.options = {
        "first_token_delay_ms": 5,
        "token_delay_ms": 1,
        "reply": "Một câu trả lời khá dài. Nó có nhiều cụm. Và nó phát lâu hơn nửa giây nhiều.",
    }
    eng, sink = build_engine(config)
    await eng.models.start()
    await eng.start()
    try:
        await eng.push_text("kể dài một chút")
        assert await wait_until(eng, is_idle, max_ms=10000, feed_silence=False)
        assert eng.counters.get("orphan_turns") is None
        assert eng.context.turns[-1].spoken_text.endswith("nhiều.")
    finally:
        await eng.close()
        await eng.models.close()
