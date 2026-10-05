"""What the model is told was said, after interruptions.

* An answer cut before a word of it was heard is not something the assistant
  said: the prompt may say it was interrupted, never what it would have said.
* One sentence broken by two pauses, each long enough to confirm a turn, is
  still one sentence — the first piece must survive the second interruption.
"""

from __future__ import annotations

import asyncio

from voiceplatform.app.simulate import Step, build_engine, feed, is_idle, wait_until
from voiceplatform.conversation.context import ConversationContext
from voiceplatform.models.base import Transcript


def test_an_interrupted_turn_renders_only_what_was_heard():
    ctx = ConversationContext("hệ thống")
    ctx.start_turn(1, "số dư của tôi")
    ctx.commit_assistant("Số dư của bạn là hai triệu đồng.", "", interrupted=True)
    ctx.start_turn(2, "kể chuyện")
    ctx.commit_assistant("Ngày xưa có một con cáo. Nó rất khôn.", "Ngày xưa có một con cáo.", interrupted=True)
    assistant = [m.content for m in ctx.messages() if m.role == "assistant"]
    assert assistant == ["(bị người dùng ngắt lời)", "Ngày xưa có một con cáo. (bị người dùng ngắt lời)"]


async def test_a_typed_interruption_before_any_audio_keeps_the_unheard_answer_out(config):
    # Slow first audio: the model has written its answer, nothing has played.
    config.models.tts.options = {"first_audio_delay_ms": 1500, "rtf": 0.02, "ms_per_char": 10}
    config.models.llm.options = {"first_token_delay_ms": 5, "token_delay_ms": 1}
    config.conversation.opener.enabled = False
    eng, _ = build_engine(config)
    await eng.models.start()
    await eng.start()
    try:
        await eng.push_text("số dư của tôi")
        assert await wait_until(eng, lambda e: e._response is not None and e._response.llm_done,
                                max_ms=2000, feed_silence=False)
        assert eng._response.first_audio_ms is None             # nothing heard
        await eng.push_text("thôi khỏi")
        assert await wait_until(eng, is_idle, max_ms=5000, feed_silence=False)
        messages = [(m.role, m.content) for m in eng.context.messages()]
        cut = messages.index(("user", "số dư của tôi"))
        assert messages[cut + 1] == ("assistant", "(bị người dùng ngắt lời)"), messages
    finally:
        await eng.close()
        await eng.models.close()


_PARTS = ["phần một", "phần hai", "phần ba"]


class _Stream:
    def __init__(self, text: str) -> None:
        self.text = text
        self.frames = 0

    async def push(self, frame):
        self.frames += 1
        return Transcript(text=self.text, is_final=False) if self.frames % 5 == 0 else None

    async def decode_now(self):
        return Transcript(text=self.text, is_final=False, confidence=1.0)

    async def finish(self):
        await asyncio.sleep(0.6)         # a final decode slower than the next words
        return Transcript(text=self.text, is_final=True, confidence=1.0)

    async def close(self):
        return None


class _PartsAsr:
    """Stream n hears part n: the three pieces of one sentence."""

    def __init__(self) -> None:
        self.opened = 0

    async def close(self) -> None:
        return None

    async def open_stream(self, *, sample_rate, language=None):
        text = _PARTS[min(self.opened, len(_PARTS) - 1)]
        self.opened += 1
        return _Stream(text)


async def test_a_sentence_in_three_pieces_keeps_its_first_piece(config):
    config.conversation.turn_detection.backend = "vad_only"
    config.conversation.turn_detection.reuse_endpoint_transcript = False
    config.models.llm.options = {"first_token_delay_ms": 5000, "token_delay_ms": 1}
    eng, _ = build_engine(config)
    await eng.models.start()
    eng.models.asr = _PartsAsr()
    await eng.start()
    try:
        await feed(eng, [Step("silence", 100), Step("speech", 600), Step("silence", 300)])
        assert eng.state.state.value == "thinking"
        await feed(eng, [Step("speech", 400), Step("silence", 300)])   # piece two, during piece one's final
        await feed(eng, [Step("speech", 400), Step("silence", 300)])   # piece three, during piece two's
        assert eng.counters.get("barge_ins") == 2
        assert await wait_until(eng, lambda e: e.context.turns and "ba" in e.context.turns[-1].user_text,
                                max_ms=4000, feed_silence=False)
        assert [t.user_text for t in eng.context.turns] == ["phần một phần hai phần ba"]
    finally:
        await eng.close()
        await eng.models.close()
