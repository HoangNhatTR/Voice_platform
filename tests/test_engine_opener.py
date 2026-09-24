"""Câu mở dựng sẵn: có tiếng dưới một giây kể cả khi câu trả lời còn lâu mới tới."""

from __future__ import annotations

import pytest

from voiceplatform.app.simulate import build_engine, is_idle, wait_until
from voiceplatform.core.events import EventType


def _openers(engine):
    trace = engine.trace.turn(engine.gen.turn_id)
    return [
        e for e in trace.events
        if e.type is EventType.FILLER and e.data.get("source") == "opener"
    ]


async def _run(config, *, warm: bool = True, question: str = "xin chào"):
    engine, sink = build_engine(config)
    await engine.models.start()
    if warm:
        await engine.models.cached_speech(config.conversation.opener.text, None)
    await engine.start()
    try:
        await engine.push_text(question)
        assert await wait_until(engine, is_idle, max_ms=10000, feed_silence=False)
        trace = engine.trace.turn(engine.gen.turn_id)
        return engine, sink, trace.metrics()
    finally:
        await engine.close()
        await engine.models.close()


async def test_a_slow_answer_still_makes_a_sound_quickly(config):
    """Vòng LLM của một lượt gọi công cụ không sinh chữ nào, nên không có gì để
    nói cho tới khi nó xong — đo được 2,2 giây im lặng trên stack thật."""
    config.conversation.opener.after_ms = 60
    config.models.llm.options = {"first_token_delay_ms": 1500, "token_delay_ms": 1}
    engine, _, metrics = await _run(config)
    assert len(_openers(engine)) == 1
    # Tiếng đầu là câu mở, không phải câu trả lời 1,5 giây sau.
    assert metrics["e2e_ttfa_ms"] < 500, metrics


async def test_a_fast_answer_is_not_padded_with_one(config):
    """Câu đệm phát vô điều kiện chỉ là độ trễ tự thêm vào."""
    config.conversation.opener.after_ms = 400
    config.models.llm.options = {"first_token_delay_ms": 5, "token_delay_ms": 1}
    engine, _, metrics = await _run(config)
    assert _openers(engine) == []
    assert metrics["e2e_ttfa_ms"] < 400


async def test_turning_it_off_leaves_the_turn_alone(config):
    config.conversation.opener.enabled = False
    config.models.llm.options = {"first_token_delay_ms": 800, "token_delay_ms": 1}
    engine, _, _ = await _run(config)
    assert _openers(engine) == []


async def test_a_cold_opener_is_skipped_rather_than_waited_for(config):
    """Chờ tổng hợp ở đây là đúng thứ câu mở sinh ra để tránh.

    Lượt đầu không có câu mở; tác vụ nền làm ấm, và lượt sau thì có.
    """
    config.conversation.opener.after_ms = 60
    config.models.llm.options = {"first_token_delay_ms": 900, "token_delay_ms": 1}
    engine, sink = build_engine(config)
    await engine.models.start()
    await engine.start()
    try:
        engine.models._speech_cache.clear()      # nguội hẳn
        await engine.push_text("câu một")
        assert await wait_until(engine, is_idle, max_ms=10000, feed_silence=False)
        assert _openers(engine) == [], "lượt nguội không được chờ tổng hợp"

        await engine.push_text("câu hai")
        assert await wait_until(engine, is_idle, max_ms=10000, feed_silence=False)
        assert len(_openers(engine)) == 1, "làm ấm nền không chạy"
    finally:
        await engine.close()
        await engine.models.close()


async def test_the_opener_never_lands_inside_the_streamed_text(config):
    """`_speak` chạy song song với `_answer` đang stream token.

    Một `assistant_delta` phát từ câu mở sẽ chen vào giữa câu model đang viết —
    đo được "Về câu bạnVâng.  hỏi lúc nãy" trên stack thật.
    """
    config.conversation.opener.after_ms = 60
    config.models.llm.options = {"first_token_delay_ms": 100, "token_delay_ms": 40}
    engine, sink = build_engine(config)
    await engine.models.start()
    await engine.models.cached_speech(config.conversation.opener.text, None)
    await engine.start()
    try:
        await engine.push_text("kể một câu dài giúp tôi")
        assert await wait_until(engine, is_idle, max_ms=10000, feed_silence=False)
        assert len(_openers(engine)) == 1, "test cần câu mở thật sự bắn"
        shown = "".join(
            m.data["text"] for m in sink.control if m.type == "assistant_delta"
        )
        assert config.conversation.opener.text not in shown
    finally:
        await engine.close()
        await engine.models.close()
