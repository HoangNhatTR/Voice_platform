"""Tách vai trò: Speech agent nói, Back end - search tra, hai bên chạy song song."""

from __future__ import annotations

import asyncio

import pytest

from voiceplatform.app.simulate import build_engine, is_idle, wait_until
from voiceplatform.core.events import EventType
from voiceplatform.tasks.search import SearchRequest, SearchResult


@pytest.fixture
def search_config(config):
    config.conversation.search.enabled = True
    config.conversation.search.max_inflight = 2
    config.conversation.search.ttl_ms = 5000
    config.models.search = type(config.models.search)(
        backend="mock", options={"delay_ms": 400, "answer": "Hà Nội hôm nay 28 độ"}
    )
    # Model hội thoại gọi công cụ search khi nghe thấy "thời tiết".
    config.models.llm.options = {
        "first_token_delay_ms": 10,
        "token_delay_ms": 1,
        "tool_triggers": {"thời tiết": "search"},
    }
    return config


@pytest.fixture
async def engine(search_config):
    eng, sink = build_engine(search_config, tools=[])
    await eng.models.start()
    await eng.start()
    yield eng, sink
    await eng.close()
    await eng.models.close()


async def test_speech_agent_answers_before_the_search_returns(engine):
    """Đúng mũi tên “phản hồi liên tục khi chưa có thông tin” trong sơ đồ."""
    eng, sink = engine
    await eng.push_text("thời tiết Hà Nội hôm nay thế nào")
    assert await wait_until(eng, is_idle, max_ms=4000, feed_silence=False)

    turn = eng.trace.turn(1)
    types = [e.type for e in turn.events]
    assert EventType.SEARCH_REQUESTED in types
    assert EventType.TTS_COMPLETE in types
    assert eng.context.turns[0].spoken_text

    # Điều cần chứng minh là THỨ TỰ: người dùng nghe thấy tiếng trước khi kết
    # quả tra cứu về. Kết quả về sớm hay muộn là chuyện của backend.
    firsts = turn.firsts
    spoke_at = firsts[EventType.TTS_FIRST_AUDIO.value]
    requested_at = firsts[EventType.SEARCH_REQUESTED.value]
    result_at = firsts.get(EventType.SEARCH_RESULT.value)
    assert spoke_at > requested_at
    if result_at is not None:
        assert spoke_at < result_at, "lượt nói đã chờ kết quả — đúng cái cần tránh"


async def test_the_result_comes_back_as_its_own_turn(engine):
    eng, sink = engine
    await eng.push_text("thời tiết Hà Nội hôm nay thế nào")
    assert await wait_until(
        eng,
        lambda e: any(
            ev.type is EventType.SEARCH_DELIVERED
            for t in e.trace.turns.values()
            for ev in t.events
        ),
        max_ms=8000,
        feed_silence=False,
    )
    assert await wait_until(eng, is_idle, max_ms=6000, feed_silence=False)

    # Lượt hai do hệ thống tự mở, và nội dung tra cứu đã tới model.
    delivery = eng.trace.turn(2)
    assert delivery is not None
    starts = [e for e in delivery.events if e.type is EventType.TURN_START]
    assert starts and starts[0].data.get("source") == "search"
    assert "28 độ" in eng.context.turns[-1].assistant_text
    # Lượt "đang tra cứu" vẫn còn nguyên trong lịch sử, không bị ghi đè.
    assert eng.context.turns[0].user_text == "thời tiết Hà Nội hôm nay thế nào"
    assert eng.context.turns[0].assistant_text
    assert eng.context.turns[-1].user_text == ""


async def test_a_search_survives_the_user_interrupting(search_config):
    """Việc tra cứu thuộc về phiên, không thuộc lượt nói đã sinh ra nó."""
    search_config.models.search.options["delay_ms"] = 600
    eng, _ = build_engine(search_config, tools=[])
    await eng.models.start()
    await eng.start()
    try:
        await eng.push_text("thời tiết Hà Nội hôm nay thế nào")
        await wait_until(eng, lambda e: e.pending.inflight == 1, max_ms=2000, feed_silence=False)
        # Người dùng chen ngang bằng một lượt khác ngay lập tức.
        await eng.push_text("thôi khoan đã")
        assert eng.pending.inflight == 1  # yêu cầu tra cứu không bị giết
        assert await wait_until(
            eng, lambda e: e.pending.stats()["ready"] or e.gen.turn_id >= 3,
            max_ms=8000, feed_silence=False,
        )
    finally:
        await eng.close()
        await eng.models.close()


async def test_too_many_in_flight_is_refused_rather_than_queued(engine):
    eng, _ = engine
    assert eng.pending.open("a", 1) is not None
    assert eng.pending.open("b", 1) is not None
    assert eng.pending.open("c", 1) is None
    assert eng.pending.stats()["dropped_full"] == 1


async def test_a_result_that_took_too_long_is_not_spoken(engine):
    eng, _ = engine
    eng.pending.ttl_ms = 10
    request = eng.pending.open("cũ rồi", 1)
    await asyncio.sleep(0.05)
    delivered = eng.pending.complete(
        SearchResult(request=request, ok=True, content="quá muộn")
    )
    assert delivered is False
    assert eng.pending.pop_ready() is None
    assert eng.pending.stats()["dropped_expired"] == 1


async def test_the_user_hears_something_the_moment_the_search_is_sent(search_config):
    """Không chờ model soạn câu: vòng LLM đó tốn hơn một giây rưỡi."""
    search_config.conversation.search.instant_ack = "Để tôi tra cứu nhé."
    search_config.models.search.options["delay_ms"] = 800
    eng, sink = build_engine(search_config, tools=[])
    await eng.models.start()
    await eng.start()
    try:
        await eng.push_text("thời tiết Hà Nội hôm nay thế nào")
        assert await wait_until(eng, is_idle, max_ms=6000, feed_silence=False)
        turn = eng.trace.turn(1)
        firsts = turn.firsts
        # Câu báo được phát ra ngay sau khi yêu cầu được gửi, trước cả khi
        # model kịp sinh token nào của vòng sau.
        assert firsts[EventType.FILLER.value] >= firsts[EventType.SEARCH_REQUESTED.value]
        assert "tra cứu" in eng.context.turns[0].spoken_text.lower()
    finally:
        await eng.close()
        await eng.models.close()
