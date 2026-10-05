"""G4 speech copy and public-source lookup, with no external services."""

from __future__ import annotations

import asyncio

import httpx

from voiceplatform.app.simulate import build_engine, is_idle, wait_until
from voiceplatform.conversation.speech_normalize import normalize_for_speech
from voiceplatform.conversation.segmenter import PhraseSegmenter
from voiceplatform.core.events import EventType
from voiceplatform.conversation.state import TurnState
from voiceplatform.models.base import Transcript
from voiceplatform.models.registry import build_search_agent
from voiceplatform.core.config import EngineSpec
from voiceplatform.tasks.search import SearchRequest, SearchResult, WikipediaSearchAgent


def test_only_unambiguous_formats_change_for_tts():
    source = "Ngày 30/09/2026, trả 1.250.000 đồng; đi 10 km. Số tài khoản 1234 5678."
    spoken = normalize_for_speech(source)
    assert spoken.startswith("ngày ba mươi tháng chín năm")
    assert "một triệu hai trăm năm mươi nghìn đồng" in spoken
    assert "mười ki lô mét" in spoken
    assert "số tài khoản một hai ba bốn năm sáu bảy tám." in spoken.lower()
    assert normalize_for_speech("Mã 123456; ngày 31/02/2026") == "Mã 123456; ngày 31/02/2026"
    assert normalize_for_speech("TP.HCM", {"TP.HCM": "thành phố Hồ Chí Minh"}) == "thành phố Hồ Chí Minh"


def test_first_phrase_does_not_separate_a_date_or_identifier():
    text = "Tôi sẽ kiểm tra ngày 30/09/2026 và số tài khoản 1234 5678 trước."
    spaces = PhraseSegmenter._safe_spaces(text)
    assert text.index("30/09/2026") not in spaces
    assert text.index("1234") not in spaces
    assert text.index("ngày") not in spaces
    assert text.index("số tài khoản") not in spaces


async def test_wikipedia_lookup_has_a_real_source_and_refuses_live_or_private_data():
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        if "Nguyễn Thị Minh Khai" in request.url.params["gsrsearch"]:
            return httpx.Response(200, json={"query": {"pages": [
                {"index": 1, "title": "Đường sắt đô thị Thành phố Hồ Chí Minh",
                 "extract": "Mạng đường sắt đô thị tại Thành phố Hồ Chí Minh.",
                 "fullurl": "https://vi.wikipedia.org/wiki/Đường_sắt_đô_thị_Thành_phố_Hồ_Chí_Minh"}
            ]}})
        return httpx.Response(200, json={"query": {"pages": [
            {"index": 1, "title": "Văn Miếu – Quốc Tử Giám", "extract": "Một di tích ở Hà Nội.",
             "fullurl": "https://vi.wikipedia.org/wiki/Văn_Miếu"},
            {"index": 2, "title": "Hồ Hoàn Kiếm", "extract": "Hồ Hoàn Kiếm nằm ở Hà Nội.",
             "fullurl": "https://vi.wikipedia.org/wiki/H%E1%BB%93_Ho%C3%A0n_Ki%E1%BA%BFm"}
        ]}})

    agent = build_search_agent(EngineSpec(backend="wikipedia_vi"))
    assert isinstance(agent, WikipediaSearchAgent)
    agent._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    try:
        result = await agent.search(SearchRequest("Hồ Hoàn Kiếm ở đâu theo Wikipedia", 1))
        assert result.ok and result.source_title == "Hồ Hoàn Kiếm"
        assert result.source_url.startswith("https://vi.wikipedia.org/wiki/")
        assert "Hà Nội" in result.content
        assert seen[0].url.params["generator"] == "search"
        # The subject, not the sentence: CirrusSearch wants every word.
        assert seen[0].url.params["gsrsearch"] == "Hồ Hoàn Kiếm"
        assert seen[0].url.params["explaintext"] == "1"
        for query in ("giá vàng hôm nay", "số dư tài khoản của tôi", "tra cứu 1234 5678 9012"):
            denied = await agent.search(SearchRequest(query, 2))
            assert not denied.ok and denied.error == "unsupported_query"
        no_match = await agent.search(SearchRequest("Kim Tự Tháp Giza", 3))
        assert not no_match.ok and no_match.error == "no_result"
        unrelated_street = await agent.search(SearchRequest(
            "Đường Nguyễn Thị Minh Khai ở thành phố Hồ Chí Minh nằm ở đâu", 4
        ))
        assert not unrelated_street.ok and unrelated_street.error == "no_result"
        assert len(seen) == 3
    finally:
        await agent.close()


async def test_search_failure_is_spoken_as_failure_without_a_second_llm_call(config):
    config.conversation.opener.enabled = False
    engine, sink = build_engine(config, tools=[])
    await engine.models.start()
    await engine.start()
    try:
        request = SearchRequest("giá vàng hôm nay", 1)
        await engine._deliver_search(SearchResult(request, False, "Nguồn không có giá trực tiếp.", error="unsupported_query"))
        assert await wait_until(engine, is_idle, max_ms=4000, feed_silence=False)
        assert "chưa tra được" in engine.context.turns[-1].assistant_text.lower()
        trace = engine.trace.turn(engine.gen.turn_id)
        assert not any(event.type is EventType.LLM_START for event in trace.events)
        assert any(p["role"] == "fallback" for p in trace.summary()["phrases"])
    finally:
        await engine.close()
        await engine.models.close()


async def test_successful_delivery_exposes_source_link(config):
    config.conversation.opener.enabled = False
    engine, sink = build_engine(config, tools=[])
    await engine.models.start()
    await engine.start()
    try:
        request = SearchRequest("Hồ Hoàn Kiếm ở đâu", 1)
        await engine._deliver_search(SearchResult(
            request, True, "Hồ Hoàn Kiếm nằm ở Hà Nội.", source="wikipedia_vi",
            source_title="Hồ Hoàn Kiếm", source_url="https://vi.wikipedia.org/wiki/Hồ_Hoàn_Kiếm",
        ))
        assert await wait_until(engine, is_idle, max_ms=4000, feed_silence=False)
        links = [message for message in sink.control if message.type == "search_source"]
        assert len(links) == 1
        assert links[0].data["generation_id"] == engine.gen.current.generation_id
        assert links[0].data["url"].startswith("https://vi.wikipedia.org/wiki/")
    finally:
        await engine.close()
        await engine.models.close()


async def test_low_asr_confidence_asks_for_one_specific_detail(config):
    config.conversation.clarify_confidence_below = 0.7
    config.conversation.opener.enabled = False
    engine, _ = build_engine(config, tools=[])
    await engine.models.start()
    await engine.start()

    class UncertainStream:
        async def finish(self):
            return Transcript("chuyển tiền vào số tài khoản 123456", is_final=True, confidence=0.2)

        async def close(self):
            return None

    try:
        engine._asr_stream = UncertainStream()
        turn_id = engine.gen.next_turn()
        engine.state.to(TurnState.THINKING, "test ASR final")
        key = engine.gen.begin(turn_id)
        await engine._respond(key)
        assert "số tài khoản" in engine.context.turns[-1].assistant_text.lower()
        assert engine.context.turns[-1].assistant_text.count("?") == 1
        assert "123456" not in engine.context.turns[-1].user_text
        assert not any(e.type is EventType.LLM_START for e in engine.trace.turn(turn_id).events)
    finally:
        await engine.close()
        await engine.models.close()


async def test_reused_endpoint_keeps_confidence_for_clarification(config):
    config.conversation.clarify_confidence_below = 0.7
    config.conversation.opener.enabled = False
    engine, _ = build_engine(config, tools=[])
    await engine.models.start()
    await engine.start()

    class ReusedStream:
        def accept_final(self, text):
            assert text == "số tiền một triệu"

        async def close(self):
            return None

    try:
        engine._asr_stream = ReusedStream()
        turn_id = engine.gen.next_turn()
        engine.state.to(TurnState.THINKING, "test endpoint reuse")
        key = engine.gen.begin(turn_id)
        await engine._respond(key, early_text="số tiền một triệu", early_confidence=0.2)
        assert "đọc lại số tiền" in engine.context.turns[-1].assistant_text.lower()
    finally:
        await engine.close()
        await engine.models.close()


async def test_later_user_turn_makes_search_result_stale(config):
    config.conversation.search.enabled = True
    config.models.search = EngineSpec(backend="mock", options={"delay_ms": 1})
    engine, _ = build_engine(config, tools=[])
    await engine.models.start()
    await engine.start()
    try:
        engine.context.start_turn(1, "câu cũ")
        request = engine.pending.open("câu cũ", 1)
        engine.context.start_turn(2, "đổi sang câu khác")
        assert engine.pending.complete(SearchResult(request, True, "kết quả cũ"))
        await asyncio.sleep(0.2)
        assert engine.pending.dropped_stale == 1
        assert not any(t.user_text == "" for t in engine.context.turns)
    finally:
        await engine.close()
        await engine.models.close()
