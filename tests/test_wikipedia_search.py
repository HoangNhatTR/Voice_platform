"""wikipedia_vi: relevance, identifiers, deadline and User-Agent. No network."""

from __future__ import annotations

import asyncio
import logging
import time

import httpx
import pytest

from voiceplatform.tasks.search import SearchRequest, WikipediaSearchAgent

HOAN_KIEM = {"index": 1, "title": "Hồ Hoàn Kiếm",
             "fullurl": "https://vi.wikipedia.org/wiki/H%E1%BB%93_Ho%C3%A0n_Ki%E1%BA%BFm",
             "extract": "Hồ Hoàn Kiếm, còn gọi là Hồ Gươm, là một hồ nước ngọt nằm ở trung tâm Hà Nội."}
HCM = {"index": 1, "title": "Hồ Chí Minh",
       "fullurl": "https://vi.wikipedia.org/wiki/H%E1%BB%93_Ch%C3%AD_Minh",
       "extract": "Hồ Chí Minh là nhà cách mạng Việt Nam."}


def _agent(pages, seen=None, *, delay_s=0.0, **options):
    agent = WikipediaSearchAgent(**options)

    async def handler(request):
        if seen is not None:
            seen.append(request.url.params["gsrsearch"])
        if delay_s:
            await asyncio.sleep(delay_s)
        return httpx.Response(200, json={"query": {"pages": pages}})

    agent._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return agent


@pytest.mark.parametrize("question,lookup", [
    ("Bạn có thể cho tôi biết giúp hồ Hoàn Kiếm nằm ở đâu được không ạ?", "hồ Hoàn Kiếm"),
    ("Hồ Hoàn Kiếm ở đâu vậy nhỉ, mình muốn đi chơi cuối tuần", "Hồ Hoàn Kiếm"),
])
async def test_polite_words_do_not_reject_the_exact_page(question, lookup):
    """The score divided by every word the user said, so a few polite words
    rejected the right page even when the API returned exactly it."""
    seen = []
    agent = _agent([HOAN_KIEM], seen)
    try:
        result = await agent.search(SearchRequest(question, 1))
        assert result.ok and result.source_title == "Hồ Hoàn Kiếm"
        assert seen == [lookup]    # the subject, not the whole sentence
    finally:
        await agent.close()


async def test_diacritics_are_meaning_not_noise():
    """Folded, "hổ" (tiger) was the "Hồ" of Hồ Chí Minh and the page was used."""
    tiger, alias = _agent([HCM]), _agent([HOAN_KIEM])
    try:
        result = await tiger.search(SearchRequest("con hổ", 1))
        assert not result.ok and result.error == "no_result"
        # A redirect: the title differs, the excerpt names the alias.
        found = await alias.search(SearchRequest("Hồ Gươm", 2))
        assert found.ok and found.source_title == "Hồ Hoàn Kiếm"
    finally:
        await tiger.close()
        await alias.close()


@pytest.mark.parametrize("question,blocked", [
    ("dân số Việt Nam năm 2019 2020", False),
    ("chiến tranh từ 1945-1975", False),
    ("dân số Hà Nội 8.435.700 người", False),
    ("số 0912 345 678 là của ai", True),
    ("tra cứu 1234 5678 9012", True),
    ("email an@example.com", True),
])
def test_identifier_filter_spares_years_and_amounts(question, blocked):
    assert WikipediaSearchAgent._has_identifier(question) is blocked


async def test_the_whole_lookup_has_one_deadline():
    agent = _agent([HOAN_KIEM], delay_s=2.0, timeout_s=0.2)
    try:
        started = time.monotonic()
        result = await agent.search(SearchRequest("Hồ Hoàn Kiếm ở đâu", 1))
        assert time.monotonic() - started < 1.0
        assert not result.ok and result.error == "TimeoutError"
    finally:
        await agent.close()


class _Records(logging.Handler):
    def __init__(self):
        super().__init__()
        self.messages = []

    def emit(self, record):
        self.messages.append(record.getMessage())


def test_user_agent_without_operator_contact_is_reported():
    handler = _Records()
    logger = logging.getLogger("voiceplatform.search")
    logger.addHandler(handler)
    try:
        WikipediaSearchAgent()
        WikipediaSearchAgent(user_agent="VoicePlatform/0.1 (ops@example.vn)")
        WikipediaSearchAgent(user_agent="VoicePlatform/0.1 (https://voice.example.vn/contact)")
    finally:
        logger.removeHandler(handler)
    assert len(handler.messages) == 1 and "no operator contact" in handler.messages[0]
