"""Forced public-source routing must not swallow ordinary "… ở đâu?" questions.

Folding diacritics made sống/sông, họ/hộ/hồ, chưa/chùa and câu/cậu/cầu one
word, so with force_source_lookup these never reached the model at all.
"""

from __future__ import annotations

import pytest

from voiceplatform.conversation.search_routing import requires_public_lookup


@pytest.mark.parametrize("text", [
    "Bạn sống ở đâu?",
    "Họ đang ở đâu rồi?",
    "Con chưa biết mẹ để chìa khoá ở đâu",
    "Câu này nằm ở đâu trong bài?",
    "Cậu đang ở đâu đấy?",
    "Hộ chiếu của tôi để ở đâu nhỉ?",
    "Anh ấy chưa về, giờ đang ở đâu?",
    "Cửa hàng gần nhất ở đâu, có cầu thang không?",
    "hồ sơ của tôi nằm ở đâu",
    "Đường để ở đâu rồi?",
    "yêu cầu của tôi gửi ở đâu",
    "hồ bơi ở đâu",
])
def test_everyday_where_questions_stay_with_the_model(text):
    assert not requires_public_lookup(text)


@pytest.mark.parametrize("text", [
    "Hồ Hoàn Kiếm nằm ở đâu?",
    "hồ hoàn kiếm nằm ở đâu",                      # ASR output: lowercase, no punctuation
    "Bạn có thể cho tôi biết hồ Hoàn Kiếm ở đâu không?",
    "Đường Nguyễn Thị Minh Khai nằm ở đâu?",
    "chùa một cột ở đâu",
    "núi bà đen thuộc tỉnh nào",
    "Địa chỉ của bảo tàng Hồ Chí Minh ở đâu?",
    "Tra cứu Wikipedia về Hồ Gươm.",
    "tìm trên mạng giúp tôi lịch sử chùa Hương",
])
def test_named_places_and_explicit_lookups_still_route(text):
    assert requires_public_lookup(text)
