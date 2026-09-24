"""Vietnamese endpointing heuristics."""

from __future__ import annotations

import pytest

from voiceplatform.conversation.turn_detector import (
    HeuristicTurnDetector,
    SemanticTurnDetector,
    VadOnlyTurnDetector,
)

BASE, MAX = 400.0, 1400.0


@pytest.fixture
def detector() -> HeuristicTurnDetector:
    return HeuristicTurnDetector(silence_ms=BASE, max_silence_ms=MAX)


async def test_finished_question_ends_on_the_base_pause(detector):
    assert await detector.required_silence_ms(text="Số dư của tôi là bao nhiêu?", utterance_ms=1200) == BASE


async def test_opener_holds_the_turn_open(detector):
    assert await detector.required_silence_ms(text="Cho tôi hỏi", utterance_ms=500) == MAX


async def test_dangling_connector_holds_the_turn_open(detector):
    assert await detector.required_silence_ms(text="Tôi muốn chuyển tiền và", utterance_ms=900) == MAX


async def test_unfinished_account_number_holds_the_turn_open(detector):
    # Someone reading a 14-digit account pauses between groups.
    assert await detector.required_silence_ms(text="chuyển tới số 1903 2688", utterance_ms=1500) == MAX


async def test_complete_account_number_does_not(detector):
    assert await detector.required_silence_ms(
        text="chuyển tới số 19032688123456", utterance_ms=2000
    ) == BASE


async def test_vad_only_ignores_the_words():
    d = VadOnlyTurnDetector(silence_ms=BASE)
    assert await d.required_silence_ms(text="Cho tôi hỏi", utterance_ms=500) == BASE


async def test_semantic_falls_back_when_the_probe_fails():
    async def broken(text: str) -> float:
        raise RuntimeError("model down")

    d = SemanticTurnDetector(probe=broken, silence_ms=BASE, max_silence_ms=MAX)
    assert await d.required_silence_ms(text="Cho tôi hỏi", utterance_ms=500) == MAX
    assert d.probe_failures == 1


async def test_semantic_confident_completion_shortens_the_wait():
    async def confident(text: str) -> float:
        return 0.95

    d = SemanticTurnDetector(probe=confident, silence_ms=BASE, max_silence_ms=MAX)
    assert await d.required_silence_ms(text="Cho tôi hỏi", utterance_ms=500) == BASE


# --- ASR đọc số ra CHỮ, không ra chữ số ------------------------------------

async def test_a_spoken_account_number_holds_the_turn_open(detector):
    """gipformer trả "không chín một", không trả "091".

    Luật cũ chỉ khớp \\d nên nó chưa từng chạy trên tiếng nói thật — đo được
    trên phiên thật: máy cắt lời ngay giữa một số tài khoản đang đọc dở.
    """
    assert await detector.required_silence_ms(
        text="số tài khoản của tôi là không chín một", utterance_ms=2000
    ) == MAX
    assert await detector.required_silence_ms(
        text="thẻ của tôi số bốn ba hai một", utterance_ms=2000
    ) == MAX


async def test_a_quantity_is_not_mistaken_for_a_number_being_read(detector):
    """Hai từ số liền nhau còn là lượng; ba từ trở lên mới gần như chắc là dãy số."""
    for finished in (
        "tôi muốn gửi tiết kiệm kỳ hạn một năm",
        "chuyển cho tôi năm triệu đồng",
        "hẹn tôi lúc hai giờ",
    ):
        assert await detector.required_silence_ms(
            text=finished, utterance_ms=2000
        ) == BASE, finished


async def test_ascii_digits_still_hold_from_the_first_one(detector):
    """Một ký tự số cũng đủ để ngờ là dãy số đang dở, khác với chữ đọc thành lời."""
    assert await detector.required_silence_ms(
        text="số của tôi là 091", utterance_ms=2000
    ) == MAX


async def test_a_complete_number_does_not_hold(detector):
    assert await detector.required_silence_ms(
        text="số tài khoản là không chín một hai ba bốn năm sáu bảy tám",
        utterance_ms=4000,
    ) == BASE


async def test_a_finished_number_ending_in_six_is_not_held_by_the_wrong_rule(detector):
    """Bỏ dấu xong "sáu" thành "sau", "tư" thành "tu" — cả hai là từ nối.

    Một số đã đọc xong mà kết thúc bằng hai chữ đó từng bị giữ thêm 920 ms vì
    luật từ nối chạy trước và không biết nó đang nhìn một dãy số.
    """
    # Chín chữ số: đủ dài để coi là đã đọc xong, và kết thúc bằng "sáu".
    assert await detector.required_silence_ms(
        text="số tài khoản là không chín một hai ba bốn năm bảy sáu", utterance_ms=4000
    ) == BASE
    assert await detector.required_silence_ms(
        text="số điện thoại không chín tám bảy sáu năm bốn ba tư", utterance_ms=4000
    ) == BASE


async def test_after_still_holds_when_it_really_is_a_connector(detector):
    """"sau" thật sự là từ nối thì vẫn phải giữ lượt mở."""
    assert await detector.required_silence_ms(
        text="tôi muốn chuyển tiền sau", utterance_ms=2000
    ) == MAX
