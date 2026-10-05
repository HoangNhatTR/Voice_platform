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


@pytest.mark.parametrize(
    "text",
    [
        "bây giờ là mấy giờ rồi",   # "rồi" cuối câu là trợ từ, không phải từ nối
        "tôi chuyển xong rồi",
        "hãy gọi lại cho tôi",      # "tôi" không phải "tới"
        "gửi cho mẹ tôi",
        "đau bụng quá má",          # "má" không phải "mà"
    ],
)
async def test_a_finished_sentence_is_not_held_by_a_folding_collision(text):
    detector = HeuristicTurnDetector(silence_ms=BASE, max_silence_ms=MAX)
    assert await detector.required_silence_ms(text=text, utterance_ms=1200) == BASE


@pytest.mark.parametrize("text", ["tôi muốn chuyển tiền tới", "chuyen tien toi"])
async def test_a_real_dangling_word_still_holds_with_or_without_diacritics(text):
    detector = HeuristicTurnDetector(silence_ms=BASE, max_silence_ms=MAX)
    assert await detector.required_silence_ms(text=text, utterance_ms=1200) == MAX


# ---------------------------------------------------------------- G3 (dev set 29/09)

@pytest.mark.parametrize("text", [
    "nếu tôi gửi tiết kiệm",           # a condition with no consequence yet
    "khi tôi chuyển tiền ra nước ngoài",
    "vì tôi làm mất thẻ",
])
async def test_a_subordinate_clause_without_its_main_clause_holds(detector, text):
    assert await detector.required_silence_ms(text=text, utterance_ms=1500) == detector.max_silence_ms


@pytest.mark.parametrize("text", [
    "nếu tôi gửi tiết kiệm thì bao lâu mới được rút",
    "khi nào ngân hàng mở cửa",        # a question, not a clause
    "vì sao lãi suất tăng",
])
async def test_a_finished_clause_or_a_question_does_not(detector, text):
    assert await detector.required_silence_ms(text=text, utterance_ms=1500) == detector.silence_ms


async def test_two_digits_after_a_number_cue_are_a_number_being_read(detector):
    assert await detector.required_silence_ms(text="mã khách hàng là bốn năm", utterance_ms=1500) == detector.max_silence_ms
    # Without a cue two number words are still a quantity.
    assert await detector.required_silence_ms(text="tôi đi hai năm", utterance_ms=1500) == detector.silence_ms


@pytest.mark.parametrize("text", ["bạn ơi", "màn ơi", "mình tự hỏi", "tôi muốn biết"])
async def test_a_vocative_or_an_object_less_verb_holds(detector, text):
    assert await detector.required_silence_ms(text=text, utterance_ms=900) == detector.max_silence_ms


def test_the_fast_tier_needs_a_stable_transcript_and_a_completion_cue():
    h = HeuristicTurnDetector(480, 1400, fast_silence_ms=300)
    assert h.evaluate("bây giờ là mấy giờ rồi") == 480
    assert h.evaluate("bây giờ là mấy giờ rồi", stable=True) == 300
    assert h.evaluate("tôi muốn khoá thẻ", stable=True) == 940       # opener + request: middle wait
    assert h.evaluate("kiểm tra số dư giúp tôi", stable=True) == 480  # no cue: base
    assert h.evaluate("số tài khoản là một hai ba bốn năm sáu bảy tám chín không", stable=True) == 480
    assert h.evaluate("khong biet khong", stable=True) == 480           # no diacritics: no fast path
    assert HeuristicTurnDetector(480, 1400).evaluate("bạn tên là gì", stable=True) == 480  # off


def test_an_opener_with_its_request_waits_less_only_on_a_stable_transcript():
    h = HeuristicTurnDetector(480, 1400)
    assert h.evaluate("tôi muốn khoá thẻ") == 1400
    assert h.evaluate("tôi muốn khoá thẻ", stable=True) == 940
    assert h.evaluate("tôi muốn", stable=True) == 1400          # the request has not come
    assert h.evaluate("cho tôi hỏi", stable=True) == 1400
    assert h.evaluate("tôi muốn chuyển tiền cho", stable=True) == 1400   # connector wins


@pytest.mark.parametrize("text", ["kiểm tra thẻ", "tôi mất thẻ", "khoá thẻ"])
async def test_a_card_is_not_a_hesitation(detector, text):
    # "thẻ" folds to "the", which is the hesitation "thế".
    assert await detector.required_silence_ms(text=text, utterance_ms=900) == detector.silence_ms


@pytest.mark.parametrize("text", ["thế", "à", "tôi muốn hỏi ờ", "ừm"])
async def test_a_real_hesitation_still_holds(detector, text):
    assert await detector.required_silence_ms(text=text, utterance_ms=900) == detector.max_silence_ms


async def test_into_is_a_connector_and_wins_over_the_opener_middle_wait():
    h = HeuristicTurnDetector(480, 1400)
    assert h.evaluate("tôi muốn đặt lịch hẹn vào", stable=True) == 1400
    assert h.evaluate("tôi nghĩ", stable=True) == 1400


@pytest.mark.parametrize("text, wait", [
    ("tài khoản của tôi có", 1400), ("có", 480), ("tôi không có", 480),
    ("tôi đang ở", 1400), ("số thẻ của tôi bắt đầu bằng bốn", 1400), ("hai cộng hai bằng bốn", 480),
    ("tôi cần mở", 1400), ("thẻ đã bị khoá", 480), ("tiền đã được chuyển", 480),
])
def test_object_verbs_places_and_a_first_digit(text, wait):
    assert HeuristicTurnDetector(480, 1400).evaluate(text, stable=True) == wait


@pytest.mark.parametrize("text", ["tôi cần tư vấn về khoản vay", "hủy lệnh chuyển", "kiểm tra khoản vay"])
def test_a_verb_used_as_a_noun_does_not_hold(text):
    assert HeuristicTurnDetector(480, 1400).evaluate(text, stable=True) == 480
