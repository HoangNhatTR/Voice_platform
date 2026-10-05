"""Vietnamese is spaced by syllable: a cut "at the last space" is a cut inside a word.

Every phrase is read by the talker as a sentence of its own, so before
02/10/2026 the 31% of joins cut without punctuation — "vì nó liên | quan",
"bầu trời có màu | xanh.", "viết | tắt" — were heard as hesitation. The
sentences below are real answers that were cut that way.
"""

from __future__ import annotations

import types

import pytest

from voiceplatform.conversation.segmenter import PhraseSegmenter, pipeline_segmenter

STREAMING = types.SimpleNamespace(streaming=True, emotion_cues=False)
ATOMIC = types.SimpleNamespace(streaming=False, emotion_cues=False)

REAL = [
    "Tôi không thể thực hiện yêu cầu này vì nó liên quan đến việc đọc hoặc thao tác trên "
    "tài khoản giả, có thể vi phạm chính sách bảo mật và an toàn.",
    "Khi đi qua khí quyển, ánh sáng màu xanh bị phân tán nhiều hơn các màu khác do hiện tượng "
    "tán xạ Rayleigh. Mắt người ta cảm nhận được màu xanh chiếm ưu thế nên bầu trời có màu xanh.",
    "Thành phố Hồ Chí Minh là viết tắt của Thành phố Hồ Chí Minh (không có từ viết tắt ngắn hơn "
    "cho tên riêng này trong ngữ cảnh hành chính).",
    "Tôi không thể thực hiện việc chuyển tiền hay giao dịch tài chính trực tiếp. Bạn vui lòng "
    "liên hệ ngân hàng hoặc người nhận để hoàn tất giao dịch.",
    "Lịch hẹn của bạn là ngày 30 tháng 9 năm 2026 lúc chín giờ sáng tại chi nhánh Hoàn Kiếm.",
]
BAD_CUTS = ["liên | quan", "có màu | xanh", "viết | tắt", "tiền hay | giao", "ngày 30 | tháng",
            "do hiện tượng | tán"]


def _segment(text: str, caps=STREAMING, timer_after: int | None = 10) -> list[str]:
    """Feed like the LLM does (a few chars per delta), firing the first-phrase timer early."""
    seg = pipeline_segmenter(caps, 48)
    out, fed = [], 0
    for i in range(0, len(text), 3):
        out += seg.push(text[i:i + 3])
        fed += 1
        if timer_after is not None and fed >= timer_after and seg.first_pending:
            out += seg.flush_first_boundary()
    return out + seg.flush()


@pytest.mark.parametrize("text", REAL)
def test_a_streaming_talker_never_gets_a_phrase_cut_inside_a_word(text):
    phrases = _segment(text)
    assert " ".join(phrases) == text
    joined = " | ".join(phrases)
    for bad in BAD_CUTS:
        assert bad not in joined, joined
    for a, b in zip(phrases, phrases[1:]):
        # Every join is punctuation, or right before a word that opens a clause.
        assert a[-1] in ".!?…,;:)" or b.split()[0] in {"vì", "và", "hay", "hoặc", "để", "nên", "khi", "mà"}, (a, b)


def test_the_timer_waits_for_a_clause_boundary_instead_of_a_space():
    seg = pipeline_segmenter(STREAMING, 48)
    # Past the 48-char cap, and the timer has fired: still no comma, no clause word.
    assert seg.push("Thành phố Hồ Chí Minh là viết tắt của Thành phố ") == []
    assert seg.flush_first_boundary() == []
    out = seg.push("Hồ Chí Minh và ") + seg.flush_first_boundary()
    assert out == ["Thành phố Hồ Chí Minh là viết tắt của Thành phố Hồ Chí Minh"]


def test_a_clause_word_still_being_written_is_not_a_boundary():
    seg = pipeline_segmenter(STREAMING, 48)
    assert seg.push("Tôi đã kiểm tra số dư của tài khoản đó và") == []   # "và" may become "vàng"
    assert seg.flush_first_boundary() == []
    seg.push("ng ")
    assert seg.flush_first_boundary() == []                           # it did
    seg2 = pipeline_segmenter(STREAMING, 48)
    seg2.push("Tôi đã kiểm tra số dư của tài khoản đó và")
    seg2.push(" ")
    assert seg2.flush_first_boundary() == ["Tôi đã kiểm tra số dư của tài khoản đó"]


@pytest.mark.parametrize("left,right", [("bởi", "vì"), ("trở", "nên"), ("đôi", "khi"), ("vẫn", "còn")])
def test_the_second_half_of_a_compound_is_not_a_clause_word(left, right):
    seg = pipeline_segmenter(STREAMING, 48)
    seg.push(f"Ở chi nhánh trung tâm giao dịch viên {left} {right} ")
    assert all(not p.endswith(left) for p in seg.flush_first_boundary())


def test_a_long_run_without_any_boundary_is_still_bounded():
    words = "một hai ba bốn năm sáu bảy tám chín mười ".split()
    text = " ".join((words * 40)[:120]) + "."
    phrases = _segment(text, timer_after=None)
    assert " ".join(phrases) == text
    assert max(len(p) for p in phrases) <= 260


def test_later_phrases_are_whole_sentences_when_the_llm_is_ahead():
    text = ("Ánh sáng mặt trời gồm nhiều màu, khi đi qua khí quyển thì ánh sáng xanh bị tán xạ nhiều hơn. "
            "Vì vậy bầu trời ban ngày có màu xanh. Lúc hoàng hôn, ánh sáng đi qua lớp khí dày hơn nên ngả sang cam đỏ.")
    phrases = _segment(text)
    assert phrases[1:] == ["khi đi qua khí quyển thì ánh sáng xanh bị tán xạ nhiều hơn.",
                           "Vì vậy bầu trời ban ngày có màu xanh.",
                           "Lúc hoàng hôn, ánh sáng đi qua lớp khí dày hơn nên ngả sang cam đỏ."]


def test_an_atomic_talker_keeps_its_short_opener_and_prefers_clause_words():
    seg = pipeline_segmenter(ATOMIC, 48)
    out = seg.push("Tôi không thể thực hiện việc chuyển tiền hay giao dịch tài chính trực tiếp ")
    assert out and len(out[0]) <= 30
    # Inside the cap a clause word wins over the last space.
    plain = PhraseSegmenter(max_chars=40, min_words=3, first_max_chars=40)
    assert plain.push("Tôi không thể làm việc này vì quy định của ngân hàng không cho phép ") == [
        "Tôi không thể làm việc này"]
