from __future__ import annotations

from voiceplatform.conversation.segmenter import PhraseSegmenter


def test_phrase_is_emitted_on_a_sentence_end():
    seg = PhraseSegmenter(min_words=3, first_min_words=99)  # no opener shortcut
    out = []
    for chunk in ["Chào bạn, ", "tôi nghe đây. ", "Bạn cần gì?"]:
        out += seg.push(chunk)
    assert out and out[0].endswith(".")


def test_the_opening_phrase_is_cut_as_early_as_it_can_be():
    """It is the whole of time-to-first-audio on a talker that batches."""
    seg = PhraseSegmenter(min_words=5)
    out = seg.push("Chào bạn. Tôi có thể giúp gì cho bạn hôm nay? ")
    assert out and out[0] == "Chào bạn."      # two words, released immediately
    assert len(" ".join(out[1:])) > 0         # the rest still follows


def test_a_short_question_is_not_cut_off_its_mark():
    """Talker normalisers flatten '?' on very short strings, so hold them.

    This holds even for the opening phrase, where everything else is relaxed
    to buy latency: a question that arrives as a statement is a worse trade
    than 200 ms.
    """
    seg = PhraseSegmenter(min_words=5)
    assert seg.push("Sao vậy? ") == []
    assert "Sao vậy?" in " ".join(seg.flush())


def test_long_run_without_punctuation_is_cut_at_a_space():
    seg = PhraseSegmenter(max_chars=40, min_words=3)
    out = seg.push("một hai ba bốn năm sáu bảy tám chín mười mười một mười hai ")
    assert out
    assert not out[0].endswith("mư")  # never split inside a word


def test_flush_returns_the_tail():
    seg = PhraseSegmenter()
    seg.push("Còn lại chút xíu")
    assert " ".join(seg.flush()) == "Còn lại chút xíu"
