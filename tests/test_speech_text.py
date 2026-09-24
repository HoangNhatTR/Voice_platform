"""Streaming markdown: the filter has to work before the construct closes."""

from __future__ import annotations

from voiceplatform.conversation.segmenter import PhraseSegmenter
from voiceplatform.conversation.text import (
    SpeechTextFilter,
    strip_emotion_cues,
    strip_markdown,
)


def test_whole_text_link_becomes_its_label():
    assert strip_markdown("Xem [Hồ Gươm](https://x.vn/a) nhé") == "Xem Hồ Gươm nhé"


def test_streaming_link_is_held_back_until_it_closes():
    f = SpeechTextFilter()
    out = f.push("Xem [Hồ ")
    assert "[" not in out and "Hồ" not in out       # nothing leaks mid-link
    out += f.push("Gươm](https://x.vn/a)")
    out += f.push(" nhé.")
    out += f.flush()
    assert out.strip() == "Xem Hồ Gươm nhé."


def test_streaming_bold_is_never_spoken_as_stars():
    f = SpeechTextFilter()
    text = "".join(f.push(part) for part in ["Giá là ", "**", "hai trăm", "**", " đồng."])
    text += f.flush()
    assert "*" not in text
    assert "hai trăm" in text


def test_unclosed_markup_is_released_rather_than_held_forever():
    f = SpeechTextFilter(max_hold_chars=20)
    out = "".join(f.push("[" + "x" * 30))
    assert out  # a truncated reply must not silence the turn


def test_emotion_cue_stripping_is_opt_in_per_engine():
    assert strip_emotion_cues("Vui quá [cười] bạn ạ") == "Vui quá bạn ạ"
    kept = PhraseSegmenter(keep_emotion_cues=True)
    kept.push("Vui quá [cười] bạn ạ. ")
    assert "[cười]" in " ".join(kept.flush())
