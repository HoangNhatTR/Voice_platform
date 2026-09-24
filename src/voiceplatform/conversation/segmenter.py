"""LLM token stream -> speakable phrases.

The whole point of the platform's latency budget lives here: synthesis must
start on the first phrase, not the first sentence and never the first
paragraph. The counterweight is that a phrase cut too short loses its
intonation — a talker's normaliser flattens "?" to "." on very short strings —
so a break needs both a boundary mark and a minimum length.
"""

from __future__ import annotations

import re

from .text import SpeechTextFilter, strip_emotion_cues

_BOUNDARY = re.compile(r"([.!?…]+)[\"')\]]*\s")
_SOFT_BOUNDARY = re.compile(r"[,;:]\s")


def pipeline_segmenter(tts_capabilities) -> "PhraseSegmenter":
    """Đúng bộ chia cụm mà đường nói dùng, không phải một bản giống nó.

    Tách ra vì bàn thử model cần chạy y hệt: chuẩn hoá văn bản và quyết định
    giữ hay bỏ cue cảm xúc là một nửa thứ người nghe thật sự nghe. Một bài thử
    tự dựng lại logic này sẽ đo một sản phẩm không tồn tại — và đó đúng là cách
    "[cười]" từng bị đọc thành chữ mà không ai thấy.
    """
    return PhraseSegmenter(
        max_chars=90,
        min_words=5,
        keep_emotion_cues=tts_capabilities.emotion_cues,
    )


class PhraseSegmenter:
    def __init__(
        self,
        max_chars: int = 90,
        min_words: int = 5,
        soft_min_chars: int = 45,
        keep_emotion_cues: bool = False,
        first_max_chars: int = 24,
        first_min_words: int = 2,
        first_soft_min_chars: int = 10,
    ) -> None:
        self.max_chars = max_chars
        self.min_words = min_words
        self.soft_min_chars = soft_min_chars
        self.keep_emotion_cues = keep_emotion_cues
        # The first phrase is measured separately because it *is* the TTFA:
        # nothing is heard until it is complete, synthesised and sent, and a
        # talker that renders a phrase in one go turns every extra word into
        # extra silence. Measured on VieNeu Nano (CPU, RTF 0.65): a 34-char
        # opener cost 2.36 s of synthesis, so the opener is kept short and the
        # later phrases carry the sentence. Below roughly two words it starts
        # to sound clipped, which is where this stops.
        self.first_max_chars = first_max_chars
        self.first_min_words = first_min_words
        self.first_soft_min_chars = first_soft_min_chars
        self._emitted = 0
        self._filter = SpeechTextFilter()
        self._buf = ""

    def push(self, delta: str) -> list[str]:
        text = self._filter.push(delta)
        if not text:
            return []
        self._buf += text
        return self._drain()

    def flush(self) -> list[str]:
        tail = self._filter.flush()
        if tail:
            self._buf += tail
        out = self._drain(final=True)
        rest = self._buf.strip()
        self._buf = ""
        if rest:
            out.append(self._clean(rest))
        return [p for p in out if p]

    def _drain(self, final: bool = False) -> list[str]:
        phrases: list[str] = []
        while True:
            cut = self._find_cut(self._buf)
            if cut is None:
                break
            phrase, self._buf = self._buf[:cut].strip(), self._buf[cut:].lstrip()
            if phrase:
                phrases.append(self._clean(phrase))
                self._emitted += 1
        return [p for p in phrases if p]

    def _find_cut(self, buf: str) -> int | None:
        if not buf:
            return None
        first = self._emitted == 0
        min_words = self.first_min_words if first else self.min_words
        soft_min = self.first_soft_min_chars if first else self.soft_min_chars
        max_chars = self.first_max_chars if first else self.max_chars

        for match in _BOUNDARY.finditer(buf):
            end = match.end()
            words = len(buf[:end].split())
            # The minimum-length rule is really about "?" and "!": a talker's
            # normaliser flattens them to a period on very short strings, so a
            # two-word question loses its intonation. A full stop carries no
            # such risk, so an opening "Chào bạn." may be cut immediately —
            # which is most of the first-audio latency on a slow talker.
            needs_length = "?" in match.group(1) or "!" in match.group(1)
            if words >= (self.min_words if needs_length else min_words):
                return end
        for match in _SOFT_BOUNDARY.finditer(buf):
            end = match.end()
            if end >= soft_min and len(buf[:end].split()) >= min_words:
                return end
        if len(buf) >= max_chars:
            # Hard cap: prefer the last space so a word is never split.
            space = buf.rfind(" ", 0, max_chars)
            return space + 1 if space > soft_min else max_chars
        return None

    def _clean(self, phrase: str) -> str:
        if not self.keep_emotion_cues:
            phrase = strip_emotion_cues(phrase)
        return " ".join(phrase.split())
