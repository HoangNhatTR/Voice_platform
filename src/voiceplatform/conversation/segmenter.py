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
_QUESTION_OPENING = re.compile(r"^(?:bạn có|bạn muốn|bạn cần|có thể|tại sao|vì sao|làm sao|khi nào|bao giờ|ở đâu|mấy|ai\b|sao\b)", re.I)
_HOLD_AFTER = {"ngày", "tháng", "năm", "số", "mã", "tài", "khoản"}
_HOLD_BEFORE = {"ngày", "số", "mã"}
# Vietnamese puts a space between SYLLABLES: "liên quan", "khí quyển", "giao
# dịch" are one word each, so "cut at the last space" lands inside a word
# about as often as between two. The talker reads every phrase as a sentence
# of its own — a final pitch fall, then a pause — and measured 02/10/2026,
# 31% of phrase joins had no punctuation ("vì nó liên | quan", "bầu trời có
# màu | xanh."): heard as hesitation. Without punctuation a phrase may end only
# right before a word that opens a clause...
_CLAUSE_START = {
    "và", "nhưng", "vì", "nên", "để", "nếu", "khi", "hoặc", "hay", "mà",
    "thì", "còn", "rồi", "tuy", "song", "nhờ", "bởi",
}
# ...and not where that word is the second half of a compound: "bởi | vì",
# "trở | nên", "đôi | khi", "vẫn | còn" would be the same mid-word cut.
_COMPOUND_LEFT = {
    "vì": {"bởi", "tại", "là"}, "nên": {"trở", "cho", "thế", "vậy"},
    "khi": {"đôi", "có", "những", "mỗi", "trước", "sau", "một"},
    "còn": {"vẫn", "hãy", "đang"}, "rồi": {"xong", "đã", "thôi"},
    "mà": {"nhưng", "thế", "vậy", "chứ"}, "hay": {"rất", "thật", "cũng", "hoặc"},
    "để": {"cho"}, "bởi": {"là"}, "thì": {"còn"},
}


def pipeline_segmenter(tts_capabilities, streaming_first_chars: int = 48) -> "PhraseSegmenter":
    """Đúng bộ chia cụm mà đường nói dùng, không phải một bản giống nó.

    Tách ra vì bàn thử model cần chạy y hệt: chuẩn hoá văn bản và quyết định
    giữ hay bỏ cue cảm xúc là một nửa thứ người nghe thật sự nghe. Một bài thử
    tự dựng lại logic này sẽ đo một sản phẩm không tồn tại — và đó đúng là cách
    "[cười]" từng bị đọc thành chữ mà không ai thấy.
    """
    if tts_capabilities.streaming:
        # A streaming talker's first audio does not grow with the phrase:
        # ZeroTTS measured 112 ms for 21 chars, 150 ms for 183 (02/10/2026).
        # The LLM writes ~130 chars/s, 7.6x faster than speech, so the next
        # sentence is ready seconds before this one is heard: later phrases
        # end at a sentence, or at a comma/clause word once already long.
        # Waiting for punctuation instead of a space cost the opener p50 0 ms,
        # p90 +306 ms on 269 recorded answers.
        return PhraseSegmenter(
            max_chars=160,
            min_words=5,
            soft_min_chars=90,
            keep_emotion_cues=tts_capabilities.emotion_cues,
            first_max_chars=streaming_first_chars,
            first_soft_min_chars=24 if streaming_first_chars > 24 else 10,
            first_min_words=4 if streaming_first_chars > 24 else 2,
            space_cuts=False,
            hard_max_chars=260,
            first_hard_max_chars=160,
        )
    # An atomic talker renders the whole phrase before its first sample, so
    # here a short opener IS the first-audio latency: keep the 24-char cap and
    # allow a space cut when no clause boundary fits, clause words first.
    return PhraseSegmenter(
        max_chars=90,
        min_words=5,
        keep_emotion_cues=tts_capabilities.emotion_cues,
        first_max_chars=24,
        first_soft_min_chars=10,
        first_min_words=2,
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
        space_cuts: bool = True,
        hard_max_chars: int | None = None,
        first_hard_max_chars: int | None = None,
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
        # Past the soft caps a phrase ends at a clause word (see _CLAUSE_START).
        # `space_cuts` also lets it end at any safe space there; without it a
        # plain space is used only past the hard cap, so one comma-less run of
        # 260 chars still cannot hold the talker forever.
        self.space_cuts = space_cuts
        self.hard_max_chars = hard_max_chars if hard_max_chars is not None else max_chars
        self.first_hard_max_chars = (first_hard_max_chars if first_hard_max_chars is not None
                                     else first_max_chars)
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

    @property
    def first_pending(self) -> bool:
        return self._emitted == 0

    def flush_first_boundary(self) -> list[str]:
        """Deadline is a request to flush, never permission to split a token.

        Keep a trailing incomplete word, adjacent name tokens and grouped
        digits together. If no safe four-word boundary exists, wait for more
        text or end-of-stream. Very short questions therefore retain '?'.
        """
        if not self.first_pending:
            return []
        if len(self._buf) < self.first_max_chars and _QUESTION_OPENING.match(self._buf.strip()):
            return []
        fits = lambda p: (p >= self.first_soft_min_chars
                          and len(self._buf[:p].split()) >= self.first_min_words)
        clause = [p for p in self._clause_spaces(self._buf) if fits(p)]
        spaces = [p for p in self._safe_spaces(self._buf) if fits(p)] if self.space_cuts else []
        if not clause and not spaces:
            return []
        cut = clause[-1] if clause else spaces[-1]
        phrase, self._buf = self._buf[:cut].strip(), self._buf[cut:].lstrip()
        self._emitted += 1
        return [self._clean(phrase)]

    @staticmethod
    def _safe_spaces(buf: str) -> list[int]:
        out = []
        for match in re.finditer(r"\s+", buf):
            before, after = buf[:match.start()].split(), buf[match.end():].split()
            # A trailing space doesn't tell us whether a name/number follows.
            if not before or not after:
                continue
            left, right = before[-1].strip('.,;:!?'), after[0].strip('.,;:!?')
            if left and right:
                if left.casefold() in _HOLD_AFTER or right.casefold() in _HOLD_BEFORE:
                    continue
                if left[0].isupper() and right[0].isupper():
                    continue
                if left[-1].isdigit() and (right[0].isdigit() or right in ('đồng','kg','km','mét','triệu','nghìn')):
                    continue
            out.append(match.end())
        return out

    @classmethod
    def _clause_spaces(cls, buf: str) -> list[int]:
        """Safe spaces after , ; : or right before a clause word (see _CLAUSE_START)."""
        out = []
        for p in cls._safe_spaces(buf):
            left_raw = buf[:p].split()[-1]
            if left_raw[-1:] in ",;:":
                out.append(p)
                continue
            # Only a finished word: "và" may still become "vàng".
            word = re.match(r"(\S+)\s", buf[p:])
            if word is None:
                continue
            right = word.group(1).strip(".,;:!?\"'()").casefold()
            left = left_raw.strip(".,;:!?\"'()").casefold()
            if right in _CLAUSE_START and left not in _COMPOUND_LEFT.get(right, ()):
                out.append(p)
        return out

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
            if words >= (self.min_words if needs_length else min(min_words, self.min_words)):
                return end
        for match in _SOFT_BOUNDARY.finditer(buf):
            end = match.end()
            if end >= soft_min and len(buf[:end].split()) >= min_words:
                return end
        if len(buf) >= max_chars:
            fits = lambda p: p >= soft_min and len(buf[:p].split()) >= min_words
            # Past the cap: a clause word inside it, else (atomic talkers) any
            # safe space inside it, else the first clause word after it. A
            # plain space only once past the hard cap — "the last space" is
            # a syllable boundary, inside a word as often as not.
            clause = [p for p in self._clause_spaces(buf) if fits(p)]
            within = [p for p in clause if p <= max_chars]
            if within:
                return within[-1]
            if self.space_cuts:
                spaces = [p for p in self._safe_spaces(buf) if fits(p) and p <= max_chars]
                if spaces:
                    return spaces[-1]
            hard = self.first_hard_max_chars if first else self.hard_max_chars
            beyond = [p for p in clause if p <= hard]
            if beyond:
                return beyond[0]
            if len(buf) >= hard:
                spaces = [p for p in self._safe_spaces(buf) if fits(p) and p <= hard]
                # A long code/name is allowed to exceed the cap. The former
                # fallback at exactly max_chars silently cut inside such tokens.
                return spaces[-1] if spaces else None
        return None

    def _clean(self, phrase: str) -> str:
        if not self.keep_emotion_cues:
            phrase = strip_emotion_cues(phrase)
        return " ".join(phrase.split())
