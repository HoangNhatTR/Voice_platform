"""Vietnamese lexical turn detection.

No model, no GPU, no added latency: it looks at the tail of the partial
transcript and holds the turn open when the sentence is obviously unfinished.
Three families of cue, each earning its place:

* dangling function words — "và", "thì", "để", "nếu"... a turn never ends there;
* opener phrases — "cho tôi hỏi", "tôi muốn"... the request has not arrived yet;
* an unfinished digit run — someone reading out an account or card number
  pauses between groups, and cutting them off mid-number is the single most
  expensive endpointing error in a banking assistant.

Everything is bounded by max_silence_ms, so a wrong guess costs a pause, never
a hang.
"""

from __future__ import annotations

import re
import unicodedata

_TRAILING_CONNECTORS = {
    "va", "voi", "thi", "la", "cho", "cua", "de", "roi", "nhung", "ma", "neu",
    "khi", "vi", "do", "tai", "tu", "den", "toi", "ve", "theo", "bang",
    "trong", "ngoai", "tren", "duoi", "sau", "truoc", "hoac", "hay", "con",
    "nen", "boi", "rang", "sang", "qua", "gom", "ke", "cung",
}
_HESITATIONS = {"a", "u", "o", "um", "uh", "e", "hm", "hmm", "uhm", "the"}
_OPENERS = (
    "cho toi hoi", "cho minh hoi", "toi muon", "minh muon", "toi can",
    "minh can", "lam on", "ban oi", "cho toi", "cho minh", "toi hoi",
    "em oi", "anh oi", "chi oi", "toi dinh", "minh dinh",
)
_DIGIT_RUN = re.compile(r"(\d[\d\s.]*)$")


def _fold(text: str) -> str:
    text = unicodedata.normalize("NFD", text.lower())
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")
    return text.replace("đ", "d")


class HeuristicTurnDetector:
    name = "heuristic"

    def __init__(
        self,
        silence_ms: float = 480.0,
        max_silence_ms: float = 1400.0,
        opener_max_words: int = 6,
        digit_tail_complete_at: int = 9,
    ) -> None:
        self.silence_ms = silence_ms
        self.max_silence_ms = max_silence_ms
        self.opener_max_words = opener_max_words
        # Below this many digits a numeric tail reads as unfinished. Vietnamese
        # account numbers run 9-16 digits and card numbers 16.
        self.digit_tail_complete_at = digit_tail_complete_at

    async def required_silence_ms(self, *, text: str, utterance_ms: float) -> float:
        return self.evaluate(text)

    def evaluate(self, text: str) -> float:
        clean = (text or "").strip()
        if not clean:
            return self.silence_ms
        folded = _fold(clean)
        stripped = folded.rstrip(" .,!?…")

        # A finished sentence mark is the strongest "done" signal there is.
        if clean.rstrip().endswith(("?", ".", "!")):
            return self.silence_ms

        words = stripped.split()
        if not words:
            return self.silence_ms
        last = words[-1].strip(".,")

        if last in _TRAILING_CONNECTORS or last in _HESITATIONS:
            return self.max_silence_ms

        match = _DIGIT_RUN.search(stripped)
        if match:
            digits = re.sub(r"\D", "", match.group(1))
            if 0 < len(digits) < self.digit_tail_complete_at:
                return self.max_silence_ms

        if len(words) <= self.opener_max_words:
            for opener in _OPENERS:
                if stripped.startswith(opener):
                    return self.max_silence_ms

        return self.silence_ms
