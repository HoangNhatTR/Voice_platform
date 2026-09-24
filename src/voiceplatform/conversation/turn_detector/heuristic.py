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

# ASR đọc số ra CHỮ, không ra chữ số: gipformer trả "không chín một" chứ không
# trả "091". Luật dãy số chỉ khớp \d nên nó chưa từng chạy trên tiếng nói thật
# — đo được 24/09: máy cắt lời ngay giữa một số tài khoản đang đọc dở.
# Dạng đã fold dấu, kèm các biến thể chỉ xuất hiện khi đọc số: mốt, tư, lăm,
# nhăm, lẻ, linh.
_SPOKEN_DIGITS = {
    "khong", "mot", "hai", "ba", "bon", "nam", "sau", "bay", "tam", "chin",
    "tu", "lam", "nham", "le", "linh", "bon", "muoi",
}
# Hai từ số liền nhau còn là lượng ("một năm", "hai giờ"); ba từ trở lên thì
# gần như chỉ có khi người ta đang ĐỌC một dãy số.
_SPOKEN_RUN_MIN = 3


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

        # Dãy số xét TRƯỚC từ nối, và khi đã chắc là dãy số thì nó quyết định
        # luôn. Bỏ dấu xong "sáu" thành "sau" và "tư" thành "tu" — cả hai đều
        # nằm trong danh sách từ nối — nên một số đã đọc XONG mà kết thúc bằng
        # sáu hay tư sẽ bị giữ thêm 920 ms vì nhầm sang luật khác.
        length, has_ascii = self._number_tail(words, stripped)
        if length and (has_ascii or length >= _SPOKEN_RUN_MIN):
            return (
                self.max_silence_ms
                if length < self.digit_tail_complete_at
                else self.silence_ms
            )

        if last in _TRAILING_CONNECTORS or last in _HESITATIONS:
            return self.max_silence_ms

        if len(words) <= self.opener_max_words:
            for opener in _OPENERS:
                if stripped.startswith(opener):
                    return self.max_silence_ms

        return self.silence_ms

    @staticmethod
    def _number_tail(words: list[str], stripped: str) -> tuple[int, bool]:
        """Độ dài dãy số ở CUỐI câu, và nó có chứa chữ số ASCII không.

        Đếm ngược từ cuối: một token toàn chữ số đóng góp số ký tự của nó, một
        từ đọc số đóng góp một. Gặp token không phải số thì dừng.
        """
        match = _DIGIT_RUN.search(stripped)
        ascii_digits = re.sub(r"\D", "", match.group(1)) if match else ""
        length = 0
        has_ascii = False
        for word in reversed(words):
            token = word.strip(".,")
            if token.isdigit():
                length += len(token)
                has_ascii = True
            elif token in _SPOKEN_DIGITS:
                length += 1
            else:
                break
        if ascii_digits and not has_ascii:
            return len(ascii_digits), True
        return length, has_ascii
