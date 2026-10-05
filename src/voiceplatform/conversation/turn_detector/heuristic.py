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

# So CÓ DẤU khi ASR trả có dấu (gipformer luôn trả có dấu). So sau khi bỏ dấu
# thì "tôi" trùng "tới", "mà" trùng "má", "còn" trùng "cơn" — "giúp tôi" là
# câu đã trọn mà bị giữ thêm 920 ms. Bản bỏ dấu chỉ dùng khi câu vào không có
# dấu nào (gõ tay, ASR khác).
#
# KHÔNG có "rồi": cuối câu nó là trợ từ hoàn thành ("mấy giờ rồi", "xong rồi"),
# không phải "rồi thì...". Đo 25/09 trên stack thật: "Bây giờ là mấy giờ rồi?"
# bị giữ 1400 ms thay vì 480 ms, ở đúng câu hỏi hay gặp nhất.
_TRAILING_CONNECTORS = {
    "và", "với", "thì", "là", "cho", "của", "để", "nhưng", "mà", "nếu",
    "khi", "vì", "do", "tại", "từ", "đến", "tới", "về", "theo", "bằng",
    "trong", "ngoài", "trên", "dưới", "sau", "trước", "hoặc", "hay", "còn",
    "nên", "bởi", "rằng", "sang", "qua", "gồm", "kể", "cùng", "vào", "ở",
}
_TRAILING_CONNECTORS_FOLDED = {
    "va", "voi", "thi", "la", "cho", "cua", "de", "nhung", "ma", "neu",
    "khi", "vi", "do", "tai", "tu", "den", "toi", "ve", "theo", "bang",
    "trong", "ngoai", "tren", "duoi", "sau", "truoc", "hoac", "hay", "con",
    "nen", "boi", "rang", "sang", "qua", "gom", "ke", "cung", "vao",
}
_HESITATIONS = {"a", "u", "o", "um", "uh", "e", "hm", "hmm", "uhm", "the"}
# Có dấu: bỏ dấu thì "thẻ" (cái thẻ) = "the" = "thế" (ngập ngừng) — mọi câu
# kết thúc bằng "khoá thẻ", "mở thẻ" bị giữ 1400 ms. Đo trên bộ dev G3 29/09:
# "Tôi muốn khoá thẻ." chờ 1,38 s ở cả năm giọng.
_HESITATIONS_MARKED = {"à", "ừ", "ờ", "ừm", "ờm", "ơ", "ư", "e", "ê", "hừm", "hm", "hmm", "um", "uh", "uhm", "thế", "a", "o", "u"}
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
# ...trừ khi câu đã nói mình đang đọc gì: "mã khách hàng là bốn năm" là một mã
# đang đọc dở, không phải "bốn năm". Bộ dev G3 29/09: 5/5 lần cắt ở câu này.
_NUMBER_CUES = ("so", "ma", "tai khoan", "dien thoai", "the", "stk", "pin", "otp", "sdt")
_NUMBER_CUE_RUN_MIN = 2

# Các lượt G3 (bộ dev 29/09) bị cắt vì câu CHƯA có vế chính:
# * mệnh đề phụ đứng đầu mà chưa có "thì/nên/nhưng": "nếu tôi gửi tiết kiệm…"
#   — 5/5 lần cắt;
# * gọi người ở cuối câu: "bạn ơi…", kể cả khi ASR nghe thành "màn ơi";
# * động từ còn thiếu tân ngữ: "mình cần hỏi…", ASR nghe "mình tự hỏi".
_SUBORDINATE_OPENERS = (
    "neu", "neu nhu", "gia su", "khi", "trong khi", "sau khi", "truoc khi",
    "mac du", "tuy", "vi", "boi vi",
)
# "khi nào", "vì sao" là câu HỎI, không phải mệnh đề phụ: "khi nào ngân hàng
# mở cửa" đã trọn.
_INTERROGATIVE_SECOND = {"nao", "sao", "dau", "gi", "the"}
_MAIN_CLAUSE_MARKS = {"thi", "nen", "nhung", "ma", "vay", "the"}
_MAIN_CLAUSE_MARKS_MARKED = {"thì", "nên", "nhưng", "mà", "vậy", "thế"}   # "thẻ" ≠ "thế"
_SUBORDINATE_MAX_WORDS = 10
_TRAILING_HOLDS = {"ơi", "hỏi", "biết", "nhờ", "muốn", "cần", "định", "nghĩ"}
_TRAILING_HOLDS_FOLDED = {"oi", "hoi", "biet", "nho", "muon", "can", "dinh", "nghi"}
# Bộ confirm G3 29/09 (sau khi bộ đó đã dùng để phát hiện lỗi — không còn là
# bộ nghiệm thu): "tài khoản của tôi có…", "tôi cần mở…", "bắt đầu bằng bốn…".
# Động từ nghiệp vụ còn thiếu tân ngữ, trừ khi là bị động/đã xong ("thẻ đã bị
# khoá", "tiền đã được chuyển" là câu trọn).
_OBJECT_VERBS = {"có", "mở", "gửi", "chuyển", "rút", "đổi", "nạp", "vay", "trả", "mua", "xem",
                 "huỷ", "hủy", "khoá", "khóa"}
_DONE_BEFORE_VERB = {"bị", "được", "đã", "rồi", "không", "chưa"}
# ...hoặc đó là DANH TỪ: "khoản vay", "lệnh chuyển", "phí rút", "tiền gửi".
# Đo trên lượt v2 29/09: "Tôi cần tư vấn về khoản vay." bị giữ 1,38 s ở 5/5 giọng.
_NOUN_BEFORE_VERB = {"khoản", "lệnh", "phí", "sổ", "tiền", "hạn", "gói", "hồ", "giấy", "mã", "cây"}
_DIGIT_AFTER = {"bang", "la", "so", "ma"}   # "bắt đầu bằng bốn", "mã là ba"

# Tín hiệu câu ĐÃ TRỌN, so có dấu: trợ từ cuối câu hỏi / câu cầu khiến và vài
# câu khép lại. Chỉ dùng để rút ngắn khoảng chờ khi transcript phủ hết lời
# nói (stable) — trên partial cũ, từ cuối nhìn thấy chưa chắc là từ cuối.
# Không có "đi": "tôi muốn đi" + ngừng + "Đà Lạt" là câu chưa xong.
_COMPLETION_FINALS = {
    "không", "chưa", "à", "ạ", "ư", "nhỉ", "nhé", "nha", "hả", "hở", "chứ",
    "vậy", "thế", "sao", "gì", "đâu", "nào", "mấy", "nhiêu", "rồi",
}
_COMPLETION_PHRASES = (
    "cảm ơn", "cám ơn", "cảm ơn bạn", "cảm ơn nhé", "tạm biệt", "thế thôi",
    "vậy thôi", "hết rồi", "được rồi",
)


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
        fast_silence_ms: float = 0.0,
    ) -> None:
        self.silence_ms = silence_ms
        self.max_silence_ms = max_silence_ms
        # 0 = no confident-completion tier (the pre-G3 behaviour).
        self.fast_silence_ms = fast_silence_ms
        self.opener_max_words = opener_max_words
        # Below this many digits a numeric tail reads as unfinished. Vietnamese
        # account numbers run 9-16 digits and card numbers 16.
        self.digit_tail_complete_at = digit_tail_complete_at

    async def required_silence_ms(self, *, text: str, utterance_ms: float, stable: bool = False) -> float:
        return self.evaluate(text, stable=stable)

    def evaluate(self, text: str, *, stable: bool = False) -> float:
        wait = self._evaluate(text, stable=stable)
        if wait == self.silence_ms and stable and self.fast_silence_ms and self._completed(text):
            return self.fast_silence_ms
        return wait

    def _completed(self, text: str) -> bool:
        clean = unicodedata.normalize("NFC", (text or "").lower()).strip().rstrip(" .,!?…")
        # Bỏ dấu hết thì "không" = "khong", "à" = "a": không phân biệt được
        # trợ từ với tiếng ngập ngừng, nên câu không dấu không đi đường nhanh.
        if not clean or _fold(clean) == clean:
            return False
        words = [w.strip(".,!?…") for w in clean.split()]
        if len(words) < 2:
            return False    # "à", "ừ" một mình là tiếng đệm, không phải câu
        # "... một hai không" là số 0 cuối một dãy, không phải "có ... không":
        # dãy số không bao giờ đi đường nhanh, dù đã đủ độ dài.
        folded = [_fold(w) for w in words]
        if self._number_tail(folded, " ".join(folded))[0] >= 2:
            return False
        if words[-1] in _COMPLETION_FINALS:
            return True
        tail = " ".join(words[-3:])
        return any(tail.endswith(phrase) for phrase in _COMPLETION_PHRASES)

    def _evaluate(self, text: str, *, stable: bool = False) -> float:
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
        cued = (length >= _NUMBER_CUE_RUN_MIN or (
            length >= 1 and len(words) > length and words[-length - 1].strip(".,") in _DIGIT_AFTER
        )) and self._number_cued(words, length)
        if length and (has_ascii or length >= _SPOKEN_RUN_MIN or cued):
            return (
                self.max_silence_ms
                if length < self.digit_tail_complete_at
                else self.silence_ms
            )

        if self._is_hesitation(clean, last) or self._is_connector(clean, last) or self._trailing_hold(clean, last):
            return self.max_silence_ms

        if self._dangling_subordinate(words, clean):
            return self.max_silence_ms

        if len(words) <= self.opener_max_words:
            for opener in _OPENERS:
                if stripped.startswith(opener):
                    # On a transcript of ALL the speech, "tôi muốn khoá thẻ"
                    # has its request: two words after the opener. It is
                    # probably done, maybe not ("… chuyển tiền" + "cho mẹ"),
                    # so wait between the two. Bộ dev G3: "Tôi muốn khoá thẻ."
                    # was held 1.38 s on every one of five voices.
                    rest = len(stripped[len(opener):].split())
                    if stable and rest >= 2:
                        return (self.silence_ms + self.max_silence_ms) / 2
                    return self.max_silence_ms

        return self.silence_ms

    @staticmethod
    def _is_hesitation(clean: str, last_folded: str) -> bool:
        if _fold(clean) == clean.lower():
            return last_folded in _HESITATIONS
        tail = unicodedata.normalize("NFC", clean.lower()).rstrip(" .,!?…").split()
        return bool(tail) and tail[-1].strip(".,") in _HESITATIONS_MARKED

    @staticmethod
    def _number_cued(words: list[str], length: int) -> bool:
        head = " ".join(w.strip(".,") for w in words[: len(words) - length])
        return any(f" {cue} " in f" {head} " for cue in _NUMBER_CUES)

    @staticmethod
    def _dangling_subordinate(words: list[str], clean: str) -> bool:
        if len(words) > _SUBORDINATE_MAX_WORDS:
            return False
        if len(words) > 1 and words[1].strip(".,") in _INTERROGATIVE_SECOND:
            return False
        head = " ".join(words[:2])
        if not any(head == o or head.startswith(o + " ") or words[0] == o for o in _SUBORDINATE_OPENERS):
            return False
        if _fold(clean) == clean.lower():
            return not any(w.strip(".,") in _MAIN_CLAUSE_MARKS for w in words[1:])
        marked = unicodedata.normalize("NFC", clean.lower()).split()
        return not any(w.strip(".,!?…") in _MAIN_CLAUSE_MARKS_MARKED for w in marked[1:])

    @staticmethod
    def _trailing_hold(clean: str, last_folded: str) -> bool:
        if _fold(clean) == clean.lower():
            return last_folded in _TRAILING_HOLDS_FOLDED
        tail = [w.strip(".,") for w in unicodedata.normalize("NFC", clean.lower()).rstrip(" .,!?…").split()]
        if not tail:
            return False
        if tail[-1] in _TRAILING_HOLDS:
            return True
        # "có" alone is "yes"; "bạn có" / "tài khoản của tôi có" are not done.
        return (
            len(tail) >= 2 and tail[-1] in _OBJECT_VERBS
            and tail[-2] not in _DONE_BEFORE_VERB and tail[-2] not in _NOUN_BEFORE_VERB
        )

    @staticmethod
    def _is_connector(clean: str, last_folded: str) -> bool:
        if _fold(clean) == clean.lower():
            return last_folded in _TRAILING_CONNECTORS_FOLDED
        tail = unicodedata.normalize("NFC", clean.lower()).rstrip(" .,!?…").split()
        return bool(tail) and tail[-1].strip(".,") in _TRAILING_CONNECTORS

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
