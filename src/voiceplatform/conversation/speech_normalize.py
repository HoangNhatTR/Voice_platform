"""Conservative Vietnamese text preparation for TTS.

The original answer stays in the transcript and history. Only a copy handed to
the talker is expanded. Ambiguous bare numbers and unknown names are untouched.
"""

from __future__ import annotations

import datetime as dt
import re

_DIGITS = ("không", "một", "hai", "ba", "bốn", "năm", "sáu", "bảy", "tám", "chín")
_SCALES = ("", "nghìn", "triệu", "tỷ", "nghìn tỷ")
_DATE = re.compile(r"(?<!\w)(?:ngày\s+)?(\d{1,2})[/-](\d{1,2})[/-](\d{4})(?!\d)", re.I)
# Not after "," or ".": those digits are the fraction of a decimal ("0,5%"),
# and expanding them alone gave "0,năm phần trăm". Decimals stay as written.
_MONEY = re.compile(r"(?<![\w.,])((?:\d{1,3}(?:\.\d{3})+|\d+))(?:\s*)(đồng|vnđ|₫)(?!\w)", re.I)
_UNIT = re.compile(r"(?<![\w.,])((?:\d{1,3}(?:\.\d{3})+|\d+))(?:\s*)(km|kg|m|cm|%|°c)(?!\w)", re.I)
_IDENTIFIER = re.compile(r"\b(số tài khoản|số điện thoại|mã otp)\s*[:#]?\s*((?:\d[\s.-]?){3,19}\d)(?!\d)", re.I)
_UNIT_NAMES = {"km": "ki lô mét", "kg": "ki lô gam", "m": "mét", "cm": "xen ti mét", "%": "phần trăm", "°c": "độ xê"}


def _under_thousand(n: int, *, pad: bool = False) -> str:
    hundreds, rest = divmod(n, 100)
    parts: list[str] = []
    if hundreds or pad:
        parts.extend((_DIGITS[hundreds], "trăm"))
    tens, ones = divmod(rest, 10)
    if tens > 1:
        parts.extend((_DIGITS[tens], "mươi"))
        if ones:
            parts.append("mốt" if ones == 1 else "lăm" if ones == 5 else _DIGITS[ones])
    elif tens == 1:
        parts.append("mười")
        if ones:
            parts.append("lăm" if ones == 5 else _DIGITS[ones])
    elif ones:
        if hundreds or pad:
            parts.append("linh")
        parts.append(_DIGITS[ones])
    return " ".join(parts)


def number_words(value: int) -> str:
    if value < 0 or value >= 10**15:
        raise ValueError("number outside supported speech range")
    if value == 0:
        return _DIGITS[0]
    groups: list[int] = []
    while value:
        value, group = divmod(value, 1000)
        groups.append(group)
    parts = []
    for index in range(len(groups) - 1, -1, -1):
        if not groups[index]:
            continue
        words = _under_thousand(groups[index], pad=index < len(groups) - 1 and groups[index] < 100)
        parts.append(" ".join(p for p in (words, _SCALES[index]) if p))
    return " ".join(parts)


def normalize_for_speech(text: str, pronunciations: dict[str, str] | None = None) -> str:
    """Expand only formats with a clear meaning; preserve the source text."""
    def identifier(match: re.Match[str]) -> str:
        digits = re.sub(r"\D", "", match.group(2))
        return f"{match.group(1)} {' '.join(_DIGITS[int(d)] for d in digits)}"

    def date(match: re.Match[str]) -> str:
        day, month, year = map(int, match.groups())
        try:
            dt.date(year, month, day)
        except ValueError:
            return match.group()
        return f"ngày {number_words(day)} tháng {number_words(month)} năm {number_words(year)}"

    def measured(match: re.Match[str], names: dict[str, str]) -> str:
        raw, unit = match.groups()
        value = int(raw.replace(".", ""))
        if value >= 10**15:
            return match.group()
        return f"{number_words(value)} {names[unit.lower()]}"

    text = _IDENTIFIER.sub(identifier, text)
    text = _DATE.sub(date, text)
    text = _MONEY.sub(lambda m: measured(m, {"đồng": "đồng", "vnđ": "đồng", "₫": "đồng"}), text)
    text = _UNIT.sub(lambda m: measured(m, _UNIT_NAMES), text)
    for written, spoken in sorted((pronunciations or {}).items(), key=lambda pair: -len(pair[0])):
        if written and spoken:
            text = re.sub(r"(?<!\w)" + re.escape(written) + r"(?!\w)", spoken, text, flags=re.I)
    return text
