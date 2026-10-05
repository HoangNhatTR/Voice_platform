"""Small, explicit routes for public-source questions that must be grounded.

The model still decides other tool calls. These narrow cases cannot safely
fall back to a model-only answer when native tool calling is busy or variable.

Matched WITH diacritics. Folding them first made sống/song, họ/hộ/hồ,
chưa/chùa and câu/cậu/cầu the same word, and 8 of 9 everyday "… ở đâu?"
questions ("Bạn sống ở đâu?", "Hộ chiếu của tôi để ở đâu nhỉ?") skipped the
model for a Wikipedia lookup. A route now needs a place noun, then its name,
then the "where" — and the name must not be a possessive or a pronoun.
"""

from __future__ import annotations

import re
import unicodedata

# Compounds that only look like a place: hồ sơ (file), hồ bơi (pool), đường phèn (sugar),
# cầu thang (stairs), yêu cầu (request), nhu cầu (need)...
_NOUN = (
    r"(?<!yêu )(?<!nhu )(?<!\w)"
    r"(?P<noun>hồ(?!\s+(?:sơ|bơi)\b)|đường(?!\s+(?:phèn|kính|ăn|cát|huyết|dây|truyền|link)\b)"
    r"|chùa|công viên|bảo tàng|cầu(?!\s+(?:thang|thủ|lông|nguyện|cứu|xin|vồng)\b)"
    r"|núi|sông|địa chỉ)"
)
# A lookahead, so "địa chỉ của bảo tàng X ở đâu" can still match at "bảo tàng".
_PLACE_WHERE = re.compile(
    r"(?=" + _NOUN + r"\s+(?P<name>\w+(?:\s+\w+){0,5}?)\s+(?:nằm\s+)?"
    r"(?:ở\s+đâu|thuộc\s+(?:tỉnh|thành\s+phố)\s+nào)(?!\w))"
)
# A name, not "của tôi", "này", "để": those ask where the user's things are
# ("Đường để ở đâu?" is the sugar, not a street).
_NOT_A_NAME = frozenset(
    "của tôi mình em anh chị bạn cậu nó họ này đó kia ấy nhà ta chúng "
    "để cất đặt gửi đang đi về giờ rồi thì là".split()
)
_EXPLICIT = re.compile(r"(?<!\w)(?:tra cứu|wikipedia|wiki|tìm trên mạng|nguồn tham khảo)(?!\w)")


def requires_public_lookup(text: str) -> bool:
    folded = unicodedata.normalize("NFC", (text or "").casefold())
    folded = re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", folded))
    if _EXPLICIT.search(folded):
        return True
    for match in _PLACE_WHERE.finditer(folded):
        if not _NOT_A_NAME.intersection(match.group("name").split()):
            return True
    return False
