"""Câu mẫu cho trang /collect: nạp, kiểm, sinh số giả, xếp kế hoạch từng người.

Ba quy tắc giữ cho nhãn đáng tin:

1. **Server dựng lại chữ, không tin chữ client gửi.** Câu của một người nói là
   hàm tất định của (speaker_id, prompt_id): số giả được sinh bằng RNG gieo từ
   hai giá trị đó, nên lúc nhận clip server tự dựng lại đúng câu đã hiện trên
   màn hình. Client chỉ gửi prompt_id.
2. **Kế hoạch cân theo họ câu.** Mỗi khối 50 câu có đúng hạn ngạch từng họ
   (`quotas` trong file JSON), và ≥40% câu `hold` là dãy số — chỗ bộ đếm lượt
   hay cắt nhầm nhất. Khối sau dùng câu khác khối trước.
3. **Fail-closed.** File câu mẫu sai (placeholder lạ, khoá không có trong câu)
   làm server từ chối khởi động thay vì sinh nhãn hỏng âm thầm.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

FAMILIES = ("hold", "continue", "complete", "backchannel", "noise", "interrupt")
PAUSE_FAMILIES = ("hold", "continue")
ACOUSTIC_FAMILIES = ("backchannel", "noise", "interrupt")
BANK_PATH = Path(__file__).with_name("collect_prompts.json")

DIGITS = ("không", "một", "hai", "ba", "bốn", "năm", "sáu", "bảy", "tám", "chín")
_PLACEHOLDER = re.compile(r"\{(\w+)\.(\w+)\}")
_FIELDS = ("text", "head", "tail", "tail2", "last2")
_PHONE_PREFIX = (3, 5, 7, 8, 9)


class BankError(ValueError):
    """File câu mẫu không dùng được."""


# ------------------------------------------------------------------ số giả

def two_digit_words(n: int) -> str:
    """1..99 thành chữ, đúng cách đọc: mười lăm, hai mươi mốt, ba mươi tư."""
    if not 1 <= n <= 99:
        raise ValueError(n)
    if n < 10:
        return DIGITS[n]
    tens, unit = divmod(n, 10)
    head = "mười" if tens == 1 else f"{DIGITS[tens]} mươi"
    if unit == 0:
        return head
    if unit == 5:
        tail = "lăm"
    elif unit == 1 and tens > 1:
        tail = "mốt"
    elif unit == 4 and tens > 1:
        tail = "tư"
    else:
        tail = DIGITS[unit]
    return f"{head} {tail}"


def _split_words(words: list[str], cut: int) -> dict[str, str]:
    head, tail = words[:cut], words[cut:]
    return {"text": " ".join(words), "head": " ".join(head), "tail": " ".join(tail),
            "tail2": " ".join(tail[-2:]), "last2": " ".join(words[-2:])}


def _digit_slot(rng: random.Random, digits: list[int]) -> dict[str, str]:
    words = [DIGITS[d] for d in digits]
    n = len(words)
    # Cắt ở giữa dãy: đọc số theo cụm rồi ngừng nhìn tiếp là đúng chỗ máy
    # dễ chốt lượt nhầm. Mỗi nửa giữ ít nhất hai chữ số.
    cut = rng.randint(max(2, n // 3), n - 2)
    return _split_words(words, cut)


def make_slot(kind: str, rng: random.Random) -> dict[str, str]:
    if kind.startswith("digits:"):
        count = int(kind.split(":", 1)[1])
        return _digit_slot(rng, [rng.randrange(10) for _ in range(count)])
    if kind == "phone":
        return _digit_slot(rng, [0, rng.choice(_PHONE_PREFIX)] + [rng.randrange(10) for _ in range(8)])
    if kind.startswith("amount:"):
        low, high = (int(x) for x in kind.split(":", 1)[1].split("-"))
        millions = rng.randint(low, high)
        hundreds = rng.randint(1, 9)
        tens = rng.randint(1, 9)
        # "hai triệu ba trăm | năm mươi nghìn": dừng giữa một số tiền đang đọc.
        head = f"{two_digit_words(millions)} triệu {DIGITS[hundreds]} trăm"
        tail = f"{two_digit_words(tens * 10)} nghìn"
        words = f"{head} {tail}".split()
        return _split_words(words, len(head.split()))
    raise BankError(f"loại slot không biết: {kind}")


def _slot_kind_ok(kind: str) -> bool:
    if kind == "phone":
        return True
    if re.fullmatch(r"digits:(\d+)", kind):
        return 4 <= int(kind.split(":")[1]) <= 16
    match = re.fullmatch(r"amount:(\d+)-(\d+)", kind)
    return bool(match) and 1 <= int(match.group(1)) <= int(match.group(2)) <= 99


# ------------------------------------------------------------------ ngân hàng câu

@dataclass(frozen=True, slots=True)
class Prompt:
    """Một câu đã dựng cho một người nói cụ thể."""

    id: str
    family: str
    digits: bool
    text: str
    part1: str
    part2: str
    keys: tuple[str, ...]
    kind: str
    instruction: str
    hint: str

    def public(self) -> dict[str, Any]:
        """Phần trang cần để hiện câu. Khoá chấm điểm ở lại server."""
        return {"id": self.id, "family": self.family, "digits": self.digits, "text": self.text,
                "part1": self.part1, "part2": self.part2, "kind": self.kind,
                "instruction": self.instruction, "hint": self.hint}


def _seed(*parts: str) -> random.Random:
    digest = hashlib.sha256("\x1f".join(parts).encode()).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))


def _fill(template: str, slots: dict[str, dict[str, str]]) -> str:
    return _PLACEHOLDER.sub(lambda m: slots[m.group(1)][m.group(2)], template)


def _plain(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", text.lower())).strip()


class Bank:
    def __init__(self, data: dict[str, Any]) -> None:
        self.version = str(data["version"])
        self.consent = data["consent"]
        self.families = data["families"]
        self.quotas: dict[str, int] = {k: int(v) for k, v in data["quotas"].items()}
        self.templates: dict[str, dict[str, Any]] = {}
        for item in data["prompts"]:
            self.templates[item["id"]] = item
        consent_text = "\n".join([self.consent["title"], *self.consent["paragraphs"], self.consent["agree"]])
        self.consent_sha256 = hashlib.sha256(consent_text.encode()).hexdigest()

    # --- dựng câu ------------------------------------------------------
    def render(self, speaker_id: str, prompt_id: str) -> Prompt:
        item = self.templates[prompt_id]
        rng = _seed("slots", speaker_id, prompt_id)
        slots = {name: make_slot(kind, rng) for name, kind in sorted(item.get("slots", {}).items())}
        family = item["family"]
        part1 = _fill(item.get("part1", ""), slots)
        part2 = _fill(item.get("part2", ""), slots)
        if family in PAUSE_FAMILIES:
            text = f"{part1.rstrip(' .')} {part2}".strip()
        else:
            text = _fill(item.get("text", ""), slots)
        keys = tuple(_fill(k, slots) for k in item.get("keys", ()))
        return Prompt(id=prompt_id, family=family, digits=bool(slots), text=text, part1=part1, part2=part2,
                      keys=keys, kind=item.get("kind", ""), instruction=item.get("instruction", ""),
                      hint=item.get("hint", ""))

    # --- kế hoạch -------------------------------------------------------
    def _pool(self, name: str) -> list[str]:
        family = "hold" if name.startswith("hold_") else name
        ids = [pid for pid, item in self.templates.items() if item["family"] == family]
        if name == "hold_digits":
            ids = [pid for pid in ids if self.templates[pid].get("slots")]
        elif name == "hold_plain":
            ids = [pid for pid in ids if not self.templates[pid].get("slots")]
        return sorted(ids)

    @property
    def block_size(self) -> int:
        return sum(self.quotas.values())

    def blocks(self) -> int:
        return min(len(self._pool(name)) // quota for name, quota in self.quotas.items())

    def plan(self, speaker_id: str) -> list[tuple[int, str]]:
        """(khối, prompt_id) theo thứ tự người nói này sẽ gặp.

        Tất định theo speaker_id: tải lại trang, đổi máy hay server khởi động
        lại đều ra đúng thứ tự cũ, nên bỏ qua câu đã thu là đủ để thu tiếp.
        """
        pools = {}
        for name in self.quotas:
            ids = self._pool(name)
            _seed("pool", speaker_id, name).shuffle(ids)
            pools[name] = ids
        order: list[tuple[int, str]] = []
        for block in range(self.blocks()):
            chosen = [pid for name, quota in self.quotas.items()
                      for pid in pools[name][block * quota:(block + 1) * quota]]
            _seed("block", speaker_id, str(block)).shuffle(chosen)
            if block == 0:
                # Câu đầu tiên là một câu trọn: làm quen micro trước khi phải
                # canh quãng dừng.
                first = next((i for i, pid in enumerate(chosen)
                              if self.templates[pid]["family"] == "complete"), 0)
                chosen.insert(0, chosen.pop(first))
            order.extend((block, pid) for pid in chosen)
        return order

    def public(self) -> dict[str, Any]:
        return {"version": self.version, "block_size": self.block_size, "blocks": self.blocks(),
                "families": self.families,
                "consent": {**self.consent, "sha256": self.consent_sha256}}


def validate(data: dict[str, Any]) -> Bank:
    """Kiểm toàn bộ file câu mẫu, kể cả câu sau khi dựng với vài người nói giả."""
    for key in ("version", "consent", "families", "quotas", "prompts"):
        if key not in data:
            raise BankError(f"thiếu khoá {key}")
    consent = data["consent"]
    if not (consent.get("version") and consent.get("title") and consent.get("paragraphs") and consent.get("agree")):
        raise BankError("consent cần version, title, paragraphs, agree")
    if set(data["families"]) != set(FAMILIES):
        raise BankError("families phải có đủ sáu họ")
    seen: set[str] = set()
    for item in data["prompts"]:
        pid, family = item.get("id", ""), item.get("family")
        if not re.fullmatch(r"[a-z]-[a-z0-9]{1,8}", pid) or pid in seen:
            raise BankError(f"id câu sai hoặc trùng: {pid!r}")
        seen.add(pid)
        if family not in FAMILIES:
            raise BankError(f"{pid}: họ {family!r} không có")
        slots = item.get("slots", {})
        for name, kind in slots.items():
            if not re.fullmatch(r"\w+", name) or not _slot_kind_ok(kind):
                raise BankError(f"{pid}: slot {name}={kind!r} không hợp lệ")
        fields = [item.get(k, "") for k in ("part1", "part2", "text")] + list(item.get("keys", []))
        for value in fields:
            for name, field in _PLACEHOLDER.findall(value):
                if name not in slots or field not in _FIELDS:
                    raise BankError(f"{pid}: placeholder {{{name}.{field}}} không có slot")
        needs = {"hold": ("part1", "part2", "keys"), "continue": ("part1", "part2", "keys"),
                 "complete": ("text",), "backchannel": ("text",), "noise": ("kind", "instruction"),
                 "interrupt": ("text", "keys")}[family]
        if any(not item.get(k) for k in needs):
            raise BankError(f"{pid}: họ {family} cần {', '.join(needs)}")
        if family == "noise" and not re.fullmatch(r"[a-z_]{2,24}", item["kind"]):
            raise BankError(f"{pid}: kind {item['kind']!r} không hợp lệ")
    bank = Bank(data)
    for name, quota in bank.quotas.items():
        if name not in ("hold_digits", "hold_plain", *FAMILIES[1:]) or quota < 0:
            raise BankError(f"hạn ngạch lạ: {name}")
        if len(bank._pool(name)) < quota:
            raise BankError(f"hạn ngạch {name}={quota} lớn hơn số câu có")
    hold = bank.quotas.get("hold_digits", 0) + bank.quotas.get("hold_plain", 0)
    if hold and bank.quotas.get("hold_digits", 0) / hold < 0.4:
        raise BankError("mỗi khối cần ≥40% câu hold là dãy số")
    for speaker in ("kiem-tra-a", "kiem-tra-b", "kiem-tra-c"):
        for pid in bank.templates:
            _check_rendered(bank.render(speaker, pid))
    return bank


def _check_rendered(prompt: Prompt) -> None:
    text = _plain(prompt.text)
    keys = [_plain(k) for k in prompt.keys]
    if any(not k or k not in text for k in keys):
        raise BankError(f"{prompt.id}: khoá {prompt.keys} không nằm trong câu {prompt.text!r}")
    if prompt.family in PAUSE_FAMILIES:
        halves = (_plain(prompt.part1), _plain(prompt.part2))
        if not all(any(k in half for k in keys) for half in halves):
            raise BankError(f"{prompt.id}: mỗi nửa câu cần ít nhất một khoá")
    needs_keys = prompt.family in (*PAUSE_FAMILIES, "interrupt")
    if needs_keys and not 1 <= len(keys) <= 6:
        raise BankError(f"{prompt.id}: cần 1–6 khoá")
    if not needs_keys and keys:
        raise BankError(f"{prompt.id}: họ {prompt.family} không dùng khoá")


@lru_cache(maxsize=4)
def load_bank(path: str | None = None) -> Bank:
    source = Path(path) if path else BANK_PATH
    try:
        data = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise BankError(f"không đọc được {source}: {exc}") from exc
    return validate(data)
