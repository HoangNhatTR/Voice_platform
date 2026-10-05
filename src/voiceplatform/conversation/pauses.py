"""Quãng nghỉ ở chỗ nối hai cụm, đặt theo dấu câu chứ không theo talker.

Talker đọc mỗi cụm như một câu riêng, nên âm thanh của mỗi cụm có sẵn im ở
đầu và im ở cuối. Phát nối đuôi, hai đoạn đó thành một quãng nghỉ 100–180 ms
ở mọi chỗ nối — giữa mệnh đề cũng dài gần bằng giữa hai câu. Ngay trong một
cụm, ZeroTTS (lấy mẫu ngẫu nhiên) có lúc dừng 450–530 ms. Đo 02/10/2026, xem
`PauseConfig`.

Chạy theo luồng: tiếng có nội dung đi ra ngay; chỉ những đoạn im được giữ lại
cho tới khi biết chúng nằm ở đầu, giữa hay cuối cụm. Không chỗ nào chờ thêm
tiếng — cắt im đầu cụm còn làm tiếng đầu tới sớm hơn.
"""

from __future__ import annotations

import re
from typing import AsyncIterator

import numpy as np

from ..core.config import PauseConfig
from ..models.base import SpeechChunk

_SENTENCE_END = re.compile(r"[.!?…][\"'”)\]]*$")
_CLAUSE_END = re.compile(r"[,;:][\"'”)\]]*$")


def tail_ms(text: str, cfg: PauseConfig) -> int:
    """Quãng nghỉ sau một cụm, theo cách cụm đó kết thúc."""
    end = text.rstrip()
    if _SENTENCE_END.search(end):
        return cfg.sentence_ms
    if _CLAUSE_END.search(end):
        return cfg.clause_ms
    return cfg.cut_ms


class PauseShaper:
    """Một cụm: xén im đầu còn `lead_ms`, nén im giữa còn `inner_max_ms`,
    thay im cuối bằng đúng `tail` (đệm thêm số 0 nếu talker để ngắn hơn)."""

    def __init__(self, sample_rate: int, cfg: PauseConfig, tail: int) -> None:
        self.sample_rate = sample_rate
        self._hop = max(1, sample_rate // 100)
        self._threshold = 10.0 ** (cfg.threshold_db / 20.0)
        self._lead = int(sample_rate * cfg.lead_ms / 1000)
        self._inner_max = int(sample_rate * cfg.inner_max_ms / 1000)
        self._tail = int(sample_rate * tail / 1000)
        self._carry = np.zeros(0, np.float32)
        self._silence: list[np.ndarray] = []
        self._voiced = False

    def _quiet(self, hop: np.ndarray) -> bool:
        return float(np.sqrt(np.mean(hop * hop))) < self._threshold

    def _pending(self) -> np.ndarray:
        return np.concatenate(self._silence) if self._silence else np.zeros(0, np.float32)

    def _release(self) -> np.ndarray:
        """Im đang giữ, ngay trước một đoạn có tiếng."""
        quiet = self._pending()
        self._silence = []
        if not self._voiced:
            return quiet[-self._lead:] if self._lead else quiet[:0]
        if quiet.size <= self._inner_max:
            return quiet
        # Giữ đuôi của tiếng trước và phần chạy đà của tiếng sau, bỏ khúc giữa.
        half = self._inner_max // 2
        return np.concatenate([quiet[:half], quiet[quiet.size - (self._inner_max - half):]])

    def _hold(self, hop: np.ndarray) -> None:
        self._silence.append(hop)
        if not self._voiced and self._lead:
            # Trước tiếng đầu tiên chỉ cần giữ đúng `lead`, phần cũ hơn bỏ luôn.
            quiet = self._pending()
            self._silence = [quiet[-self._lead:]]
        elif not self._voiced:
            self._silence = []

    def push(self, samples: np.ndarray) -> np.ndarray:
        x = np.asarray(samples, dtype=np.float32)
        if self._carry.size:
            x = np.concatenate([self._carry, x])
        whole = x.size // self._hop * self._hop
        self._carry = x[whole:].copy()
        out: list[np.ndarray] = []
        for start in range(0, whole, self._hop):
            hop = x[start:start + self._hop]
            if self._quiet(hop):
                self._hold(hop)
                continue
            out.append(self._release())
            self._voiced = True
            out.append(hop)
        return np.concatenate(out) if out else np.zeros(0, np.float32)

    def finish(self) -> np.ndarray:
        out: list[np.ndarray] = []
        if self._carry.size:
            if self._quiet(self._carry):
                self._hold(self._carry)
            else:
                out.append(self._release())
                self._voiced = True
                out.append(self._carry)
            self._carry = np.zeros(0, np.float32)
        if not self._voiced:
            return np.zeros(0, np.float32)      # cụm toàn im: không có gì để nói
        quiet = self._pending()[: self._tail]
        self._silence = []
        out.append(quiet)
        if quiet.size < self._tail:
            out.append(np.zeros(self._tail - quiet.size, np.float32))
        return np.concatenate(out)


async def shape_pauses(
    source: AsyncIterator[SpeechChunk], text: str, cfg: PauseConfig
) -> AsyncIterator[SpeechChunk]:
    """`source` với quãng nghỉ đã chuẩn hoá; đóng `source` khi bị đóng giữa chừng."""
    shaper: PauseShaper | None = None
    try:
        async for chunk in source:
            if shaper is None:
                shaper = PauseShaper(chunk.sample_rate, cfg, tail_ms(text, cfg))
            out = shaper.push(chunk.samples)
            if out.size:
                yield SpeechChunk(samples=out, sample_rate=chunk.sample_rate)
        if shaper is not None:
            out = shaper.finish()
            if out.size:
                yield SpeechChunk(samples=out, sample_rate=shaper.sample_rate)
    finally:
        aclose = getattr(source, "aclose", None)
        if aclose is not None:
            await aclose()
