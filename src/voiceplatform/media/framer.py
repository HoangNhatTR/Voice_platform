"""Re-frame arbitrary inbound chunks into the fixed frame the planes expect.

Everything downstream counts frames — end-of-turn, barge-in, latency budgets —
so a client that ships 37 ms blobs must not change what a "frame" means.
"""

from __future__ import annotations

import numpy as np

from ..core.audio import AudioFrame, LinearResampler


class Framer:
    def __init__(self, sample_rate: int, frame_samples: int) -> None:
        self.sample_rate = sample_rate
        self.frame_samples = frame_samples
        self._buf = np.zeros(0, dtype=np.float32)
        self._seq = 0
        # One resampler for the whole stream, not one per chunk: see
        # core.audio.LinearResampler for why per-chunk drifts.
        self._resampler: LinearResampler | None = None

    def reset(self) -> None:
        self._buf = np.zeros(0, dtype=np.float32)
        if self._resampler is not None:
            self._resampler.reset()

    def push(self, samples: np.ndarray, src_rate: int | None = None) -> list[AudioFrame]:
        if src_rate is not None and src_rate != self.sample_rate:
            if self._resampler is None or self._resampler.src_rate != src_rate:
                self._resampler = LinearResampler(src_rate, self.sample_rate)
            samples = self._resampler.process(np.asarray(samples, dtype=np.float32))
        if samples.size:
            self._buf = np.concatenate([self._buf, samples.astype(np.float32, copy=False)])
        out: list[AudioFrame] = []
        while self._buf.size >= self.frame_samples:
            chunk = self._buf[: self.frame_samples]
            self._buf = self._buf[self.frame_samples :]
            out.append(
                AudioFrame(samples=chunk, sample_rate=self.sample_rate, seq=self._seq)
            )
            self._seq += 1
        return out

    def flush(self) -> AudioFrame | None:
        """Pad and emit whatever is left (end of stream only)."""
        if self._buf.size == 0:
            return None
        pad = np.zeros(self.frame_samples - self._buf.size, dtype=np.float32)
        chunk = np.concatenate([self._buf, pad])
        self._buf = np.zeros(0, dtype=np.float32)
        frame = AudioFrame(samples=chunk, sample_rate=self.sample_rate, seq=self._seq)
        self._seq += 1
        return frame
