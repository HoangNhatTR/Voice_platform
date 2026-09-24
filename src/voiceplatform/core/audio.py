"""Audio primitives shared by the media and model planes.

One format travels the whole pipeline: mono float32 in [-1, 1] at a single
session sample rate. Conversion to int16 happens only at the transport edge, so
no stage has to guess what it was handed.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .clock import now_ms
from .ids import GenerationKey

Samples = np.ndarray  # float32, shape (n,)


@dataclass(slots=True)
class AudioFrame:
    """A block of mono PCM with the time it entered the system."""

    samples: Samples
    sample_rate: int
    seq: int = 0
    captured_at_ms: float = field(default_factory=now_ms)
    key: GenerationKey | None = None  # set on outbound (assistant) audio

    @property
    def duration_ms(self) -> float:
        return 1000.0 * len(self.samples) / float(self.sample_rate)

    @property
    def rms(self) -> float:
        if self.samples.size == 0:
            return 0.0
        return float(np.sqrt(np.mean(np.square(self.samples, dtype=np.float64))))

    def to_int16_bytes(self) -> bytes:
        clipped = np.clip(self.samples, -1.0, 1.0)
        return (clipped * 32767.0).astype("<i2").tobytes()

    @classmethod
    def from_int16_bytes(
        cls,
        payload: bytes,
        sample_rate: int,
        seq: int = 0,
        key: GenerationKey | None = None,
    ) -> "AudioFrame":
        pcm = np.frombuffer(payload, dtype="<i2").astype(np.float32) / 32768.0
        return cls(samples=pcm, sample_rate=sample_rate, seq=seq, key=key)

    def with_key(self, key: GenerationKey) -> "AudioFrame":
        return AudioFrame(
            samples=self.samples,
            sample_rate=self.sample_rate,
            seq=self.seq,
            captured_at_ms=self.captured_at_ms,
            key=key,
        )


def silence(duration_ms: float, sample_rate: int) -> Samples:
    return np.zeros(int(sample_rate * duration_ms / 1000.0), dtype=np.float32)


def resample_linear(samples: Samples, src_rate: int, dst_rate: int) -> Samples:
    """Good enough for rate matching at the edge; not a broadcast resampler.

    Real engines get audio at their own native rate from the adapter, so this
    only covers the browser -> session rate step when the client cannot do it.
    """
    if src_rate == dst_rate or samples.size == 0:
        return samples.astype(np.float32, copy=False)
    duration = samples.size / float(src_rate)
    dst_n = int(round(duration * dst_rate))
    if dst_n <= 0:
        return np.zeros(0, dtype=np.float32)
    src_t = np.arange(samples.size, dtype=np.float64) / float(src_rate)
    dst_t = np.arange(dst_n, dtype=np.float64) / float(dst_rate)
    return np.interp(dst_t, src_t, samples).astype(np.float32)


class LinearResampler:
    """Linear resampling that stays continuous across successive chunks.

    `resample_linear` restarts its grid on every call. That is right for a
    whole utterance and wrong for a stream: each chunk rounds its own length,
    so the output drifts against the input clock, and the seam between two
    chunks is a step rather than a slope. Every counter in the conversation
    plane is in frames, so drift is not cosmetic.

    Harmless today because the browser client downsamples before sending and
    `src_rate == dst_rate`. It stops being harmless on a leg that cannot
    resample for us — the WebRTC and telephony case this seat exists for.
    """

    __slots__ = ("src_rate", "dst_rate", "_prev", "_pos")

    def __init__(self, src_rate: int, dst_rate: int) -> None:
        self.src_rate = src_rate
        self.dst_rate = dst_rate
        self._prev = 0.0   # last sample of the previous chunk, at index -1
        self._pos = 0.0    # where the next output sample falls in this chunk

    def reset(self) -> None:
        self._prev = 0.0
        self._pos = 0.0

    def process(self, samples: Samples) -> Samples:
        if self.src_rate == self.dst_rate:
            return samples.astype(np.float32, copy=False)
        n = samples.size
        if n == 0:
            return np.zeros(0, dtype=np.float32)
        step = self.src_rate / float(self.dst_rate)
        count = int(np.floor((n - 1 - self._pos) / step)) + 1
        if count <= 0:
            # Chunk shorter than one output step; carry it and wait.
            self._pos -= n
            self._prev = float(samples[-1])
            return np.zeros(0, dtype=np.float32)
        positions = self._pos + step * np.arange(count)
        # One sample of history in front, so a position in [-1, 0) interpolates
        # across the chunk boundary instead of clamping to the first sample.
        extended = np.concatenate([np.array([self._prev], dtype=np.float32), samples])
        out = np.interp(
            positions + 1.0, np.arange(extended.size, dtype=np.float64), extended
        ).astype(np.float32)
        self._pos = self._pos + step * count - n
        self._prev = float(samples[-1])
        return out


class RingBuffer:
    """Fixed-capacity float32 ring used for pre-roll and jitter smoothing."""

    def __init__(self, capacity_samples: int) -> None:
        self._buf = np.zeros(capacity_samples, dtype=np.float32)
        self._capacity = capacity_samples
        self._write = 0
        self._filled = 0

    def __len__(self) -> int:
        return self._filled

    @property
    def capacity(self) -> int:
        return self._capacity

    def write(self, samples: Samples) -> None:
        n = samples.size
        if n == 0:
            return
        if n >= self._capacity:
            self._buf[:] = samples[-self._capacity :]
            self._write = 0
            self._filled = self._capacity
            return
        end = self._write + n
        if end <= self._capacity:
            self._buf[self._write : end] = samples
        else:
            head = self._capacity - self._write
            self._buf[self._write :] = samples[:head]
            self._buf[: n - head] = samples[head:]
        self._write = end % self._capacity
        self._filled = min(self._capacity, self._filled + n)

    def read_last(self, n: int) -> Samples:
        n = min(n, self._filled)
        if n == 0:
            return np.zeros(0, dtype=np.float32)
        start = (self._write - n) % self._capacity
        if start + n <= self._capacity:
            return self._buf[start : start + n].copy()
        head = self._capacity - start
        return np.concatenate([self._buf[start:], self._buf[: n - head]])

    def clear(self) -> None:
        self._write = 0
        self._filled = 0


class Utterance:
    """Accumulates one user utterance, pre-roll included."""

    def __init__(self, sample_rate: int) -> None:
        self.sample_rate = sample_rate
        self._chunks: list[Samples] = []
        self.started_at_ms: float | None = None

    def add(self, samples: Samples) -> None:
        if self.started_at_ms is None:
            self.started_at_ms = now_ms()
        if samples.size:
            self._chunks.append(samples.astype(np.float32, copy=False))

    @property
    def duration_ms(self) -> float:
        n = sum(c.size for c in self._chunks)
        return 1000.0 * n / float(self.sample_rate)

    def audio(self) -> Samples:
        if not self._chunks:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(self._chunks)

    def reset(self) -> None:
        self._chunks.clear()
        self.started_at_ms = None
