"""Audio primitives shared by the media and model planes.

One format travels the whole pipeline: mono float32 in [-1, 1] at a single
session sample rate. Conversion to int16 happens only at the transport edge, so
no stage has to guess what it was handed.
"""

from __future__ import annotations

import math
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


class BandLimitedResampler:
    """Windowed-sinc polyphase resampling, continuous across chunks.

    Neither linear resampler low-passes first, so going DOWN in rate folds
    everything above the new Nyquist back into the band. 48 kHz -> 24 kHz
    with `resample_linear` is plain decimation: a 15 kHz sibilant came out at
    9 kHz at full level. This one cuts at 90% of the lower Nyquist (flat to
    10 kHz, -60 dB at 12 kHz for 48 -> 24 kHz with the defaults).

    State carries over: a stream fed in chunks of any size gives exactly the
    samples one call on the whole stream gives, so seams cannot click and the
    output count follows the input clock. Causal, so audio is delayed by
    (taps-1)/2 input samples (1 ms at 48 kHz), and that last millisecond stays
    in the filter until more input comes.
    """

    def __init__(self, src_rate: int, dst_rate: int, taps: int = 96, beta: float = 8.0) -> None:
        if src_rate <= 0 or dst_rate <= 0 or taps < 2:
            raise ValueError("invalid resampler settings")
        g = math.gcd(src_rate, dst_rate)
        self.src_rate, self.dst_rate = src_rate, dst_rate
        self.up, self.down = dst_rate // g, src_rate // g
        self._taps = taps
        n = taps * self.up
        cutoff = 0.9 * 0.5 / max(self.up, self.down)   # cycles per upsampled sample
        t = np.arange(n) - (n - 1) / 2.0
        h = 2 * cutoff * np.sinc(2 * cutoff * t) * np.kaiser(n, beta)
        self._h = h * (self.up / h.sum())
        # phases[p, j] weighs input x[m - j] for output phase p.
        self._phases = np.ascontiguousarray(self._h.reshape(taps, self.up).T)
        self.reset()

    def reset(self) -> None:
        self._history = np.zeros(self._taps - 1)
        self._t = (self._taps - 1) * self.up   # next output, in upsampled steps

    def process(self, samples: Samples) -> Samples:
        if self.src_rate == self.dst_rate:
            return samples.astype(np.float32, copy=False)
        buf = np.concatenate([self._history, np.asarray(samples, dtype=np.float64).reshape(-1)])
        k = self._taps
        ts = np.arange(self._t, buf.size * self.up, self.down)
        if ts.size:
            m = ts // self.up
            if self.up == 1:
                out = np.convolve(buf, self._h, mode="valid")[m - (k - 1)]
            else:
                idx = m[:, None] - np.arange(k)[None, :]
                out = np.einsum("ij,ij->i", buf[idx], self._phases[ts % self.up])
            self._t = int(ts[-1]) + self.down
        else:
            out = np.zeros(0)
        self._t -= (buf.size - (k - 1)) * self.up
        self._history = buf[buf.size - (k - 1):]
        return out.astype(np.float32)


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
