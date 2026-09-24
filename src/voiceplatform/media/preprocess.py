"""Echo cancellation and noise suppression seats.

Honest state of the art here: browser `getUserMedia({echoCancellation: true})`
runs the same WebRTC APM this would otherwise host, with the far-end reference
it already has. Server-side AEC only earns its keep for SIP/telephony legs,
where the client cannot do it. Both are backends of one interface so the
conversation plane never learns which one ran.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np

from ..core.audio import AudioFrame


class Preprocessor(Protocol):
    name: str

    def process(self, frame: AudioFrame) -> AudioFrame:
        """Return the cleaned frame (may be the same object)."""

    def far_end(self, frame: AudioFrame) -> None:
        """Feed assistant audio as the echo reference."""

    def reset(self) -> None: ...


class PassthroughPreprocessor:
    """Used when AEC/NS happen on the client."""

    name = "passthrough"

    def process(self, frame: AudioFrame) -> AudioFrame:
        return frame

    def far_end(self, frame: AudioFrame) -> None:
        return None

    def reset(self) -> None:
        return None


class DuckingPreprocessor:
    """A last-resort half-duplex guard, not echo cancellation.

    When no AEC exists anywhere (raw PCM from a phone leg), this attenuates the
    near end while the assistant is speaking so its own voice does not trip
    barge-in. It costs true full duplex: real interruptions need to be louder.
    Prefer client AEC; reach for this only when there is none.
    """

    name = "ducking"

    def __init__(self, attenuation: float = 0.35, hold_frames: int = 8) -> None:
        self.attenuation = attenuation
        self.hold_frames = hold_frames
        self._active = 0

    def process(self, frame: AudioFrame) -> AudioFrame:
        if self._active <= 0:
            return frame
        self._active -= 1
        return AudioFrame(
            samples=(frame.samples * self.attenuation).astype(np.float32),
            sample_rate=frame.sample_rate,
            seq=frame.seq,
            captured_at_ms=frame.captured_at_ms,
        )

    def far_end(self, frame: AudioFrame) -> None:
        self._active = self.hold_frames

    def reset(self) -> None:
        self._active = 0


def build_preprocessor(backend: str) -> Preprocessor:
    if backend in {"client", "none"}:
        return PassthroughPreprocessor()
    if backend == "ducking":
        return DuckingPreprocessor()
    raise ValueError(f"unknown aec backend: {backend}")
