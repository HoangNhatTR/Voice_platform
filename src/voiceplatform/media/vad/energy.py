"""RMS gate with an adaptive noise floor.

The default: no model to load, no GPU, deterministic in tests. It is weaker
than Silero on noisy input, which is why the noise floor tracks slowly upward
and quickly downward instead of using a fixed threshold.
"""

from __future__ import annotations

from ...core.audio import AudioFrame
from .base import Vad


class EnergyVad(Vad):
    name = "energy"

    def __init__(
        self,
        threshold: float = 0.012,
        floor_attack: float = 0.02,
        floor_release: float = 0.2,
        margin: float = 2.5,
    ) -> None:
        self._threshold = threshold
        self._floor_attack = floor_attack
        self._floor_release = floor_release
        self._margin = margin
        self._floor = threshold / margin

    def reset(self) -> None:
        self._floor = self._threshold / self._margin

    def probability(self, frame: AudioFrame) -> float:
        rms = frame.rms
        gate = max(self._threshold, self._floor * self._margin)
        if rms < gate:
            # Quiet frame: let the floor follow the room up slowly.
            self._floor += self._floor_attack * (rms - self._floor)
        else:
            # Loud frame: pull the floor down fast so speech never raises it.
            self._floor += self._floor_release * (min(rms, self._floor) - self._floor)
        if gate <= 0:
            return 1.0 if rms > 0 else 0.0
        return max(0.0, min(1.0, rms / (gate * 2.0)))
