"""Barge-in: the user speaking over the assistant.

Two settings decide whether a session feels alive or unusable, and both have
already cost this team a day of measurement:

* how many consecutive speech frames count as an interruption. At one frame a
  real session lost 15 of 21 turns, because a user still finishing a sentence
  cancelled the answer being synthesised for them.
* whether the counter resets. A run counter that only ever increments turns two
  unrelated coughs a minute apart into an interruption. It resets on every
  frame below threshold, unconditionally (see media.vad.gate.RunCounter).

The guard window exists because the loudest echo of the assistant's own voice
arrives right at playback onset, before any AEC has converged.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..core.audio import AudioFrame
from ..core.config import BargeInConfig
from ..media.vad.gate import RunCounter


@dataclass(slots=True)
class BargeInStats:
    fired: int = 0
    suppressed_guard: int = 0
    suppressed_level: int = 0


class BargeInDetector:
    def __init__(self, config: BargeInConfig) -> None:
        self.config = config
        self._counter = RunCounter(threshold=0.5, required=config.speech_frames)
        self._armed_at_ms: float | None = None
        self.stats = BargeInStats()

    @property
    def armed(self) -> bool:
        return self._armed_at_ms is not None

    def arm(self, at_ms: float) -> None:
        """Assistant audio started going out.

        `at_ms` is the session's *audio* clock, not wall time: the guard window
        has to be measured in the same units as the frames it guards, or a
        replay faster than real time skips the guard entirely.
        """
        self._armed_at_ms = at_ms
        self._counter.reset()

    def disarm(self) -> None:
        self._armed_at_ms = None
        self._counter.reset()

    def update(self, probability: float, frame: AudioFrame, now_ms: float) -> bool:
        if not self.config.enabled or self._armed_at_ms is None:
            return False
        if now_ms - self._armed_at_ms < self.config.guard_ms:
            # Still inside the echo guard: do not even count the frame, or the
            # run survives the guard and fires the instant it lifts.
            self._counter.reset()
            self.stats.suppressed_guard += 1
            return False
        if frame.rms < self.config.min_rms:
            self._counter.reset()
            self.stats.suppressed_level += 1
            return False
        if self._counter.update(probability):
            self.stats.fired += 1
            return True
        return False
