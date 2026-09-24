"""Frame probabilities -> speech start / end edges.

The counters here are the bug surface of every voice agent: a run counter that
is not reset on the opposite frame accumulates across pauses, and an assistant
gets interrupted by two coughs a minute apart. Both runs reset each other, on
every frame, unconditionally.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class GateEdge(str, Enum):
    START = "start"
    END = "end"


@dataclass(slots=True)
class GateState:
    active: bool = False
    speech_run: int = 0
    silence_run: int = 0
    frames_seen: int = 0


class SpeechGate:
    """Hysteresis over per-frame speech probability."""

    def __init__(self, threshold: float, start_frames: int, end_frames: int) -> None:
        if start_frames < 1 or end_frames < 1:
            raise ValueError("start_frames and end_frames must be >= 1")
        self.threshold = threshold
        self.start_frames = start_frames
        self.end_frames = end_frames
        self.state = GateState()

    def reset(self, active: bool = False) -> None:
        self.state = GateState(active=active)

    @property
    def active(self) -> bool:
        return self.state.active

    def update(self, probability: float) -> GateEdge | None:
        st = self.state
        st.frames_seen += 1
        if probability >= self.threshold:
            st.speech_run += 1
            st.silence_run = 0
        else:
            st.silence_run += 1
            st.speech_run = 0

        if not st.active and st.speech_run >= self.start_frames:
            st.active = True
            st.silence_run = 0
            return GateEdge.START
        if st.active and st.silence_run >= self.end_frames:
            st.active = False
            st.speech_run = 0
            return GateEdge.END
        return None


class RunCounter:
    """A bare consecutive-frame counter (barge-in uses one of its own).

    Separate from SpeechGate on purpose: barge-in needs a different threshold
    and a different run length than end-of-turn, and sharing one counter is how
    the two decisions start corrupting each other.
    """

    __slots__ = ("threshold", "required", "run", "_fired")

    def __init__(self, threshold: float, required: int) -> None:
        self.threshold = threshold
        self.required = max(1, required)
        self.run = 0
        self._fired = False

    def reset(self) -> None:
        self.run = 0
        self._fired = False

    def update(self, probability: float) -> bool:
        """True exactly once per continuous run that reaches `required`."""
        if probability >= self.threshold:
            self.run += 1
        else:
            self.run = 0
            self._fired = False
            return False
        if self.run >= self.required and not self._fired:
            self._fired = True
            return True
        return False
