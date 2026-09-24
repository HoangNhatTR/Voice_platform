"""Where the conversation plane sends its output.

A transport implements this; tests implement it in ten lines. The engine never
imports a transport, which is what keeps WebSocket, WebRTC and a future SIP leg
interchangeable.
"""

from __future__ import annotations

from typing import Protocol

from ..core.audio import AudioFrame
from ..core.ids import GenerationKey
from ..media.transport.base import ControlMessage


class AudioSink(Protocol):
    async def send_audio(self, frame: AudioFrame, key: GenerationKey) -> None: ...
    async def send_control(self, message: ControlMessage) -> None: ...


class CollectingSink:
    """In-memory sink for tests and offline runs."""

    def __init__(self) -> None:
        self.audio: list[tuple[GenerationKey, AudioFrame]] = []
        self.control: list[ControlMessage] = []
        # One ordered list across both streams. Ordering is the thing most
        # worth asserting: "no audio for this generation *after* its
        # playback_reset" is the real contract, and it cannot be checked from
        # two separate lists.
        self.timeline: list[tuple[str, object]] = []

    async def send_audio(self, frame: AudioFrame, key: GenerationKey) -> None:
        self.audio.append((key, frame))
        self.timeline.append(("audio", key))

    async def send_control(self, message: ControlMessage) -> None:
        self.control.append(message)
        self.timeline.append(("control", message))

    def audio_after_control(self, control_type: str) -> list[GenerationKey]:
        """Audio keys emitted after the first control message of this type."""
        for index, (kind, item) in enumerate(self.timeline):
            if kind == "control" and item.type == control_type:  # type: ignore[union-attr]
                return [k for kind2, k in self.timeline[index + 1 :] if kind2 == "audio"]
        return []

    def control_types(self) -> list[str]:
        return [m.type for m in self.control]

    def audio_samples_for(self, generation_id: int) -> int:
        return sum(f.samples.size for k, f in self.audio if k.generation_id == generation_id)
