"""Transport contract between the user plane and the platform.

The conversation plane must not know whether audio arrived over WebSocket,
WebRTC or a SIP bridge. It sends control messages and audio frames; it receives
audio frames and control messages. That is the whole surface.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from ...core.audio import AudioFrame
from ...core.ids import GenerationKey


@dataclass(slots=True)
class ControlMessage:
    type: str
    data: dict[str, Any] = field(default_factory=dict)


class Transport(Protocol):
    """One connected client."""

    session_id: str

    async def send_audio(self, frame: AudioFrame, key: GenerationKey) -> None:
        """Deliver assistant audio, tagged so the client can fence it."""

    async def send_control(self, message: ControlMessage) -> None: ...

    async def close(self, reason: str = "") -> None: ...
