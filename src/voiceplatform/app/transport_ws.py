"""WebSocket transport.

Chosen as the first leg on purpose: it is the shortest path to a session you
can actually hear, and it keeps the browser's own AEC/NS in play through
getUserMedia. WebRTC belongs underneath the same `Transport` interface for
lossy networks, phone legs and adaptive jitter handling — see
media/transport/webrtc.py.

Outbound audio carries a 12-byte header so the client can fence it:

    uint32 turn_id | uint32 generation_id | uint32 seq | int16 PCM...

A client that plays a frame whose generation is no longer current is the reason
an interrupted assistant keeps talking for another half second.
"""

from __future__ import annotations

import json
import struct

from ..core.audio import AudioFrame
from ..core.ids import GenerationKey
from ..media.transport.base import ControlMessage
from ..observability.logging import get_logger

log = get_logger("ws")

HEADER = struct.Struct("<III")


class WebSocketTransport:
    def __init__(self, websocket, session_id: str) -> None:
        self.ws = websocket
        self.session_id = session_id
        self._seq = 0
        self.closed = False

    async def send_audio(self, frame: AudioFrame, key: GenerationKey) -> None:
        if self.closed:
            return
        header = HEADER.pack(key.turn_id, key.generation_id, self._seq)
        self._seq += 1
        try:
            await self.ws.send_bytes(header + frame.to_int16_bytes())
        except Exception as exc:  # client vanished mid-turn
            self.closed = True
            log.info("audio send failed, marking closed: %s", exc)

    async def send_control(self, message: ControlMessage) -> None:
        if self.closed:
            return
        payload = {"type": message.type, **message.data}
        try:
            await self.ws.send_text(json.dumps(payload, ensure_ascii=False))
        except Exception as exc:
            self.closed = True
            log.info("control send failed, marking closed: %s", exc)

    async def close(self, reason: str = "") -> None:
        if self.closed:
            return
        self.closed = True
        try:
            await self.ws.close()
        except Exception:  # pragma: no cover
            pass
