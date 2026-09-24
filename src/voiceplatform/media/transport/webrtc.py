"""WebRTC transport seat.

Not implemented yet, and deliberately visible rather than absent: WebSocket
carries the same frames today, and the interface above is what a WebRTC leg
(aiortc, or a LiveKit/Pipecat worker) plugs into when it lands. Building it
means an `aiortc` RTCPeerConnection whose inbound MediaStreamTrack feeds the
same `Framer`, and an outbound track fed from the same audio sink.
"""

from __future__ import annotations

from ...core.errors import TransportError


class WebRtcTransport:  # pragma: no cover - placeholder
    def __init__(self, *args, **kwargs) -> None:
        raise TransportError(
            "WebRTC transport is not implemented yet. Use the WebSocket "
            "transport (ws://.../v1/realtime), or plug an aiortc/LiveKit leg "
            "into media.transport.base.Transport."
        )
