"""Turn detection: is the user finished, or only pausing?

Separate from VAD on purpose. VAD answers "is there sound"; this answers "may I
speak now", and the second question needs the words. A detector returns how
much silence it wants before the turn is declared over, given what has been
transcribed so far, so the engine's timing logic stays in one place.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class TurnDetector(Protocol):
    name: str

    async def required_silence_ms(self, *, text: str, utterance_ms: float) -> float:
        """Silence needed, in ms, before this utterance counts as finished."""
