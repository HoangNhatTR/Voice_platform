from __future__ import annotations


class VadOnlyTurnDetector:
    """The baseline: a fixed pause ends the turn.

    Kept because it is the honest control condition for measuring any smarter
    detector, not because it is good. It cuts "Cho tôi hỏi..." in half.
    """

    name = "vad_only"

    def __init__(self, silence_ms: float = 480.0) -> None:
        self.silence_ms = silence_ms

    async def required_silence_ms(self, *, text: str, utterance_ms: float, stable: bool = False) -> float:
        return self.silence_ms
