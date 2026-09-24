"""Voice activity detection contract.

A VAD answers one narrow question per frame — is there speech here — and
nothing about whether the speaker is finished. That decision belongs to the
conversation plane; keeping them apart is what lets a semantic turn detector be
swapped in without touching audio code.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from ...core.audio import AudioFrame


@runtime_checkable
class Vad(Protocol):
    name: str

    def reset(self) -> None:
        """Forget all internal state (new session or new turn)."""

    def probability(self, frame: AudioFrame) -> float:
        """Speech probability in [0, 1] for this frame."""
