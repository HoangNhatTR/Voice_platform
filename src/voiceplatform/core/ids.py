"""Session / turn / generation identity.

Every artifact that travels through the platform carries a GenerationKey.
Anything arriving under a key that is no longer current is dropped, which is
the single mechanism that keeps interrupted turns from leaking audio or text
into the next turn.
"""

from __future__ import annotations

import itertools
import uuid
from dataclasses import dataclass

_SESSION_COUNTER = itertools.count(1)


def new_session_id() -> str:
    return f"s{next(_SESSION_COUNTER):04d}-{uuid.uuid4().hex[:8]}"


@dataclass(frozen=True, slots=True)
class GenerationKey:
    """Identity of one assistant response attempt."""

    session_id: str
    turn_id: int
    generation_id: int

    def __str__(self) -> str:  # pragma: no cover - display only
        return f"{self.session_id}/t{self.turn_id}/g{self.generation_id}"

    def as_dict(self) -> dict[str, object]:
        return {
            "session_id": self.session_id,
            "turn_id": self.turn_id,
            "generation_id": self.generation_id,
        }
