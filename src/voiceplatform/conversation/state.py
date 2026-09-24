"""The turn state machine.

Explicit and strict: an illegal transition raises instead of quietly leaving
the session in a state nothing can leave. Voice bugs are almost always state
bugs, and a machine that refuses the impossible move names the bug at the
moment it happens instead of three seconds later in the audio.
"""

from __future__ import annotations

from collections.abc import Callable
from enum import Enum

from ..core.clock import now_ms
from ..core.errors import IllegalTransition


class TurnState(str, Enum):
    IDLE = "idle"            # nobody is speaking; waiting for the user
    LISTENING = "listening"  # user speech in progress (or just paused)
    THINKING = "thinking"    # turn confirmed; ASR final / LLM / tools running
    SPEAKING = "speaking"    # assistant audio is going out
    CLOSED = "closed"


_ALLOWED: dict[TurnState, set[TurnState]] = {
    TurnState.IDLE: {TurnState.LISTENING, TurnState.THINKING, TurnState.CLOSED},
    # THINKING -> LISTENING is a barge-in before the first audio.
    TurnState.LISTENING: {TurnState.THINKING, TurnState.IDLE, TurnState.CLOSED},
    TurnState.THINKING: {
        TurnState.SPEAKING,
        TurnState.LISTENING,
        TurnState.IDLE,
        TurnState.CLOSED,
    },
    TurnState.SPEAKING: {TurnState.IDLE, TurnState.LISTENING, TurnState.CLOSED},
    TurnState.CLOSED: set(),
}


class TurnStateMachine:
    def __init__(self, on_change: Callable[[TurnState, TurnState, str], None] | None = None) -> None:
        self._state = TurnState.IDLE
        self._since_ms = now_ms()
        self._on_change = on_change

    @property
    def state(self) -> TurnState:
        return self._state

    @property
    def since_ms(self) -> float:
        return self._since_ms

    @property
    def elapsed_ms(self) -> float:
        return now_ms() - self._since_ms

    def can(self, target: TurnState) -> bool:
        return target in _ALLOWED[self._state]

    def to(self, target: TurnState, reason: str = "") -> TurnState:
        if target is self._state:
            return self._state
        if not self.can(target):
            raise IllegalTransition(
                f"{self._state.value} -> {target.value} is not allowed ({reason})"
            )
        previous, self._state = self._state, target
        self._since_ms = now_ms()
        if self._on_change is not None:
            self._on_change(previous, target, reason)
        return target

    def is_assistant_active(self) -> bool:
        return self._state in {TurnState.THINKING, TurnState.SPEAKING}
