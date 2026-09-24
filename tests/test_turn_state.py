from __future__ import annotations

import pytest

from voiceplatform.core.errors import IllegalTransition
from voiceplatform.conversation.state import TurnState, TurnStateMachine


def test_normal_cycle():
    machine = TurnStateMachine()
    machine.to(TurnState.LISTENING)
    machine.to(TurnState.THINKING)
    machine.to(TurnState.SPEAKING)
    machine.to(TurnState.IDLE)
    assert machine.state is TurnState.IDLE


def test_barge_in_from_speaking_goes_back_to_listening():
    machine = TurnStateMachine()
    machine.to(TurnState.LISTENING)
    machine.to(TurnState.THINKING)
    machine.to(TurnState.SPEAKING)
    machine.to(TurnState.LISTENING, "barge-in")
    assert machine.state is TurnState.LISTENING


def test_illegal_transition_raises_instead_of_wedging():
    machine = TurnStateMachine()
    machine.to(TurnState.LISTENING)
    with pytest.raises(IllegalTransition):
        machine.to(TurnState.SPEAKING)  # nothing was ever generated


def test_change_callback_sees_both_ends():
    seen = []
    machine = TurnStateMachine(on_change=lambda a, b, r: seen.append((a, b, r)))
    machine.to(TurnState.LISTENING, "speech")
    assert seen == [(TurnState.IDLE, TurnState.LISTENING, "speech")]
