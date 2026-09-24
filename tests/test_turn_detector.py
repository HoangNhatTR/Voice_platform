"""Vietnamese endpointing heuristics."""

from __future__ import annotations

import pytest

from voiceplatform.conversation.turn_detector import (
    HeuristicTurnDetector,
    SemanticTurnDetector,
    VadOnlyTurnDetector,
)

BASE, MAX = 400.0, 1400.0


@pytest.fixture
def detector() -> HeuristicTurnDetector:
    return HeuristicTurnDetector(silence_ms=BASE, max_silence_ms=MAX)


async def test_finished_question_ends_on_the_base_pause(detector):
    assert await detector.required_silence_ms(text="Số dư của tôi là bao nhiêu?", utterance_ms=1200) == BASE


async def test_opener_holds_the_turn_open(detector):
    assert await detector.required_silence_ms(text="Cho tôi hỏi", utterance_ms=500) == MAX


async def test_dangling_connector_holds_the_turn_open(detector):
    assert await detector.required_silence_ms(text="Tôi muốn chuyển tiền và", utterance_ms=900) == MAX


async def test_unfinished_account_number_holds_the_turn_open(detector):
    # Someone reading a 14-digit account pauses between groups.
    assert await detector.required_silence_ms(text="chuyển tới số 1903 2688", utterance_ms=1500) == MAX


async def test_complete_account_number_does_not(detector):
    assert await detector.required_silence_ms(
        text="chuyển tới số 19032688123456", utterance_ms=2000
    ) == BASE


async def test_vad_only_ignores_the_words():
    d = VadOnlyTurnDetector(silence_ms=BASE)
    assert await d.required_silence_ms(text="Cho tôi hỏi", utterance_ms=500) == BASE


async def test_semantic_falls_back_when_the_probe_fails():
    async def broken(text: str) -> float:
        raise RuntimeError("model down")

    d = SemanticTurnDetector(probe=broken, silence_ms=BASE, max_silence_ms=MAX)
    assert await d.required_silence_ms(text="Cho tôi hỏi", utterance_ms=500) == MAX
    assert d.probe_failures == 1


async def test_semantic_confident_completion_shortens_the_wait():
    async def confident(text: str) -> float:
        return 0.95

    d = SemanticTurnDetector(probe=confident, silence_ms=BASE, max_silence_ms=MAX)
    assert await d.required_silence_ms(text="Cho tôi hỏi", utterance_ms=500) == BASE
