from .base import TurnDetector
from .heuristic import HeuristicTurnDetector
from .semantic import SemanticTurnDetector
from .vad_only import VadOnlyTurnDetector

__all__ = [
    "TurnDetector",
    "HeuristicTurnDetector",
    "SemanticTurnDetector",
    "VadOnlyTurnDetector",
    "build_turn_detector",
]


def build_turn_detector(backend: str, *, silence_ms: float, max_silence_ms: float) -> TurnDetector:
    if backend == "vad_only":
        return VadOnlyTurnDetector(silence_ms=silence_ms)
    if backend == "heuristic":
        return HeuristicTurnDetector(silence_ms=silence_ms, max_silence_ms=max_silence_ms)
    if backend == "semantic":
        return SemanticTurnDetector(silence_ms=silence_ms, max_silence_ms=max_silence_ms)
    raise ValueError(f"unknown turn detection backend: {backend}")
