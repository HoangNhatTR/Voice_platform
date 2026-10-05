from .base import TurnDetector
from .heuristic import HeuristicTurnDetector
from .semantic import SemanticTurnDetector
from .vad_only import VadOnlyTurnDetector
from .text_model import load_text_turn_model

__all__ = [
    "TurnDetector",
    "HeuristicTurnDetector",
    "SemanticTurnDetector",
    "VadOnlyTurnDetector",
    "build_turn_detector",
]


def build_turn_detector(
    backend: str, *, silence_ms: float, max_silence_ms: float, fast_silence_ms: float = 0.0,
    semantic_model_path: str = "", semantic_threshold: float = 0.6,
    semantic_probe_timeout_ms: float = 60.0,
) -> TurnDetector:
    if backend == "vad_only":
        return VadOnlyTurnDetector(silence_ms=silence_ms)
    if backend == "heuristic":
        return HeuristicTurnDetector(
            silence_ms=silence_ms, max_silence_ms=max_silence_ms, fast_silence_ms=fast_silence_ms
        )
    if backend == "semantic":
        if not semantic_model_path:
            raise ValueError("semantic turn detection requires semantic_model_path")
        model = load_text_turn_model(semantic_model_path)

        async def probe(text: str) -> float:
            return model.score(text)

        return SemanticTurnDetector(
            probe=probe, silence_ms=silence_ms, max_silence_ms=max_silence_ms,
            fast_silence_ms=fast_silence_ms, complete_threshold=semantic_threshold,
            probe_timeout_ms=semantic_probe_timeout_ms,
        )
    raise ValueError(f"unknown turn detection backend: {backend}")
