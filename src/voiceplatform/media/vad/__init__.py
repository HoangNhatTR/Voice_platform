from .base import Vad
from .energy import EnergyVad
from .gate import GateEdge, RunCounter, SpeechGate

__all__ = ["Vad", "EnergyVad", "SpeechGate", "GateEdge", "RunCounter", "build_vad"]


def build_vad(backend: str, *, threshold: float, energy_threshold: float) -> Vad:
    if backend == "energy":
        return EnergyVad(threshold=energy_threshold)
    if backend == "silero":
        from .silero import SileroVad

        return SileroVad(threshold=threshold)
    raise ValueError(f"unknown vad backend: {backend}")
