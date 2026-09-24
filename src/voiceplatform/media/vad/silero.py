"""Silero VAD adapter.

Loaded lazily: importing torch costs seconds and the platform must start (and
its tests must run) on a machine with no model at all.
"""

from __future__ import annotations

import numpy as np

from ...core.audio import AudioFrame, resample_linear
from ...core.errors import ModelUnavailable
from .base import Vad

_SILERO_RATE = 16000
_SILERO_WINDOW = 512  # samples the model expects at 16 kHz


class SileroVad(Vad):
    name = "silero"

    def __init__(self, threshold: float = 0.5) -> None:
        self.threshold = threshold
        self._model = None
        self._tail = np.zeros(0, dtype=np.float32)
        self._last = 0.0

    def _ensure(self) -> None:
        if self._model is not None:
            return
        try:
            import torch
            from silero_vad import load_silero_vad
        except ImportError as exc:
            raise ModelUnavailable(
                "silero-vad and torch are required for the silero VAD backend"
            ) from exc
        self._torch = torch
        self._model = load_silero_vad()
        self._model.eval()

    def reset(self) -> None:
        self._tail = np.zeros(0, dtype=np.float32)
        self._last = 0.0
        if self._model is not None:
            try:
                self._model.reset_states()
            except Exception:  # pragma: no cover - version dependent
                pass

    def probability(self, frame: AudioFrame) -> float:
        self._ensure()
        pcm = frame.samples
        if frame.sample_rate != _SILERO_RATE:
            pcm = resample_linear(pcm, frame.sample_rate, _SILERO_RATE)
        self._tail = np.concatenate([self._tail, pcm])
        # Frames are 20 ms (320 samples at 16 kHz) but the model wants 512, so
        # a probability only updates every other frame; holding the last value
        # is what keeps the gate's frame counting meaningful.
        while self._tail.size >= _SILERO_WINDOW:
            window = self._tail[:_SILERO_WINDOW]
            self._tail = self._tail[_SILERO_WINDOW:]
            with self._torch.no_grad():
                tensor = self._torch.from_numpy(window.copy())
                self._last = float(self._model(tensor, _SILERO_RATE).item())
        return self._last
