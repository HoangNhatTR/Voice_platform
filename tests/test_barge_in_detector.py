"""The barge-in detector on its own, where the timing is exact."""

from __future__ import annotations

import numpy as np

from voiceplatform.conversation.barge_in import BargeInDetector
from voiceplatform.core.audio import AudioFrame
from voiceplatform.core.config import BargeInConfig

RATE = 16000
FRAME_MS = 20


def loud_frame(level: float = 0.2) -> AudioFrame:
    rng = np.random.default_rng(0)
    n = int(RATE * FRAME_MS / 1000)
    return AudioFrame(samples=(level * rng.normal(0, 1, n)).astype(np.float32), sample_rate=RATE)


def quiet_frame() -> AudioFrame:
    n = int(RATE * FRAME_MS / 1000)
    return AudioFrame(samples=np.zeros(n, dtype=np.float32), sample_rate=RATE)


def detector(**kwargs) -> BargeInDetector:
    config = BargeInConfig(speech_frames=4, guard_ms=60, min_rms=0.02, **kwargs)
    return BargeInDetector(config)


def test_nothing_fires_before_the_assistant_speaks():
    d = detector()
    assert d.update(1.0, loud_frame(), 0.0) is False
    assert d.stats.fired == 0


def test_the_guard_window_suppresses_the_onset_echo():
    d = detector()
    d.arm(at_ms=0.0)
    for t in (10.0, 30.0, 50.0):
        assert d.update(1.0, loud_frame(), t) is False
    assert d.stats.suppressed_guard == 3


def test_speech_inside_the_guard_does_not_carry_over():
    """Or the interruption fires the instant the guard lifts."""
    d = detector()
    d.arm(at_ms=0.0)
    for t in (10.0, 30.0, 50.0):
        d.update(1.0, loud_frame(), t)
    assert d.update(1.0, loud_frame(), 70.0) is False   # run restarts at 1
    assert d.update(1.0, loud_frame(), 90.0) is False
    assert d.update(1.0, loud_frame(), 110.0) is False
    assert d.update(1.0, loud_frame(), 130.0) is True   # the fourth


def test_a_quiet_frame_resets_the_run():
    d = detector()
    d.arm(at_ms=0.0)
    t = 100.0
    for _ in range(3):
        d.update(1.0, loud_frame(), t)
        t += FRAME_MS
    d.update(0.0, quiet_frame(), t)
    t += FRAME_MS
    fired = [d.update(1.0, loud_frame(), t + i * FRAME_MS) for i in range(3)]
    assert fired == [False, False, False]   # the earlier burst did not count


def test_quiet_speech_below_the_level_floor_is_ignored():
    d = detector()
    d.arm(at_ms=0.0)
    for i in range(8):
        assert d.update(1.0, loud_frame(level=0.001), 100.0 + i * FRAME_MS) is False
    assert d.stats.suppressed_level == 8


def test_disarm_stops_everything():
    d = detector()
    d.arm(at_ms=0.0)
    d.disarm()
    assert d.update(1.0, loud_frame(), 200.0) is False
