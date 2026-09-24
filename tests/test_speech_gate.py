"""The counters that decide when someone started and stopped talking."""

from __future__ import annotations

from voiceplatform.media.vad.gate import GateEdge, RunCounter, SpeechGate


def test_start_requires_consecutive_speech_frames():
    gate = SpeechGate(threshold=0.5, start_frames=3, end_frames=3)
    assert gate.update(0.9) is None
    assert gate.update(0.9) is None
    assert gate.update(0.9) is GateEdge.START
    assert gate.active


def test_speech_run_resets_on_a_single_quiet_frame():
    """The bug this guards: two bursts separated by silence must not add up."""
    gate = SpeechGate(threshold=0.5, start_frames=3, end_frames=3)
    gate.update(0.9)
    gate.update(0.9)
    gate.update(0.1)          # one quiet frame wipes the run
    assert gate.update(0.9) is None
    assert gate.update(0.9) is None
    assert gate.update(0.9) is GateEdge.START


def test_end_requires_consecutive_silence_and_resets_too():
    gate = SpeechGate(threshold=0.5, start_frames=1, end_frames=3)
    assert gate.update(0.9) is GateEdge.START
    gate.update(0.1)
    gate.update(0.1)
    gate.update(0.9)          # speech resumed: the pause did not count
    assert gate.update(0.1) is None
    assert gate.update(0.1) is None
    assert gate.update(0.1) is GateEdge.END
    assert not gate.active


def test_run_counter_fires_once_per_run():
    counter = RunCounter(threshold=0.5, required=2)
    assert counter.update(0.9) is False
    assert counter.update(0.9) is True
    assert counter.update(0.9) is False   # same run, already reported
    counter.update(0.0)
    assert counter.update(0.9) is False
    assert counter.update(0.9) is True    # new run


def test_run_counter_does_not_accumulate_across_gaps():
    counter = RunCounter(threshold=0.5, required=4)
    for _ in range(3):
        counter.update(0.9)
        counter.update(0.0)               # every burst is interrupted
    assert counter.run == 0
    assert counter.update(0.9) is False
