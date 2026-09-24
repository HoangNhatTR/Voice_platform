"""Re-framing and rate matching: the media plane's clock must not drift."""

from __future__ import annotations

import numpy as np

from voiceplatform.core.audio import LinearResampler
from voiceplatform.media.framer import Framer


def _produced(framer: Framer, chunks: int, chunk: int, src_rate: int) -> int:
    total = 0
    for _ in range(chunks):
        for frame in framer.push(np.zeros(chunk, dtype=np.float32), src_rate=src_rate):
            total += frame.samples.size
    return total + framer._buf.size


def test_resampling_does_not_drift_across_chunks():
    """A rate that does not divide evenly used to gain samples every chunk.

    Resampling each chunk on its own grid rounds each one independently: at
    44.1 kHz in 1024-sample blocks that was +20 samples per second, which is a
    frame and a half of invented audio every second and everything downstream
    counts frames.
    """
    framer = Framer(16000, 320)
    chunks, chunk, src_rate = 43, 1024, 44100
    produced = _produced(framer, chunks, chunk, src_rate)
    ideal = chunks * chunk * 16000 / src_rate
    assert abs(produced - ideal) <= 1


def test_the_seam_between_two_chunks_is_not_a_step():
    """Interpolation has to see the previous chunk's last sample."""
    src_rate, dst_rate, chunk = 44100, 16000, 441
    t = np.arange(src_rate, dtype=np.float64) / src_rate
    tone = np.sin(2 * np.pi * 100 * t).astype(np.float32)

    resampler = LinearResampler(src_rate, dst_rate)
    out = np.concatenate(
        [resampler.process(tone[i : i + chunk]) for i in range(0, tone.size, chunk)]
    )
    step = 2 * np.pi * 100 / dst_rate
    # A clean 100 Hz tone never moves more than one phase step per sample.
    assert float(np.abs(np.diff(out)).max()) < step * 1.5
    assert abs(out.size - dst_rate) <= 1


def test_a_matching_rate_is_left_alone():
    framer = Framer(16000, 320)
    frames = framer.push(np.ones(640, dtype=np.float32), src_rate=16000)
    assert [f.samples.size for f in frames] == [320, 320]
    assert float(frames[0].samples.min()) == 1.0
