"""Phrase joins get a pause chosen by punctuation, not by the talker's padding.

Measured 02/10/2026 on ZeroTTS: every phrase carries its own leading and
trailing silence, so back to back they made a 100–180 ms pause at EVERY join —
mid-clause as long as between sentences — plus random 450–530 ms stops inside
a phrase. Heard as hesitation.
"""

from __future__ import annotations

import numpy as np
import pytest

from voiceplatform.app.simulate import build_engine, is_idle, wait_until
from voiceplatform.conversation.pauses import PauseShaper, shape_pauses, tail_ms
from voiceplatform.core.config import PauseConfig
from voiceplatform.models.base import SpeechChunk
from voiceplatform.models.tts.mock import MockTtsEngine

RATE = 24000
CFG = PauseConfig(enabled=True)


def _tone(ms: float) -> np.ndarray:
    t = np.arange(int(RATE * ms / 1000)) / RATE
    return (0.3 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)


def _quiet(ms: float) -> np.ndarray:
    return np.zeros(int(RATE * ms / 1000), np.float32)


def _shape(audio: np.ndarray, text: str, cuts=()) -> np.ndarray:
    shaper = PauseShaper(RATE, CFG, tail_ms(text, CFG))
    parts = np.split(audio, sorted(cuts)) if cuts else [audio]
    return np.concatenate([shaper.push(p) for p in parts] + [shaper.finish()])


def _ms(samples: int) -> float:
    return 1000.0 * samples / RATE


def _edges(x: np.ndarray) -> tuple[float, float]:
    loud = np.flatnonzero(np.abs(x) > 1e-3)
    return _ms(loud[0]), _ms(x.size - 1 - loud[-1])


@pytest.mark.parametrize("text,tail", [
    ("Hồ Hoàn Kiếm nằm ở trung tâm Hà Nội.", CFG.sentence_ms),
    ("Ánh sáng mặt trời gồm nhiều màu,", CFG.clause_ms),
    ("Tôi không thể thực hiện việc chuyển tiền", CFG.cut_ms),
])
def test_the_pause_after_a_phrase_follows_how_it_ends(text, tail):
    out = _shape(np.concatenate([_quiet(200), _tone(500), _quiet(90)]), text)
    lead, trail = _edges(out)
    assert lead == pytest.approx(CFG.lead_ms, abs=10)        # talker's onset pad trimmed
    assert trail == pytest.approx(tail, abs=10)              # padded or cut to the target


def test_a_long_stop_inside_a_phrase_is_compressed_and_a_comma_pause_is_kept():
    audio = np.concatenate([_tone(300), _quiet(520), _tone(300), _quiet(200), _tone(300), _quiet(80)])
    out = _shape(audio, "Một câu.")
    voiced = np.abs(out) > 1e-3
    gaps, run = [], 0
    for v in voiced:
        if v:
            if run:
                gaps.append(_ms(run))
            run = 0
        else:
            run += 1
    inner = [g for g in gaps if g > 5]
    assert inner == [pytest.approx(CFG.inner_max_ms, abs=10), pytest.approx(200, abs=10)]
    assert _ms(voiced.sum()) == pytest.approx(900, abs=15)  # no speech lost


def test_chunking_does_not_change_the_output():
    rng = np.random.default_rng(7)
    audio = np.concatenate([_quiet(150), _tone(400), _quiet(450), _tone(250), _quiet(120)])
    whole = _shape(audio, "Kết thúc.")
    for _ in range(20):
        cuts = rng.integers(1, audio.size - 1, size=rng.integers(1, 12))
        assert np.array_equal(_shape(audio, "Kết thúc.", set(cuts.tolist())), whole)


def test_a_silent_phrase_produces_nothing():
    assert _shape(_quiet(600), "[cười]").size == 0


async def test_shape_pauses_closes_its_source_when_closed_early():
    closed = []

    async def source():
        try:
            for _ in range(10):
                yield SpeechChunk(samples=_tone(100), sample_rate=RATE)
        finally:
            closed.append(True)

    stream = shape_pauses(source(), "Câu.", CFG)
    await anext(stream)
    await stream.aclose()
    assert closed == [True]


async def test_the_engine_sends_shaped_audio_for_synthesised_phrases(config, monkeypatch):
    original = MockTtsEngine.synthesize

    async def padded(self, text, *, voice=None):
        # A talker that pads every phrase like ZeroTTS: onset and trailing silence.
        yield SpeechChunk(samples=_quiet(250), sample_rate=self.sample_rate)
        async for chunk in original(self, text, voice=voice):
            yield chunk
        yield SpeechChunk(samples=_quiet(250), sample_rate=self.sample_rate)

    monkeypatch.setattr(MockTtsEngine, "synthesize", padded)
    totals = {}
    for enabled in (False, True):
        config.conversation.pauses.enabled = enabled
        config.models.llm.options = {"first_token_delay_ms": 10, "token_delay_ms": 1,
                                     "reply": "Hồ Hoàn Kiếm nằm ở trung tâm Hà Nội."}
        eng, sink = build_engine(config)
        await eng.models.start()
        await eng.start()
        try:
            await eng.push_text("hồ hoàn kiếm ở đâu")
            assert await wait_until(eng, is_idle, max_ms=8000, feed_silence=False)
            audio = np.concatenate([f.samples for _, f in sink.audio])
            totals[enabled] = audio
        finally:
            await eng.close()
            await eng.models.close()
    rate = sink.audio[0][1].sample_rate
    raw_lead = np.flatnonzero(np.abs(totals[False]) > 1e-3)[0]
    shaped_lead = np.flatnonzero(np.abs(totals[True]) > 1e-3)[0]
    assert 1000 * raw_lead / rate >= 240
    assert 1000 * shaped_lead / rate == pytest.approx(config.conversation.pauses.lead_ms, abs=12)
