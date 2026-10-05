"""The client's clock estimate must stay replaceable for the whole session.

The browser re-syncs once a minute because the two monotonic clocks drift.
Before 02/10 the engine kept an estimate only if its uncertainty was strictly
lower than the stored one, so the first tight estimate (often < 1 ms on a LAN)
was never replaced and every later re-sync was ignored for the full hour.
"""

from __future__ import annotations

import pytest

import voiceplatform.conversation.engine as engine_module
from voiceplatform.app.simulate import build_engine


@pytest.fixture()
def clock(monkeypatch):
    now = {"ms": 1_000_000.0}
    monkeypatch.setattr(engine_module, "now_ms", lambda: now["ms"])
    return now


def test_a_fresh_estimate_is_kept_against_a_looser_one(clock):
    eng, _ = build_engine()
    eng.set_playback_clock(100.0, 0.5)
    clock["ms"] += 1_000
    eng.set_playback_clock(250.0, 2.0)
    assert eng.playback_clock == (100.0, 0.5)


def test_an_aged_estimate_gives_way_to_a_new_sync(clock):
    eng, _ = build_engine()
    eng.set_playback_clock(100.0, 0.5)
    # One minute at worst-case drift adds 6 ms: a 2 ms re-sync is now better.
    clock["ms"] += 60_000
    eng.set_playback_clock(103.0, 2.0)
    assert eng.playback_clock == (103.0, 2.0)
    # ...and the replacement is fresh again.
    clock["ms"] += 1_000
    eng.set_playback_clock(90.0, 2.5)
    assert eng.playback_clock == (103.0, 2.0)


def test_invalid_estimates_are_still_refused(clock):
    eng, _ = build_engine()
    eng.set_playback_clock(100.0, 0.5)
    clock["ms"] += 3_600_000
    for offset, uncertainty in ((float("nan"), 1.0), (5.0, -1.0), (5.0, 1001.0), ("5", 1.0)):
        eng.set_playback_clock(offset, uncertainty)
    assert eng.playback_clock == (100.0, 0.5)
