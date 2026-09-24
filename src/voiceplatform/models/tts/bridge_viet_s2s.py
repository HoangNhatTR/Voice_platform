"""TTS engines borrowed from speech2speech (VieNeu, viXTTS, F5, subprocess).

`supports_emotion_cues` and `normalizes_text` are read off the loaded backend
rather than the config: a config that claims cue support the checkpoint lacks
is exactly how "[cười]" ends up spoken as the word.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

import numpy as np

from ..base import SpeechChunk, TtsCapabilities
from ..bridge import load_viet_s2s, split_options


class BridgeTtsEngine:
    def __init__(self, backend: str = "vieneu", **options) -> None:
        root, opts = split_options(options)
        self.name = f"s2s:{backend}"
        expected_rtf = float(opts.pop("expected_rtf", 0.6))
        load_viet_s2s(root)
        from viet_s2s.backends import create_talker
        from viet_s2s.config import TalkerConfig

        cfg = TalkerConfig(backend=backend, **opts)
        self.backend = create_talker(cfg)
        self.sample_rate = int(getattr(self.backend, "sample_rate", 24000))
        self.normalizes_text = bool(getattr(self.backend, "normalizes_text", False))
        self.capabilities = TtsCapabilities(
            streaming=True,
            emotion_cues=bool(getattr(self.backend, "supports_emotion_cues", False)),
            voices=tuple(filter(None, [getattr(cfg, "voice", None)])),
            native_sample_rate=self.sample_rate,
            expected_rtf=expected_rtf,
        )
        self._loaded = False

    async def start(self) -> None:
        if not self._loaded:
            await self.backend.load()
            self._loaded = True
            self.sample_rate = int(getattr(self.backend, "sample_rate", self.sample_rate))
            self.capabilities.native_sample_rate = self.sample_rate
            self.capabilities.emotion_cues = bool(
                getattr(self.backend, "supports_emotion_cues", False)
            )

    async def synthesize(self, text: str, *, voice: str | None = None) -> AsyncIterator[SpeechChunk]:
        await self.start()
        first = True
        async for block in self.backend.synthesize_stream(text, voice=voice):
            samples = np.asarray(block, dtype=np.float32).reshape(-1)
            if samples.size == 0:
                continue
            yield SpeechChunk(
                samples=samples,
                sample_rate=self.sample_rate,
                text=text if first else "",
            )
            first = False

    async def close(self) -> None:
        close = getattr(self.backend, "close", None)
        if close is not None:
            await close()
        self._loaded = False
