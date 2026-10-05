"""ASR ceiling for the G3 stimuli: each clip decoded whole, offline, by the same ASR.

A half-sentence that is missing from what the LLM got can be a turn-taking
loss (the pipeline cut, merged or dropped it) or a recognition error (the ASR
never hears "bạn ơi" as "bạn ơi", even given the whole clip). Scoring against
this reference separates the two: only what the whole-clip decode HAS and the
pipeline LOST is charged to turn-taking.

  PYTHONPATH=src python scripts/asr_reference_g3.py \
      --stimuli docs/audits/2026-09-29/g3/stimuli.json --output docs/audits/2026-09-29/g3/asr-reference.json
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
from pathlib import Path

import numpy as np

from voiceplatform.core.config import EngineSpec
from voiceplatform.models.registry import build_asr


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--stimuli", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--config", default="configs/local-cpu.yaml")
    args = parser.parse_args()
    from voiceplatform.core.config import Config

    spec = Config.load(args.config).models.asr
    options = {k: v for k, v in spec.options.items() if k != "partial_every_frames"}
    engine = build_asr(EngineSpec(backend=spec.backend, options=options))
    await engine.start()
    stimuli = json.loads(Path(args.stimuli).read_text())
    out = {"stimuli_sha256": stimuli.get("sha256"), "asr": spec.backend, "options": options, "text": {}}
    for family in ("hold", "complete", "continue", "backchannel", "interrupt"):
        for case in stimuli["cases"].get(family, []):
            pcm = np.frombuffer(base64.b64decode(case["pcm"]), "<i2").astype(np.float32) / 32768.0
            # The same trailing silence the pipeline's final decode sees.
            pcm = np.concatenate([np.zeros(5120, np.float32), pcm, np.zeros(7680, np.float32)])
            result = await engine.backend.transcribe(pcm, 16000, partial=False)
            out["text"][case["id"]] = (getattr(result, "text", "") or "").strip()
    await engine.close()
    Path(args.output).write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(len(out["text"]), "clips")


if __name__ == "__main__":
    asyncio.run(main())
