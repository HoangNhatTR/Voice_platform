"""Cho một file WAV chạy qua engine như thể đó là micro, ghi câu trả lời ra WAV.

Đây là cách kiểm chứng stack thật mà không cần trình duyệt, không cần micro và
không phụ thuộc vào loa: nếu file ra có tiếng thì đường ASR -> LLM -> TTS đã
thông, còn nếu không thì timeline in kèm chỉ đúng chặng đang hỏng.

    PYTHON=/home/ai01/AIHoang/speech2speech/.venv/bin/python \
      PYTHONPATH=src $PYTHON scripts/replay_wav.py \
        --config configs/local-cpu.yaml --wav <input.wav> --out reply.wav
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from voiceplatform.app.simulate import is_idle, wait_until
from voiceplatform.conversation.engine import ConversationEngine
from voiceplatform.conversation.sink import CollectingSink
from voiceplatform.core.config import Config
from voiceplatform.models.registry import ModelPlane
from voiceplatform.tasks.executor import TaskExecutor
from voiceplatform.tasks.registry import build_registry


async def main(args: argparse.Namespace) -> int:
    import soundfile as sf

    config = Config.load(args.config) if args.config else Config()
    config.observability.write_traces = False

    data, rate = sf.read(args.wav, dtype="float32")
    mic = data[:, 0] if data.ndim > 1 else data
    print(f"vào : {args.wav} — {len(mic)/rate:.1f}s @ {rate} Hz, "
          f"rms {float(np.sqrt(np.mean(mic**2))):.4f}")

    sink = CollectingSink()
    models = ModelPlane(config.models, output_sample_rate=config.audio.output_sample_rate)
    engine = ConversationEngine(
        config, models, sink, executor=TaskExecutor(build_registry(config.tasks.tools))
    )
    print("nạp model...")
    await models.start()
    await engine.start()

    frame = int(rate * config.audio.frame_ms / 1000)
    try:
        for i in range(0, len(mic) - frame, frame):
            await engine.push_audio(mic[i : i + frame], src_rate=rate)
            await asyncio.sleep(config.audio.frame_ms / 1000.0 if args.realtime else 0)
        # Im lặng đuôi để chốt lượt, rồi chờ trả lời xong.
        silence = np.zeros(frame, dtype=np.float32)
        for _ in range(config.frames_for_ms(1500)):
            await engine.push_audio(silence, src_rate=rate)
            await asyncio.sleep(config.audio.frame_ms / 1000.0 if args.realtime else 0)
        await wait_until(engine, is_idle, max_ms=args.timeout_ms)

        turn = engine.trace.turn(engine.gen.turn_id) or engine.trace.turn(1)
        if turn is not None:
            base = turn.events[0].ts_ms
            print("\n--- timeline ---")
            for ev in turn.events:
                if ev.type.value in {"state_changed", "asr_partial", "assistant_delta"}:
                    continue
                print(f"{ev.ts_ms - base:8.0f} ms  {ev.type.value:20s} {str(ev.data)[:70]}")
            print("\n--- số đo ---")
            for key, value in turn.metrics().items():
                print(f"  {key:22s} {value}")

        if sink.audio:
            out_rate = sink.audio[0][1].sample_rate
            audio = np.concatenate([f.samples for _, f in sink.audio])
            sf.write(args.out, audio, out_rate)
            print(f"\nra  : {args.out} — {audio.size/out_rate:.2f}s @ {out_rate} Hz, "
                  f"peak {float(np.abs(audio).max()):.3f}")
            return 0
        print("\nKHÔNG có audio trả lời — xem timeline ở trên để biết chặng nào dừng.")
        return 1
    finally:
        await engine.close()
        await models.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c")
    parser.add_argument("--wav", required=True)
    parser.add_argument("--out", default="reply.wav")
    parser.add_argument("--realtime", action="store_true")
    parser.add_argument("--timeout-ms", type=int, default=60000)
    raise SystemExit(asyncio.run(main(parser.parse_args())))
