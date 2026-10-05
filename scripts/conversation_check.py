"""Talk to a RUNNING server the way a person would, and check turn-taking.

Unit tests drive the engine with synthetic noise; this drives the real stack
with real Vietnamese speech. The "user" is the server's own TTS (/try/tts, in
a different voice), streamed into /v1/realtime at real-time pace like a
microphone, with silence in between — so ASR, the turn detector, barge-in and
the client-side fencing all see what they see in a live session.

Five situations, each in its own session:

  full_turn      a question gets a spoken answer
  pause          "chuyển tiền cho … (pause) … số tài khoản …" is ONE turn
  interrupt      talking over the answer stops it, and the new question is answered
  cough          a noise burst over the answer: it stops, then carries on
  backchannel    "ừ" over the answer: it stops, then carries on

Usage (the server must be reachable from loopback: /try/tts is local-only):

  PYTHONPATH=src python scripts/conversation_check.py --base https://127.0.0.1:18100
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import io
import json
import ssl
import struct
import sys
import time
import urllib.request
import wave
from dataclasses import dataclass, field

import numpy as np

HEADER = struct.Struct("<III")
FRAME_MS = 20
USER_VOICE = "quangminh"   # a different voice from the assistant's


# ---------------------------------------------------------------- plumbing

def _ssl() -> ssl.SSLContext:
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE   # the LAN cert is self-signed on purpose
    return ctx


def _http(base: str, path: str, body: dict | None = None) -> dict:
    data = json.dumps(body).encode() if body is not None else None
    request = urllib.request.Request(
        base + path, data=data, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(request, context=_ssl(), timeout=120) as response:
        return json.loads(response.read())


def synth(base: str, text: str) -> tuple[np.ndarray, int]:
    reply = _http(base, "/try/tts", {"text": text, "voice": USER_VOICE})
    with wave.open(io.BytesIO(base64.b64decode(reply["wav_base64"]))) as handle:
        rate = handle.getframerate()
        pcm = np.frombuffer(handle.readframes(handle.getnframes()), dtype="<i2")
    return pcm.astype(np.float32) / 32768.0, rate


def noise_burst(ms: int, rate: int) -> np.ndarray:
    rng = np.random.default_rng(7)
    n = int(rate * ms / 1000)
    t = np.arange(n) / rate
    return (0.15 * (np.sin(2 * np.pi * 150 * t) + rng.normal(0, 0.5, n))).astype(np.float32)


@dataclass
class Session:
    ws: object
    rate: int
    session_id: str = ""
    controls: list[tuple[float, dict]] = field(default_factory=list)
    # generation id -> [first frame time, last frame time, frames]
    audio: dict[int, list[float]] = field(default_factory=dict)
    queue: list[np.ndarray] = field(default_factory=list)
    _t0: float = field(default_factory=time.monotonic)

    def now(self) -> float:
        return time.monotonic() - self._t0

    # --- microphone: one frame every 20 ms, silence when nothing queued ----
    async def mic(self) -> None:
        size = self.rate * FRAME_MS // 1000
        pending = np.zeros(0, np.float32)
        quiet = np.zeros(size, np.float32)
        tick = time.monotonic()
        while True:
            if pending.size == 0 and self.queue:
                pending = self.queue.pop(0)
            if pending.size:
                chunk, pending = pending[:size], pending[size:]
                if chunk.size < size:
                    chunk = np.pad(chunk, (0, size - chunk.size))
            else:
                chunk = quiet
            await self.ws.send((np.clip(chunk, -1, 1) * 32767).astype("<i2").tobytes())
            tick += FRAME_MS / 1000
            await asyncio.sleep(max(0.0, tick - time.monotonic()))

    async def reader(self) -> None:
        async for message in self.ws:
            at = self.now()
            if isinstance(message, bytes):
                _turn, generation, _seq = HEADER.unpack_from(message, 0)
                span = self.audio.setdefault(generation, [at, at, 0])
                span[1] = at
                span[2] += 1
            else:
                data = json.loads(message)
                if data.get("type") == "ready":
                    self.session_id = data["session_id"]
                self.controls.append((at, data))

    async def say(self, samples: np.ndarray) -> float:
        """Queue a clip; return once it has gone out. Returns its start time."""
        started = self.now()
        self.queue.append(samples)
        await asyncio.sleep(samples.size / self.rate + 0.05)
        return started

    def after(self, t: float, kind: str) -> list[tuple[float, dict]]:
        return [(at, c) for at, c in self.controls if at >= t and c.get("type") == kind]

    def generations_after(self, t: float) -> list[int]:
        return sorted(g for g, span in self.audio.items() if span[0] >= t)

    async def until(self, predicate, timeout: float) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if predicate():
                return True
            await asyncio.sleep(0.02)
        return predicate()

    def idle_since(self, t: float) -> bool:
        return any(c.get("state") == "idle" for _, c in self.after(t, "state"))

    def final_text(self) -> str:
        finals = [c["text"] for _, c in self.controls if c.get("type") == "transcript" and c.get("final")]
        return finals[-1] if finals else ""


async def open_session(base: str, rate: int):
    import websockets

    url = base.replace("https://", "wss://").replace("http://", "ws://") + "/v1/realtime"
    ws = await websockets.connect(url, ssl=_ssl() if url.startswith("wss") else None, max_size=None)
    session = Session(ws=ws, rate=rate)
    await ws.send(json.dumps({"type": "hello", "sample_rate": rate}))
    tasks = [asyncio.create_task(session.reader()), asyncio.create_task(session.mic())]
    await session.until(lambda: session.session_id, 10)
    # Give the session's own warm-up a moment, as a person opening the page would.
    await asyncio.sleep(1.5)
    return session, tasks


async def close_session(session: Session, tasks) -> dict:
    counters: dict = {}
    try:
        await asyncio.sleep(0.3)
        counters = _counters(session)
    finally:
        for task in tasks:
            task.cancel()
        await session.ws.close()
    return counters


BASE = ""


def _counters(session: Session) -> dict:
    try:
        return _http(BASE, f"/sessions/{session.session_id}/turns?limit=1").get("counters", {})
    except Exception:
        return {}


# ---------------------------------------------------------------- scenarios

async def speak_over_answer(session: Session, question, over: np.ndarray, *, after_audio_s: float = 1.2):
    """Ask, wait until the answer has been audible for a while, then send `over`."""
    await session.say(question)
    asked = session.now()
    if not await session.until(lambda: session.generations_after(asked), 15):
        return None
    first_gen = session.generations_after(asked)[0]
    started = session.audio[first_gen][0]
    await session.until(lambda: session.now() - started >= after_audio_s, 10)
    t_over = await session.say(over)
    return first_gen, t_over


# Turns the engine heard as an interjection and left unanswered on purpose —
# "ừ", a cough, nothing transcribed — so that the answer they cut resumes.
# Counting them as failed turns made the backchannel case fail exactly when the
# engine got it right (02/10/2026).
_UNANSWERED_ON_PURPOSE = {"backchannel", "empty transcript", "too short", "not speech"}


def validate_turns(payload: dict, *, require_answer: bool = False) -> tuple[bool, str]:
    turns = payload.get("turns", [])
    answered = False
    for turn in turns:
        events = turn.get("events", [])
        failures = [e for e in events if e.get("type") in ("error", "tool_failed")]
        if failures:
            return False, "lỗi pipeline: " + str(failures[0].get("data", {}))
        types = {e.get("type") for e in events}
        if "turn_confirmed" not in types or "cancel" in types:
            continue
        if any(e.get("type") == "turn_end" and (e.get("data") or {}).get("reason") in _UNANSWERED_ON_PURPOSE
               for e in events):
            continue
        ended = any(e.get("type") == "turn_end" and e.get("data", {}).get("answered") for e in events)
        if not ended or "llm_first_token" not in types or "tts_first_audio" not in types:
            return False, "lượt thiếu nội dung LLM, audio hoặc turn_end thành công"
        answered = True
    if require_answer and not answered:
        return False, "không có lượt trả lời hoàn chỉnh"
    return True, "pipeline hoàn thành không có fallback/lỗi"


def turn_evidence(session) -> dict:
    return _http(BASE, f"/sessions/{session.session_id}/turns?limit=50")


async def check_full_turn(session, clips) -> tuple[bool, str]:
    end = (await session.say(clips["question"])) + clips["question"].size / session.rate
    got = await session.until(lambda: session.generations_after(end - 0.5), 15)
    if not got:
        return False, "không có tiếng trả lời trong 15 s"
    first = session.audio[session.generations_after(end - 0.5)[0]][0]
    if not await session.until(lambda: session.idle_since(first), 30):
        return False, "lượt không kết thúc trong 30 s"
    valid, reason = validate_turns(turn_evidence(session), require_answer=True)
    if not valid:
        return False, reason
    return True, f"tiếng đầu sau {1000 * (first - end):.0f} ms kể từ lúc hết câu · ASR: “{session.final_text()}”"


async def check_pause(session, clips) -> tuple[bool, str]:
    rate = session.rate
    t0 = await session.say(clips["part1"])
    await session.say(np.zeros(int(rate * 0.7), np.float32))
    await session.say(clips["part2"])
    await session.until(lambda: session.after(t0, "speaking"), 20)
    await session.until(lambda: session.idle_since(session.now() - 0.1), 30)
    text = session.final_text().lower()
    both = "chuyển" in text and ("tài khoản" in text or "không chín" in text)
    return both, f"transcript cuối: “{session.final_text()}”"


async def check_interrupt(session, clips) -> tuple[bool, str]:
    result = await speak_over_answer(session, clips["long"], clips["interrupt"])
    if result is None:
        return False, "câu hỏi dài không được trả lời"
    old_gen, t_over = result
    if not await session.until(lambda: session.after(t_over, "playback_reset"), 5):
        return False, "nói chen vào mà máy không dừng"
    reset_at = session.after(t_over, "playback_reset")[0][0]
    leaked = session.audio[old_gen][1] > reset_at + 0.05
    answered = await session.until(
        lambda: [g for g in session.generations_after(reset_at) if g != old_gen], 20
    )
    await session.until(lambda: session.idle_since(reset_at + 0.5), 30)
    text = session.final_text()
    ok = answered and not leaked and "giờ" in text.lower()
    return ok, (
        f"dừng sau {1000 * (reset_at - t_over):.0f} ms kể từ lúc bắt đầu nói chen · "
        f"tiếng cũ lọt sau reset: {'CÓ' if leaked else 'không'} · "
        f"trả lời câu mới: {'có' if answered else 'KHÔNG'} (“{text}”)"
    )


async def check_resume(session, clips, over_key: str, counter: str) -> tuple[bool, str]:
    result = await speak_over_answer(session, clips["long"], clips[over_key])
    if result is None:
        return False, "câu hỏi dài không được trả lời"
    old_gen, t_over = result
    if not await session.until(lambda: session.after(t_over, "playback_reset"), 5):
        return True, "không kích hoạt ngắt lời (không sao: máy nói liền mạch)"
    reset_at = session.after(t_over, "playback_reset")[0][0]
    resumed = await session.until(
        lambda: [g for g in session.generations_after(reset_at) if g != old_gen], 8
    )
    gap = None
    if resumed:
        new_gen = [g for g in session.generations_after(reset_at) if g != old_gen][0]
        gap = session.audio[new_gen][0] - reset_at
    await session.until(lambda: session.idle_since(reset_at + 0.5), 30)
    counters = _counters(session)
    detail = (
        f"dừng rồi nói tiếp sau {1000 * gap:.0f} ms" if gap is not None else "dừng rồi IM LUÔN"
    ) + f" · counters: resumed={counters.get('resumed')}, {counter}={counters.get(counter)}"
    # New audio alone is not enough: the reply to a misheard "từ" is new audio
    # too. Only the resume path counts.
    return resumed and counters.get("resumed") == 1, detail


# ---------------------------------------------------------------- main

async def main() -> int:
    global BASE
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base", default="https://127.0.0.1:18100")
    parser.add_argument("--only", nargs="*", help="chỉ chạy các tình huống này")
    parser.add_argument("--output", help="ghi evidence JSON")
    args = parser.parse_args()
    BASE = args.base.rstrip("/")

    required = {
        "full_turn": {"question"}, "pause": {"part1", "part2"},
        "interrupt": {"long", "interrupt"}, "cough": {"long"},
        "backchannel": {"long", "backchannel"},
    }
    selected = args.only or list(required)
    if set(selected) - required.keys():
        parser.error("unknown scenario")
    needed = set().union(*(required[name] for name in selected))
    readiness = _http(BASE, "/readyz")
    if not readiness.get("ok"):
        raise RuntimeError("pipeline is not ready")
    evidence = {"readiness": readiness, "scenarios": []}
    print("tổng hợp giọng người dùng qua /try/tts ...", flush=True)
    texts = {
        "question": "Bây giờ là mấy giờ rồi?",
        "part1": "Chuyển tiền cho",
        "part2": "số tài khoản không chín một hai ba bốn năm",
        # Không cần tra cứu: đo cơ chế ngắt lời, không đo độ trễ của search.
        "long": "Bạn kể cho tôi nghe một câu chuyện cổ tích ngắn về con cáo và con quạ đi.",
        "interrupt": "Thôi, bây giờ là mấy giờ rồi?",
        "backchannel": "Ừ.",
    }
    clips: dict[str, np.ndarray] = {}
    rate = 0
    for key, text in texts.items():
        if key in needed:
            clips[key], rate = synth(BASE, text)
    clips["cough"] = noise_burst(160, rate)

    scenarios = {
        "full_turn": lambda s: check_full_turn(s, clips),
        "pause": lambda s: check_pause(s, clips),
        "interrupt": lambda s: check_interrupt(s, clips),
        "cough": lambda s: check_resume(s, clips, "cough", "utterance_discarded_short"),
        "backchannel": lambda s: check_resume(s, clips, "backchannel", "backchannels"),
    }
    failed = 0
    for name, run in scenarios.items():
        if name not in selected:
            continue
        session, tasks = await open_session(BASE, rate)
        try:
            ok, detail = await run(session)
        except Exception as exc:  # report and keep going
            ok, detail = False, f"{type(exc).__name__}: {exc}"
        timeline = None
        try:
            timeline = turn_evidence(session)
            valid, reason = validate_turns(timeline, require_answer=name == "full_turn")
            if not valid:
                ok, detail = False, detail + " · " + reason
        except Exception as exc:
            ok, detail = False, detail + f" · không lấy được evidence: {type(exc).__name__}"
        counters = await close_session(session, tasks)
        if counters.get("orphan_turns"):
            ok, detail = False, detail + f" · ORPHAN TURN x{counters['orphan_turns']}"
        evidence["scenarios"].append({"name": name, "ok": ok, "detail": detail, "timeline": timeline, "counters": counters,
                                      "assistant_text": "".join(m.get("text", "") for _, m in session.controls if m.get("type") == "assistant_delta"),
                                      "audio_frames": sum(row[2] for row in session.audio.values())})
        failed += not ok
        print(f"{'PASS' if ok else 'FAIL'}  {name:<12} {detail}")
        await asyncio.sleep(1.0)
    print("CONVERSATION OK" if not failed else f"CONVERSATION: {failed} FAIL")
    if args.output:
        from pathlib import Path
        Path(args.output).write_text(json.dumps(evidence, ensure_ascii=False, indent=2) + "\n")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
