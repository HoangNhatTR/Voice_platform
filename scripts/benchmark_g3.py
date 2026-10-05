"""Nghiệm thu G3: chờ lượt, ngắt lời, nói tiếp — trên pipeline thật.

Nói vào /v1/realtime như một micro (khung 20 ms, đúng nhịp thời gian thực),
nghe bằng một player mô phỏng ĐÚNG logic của web/playback-worklet.js (đệm khởi
động, phát liên tục, marker cuối cụm, reset theo generation) và gửi playback
feedback như trình duyệt. Server và harness chạy cùng máy nên mọi mốc thời gian
dùng chung CLOCK_MONOTONIC; không trừ chéo đồng hồ hai máy.

Không phải micro người thật: không phòng, không AEC thật, không mạng LAN. Tiếng
vọng loa ngoài được mô phỏng bằng cách trộn chính audio player đã "phát" vào
micro ở một mức dB và độ trễ cho trước (--echo-db). Giới hạn này ghi thẳng vào
manifest của mỗi lần chạy.

  PYTHONPATH=src python scripts/benchmark_g3.py --base https://127.0.0.1:19101 \
      --stimuli docs/audits/2026-09-29/g3/stimuli.json --output /tmp/g3-run \
      --families hold,complete,continue,backchannel,noise,interrupt
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import collections
import hashlib
import json
import ssl
import struct
import sys
import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

HEADER = struct.Struct("<III")
FRAME_MS = 20
MIC_RATE = 16000
ONSET_RMS = 0.01          # "người dùng bắt đầu nói": khung 20 ms đầu tiên vượt ngưỡng
STARTUP_MS = 160          # phải khớp playback_buffer_ms server gửi trong `ready`


def now() -> float:
    return time.monotonic() * 1000.0


def _ssl() -> ssl.SSLContext:
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    return ctx


def http_json(base: str, path: str, token: str | None = None) -> Any:
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    request = urllib.request.Request(base + path, headers=headers)
    with urllib.request.urlopen(request, context=_ssl(), timeout=30) as response:
        return json.loads(response.read())


def decode(pcm_b64: str) -> np.ndarray:
    return np.frombuffer(base64.b64decode(pcm_b64), "<i2").astype(np.float32) / 32768.0


# ---------------------------------------------------------------- player

class EmulatedWorklet:
    """web/playback-worklet.js, dịch sang Python từng nhánh một.

    Chạy theo khối ~5 ms thay vì quantum 128 mẫu; mốc sự kiện nội suy theo vị
    trí mẫu trong khối nên sai số thời gian dưới một khối.
    """

    def __init__(self, send, startup_ms: int = STARTUP_MS) -> None:
        self.send = send
        self.rate = 24000
        self.startup = int(self.rate * startup_ms / 1000)
        self.queue: collections.deque = collections.deque()
        self.buffered = 0
        self.floor = 0
        self.playing = False
        self.meta: dict | None = None
        self.last_meta: dict | None = None
        self.signal = False
        self.gap_at: float | None = None
        self.gap_notified = False
        self.clock_ms: float | None = None      # playhead, monotonic ms
        self.events: list[dict] = []
        # (start_ms, samples) của những gì đã "ra loa": nguồn cho tiếng vọng.
        self.output: collections.deque = collections.deque(maxlen=400)

    def set_rate(self, rate: int) -> None:
        if rate != self.rate:
            self.rate = rate
            self.startup = int(rate * STARTUP_MS / 1000)

    def report(self, event: str, at_ms: float, meta: dict | None = None, **extra) -> None:
        meta = meta or self.meta or self.last_meta
        if not meta:
            return
        row = {"event": event, "client_ms": at_ms, **meta,
               "buffer_ms": self.buffered * 1000 / self.rate, **extra}
        self.events.append(row)
        self.send({"type": "playback", **row, "audio_time_s": at_ms / 1000.0,
                   "clock_basis": "audio_output_timestamp", "source": "g3_emulated_worklet"})

    # --- messages from the socket -----------------------------------------
    def reset(self, minimum_generation: int, at_ms: float) -> None:
        if self.meta:
            self.report("playback_stopped", at_ms, reason="reset")
        self.queue.clear()
        self.buffered = 0
        self.meta = None
        self.playing = False
        self.gap_at = None
        self.gap_notified = False
        self.floor = minimum_generation

    def push_pcm(self, meta: dict, pcm: np.ndarray) -> None:
        if meta.get("generation_id", 0) < self.floor:
            return
        self.queue.append({"type": "pcm", "meta": meta, "pcm": pcm, "offset": 0})
        self.buffered += pcm.size

    def push_marker(self, kind: str, meta: dict) -> None:
        if meta.get("generation_id", 0) < self.floor:
            return
        self.queue.append({"type": kind, "meta": meta})

    # --- render ------------------------------------------------------------
    def _markers(self, at: float) -> None:
        while self.queue and self.queue[0]["type"] != "pcm":
            packet = self.queue.popleft()
            if packet["type"] == "end":
                self.report("playback_stopped", at, packet["meta"])
                self.last_meta = packet["meta"]
                self.meta = None
                self.signal = False
                self.gap_at = None
                self.gap_notified = False
                if self.buffered == 0:
                    self.playing = False
            elif packet["type"] == "generation_end":
                self.report("playback_generation_end", at, packet["meta"])
                self.playing = False

    def render(self, until_ms: float) -> None:
        if self.clock_ms is None:
            self.clock_ms = until_ms
            return
        n = int((until_ms - self.clock_ms) * self.rate / 1000)
        if n <= 0:
            return
        t0 = self.clock_ms
        at = lambda k: t0 + k * 1000.0 / self.rate  # noqa: E731
        self._markers(at(0))
        complete = any(p["type"] != "pcm" for p in self.queue)
        if not self.playing and (self.buffered >= self.startup or (complete and self.buffered)):
            self.playing = True
            if self.gap_notified and self.gap_at is not None:
                self.report("playback_resumed", at(0), gap_ms=at(0) - self.gap_at)
            self.gap_at = None
            self.gap_notified = False
        out = np.zeros(n, np.float32)
        written = 0
        while self.playing and written < n and self.queue:
            self._markers(at(written))
            if not self.queue or not self.playing:
                break
            packet = self.queue[0]
            if not self.meta or self.meta.get("phrase_id") != packet["meta"].get("phrase_id"):
                self.meta = packet["meta"]
                self.last_meta = packet["meta"]
                self.signal = False
                self.report("playback_started", at(written))
            take = min(n - written, packet["pcm"].size - packet["offset"])
            part = packet["pcm"][packet["offset"]:packet["offset"] + take]
            if not self.signal:
                loud = np.nonzero(np.abs(part) >= 0.003)[0]
                if loud.size:
                    self.signal = True
                    self.report("playback_signal_started", at(written + int(loud[0])))
            out[written:written + take] = part
            written += take
            packet["offset"] += take
            self.buffered -= take
            if packet["offset"] == packet["pcm"].size:
                self.queue.popleft()
        self._markers(at(written))
        if self.meta and self.buffered == 0 and written < n:
            if self.gap_at is None:
                self.gap_at = at(written)
            self.playing = False
            if not self.gap_notified and at(n) - self.gap_at >= 20:
                self.gap_notified = True
                self.report("playback_underrun", self.gap_at)
        self.output.append((t0, self.rate, out))
        self.clock_ms = t0 + n * 1000.0 / self.rate

    def echo(self, start_ms: float, end_ms: float, rate: int) -> np.ndarray:
        """What left the speaker in [start, end), resampled to the mic rate."""
        n_out = int(round((end_ms - start_ms) * rate / 1000))
        acc = np.zeros(n_out, np.float32)
        for t0, r, samples in self.output:
            t1 = t0 + samples.size * 1000.0 / r
            if t1 <= start_ms or t0 >= end_ms or samples.size == 0:
                continue
            times = start_ms + np.arange(n_out) * 1000.0 / rate
            idx = ((times - t0) * r / 1000).astype(np.int64)
            ok = (idx >= 0) & (idx < samples.size)
            acc[ok] += samples[idx[ok]]
        return acc

    @property
    def active(self) -> bool:
        return self.meta is not None or self.buffered > 0


# ---------------------------------------------------------------- client

@dataclass
class Clip:
    label: str
    pcm: np.ndarray
    queued_ms: float = 0.0
    start_ms: float | None = None      # first frame sent
    onset_ms: float | None = None      # first voiced frame sent
    end_ms: float | None = None        # last voiced frame sent
    done_ms: float | None = None       # last frame of the clip sent
    voiced_marks: list[float] = field(default_factory=list)


class Client:
    def __init__(self, base: str, echo_db: float | None, echo_delay_ms: float,
                 echo_onset_db: float | None = None, echo_converge_ms: float = 300.0) -> None:
        self.base = base
        self.ws = None
        self.session_id = ""
        self.token = None
        self.controls: list[tuple[float, dict]] = []
        self.frames: dict[int, list[float]] = {}   # generation -> [first, last, count]
        self.meta: dict | None = None
        self.player = EmulatedWorklet(self._send_json)
        self.queue: collections.deque[Clip] = collections.deque()
        self.current: Clip | None = None
        self.echo_gain = None if echo_db is None else 10 ** (echo_db / 20)
        self.echo_delay_ms = echo_delay_ms
        # Browser AEC needs a moment on each new stretch of far-end audio: the
        # residual starts at echo_onset_db and settles to echo_db.
        self.echo_onset_gain = None if echo_onset_db is None else 10 ** (echo_onset_db / 20)
        self.echo_converge_ms = echo_converge_ms
        self.state = "offline"
        self.tasks: list[asyncio.Task] = []
        self.mic_lateness_ms = 0.0
        self.outbox: asyncio.Queue = asyncio.Queue()
        self.clock_offset = None
        self.clock_uncertainty = float("inf")

    def _send_json(self, message: dict) -> None:
        self.outbox.put_nowait(json.dumps(message))

    async def open(self) -> None:
        import websockets

        url = self.base.replace("https://", "wss://") + "/v1/realtime"
        self.ws = await websockets.connect(url, ssl=_ssl(), max_size=None)
        self.tasks = [asyncio.create_task(t) for t in (self._reader(), self._mic(), self._render(), self._sender())]
        await self.until(lambda: self.session_id, 15)
        if not self.session_id:
            raise RuntimeError(f"no ready: {self.controls[-3:]}")
        self._send_json({"type": "hello", "sample_rate": MIC_RATE, "playback_feedback": True})
        for i in range(5):
            self._send_json({"type": "clock_sync", "id": i, "client_send_ms": now()})
            await asyncio.sleep(0.08)
        await self.until(lambda: self.clock_offset is not None, 5)
        await asyncio.sleep(1.5)   # the session's own prewarm, as a page would

    async def close(self) -> None:
        try:
            self._send_json({"type": "bye"})
            await asyncio.sleep(0.2)
        except Exception:
            pass
        for task in self.tasks:
            task.cancel()
        await asyncio.gather(*self.tasks, return_exceptions=True)
        try:
            await self.ws.close()
        except Exception:
            pass

    async def _sender(self) -> None:
        while True:
            message = await self.outbox.get()
            await self.ws.send(message)

    async def _reader(self) -> None:
        async for message in self.ws:
            at = now()
            if isinstance(message, bytes):
                turn_id, generation, _seq = HEADER.unpack_from(message, 0)
                span = self.frames.setdefault(generation, [at, at, 0])
                span[1] = at
                span[2] += 1
                pcm = np.frombuffer(message[HEADER.size:], "<i2").astype(np.float32) / 32768.0
                meta = dict(self.meta or {"generation_id": generation, "turn_id": turn_id, "phrase_id": f"?{generation}", "role": "?"})
                self.player.push_pcm(meta, pcm)
                continue
            data = json.loads(message)
            self.controls.append((at, data))
            kind = data.get("type")
            if kind == "ready":
                self.session_id = data["session_id"]
                self.token = data.get("session_token")
                self.player.set_rate(int(data.get("output_sample_rate") or 24000))
            elif kind == "state":
                self.state = data.get("state")
            elif kind == "speaking":
                # Server gửi "speaking" thành message riêng, không qua "state".
                self.state = "speaking"
                self.player.set_rate(int(data.get("sample_rate") or self.player.rate))
            elif kind == "audio_segment":
                self.meta = {k: data[k] for k in ("phrase_id", "role", "generation_id", "turn_id")}
            elif kind == "audio_end":
                self.player.push_marker("end", dict(self.meta or {}))
            elif kind == "audio_generation_end":
                self.player.push_marker("generation_end", dict(self.meta or {"generation_id": data.get("generation_id")}))
            elif kind == "playback_reset":
                self.player.reset(int(data.get("generation_id", 0)) + 1, at)
            elif kind == "clock_sync":
                rtt = max(0.0, at - data["client_send_ms"] - (data["server_send_ms"] - data["server_receive_ms"]))
                if rtt / 2 < self.clock_uncertainty:
                    self.clock_uncertainty = rtt / 2
                    self.clock_offset = ((data["server_receive_ms"] - data["client_send_ms"]) + (data["server_send_ms"] - at)) / 2
                    self._send_json({"type": "clock_sync_result", "offset_ms": self.clock_offset,
                                     "uncertainty_ms": self.clock_uncertainty})

    async def _render(self) -> None:
        while True:
            self.player.render(now())
            await asyncio.sleep(0.005)

    async def _mic(self) -> None:
        size = MIC_RATE * FRAME_MS // 1000
        pending = np.zeros(0, np.float32)
        tick = now()
        while True:
            if pending.size == 0 and self.current is not None:
                self.current.done_ms = now()
                self.current = None
            if pending.size == 0 and self.queue:
                self.current = self.queue.popleft()
                pending = self.current.pcm
            if pending.size:
                chunk, pending = pending[:size], pending[size:]
                if chunk.size < size:
                    chunk = np.pad(chunk, (0, size - chunk.size))
            else:
                chunk = np.zeros(size, np.float32)
            sent_at = now()
            clip = self.current
            if clip is not None:
                if clip.start_ms is None:
                    clip.start_ms = sent_at
                if float(np.sqrt(np.mean(chunk ** 2))) >= ONSET_RMS:
                    if clip.onset_ms is None:
                        clip.onset_ms = sent_at
                    clip.end_ms = sent_at + FRAME_MS
                    clip.voiced_marks.append(sent_at)
            if self.echo_gain is not None:
                end = sent_at - self.echo_delay_ms
                gain = self.echo_gain
                if self.echo_onset_gain is not None:
                    onsets = [e["client_ms"] for e in self.player.events[-20:] if e["event"] == "playback_started"]
                    since = end - onsets[-1] if onsets else float("inf")
                    if 0 <= since < self.echo_converge_ms:
                        mix = since / self.echo_converge_ms
                        gain = self.echo_onset_gain * (1 - mix) + self.echo_gain * mix
                chunk = chunk + gain * self.player.echo(end - FRAME_MS, end, MIC_RATE)[:size]
            await self.ws.send((np.clip(chunk, -1, 1) * 32767).astype("<i2").tobytes())
            tick += FRAME_MS
            self.mic_lateness_ms = max(self.mic_lateness_ms, now() - tick)
            await asyncio.sleep(max(0.0, (tick - now()) / 1000.0))

    # --- helpers -----------------------------------------------------------
    def say(self, label: str, pcm: np.ndarray) -> Clip:
        clip = Clip(label=label, pcm=pcm, queued_ms=now())
        self.queue.append(clip)
        return clip

    async def spoken(self, clip: Clip, timeout: float = 30) -> None:
        await self.until(lambda: clip.done_ms is not None, timeout)

    async def until(self, predicate, timeout: float) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if predicate():
                return True
            await asyncio.sleep(0.01)
        return bool(predicate())

    def after(self, t: float, kind: str, **match) -> list[tuple[float, dict]]:
        return [(at, c) for at, c in self.controls
                if at >= t and c.get("type") == kind and all(c.get(k) == v for k, v in match.items())]

    def playback_after(self, t: float, event: str, **match) -> list[dict]:
        return [e for e in self.player.events
                if e["client_ms"] >= t and e["event"] == event and all(e.get(k) == v for k, v in match.items())]

    async def idle(self, timeout: float = 30) -> bool:
        return await self.until(lambda: self.state == "idle" and not self.player.active, timeout)

    async def stop_answer(self) -> None:
        """The interrupt button: end whatever is playing and wait for idle."""
        if self.state != "idle" or self.player.active:
            self._send_json({"type": "interrupt"})
        await self.idle(20)
        await asyncio.sleep(0.3)

    def turns(self, limit: int = 50) -> list[dict]:
        return http_json(self.base, f"/sessions/{self.session_id}/turns?limit={limit}", self.token)["turns"]

    def counters(self) -> dict:
        return http_json(self.base, f"/sessions/{self.session_id}/turns?limit=1", self.token).get("counters", {})


def clip_row(clip: Clip) -> dict:
    return {"label": clip.label, "start_ms": clip.start_ms, "onset_ms": clip.onset_ms,
            "end_ms": clip.end_ms, "done_ms": clip.done_ms, "duration_ms": 1000 * clip.pcm.size / MIC_RATE}


def turns_between(turns: list[dict], start_turn: int) -> list[dict]:
    return [t for t in turns if t["turn_id"] > start_turn]


def last_turn_id(client: Client) -> int:
    turns = client.turns(limit=1)
    return max((t["turn_id"] for t in turns), default=0)


# ---------------------------------------------------------------- families

async def utterance_case(client: Client, case: dict) -> dict:
    """hold / complete / continue: say it, let the pipeline decide, then stop."""
    await client.idle(20)
    before = last_turn_id(client)
    started = now()
    pcm = decode(case["pcm"])
    clip = client.say(case["id"], pcm)
    await client.spoken(clip, 60)
    # The last turn is the one whose answer starts AFTER the whole clip was said.
    got = await client.until(lambda: any(g for g, span in client.frames.items() if span[0] >= clip.end_ms), 20)
    answered_at = min((span[0] for g, span in client.frames.items() if span[0] >= clip.end_ms), default=None)
    if got:
        # Đợi tới lúc NỘI DUNG ra loa (câu mở "Vâng." không tính) để đo được
        # độ trễ nội dung, rồi mới bấm dừng cho ca kế tiếp.
        await client.until(lambda: client.playback_after(clip.end_ms, "playback_started", role="content"), 8)
        await asyncio.sleep(0.2)
    await client.stop_answer()
    part2_onset = None
    if "pause_at_ms" in case and clip.start_ms is not None:
        # First voiced frame after the inserted pause.
        boundary = clip.start_ms + case["pause_at_ms"] + case["pause_ms"] - FRAME_MS
        part2_onset = next((m for m in clip.voiced_marks if m >= boundary), None)
    return {
        "case": {k: v for k, v in case.items() if k != "pcm"}, "clip": clip_row(clip),
        "part2_onset_ms": part2_onset, "answer_first_frame_ms": answered_at,
        "controls": [(at, c) for at, c in client.controls if at >= started and c.get("type") != "clock_sync"],
        "playback": [e for e in client.player.events if e["client_ms"] >= started],
        "turns": turns_between(client.turns(), before), "started_ms": started,
    }


async def ask_long(client: Client, stimuli: dict, index: int) -> tuple[int | None, float | None, int]:
    """Ask for a long answer; return (generation, first playback start, turn before)."""
    await client.idle(20)
    before = last_turn_id(client)
    voices = sorted(stimuli["long_question"]["pcm_by_voice"])
    clip = client.say("long_question", decode(stimuli["long_question"]["pcm_by_voice"][voices[index % len(voices)]]))
    await client.spoken(clip, 30)
    asked = clip.end_ms or now()
    ok = await client.until(lambda: client.playback_after(asked, "playback_started"), 20)
    if not ok:
        return None, None, before
    first = client.playback_after(asked, "playback_started")[0]
    return first["generation_id"], first["client_ms"], before


async def echo_answer(client: Client, stimuli: dict, index: int, listen_s: float) -> dict:
    """No user speech at all: only the speaker's own echo in the mic."""
    generation, play_start, before = await ask_long(client, stimuli, index)
    if generation is None:
        await client.stop_answer()
        return {"case": {"id": f"echo-{index:03d}"}, "error": "no long answer"}
    await client.until(lambda: client.state == "idle" or now() - play_start >= 1000 * listen_s, listen_s + 5)
    resets = [(at, c) for at, c in client.after(play_start, "playback_reset")]
    ended = now()
    await client.stop_answer()
    return {"case": {"id": f"echo-{index:03d}", "expect": "no_barge_in"}, "family": "echo",
            "play_start_ms": play_start, "listened_ms": ended - play_start,
            "resets": [at for at, _ in resets], "turns": turns_between(client.turns(), before)}


async def over_answer_chain(client: Client, stimuli: dict, chain: list[dict], index: int, family: str) -> list[dict]:
    generation, play_start, before = await ask_long(client, stimuli, index)
    rows: list[dict] = []
    if generation is None:
        await client.stop_answer()
        return [{"case": {k: v for k, v in c.items() if k != "pcm"}, "error": "no long answer"} for c in chain]
    anchor = play_start
    current_gen = generation
    for case in chain:
        target = anchor + 1000 * case["offset_s"]
        await client.until(lambda: now() >= target, 10)
        # Chỉ đo khi trợ lý thật sự đang phát tiếng: câu trả lời đã hết thì ca
        # này không còn là "chen vào lúc máy đang nói".
        if not (client.player.active and client.state == "speaking"):
            rows.append({"case": {k: v for k, v in case.items() if k != "pcm"}, "skipped": "answer not playing",
                         "state": client.state})
            continue
        started = now()
        playing_gen = (client.player.meta or {}).get("generation_id", current_gen)
        clip = client.say(case["id"], decode(case["pcm"]))
        await client.spoken(clip, 20)
        # Settle: a reset (or none), then a resume, a new answer, or silence.
        onset = clip.onset_ms or clip.start_ms
        await client.until(lambda: client.after(onset - 5, "playback_reset"), 2.5)
        reset = client.after(onset - 5, "playback_reset")
        outcome = "not_fired"
        resume_ctl = None
        new_gen = None
        if reset:
            reset_at = reset[0][0]

            def settled() -> bool:
                return bool(client.after(reset_at, "state", state="thinking")) and any(
                    g > playing_gen and span[0] >= reset_at for g, span in client.frames.items()
                ) or (client.state == "idle" and now() - (clip.done_ms or now()) > 2500)

            await client.until(settled, 20)
            resumes = client.after(reset_at, "state", state="thinking", source="resume")
            fresh = [g for g, span in client.frames.items() if g > playing_gen and span[0] >= reset_at]
            if resumes:
                outcome, resume_ctl = "resumed", resumes[0][0]
            elif fresh:
                outcome = "answered"
            else:
                outcome = "abandoned"
            new_gen = min(fresh) if fresh else None
        new_play = None
        if new_gen is not None:
            await client.until(lambda: client.playback_after(started, "playback_started", generation_id=new_gen), 10)
            starts = client.playback_after(started, "playback_started", generation_id=new_gen)
            new_play = starts[0]["client_ms"] if starts else None
        old_frames_after_reset = 0
        if reset:
            span = client.frames.get(playing_gen)
            old_frames_after_reset = int(span is not None and span[1] > reset[0][0] + 1)
        stops = [e for e in client.player.events if e["event"] == "playback_stopped"
                 and e.get("reason") == "reset" and e["client_ms"] >= onset - 5]
        rows.append({
            "case": {k: v for k, v in case.items() if k != "pcm"}, "family": family, "clip": clip_row(clip),
            "playing_generation": playing_gen, "reset_ms": reset[0][0] if reset else None,
            "emulated_stop_ms": stops[0]["client_ms"] if stops else None,
            "outcome": outcome, "resume_control_ms": resume_ctl, "new_generation": new_gen,
            "new_playback_ms": new_play, "old_frames_after_reset": old_frames_after_reset,
            "started_ms": started,
        })
        if outcome in ("answered", "abandoned"):
            break
        if new_gen is not None:
            current_gen = new_gen
        anchor = now()
    await asyncio.sleep(0.5)
    await client.stop_answer()
    turns = turns_between(client.turns(), before)
    for row in rows:
        row["turns"] = turns
    return rows


# ---------------------------------------------------------------- main

async def run(args) -> int:
    base = args.base.rstrip("/")
    stimuli = json.loads(Path(args.stimuli).read_text())
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    health = http_json(base, "/readyz")
    if not health.get("ok"):
        raise SystemExit("server not ready")
    if http_json(base, "/sessions"):
        raise SystemExit("server has live sessions; use an isolated benchmark instance")
    manifest = {
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "base": base, "readyz": health,
        "config": http_json(base, "/config"), "stimuli_sha256": stimuli.get("sha256"),
        "families": args.families, "limit": args.limit, "echo_db": args.echo_db,
        "echo_delay_ms": args.echo_delay_ms, "echo_onset_db": args.echo_onset_db,
        "echo_converge_ms": args.echo_converge_ms, "per_session": args.per_session,
        "harness_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "basis": (stimuli.get("basis", "unspecified input") + "; "
                  "Python emulation of web/playback-worklet.js; same-host monotonic clock; "
                  "no real AEC or LAN"),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1))
    families = args.families.split(",")
    for family in families:
        cases = stimuli["cases"].get(family) or [{"id": f"echo-{i:03d}"} for i in range(args.echo_answers)]
        if args.limit:
            cases = cases[: args.limit]
        rows: list[dict] = []
        client = None
        done_in_session = 0
        index = 0
        t_family = time.monotonic()
        chain_len = 1 if family == "interrupt" else args.chain
        batches = [cases[i:i + chain_len] for i in range(0, len(cases), chain_len)] \
            if family in ("backchannel", "noise", "interrupt") else [[c] for c in cases]
        for batch in batches:
            if client is None or done_in_session >= args.per_session:
                if client is not None:
                    rows.append({"session_counters": client.counters(), "session_id": client.session_id,
                                 "mic_lateness_ms": client.mic_lateness_ms})
                    await client.close()
                    await asyncio.sleep(0.5)
                client = Client(base, args.echo_db, args.echo_delay_ms, args.echo_onset_db, args.echo_converge_ms)
                await client.open()
                done_in_session = 0
            try:
                if family == "echo":
                    if args.echo_db is None:
                        raise SystemExit("--families echo needs --echo-db")
                    rows.append(await echo_answer(client, stimuli, index, args.echo_listen_s))
                elif family in ("backchannel", "noise", "interrupt"):
                    result = await over_answer_chain(client, stimuli, batch, index, family)
                    rows.extend(result)
                else:
                    rows.append(await utterance_case(client, batch[0]))
            except Exception as exc:
                rows.append({"case": {k: v for k, v in batch[0].items() if k != "pcm"},
                             "error": f"{type(exc).__name__}: {exc}"})
                try:
                    await client.close()
                except Exception:
                    pass
                client = None
            index += 1
            done_in_session += len(batch)
            done_cases = sum(len(b) for b in batches[:index])
            print(f"{family} {done_cases}/{len(cases)}"
                  f" last={rows[-1].get('outcome') or rows[-1].get('error') or 'ok'}", flush=True)
        if client is not None:
            rows.append({"session_counters": client.counters(), "session_id": client.session_id,
                         "mic_lateness_ms": client.mic_lateness_ms})
            await client.close()
        (out / f"{family}.json").write_text(json.dumps({
            "family": family, "elapsed_s": round(time.monotonic() - t_family, 1), "rows": rows,
        }, ensure_ascii=False))
    manifest["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base", default="https://127.0.0.1:19101")
    parser.add_argument("--stimuli", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--families", default="hold,complete,continue,backchannel,noise,interrupt")
    parser.add_argument("--limit", type=int, default=0, help="số ca đầu mỗi họ (0 = tất cả)")
    parser.add_argument("--chain", type=int, default=3, help="số lần chen vào mỗi câu trả lời dài")
    parser.add_argument("--per-session", type=int, default=12, help="mở phiên mới sau N ca")
    parser.add_argument("--echo-db", type=float, default=None, help="trộn tiếng vọng của loa vào micro")
    parser.add_argument("--echo-delay-ms", type=float, default=80.0)
    parser.add_argument("--echo-onset-db", type=float, default=None, help="tiếng vọng lúc AEC chưa hội tụ")
    parser.add_argument("--echo-converge-ms", type=float, default=300.0)
    parser.add_argument("--echo-answers", type=int, default=10, help="họ echo: số câu trả lời dài để nghe")
    parser.add_argument("--echo-listen-s", type=float, default=12.0)
    return asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    sys.exit(main())
