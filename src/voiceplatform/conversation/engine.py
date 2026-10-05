"""The realtime conversation engine.

This is the product. Everything else is an adapter: swap PhoWhisper for
Gipformer, the cascade for a native S2S model, WebSocket for WebRTC, and this
file does not change.

What it owns:

* when a turn starts and, harder, when it ends;
* barge-in, and stopping everything fast enough to matter;
* generation fencing, so a cancelled turn cannot leak audio or text forward;
* the fast path (ASR -> LLM -> TTS) staying responsive while the slow path
  (tools, retrieval, backends) takes as long as it takes;
* a timeline of every stage, because latency you cannot see you cannot fix.
"""

from __future__ import annotations

import asyncio
import contextlib
import re
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..core.audio import AudioFrame, RingBuffer
from ..core.clock import now_ms
from ..core.config import Config
from ..core.errors import ModelTimeout, ModelUnavailable
from ..core.events import Event, EventType
from ..core.ids import GenerationKey, new_session_id
from ..media.framer import Framer
from ..media.preprocess import build_preprocessor
from ..media.transport.base import ControlMessage
from ..media.vad import build_vad
from ..media.vad.gate import GateEdge, SpeechGate
from ..models.base import Message, Transcript
from ..models.registry import ModelPlane
from ..observability.logging import get_logger
from ..observability.probe import Probe, observing
from ..observability.trace import SessionTrace
from ..tasks.base import TaskContext
from ..tasks.executor import TaskExecutor
from ..tasks.registry import ToolRegistry
from ..tasks.search import SearchAgent, SearchRequest, SearchResult
from .barge_in import BargeInDetector
from .context import ConversationContext
from .pending import PendingSearches
from .segmenter import PhraseSegmenter, pipeline_segmenter
from .pauses import shape_pauses
from .first_phrase import with_first_phrase_deadline
from .speech_normalize import normalize_for_speech
from .sink import AudioSink
from .speculation import Speculation
from .state import TurnState, TurnStateMachine
from .turn_detector import build_turn_detector

log = get_logger("engine")


async def _aiter(items: list[Any]):
    """Cho audio dựng sẵn đi qua đúng đường mà audio vừa tổng hợp đi."""
    for item in items:
        yield item


async def _held_audio(frames: list[AudioFrame], skip: int, done: asyncio.Event | None, progress: asyncio.Event):
    """Audio a cut turn already has, from sample `skip` on.

    `frames` may still be growing: the cut phrase can be mid-synthesis when
    the interruption lands, and its talker is left to finish it into this
    list (see ResponseState.holding). `done` is set when it has.
    """
    index = 0
    position = 0
    while True:
        while index < len(frames):
            frame = frames[index]
            index += 1
            size = frame.samples.size
            if position + size <= skip:
                position += size
                continue
            start = max(0, skip - position)
            position += size
            yield frame if start == 0 else AudioFrame(samples=frame.samples[start:], sample_rate=frame.sample_rate)
        if done is None or done.is_set():
            if index >= len(frames):
                return
            continue
        progress.clear()
        if index < len(frames) or done.is_set():
            continue
        await progress.wait()


def _quiet_point(frames: list[AudioFrame], target: int, window_ms: float = 20.0) -> int:
    """The quietest spot near `target` samples in: resume there, not mid-syllable."""
    if target <= 0 or not frames:
        return 0
    rate = frames[0].sample_rate
    pcm = np.concatenate([f.samples for f in frames])
    window = max(1, int(rate * window_ms / 1000))
    lo = max(0, target - int(0.2 * rate))
    hi = min(pcm.size - window, target + int(0.1 * rate))
    if hi <= lo:
        return max(0, min(target, pcm.size))
    step = max(1, int(rate * 0.005))
    starts = np.arange(lo, hi, step)
    energy = [float(np.mean(pcm[i:i + window] ** 2)) for i in starts]
    return int(starts[int(np.argmin(energy))])

_MAX_TOOL_ROUNDS = 2
# Đệm chống jitter của bộ phát CŨ (web/client.js playFrame và playback.js
# legacy: `cushion`), tức client không bật playback feedback. Client đang chạy
# thật phát qua AudioWorklet và đệm `playback_buffer_ms` (xem ResponseState).
_CLIENT_CUSHION_S = 0.06
# How long past the scheduled end of playback a turn waits for the client's
# `playback_generation_end` before ending anyway: network, output latency and
# the error of the schedule mirror, not a model deadline.
_PLAYBACK_DONE_MARGIN_S = 2.0
# Worst-case relative drift of the browser's and the server's monotonic
# clocks (±50 ppm each): how fast a stored clock estimate loses accuracy.
_CLOCK_DRIFT_MS_PER_S = 0.1
_FALLBACK_REPLY = "Xin lỗi, tôi đang gặp trục trặc. Bạn nhắc lại giúp tôi nhé."

# Lời đệm của người NGHE: nói lúc trợ lý đang nói là để báo "tôi vẫn nghe",
# không phải để giành lượt. Chỉ so trên cả câu (sau khi bỏ dấu câu), nên
# "vâng, nhưng mà..." vẫn là một lượt thật.
_BACKCHANNELS = frozenset(
    {
        "ừ", "ừm", "ừ ừ", "ừa", "ờ", "ờ ờ", "à", "ừ hử", "hm", "hmm", "uh",
        "um", "vâng", "vâng ạ", "dạ", "dạ vâng", "vâng vâng", "dạ dạ", "ok",
        "oke", "okay", "ô kê", "đúng rồi", "ừ đúng rồi", "vâng đúng rồi",
        "ồ", "ô", "à há", "ờ há", "ừ hứ", "uh huh",
    }
)


# Words that make a short interjection a request even when it starts like a
# backchannel: "ừ, dừng", "vâng nhưng…".
_COMMAND_WORDS = frozenset({
    "dừng", "thôi", "khoan", "đợi", "chờ", "không", "nhưng", "mà", "sai", "nhầm",
    "stop", "lại", "chậm", "ngắn", "hỏi", "gì", "sao", "hả", "nào", "đâu",
})


def _commands(text: str) -> bool:
    words = "".join(c if c.isalnum() or c.isspace() else " " for c in text.lower()).split()
    return any(w in _COMMAND_WORDS for w in words)


def _is_backchannel(text: str) -> bool:
    words = "".join(c if c.isalnum() or c.isspace() else " " for c in text.lower()).split()
    return bool(words) and " ".join(words) in _BACKCHANNELS


class FencedSink:
    """The last gate before audio leaves the process.

    Cancellation is asynchronous: `task.cancel()` only lands at the next await,
    so a loop that checks the fence at the top can still write one more frame
    before it stops. Checking here — immediately before the write, with nothing
    awaited in between — narrows that window to the write itself. Control
    messages are never fenced: `playback_reset` exists precisely to be sent
    after the generation it refers to is dead.
    """

    def __init__(self, sink: AudioSink, gen: "GenerationManager") -> None:
        self._sink = sink
        self._gen = gen
        self.dropped = 0

    async def send_audio(self, frame: AudioFrame, key: GenerationKey) -> None:
        if not self._gen.check(key):
            self.dropped += 1
            return
        await self._sink.send_audio(frame, key)

    async def send_control(self, message: ControlMessage) -> None:
        await self._sink.send_control(message)


@dataclass(slots=True)
class Phrase:
    """Một cụm chờ phát. `audio` khác None nghĩa là đã tổng hợp sẵn."""

    text: str
    audio: list[Any] | None = None
    filler: bool = False
    role: str = "content"
    phrase_id: str = ""
    ready_at_ms: float = field(default_factory=now_ms)

    def __post_init__(self):
        if self.filler and self.role == "content":
            self.role = "filler"
        if self.role == "content" and re.fullmatch(r"(?:(?:vâng|dạ|ừ|được|ok|okay)[.!?,…\s]*)+", self.text.strip(), re.IGNORECASE):
            self.role, self.filler = "ack", True
    # Câu đệm (câu mở, lời chờ công cụ, câu báo tra cứu): không phải nội dung
    # trả lời. Người dùng mới nghe câu đệm thì coi như CHƯA nghe câu trả lời.


@dataclass(slots=True)
class _Sent:
    """A phrase fully handed to the client, and when the client plays it."""

    phrase: Phrase
    start: float   # time.monotonic() of its first sample leaving the speaker
    end: float
    frames: list[AudioFrame] = field(default_factory=list)   # what was sent, for a resume


@dataclass(slots=True)
class ResponseState:
    """What the current generation has produced so far.

    "Sent" is not "heard". The talker runs faster than real time (ZeroTTS at
    RTF ~0.5), so the server finishes sending a phrase long before the client
    finishes playing it. `schedule()` replays the client's player here, and
    `play_end` is where its buffer runs dry: it is what lets history record
    what the user actually heard, and a resume start from the phrase they
    were actually hearing.

    Two players. The live one (playback feedback on) is an AudioWorklet: it
    plays nothing until `startup_s` is buffered or the phrase's end marker
    arrives, then plays back to back, and running dry — an underrun, or a
    phrase boundary with nothing queued behind it — re-arms that buffer. The
    legacy scheduler (`startup_s == 0`) starts each frame at
    max(now + cushion, end of the previous one).
    """

    key: GenerationKey
    generated: list[str] = field(default_factory=list)
    spoken: list[str] = field(default_factory=list)   # heard in full
    history_spoken: list[str] = field(default_factory=list)  # excludes cached fillers
    first_audio_ms: float | None = None
    queue: asyncio.Queue | None = None
    speak_task: asyncio.Task | None = None
    sent: list[_Sent] = field(default_factory=list)
    current: Phrase | None = None          # dequeued, not fully sent yet
    current_start: float | None = None
    play_end: float = 0.0                  # mirror of the client's nextStartAt
    playback_done: asyncio.Event = field(default_factory=asyncio.Event)
    roles_sent: set[str] = field(default_factory=set)
    output_role: str = "content"
    llm_done: bool = False                 # every LLM round completed
    # The talker has taken the end-of-answer marker off `queue`. A resume
    # must not wait on that queue again: nothing more will ever arrive, and
    # the resumed turn hung until the orphan sweep (20 s) — found 29/09 by a
    # cough over the playback TAIL, after the last frame had been sent.
    queue_ended: bool = False
    # Heard in full before this response took over (a resume): the history
    # of an answer cut twice must keep what the first part said.
    heard_before: list[str] = field(default_factory=list)
    # Playback feedback: when the client REALLY started / finished each
    # phrase (monotonic seconds, server clock). The schedule mirror assumes a
    # 60 ms cushion; the worklet buffers 160 ms first and underruns shift
    # everything after them, so where feedback exists it wins.
    observed_start: dict[str, float] = field(default_factory=dict)
    observed_end: dict[str, float] = field(default_factory=dict)
    offset_s: float = 0.0
    cut_observed_s: float | None = None    # client stop on playback_reset
    onset_seen: bool = False
    # Frames of the phrase being sent. While `holding`, the talker finishes
    # the phrase into this list instead of sending it: a false interruption
    # then continues from where playback stopped.
    current_frames: list[AudioFrame] = field(default_factory=list)
    holding: bool = False
    hold_done: asyncio.Event = field(default_factory=asyncio.Event)
    hold_progress: asyncio.Event = field(default_factory=asyncio.Event)
    # The client's startup buffer (`ready.playback_buffer_ms`); 0 = legacy.
    startup_s: float = 0.0
    buffering: bool = True        # the worklet is filling that buffer, not playing
    buffered_s: float = 0.0
    current_placed: bool = False  # the current phrase has a frame on the timeline
    # Where the current phrase's first frame sits in a buffer not yet playing:
    # its start is known only once that buffer starts.
    pending_offset: float | None = None
    # False for a reply that has no turn of its own in the history (a fallback
    # after the transcript itself failed): committing it would overwrite the
    # previous turn's answer.
    in_history: bool = True

    def begin_phrase(self, phrase: Phrase) -> None:
        self.current = phrase
        self.current_frames = []
        self.current_start = None
        self.current_placed = False
        self.pending_offset = None

    def schedule(self, duration_s: float, now: float | None = None) -> float | None:
        """Place one frame of the current phrase on the client's timeline.

        Returns its start, or None while the client is still buffering
        (`current_start` is filled in when the buffer starts).
        """
        now = time.monotonic() if now is None else now
        first, self.current_placed = not self.current_placed, True
        if not self.startup_s:
            start = max(now + _CLIENT_CUSHION_S, self.play_end)
        else:
            if not self.buffering and self.play_end <= now:
                self.buffering, self.buffered_s = True, 0.0   # ran dry: buffer again
            if self.buffering:
                offset = self.buffered_s
                self.buffered_s += duration_s
                if first:
                    self.pending_offset = offset
                if self.buffered_s + 1e-9 < self.startup_s:   # the worklet counts samples
                    return None
                self._start_playing(now)
                return now + offset
            start = self.play_end
        self.play_end = start + duration_s
        if first:
            self.current_start = start
        return start

    def end_phrase(self, now: float | None = None) -> None:
        """The phrase's end marker went out: a worklet still buffering starts on it."""
        if self.startup_s and self.buffering and self.buffered_s > 0:
            self._start_playing(time.monotonic() if now is None else now)

    def _start_playing(self, at: float) -> None:
        self.play_end = at + self.buffered_s
        self.buffering, self.buffered_s = False, 0.0
        if self.pending_offset is not None:
            self.current_start = at + self.pending_offset
            self.pending_offset = None

    def observe(self, phrase_id: str, event: str, at_s: float) -> None:
        if event == "playback_started":
            self.observed_start.setdefault(phrase_id, at_s)
            scheduled = next((x.start for x in self.sent if x.phrase.phrase_id == phrase_id), None)
            if scheduled is None and self.current is not None and self.current.phrase_id == phrase_id:
                scheduled = self.current_start
            if scheduled is not None:
                self.offset_s = at_s - scheduled
        elif event == "playback_stopped":
            self.observed_end.setdefault(phrase_id, at_s)

    def _start(self, sent: _Sent) -> float:
        return self.observed_start.get(sent.phrase.phrase_id, sent.start + self.offset_s)

    def _end(self, sent: _Sent) -> float:
        return self.observed_end.get(sent.phrase.phrase_id, sent.end + self.offset_s)

    def current_started(self) -> float | None:
        if self.current is None:
            return None
        observed = self.observed_start.get(self.current.phrase_id)
        if observed is not None:
            return observed
        return None if self.current_start is None else self.current_start + self.offset_s

    def heard_by(self, at: float) -> list[Phrase]:
        return [s.phrase for s in self.sent if self._end(s) <= at]

    def unheard_by(self, at: float) -> list[Phrase]:
        """Content the user has not heard in full: replay from the cut phrase."""
        rest = [s.phrase for s in self.sent if self._end(s) > at and not s.phrase.filler]
        if self.current is not None and not self.current.filler:
            rest.append(self.current)
        return rest

    def content_heard_by(self, at: float) -> bool:
        if any(not s.phrase.filler and self._start(s) <= at for s in self.sent):
            return True
        started = self.current_started()
        return (
            self.current is not None
            and not self.current.filler
            and started is not None
            and started <= at
        )

    @property
    def generated_text(self) -> str:
        return "".join(self.generated)

    @property
    def spoken_text(self) -> str:
        return " ".join(self.spoken)


@dataclass(slots=True)
class _Interrupted:
    """A turn a user-speech barge-in just cut, kept until that speech resolves.

    Its LLM is NOT cancelled with it: the producer tasks are moved out of the
    generation and keep writing into the old phrase queue ("draining"), so a
    resume continues the answer instead of asking the model again from the top.
    A real new turn cancels the drain.
    """

    key: GenerationKey
    response: ResponseState | None
    user_text: str
    at_ms: float           # audio clock
    replay: list[Phrase]   # content not heard in full when it was cut
    content_heard: bool
    drain: list[asyncio.Task] = field(default_factory=list)
    at_s: float = 0.0      # time.monotonic() of the cut
    hold: asyncio.Task | None = None   # the talker finishing the cut phrase into memory
    # Earlier unanswered turns whose words `user_text` already carries: a merge
    # drops them from the history along with this one.
    absorbed: tuple[int, ...] = ()


class ConversationEngine:
    def __init__(
        self,
        config: Config,
        models: ModelPlane,
        sink: AudioSink,
        *,
        session_id: str | None = None,
        executor: TaskExecutor | None = None,
        voice: str | None = None,
        search_agent: SearchAgent | None = None,
    ) -> None:
        self._phrase_sequence = 0
        self._known_phrases = {}
        self._last_voice_ingress_ms = None
        self._last_voice_audio_ms = None
        self.playback_feedback_enabled = False
        self.playback_clock = None
        self._playback_clock_at_ms = 0.0
        self._last_playback_feedback_ms = 0.0
        self.config = config
        self.models = models
        self.session_id = session_id or new_session_id()
        self._executor: TaskExecutor | None = None
        self.executor = executor
        self.voice = voice

        from .generation import GenerationManager  # local: avoids import cycle noise

        self.gen = GenerationManager(self.session_id)
        self.sink = FencedSink(sink, self.gen)
        self.trace = SessionTrace(self.session_id, keep_turns=config.observability.keep_turns)
        self.state = TurnStateMachine(on_change=self._on_state_change)
        self.context = ConversationContext(
            config.conversation.system_prompt, config.conversation.history_turns,
            tool_instruction=config.conversation.tool_instruction,
        )

        audio = config.audio
        self.framer = Framer(audio.sample_rate, config.frame_samples)
        self.preprocessor = build_preprocessor(config.media.aec.backend)
        self.vad = build_vad(
            config.media.vad.backend,
            threshold=config.media.vad.threshold,
            energy_threshold=config.media.vad.energy_threshold,
        )
        self.gate = SpeechGate(
            threshold=config.media.vad.threshold if config.media.vad.backend == "silero" else 0.5,
            start_frames=config.media.vad.start_frames,
            end_frames=config.media.vad.end_frames,
        )
        self.barge_in = BargeInDetector(config.conversation.barge_in)
        # A speech model of the interruption's own (barge_in.vad: silero),
        # kept apart from the gate so endpointing is untouched by the A/B.
        self.speech_vad = None
        bi = config.conversation.barge_in
        self._silero_triggers = bi.vad == "silero"
        if self._silero_triggers or bi.speech_model == "silero":
            from ..media.vad.silero import SileroVad

            self.speech_vad = SileroVad(threshold=0.5)
        self._interjection_speech_ms = 0.0
        self._burst_speech_ms = 0.0     # speech-model speech since the loud burst began
        self.turn_detector = build_turn_detector(
            config.conversation.turn_detection.backend,
            silence_ms=config.conversation.turn_detection.silence_ms,
            max_silence_ms=config.conversation.turn_detection.max_silence_ms,
            fast_silence_ms=config.conversation.turn_detection.fast_silence_ms,
            semantic_model_path=config.conversation.turn_detection.semantic_model_path,
            semantic_threshold=config.conversation.turn_detection.semantic_threshold,
            semantic_probe_timeout_ms=config.conversation.turn_detection.semantic_probe_timeout_ms,
        )
        self.preroll = RingBuffer(
            int(audio.sample_rate * config.media.vad.pre_roll_ms / 1000) or 1
        )
        # Longer history for an interjection that was detected late; see
        # BargeInConfig.burst_preroll_max_ms.
        burst_ms = max(config.conversation.barge_in.burst_preroll_max_ms, config.media.vad.pre_roll_ms)
        self.burst_ring = RingBuffer(int(audio.sample_rate * burst_ms / 1000) or 1)
        self._quiet_ms = 1e9
        self._burst_onset_ms: float | None = None

        # per-turn scratch
        # Turn timing runs on an audio clock (frames consumed), not wall time.
        # In a live session the two agree; everywhere else — tests, replays,
        # a burst of frames after a jitter stall — only audio time is right.
        self._audio_clock_ms = 0.0
        self._asr_stream: Any = None
        self._utterance_ms = 0.0   # everything handed to ASR, pre-roll included
        self._speech_ms = 0.0      # frames that actually looked like speech
        self._partial_text = ""
        # Did any periodic partial of this turn read as a backchannel? Kept
        # apart from _partial_text, which the endpoint decode overwrites: on a
        # real "ừ" gipformer's partial says "ừ" and the full decode says "từ".
        self._partial_backchannel = False
        self._endpoint_at_ms: float | None = None
        self._required_silence_ms = float(config.conversation.turn_detection.silence_ms)
        # Endpoint decode: (speech ms it covered, text). Covering every speech
        # frame is what makes it "stable" — usable to shorten the wait and to
        # stand in for the final decode.
        self._endpoint_seq = 0
        self._endpoint_task: asyncio.Task | None = None
        self._endpoint_text: tuple[float, str, float | None] | None = None
        self._speculation: Speculation | None = None
        self._spec_attempts = 0
        # (generation, audio-clock ms before which its first audio may not go out)
        self._commit_gate: tuple[GenerationKey, float] | None = None
        self._response: ResponseState | None = None
        self._watchdog_task: asyncio.Task | None = None
        self._delivery_task: asyncio.Task | None = None
        self._ack_audio: list[Any] | None = None
        self._last_filler_ms = float("-inf")
        self._interrupted: _Interrupted | None = None
        self._progress_ms = now_ms()   # last audio frame out / LLM text in
        # Producer of a cut turn -> generation its text now belongs to (None
        # while undecided). Lets a drained LLM keep streaming text to the
        # client under the resumed generation's id.
        self._drain_targets: dict[GenerationKey, GenerationKey | None] = {}
        self._drain_tasks: dict[GenerationKey, list[asyncio.Task]] = {}   # what each drain kept running
        self._confirmed_text: tuple[int, str] = (0, "")
        self._closed = False
        self.counters: dict[str, int] = {}

        # --- nửa "Back end - search" của sơ đồ --------------------------
        self.search_agent = search_agent
        self.pending = PendingSearches(
            max_inflight=config.conversation.search.max_inflight,
            ttl_ms=config.conversation.search.ttl_ms,
        )
        if self.search_agent is not None and config.conversation.search.enabled:
            if self._executor is None:
                self._executor = TaskExecutor(ToolRegistry())
                self._executor.bind(self._emit_tool_event)
            from ..tasks.builtin.search_tool import SearchTool

            self._executor.registry.register(
                SearchTool(
                    self._dispatch_search,
                    instant_ack=config.conversation.search.instant_ack,
                )
            )

    @property
    def executor(self) -> TaskExecutor | None:
        return self._executor

    @executor.setter
    def executor(self, value: TaskExecutor | None) -> None:
        self._executor = value
        if value is not None:
            value.bind(self._emit_tool_event)

    def _emit_tool_event(self, type: EventType, key: GenerationKey, data: dict[str, Any]) -> None:
        self.trace.record(Event.for_key(type, key, **data))

    # ------------------------------------------------------------------ #
    # lifecycle
    # ------------------------------------------------------------------ #
    def _probe(self, stage, key=None, **fields):
        turn_id = key.turn_id if key else self.gen.turn_id or None
        generation_id = key.generation_id if key else None
        def emit(kind, stamp, data):
            self.trace.record(Event(type=kind, session_id=self.session_id, turn_id=turn_id,
                                    generation_id=generation_id, ts_ms=stamp, data=data))
        return Probe(stage, emit, fields)

    def _prepare_phrase(self, phrase, key):
        if not phrase.phrase_id or self._known_phrases.get(phrase.phrase_id, (None,))[0] != key:
            self._phrase_sequence += 1
            phrase.phrase_id = f"g{key.generation_id}-p{self._phrase_sequence}"
            phrase.ready_at_ms = now_ms()
            self._known_phrases[phrase.phrase_id] = (key, phrase.role)
            while len(self._known_phrases) > 4096:
                self._known_phrases.pop(next(iter(self._known_phrases)))
            self.trace.record(Event.for_key(EventType.PHRASE_READY, key,
                phrase_id=phrase.phrase_id, role=phrase.role, text=phrase.text,
                chars=len(phrase.text), words=len(phrase.text.split()), cached=phrase.audio is not None))
        return phrase

    async def _enqueue_phrase(self, queue, phrase, key):
        await queue.put(self._prepare_phrase(phrase, key))

    def _filler_allowed(self, cooldown_ms: int, *, reserve: bool = True) -> bool:
        stamp = now_ms()
        if stamp - self._last_filler_ms < cooldown_ms:
            return False
        if reserve:
            self._last_filler_ms = stamp
        return True

    def set_playback_clock(self, offset_ms, uncertainty_ms):
        import math
        if (type(offset_ms) not in (float, int) or type(uncertainty_ms) not in (float, int)
                or not math.isfinite(offset_ms) or not math.isfinite(uncertainty_ms)
                or abs(offset_ms) > 1e13 or not 0 <= uncertainty_ms <= 1000):
            return
        # The two monotonic clocks drift apart (crystal tolerance at both ends):
        # however tight, an estimate kept for an hour can be off by hundreds of
        # ms. Its uncertainty therefore grows with age, so the client's
        # once-a-minute re-sync can replace it; compared unaged, it never could.
        now = now_ms()
        if self.playback_clock is None or uncertainty_ms < self.playback_clock[1] + (
                _CLOCK_DRIFT_MS_PER_S * (now - self._playback_clock_at_ms) / 1000):
            self.playback_clock = (offset_ms, uncertainty_ms)
            self._playback_clock_at_ms = now
            self._emit(EventType.CLIENT_CLOCK_SYNC, uncertainty_ms=uncertainty_ms)

    def playback_feedback(self, data):
        import math
        event_names = {e.value: e for e in (EventType.PLAYBACK_STARTED, EventType.PLAYBACK_SIGNAL_STARTED, EventType.PLAYBACK_GENERATION_END, EventType.PLAYBACK_STOPPED,
            EventType.PLAYBACK_UNDERRUN, EventType.PLAYBACK_RESUMED, EventType.PLAYBACK_BUFFER)}
        kind = event_names.get(data.get("event"))
        known = self._known_phrases.get(data.get("phrase_id", ""))
        client_ms = data.get("client_ms")
        if (kind is None or known is None or self.playback_clock is None or
                type(client_ms) not in (int, float) or not math.isfinite(client_ms)):
            return
        key, role = known
        if data.get("generation_id") != key.generation_id:
            return
        stamp = client_ms + self.playback_clock[0]
        if stamp < now_ms()-600000 or stamp > now_ms()+1000:
            return
        fields = {"phrase_id": data["phrase_id"], "role": role,
                  "clock_uncertainty_ms": self.playback_clock[1], "source": "legacy_scheduler_estimate" if data.get("source") == "legacy_scheduler_estimate" else "browser_audio_render"}
        for name in ("buffer_ms", "gap_ms", "audio_time_s", "output_latency_ms"):
            value = data.get(name)
            if type(value) in (int, float) and math.isfinite(value) and 0 <= value <= 600000:
                fields[name] = value
        reason = data.get("reason")
        if reason in ("reset",):
            fields["reason"] = reason
        self.trace.record(Event(type=kind, session_id=self.session_id, turn_id=key.turn_id,
                                generation_id=key.generation_id, ts_ms=stamp, data=fields))
        if kind is EventType.PLAYBACK_GENERATION_END and self._response is not None and self._response.key == key:
            self._response.playback_done.set()
        interrupted = self._interrupted.response if self._interrupted is not None else None
        for response in (self._response, interrupted):
            if response is None or response.key != key:
                continue
            if kind is EventType.PLAYBACK_STARTED:
                response.observe(data["phrase_id"], "playback_started", stamp / 1000.0)
                if response is self._response and not response.onset_seen:
                    response.onset_seen = True
                    self._anchor_guard(stamp)
            elif kind is EventType.PLAYBACK_STOPPED:
                if reason == "reset":
                    if response.cut_observed_s is None:
                        response.cut_observed_s = stamp / 1000.0
                else:
                    response.observe(data["phrase_id"], "playback_stopped", stamp / 1000.0)

    def _anchor_guard(self, client_start_ms: float) -> None:
        """Re-anchor the echo guard to the moment the client started playing.

        The frame carrying the first echo reaches this process one mic frame
        plus the uplink after the speaker starts; the audio clock of that
        frame is ~now + (start - now) + uplink.
        """
        cfg = self.config.conversation.barge_in
        if not cfg.guard_from_playback or not self.barge_in.armed:
            return
        uplink_ms = 2 * self.config.audio.frame_ms
        until = self._audio_clock_ms + (client_start_ms + cfg.guard_ms + uplink_ms - now_ms())
        self.barge_in.set_guard_until(until)
        self._emit(EventType.BARGE_IN_GUARD, anchor="playback",
                   guard_left_ms=round(until - self._audio_clock_ms, 1))

    async def start(self) -> None:
        if self.speech_vad is not None:
            # ~110 ms to build the ONNX session: not on the first frame.
            await asyncio.to_thread(self.speech_vad.load)
        self._emit(EventType.SESSION_OPEN, models=self.models.describe())
        await self.sink.send_control(
            ControlMessage(
                "ready",
                {
                    "session_id": self.session_id,
                    "measurement_schema": 2,
                    # One number for both ends: the client buffers this much
                    # before playing, and ResponseState models exactly that.
                    "playback_buffer_ms": self.config.conversation.barge_in.playback_startup_ms,
                    "session_token": getattr(self, "session_token", None),
                    "input_sample_rate": self.config.audio.sample_rate,
                    "frame_ms": self.config.audio.frame_ms,
                    "output_sample_rate": int(
                        getattr(self.models.tts, "capabilities").native_sample_rate
                    ),
                    "models": self.models.describe(),
                    "voice": self.voice,
                },
            )
        )
        if self._watchdog_task is None:
            self._watchdog_task = asyncio.create_task(self._watchdog(), name="orphan-sweep")
        if self._delivery_task is None and self.search_agent is not None:
            self._delivery_task = asyncio.create_task(
                self._delivery_loop(), name="search-delivery"
            )
        # Một lần cho cả câu mở lẫn câu báo tra cứu. Nền, không chặn phiên:
        # chưa kịp ấm thì lượt đầu chỉ chậm hơn, không hỏng.
        # spawn_detached, không phải create_task trần: asyncio chỉ giữ tham
        # chiếu yếu tới task, và phiên đóng giữa chừng phải huỷ được nó.
        self.gen.spawn_detached(self._prewarm(), name="prewarm-speech")
        touch = getattr(self.models, "touch", None)
        if touch is not None:
            self.gen.spawn_detached(touch(), name="touch-models")

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        background = [t for t in (self._watchdog_task, self._delivery_task) if t is not None]
        for task in background:
            if task is not None:
                task.cancel()
        await asyncio.gather(*background, return_exceptions=True)
        self._watchdog_task = None
        self._delivery_task = None
        await self._discard_speculation("session closed")
        await self.gen.cancel_all()
        await self._close_asr()
        if self.state.state is not TurnState.CLOSED:
            self.state.to(TurnState.CLOSED, "session closed")
        self._emit(EventType.SESSION_CLOSE, counters=self.counters)
        if self.config.observability.write_traces:
            try:
                self.trace.write_jsonl(self.config.observability.trace_dir)
            except OSError as exc:  # pragma: no cover - disk only
                log.warning("could not write trace: %s", exc)

    def set_voice(self, voice: str | None) -> str | None:
        """Giọng của RIÊNG phiên này. Ăn từ cụm kế tiếp."""
        wanted = (voice or "").strip() or None
        available = getattr(self.models.tts.capabilities, "voices", None) or []
        if wanted and available and wanted not in available:
            raise ValueError(f"giọng {wanted!r} không có. Engine khai: {', '.join(available)}")
        self.voice = wanted
        if wanted:
            self.gen.spawn_detached(self._prewarm(), name="prewarm-voice")
        return wanted

    # ------------------------------------------------------------------ #
    # inbound audio
    # ------------------------------------------------------------------ #
    async def push_audio(self, samples: np.ndarray, src_rate: int | None = None) -> None:
        """Entry point from the transport. Never blocks on model work."""
        if self._closed:
            return
        for frame in self.framer.push(samples, src_rate):
            await self._on_frame(frame)

    async def _on_frame(self, raw: AudioFrame) -> None:
        frame = self.preprocessor.process(raw)
        self._audio_clock_ms += frame.duration_ms
        self._count("frames_in")
        probability = self.vad.probability(frame)
        if probability >= self.gate.threshold:
            self._last_voice_ingress_ms = raw.captured_at_ms
            self._last_voice_audio_ms = self._audio_clock_ms

        voiced = None      # the speech model's verdict on this frame, if there is one
        if self.speech_vad is not None:
            voiced = self.speech_vad.probability(frame) >= 0.5
            if voiced and self._interrupted is not None and self.state.state is TurnState.LISTENING:
                self._interjection_speech_ms += frame.duration_ms
        trigger = (1.0 if voiced else 0.0) if self._silero_triggers else probability

        # 1. Interruption first: it must not wait on anything below.
        if self.state.is_assistant_active() and self.barge_in.update(
            trigger, frame, self._audio_clock_ms
        ):
            await self._handle_barge_in()
            # The interjection so far — the burst that fired the stop — counts.
            if voiced is not None:
                self._interjection_speech_ms = self._burst_speech_ms + (frame.duration_ms if voiced else 0.0)

        self.preroll.write(frame.samples)
        self.burst_ring.write(frame.samples)
        if frame.rms >= self.config.conversation.barge_in.min_rms:
            if self._quiet_ms >= 300.0:
                self._burst_onset_ms = self._audio_clock_ms - frame.duration_ms
                self._burst_speech_ms = 0.0
            self._quiet_ms = 0.0
        else:
            self._quiet_ms += frame.duration_ms
        if voiced:
            self._burst_speech_ms += frame.duration_ms
        edge = self.gate.update(probability)

        if self.state.state is TurnState.LISTENING and self._asr_stream is not None:
            self._utterance_ms += frame.duration_ms
            if probability >= self.gate.threshold:
                self._speech_ms += frame.duration_ms
            try:
                partial = await self._asr_stream.push(frame)
            except Exception as exc:
                log.warning("asr push failed: %s", exc)
                partial = None
            if partial is not None and partial.text:
                await self._on_partial(partial)

        if edge is GateEdge.START:
            if self.state.state is TurnState.IDLE:
                # Open the turn first: an event emitted before the turn id is
                # bumped lands in the previous turn's timeline.
                await self._begin_listening()
                self._emit(EventType.VAD_START, rms=round(frame.rms, 5))
            elif self.state.state is TurnState.LISTENING:
                self._emit(EventType.VAD_START, rms=round(frame.rms, 5))
                # Speech resumed inside the same utterance: the pause we were
                # timing was a pause, not an ending.
                self._endpoint_at_ms = None
                self._endpoint_seq += 1
                self._count("endpoint_cancelled")
                await self._discard_speculation("speech resumed")
        elif edge is GateEdge.END:
            self._emit(EventType.VAD_END, utterance_ms=round(self._utterance_ms, 1))
            if self.state.state is TurnState.LISTENING:
                await self._mark_endpoint()

        if self.state.state is TurnState.LISTENING:
            await self._maybe_confirm_turn()

    def _endpoint_stable(self) -> bool:
        """The endpoint decode still covers every speech frame of this turn."""
        return self._endpoint_text is not None and self._endpoint_text[0] == self._speech_ms

    async def _on_partial(self, transcript: Transcript) -> None:
        if self._endpoint_text is not None and (
            len(transcript.text.split()) <= len(self._endpoint_text[1].split())
        ):
            # A periodic decode started before the pause can land after the
            # endpoint decode; its audio is a prefix of what that one heard.
            # Letting it replace the decision text judged the pause on words
            # the speaker had already gone past — measured in a test: the
            # first frame of resumed speech reverted "... nghìn đến" to
            # "chuyển năm trăm" and confirmed the half-sentence on that frame.
            self._emit(EventType.ASR_PARTIAL, text=transcript.text, superseded=True)
            return
        self._partial_text = transcript.text
        self._partial_backchannel = self._partial_backchannel or _is_backchannel(transcript.text)
        if EventType.ASR_FIRST_PARTIAL.value not in self._turn_firsts():
            self._emit(EventType.ASR_FIRST_PARTIAL, text=transcript.text)
        self._emit(EventType.ASR_PARTIAL, text=transcript.text)
        await self.sink.send_control(
            ControlMessage("transcript", {"text": transcript.text, "final": False})
        )
        # A longer partial can change the endpointing decision mid-pause.
        if self._endpoint_at_ms is not None:
            self._required_silence_ms = await self._required_silence()

    def _turn_firsts(self) -> dict[str, float]:
        turn = self.trace.turn(self.gen.turn_id)
        return turn.firsts if turn else {}

    # ------------------------------------------------------------------ #
    # turn boundaries
    # ------------------------------------------------------------------ #
    async def _begin_listening(self) -> None:
        # A fresh turn from IDLE: whatever a past barge-in left behind is over.
        self._end_drain(self._interrupted)
        self._interrupted = None
        turn_id = self.gen.next_turn()
        self.state.to(TurnState.LISTENING, "speech started")
        self._utterance_ms = 0.0
        self._speech_ms = 0.0
        self._partial_text = ""
        self._partial_backchannel = False
        self._endpoint_at_ms = None
        self._reset_endpoint_state()
        await self._discard_speculation("new turn")
        self._required_silence_ms = float(self.config.conversation.turn_detection.silence_ms)
        self._emit(EventType.TURN_START, turn_id=turn_id)
        try:
            with observing(self._probe("asr")):
                self._asr_stream = await self.models.asr.open_stream(
                    sample_rate=self.config.audio.sample_rate
                )
        except (ModelUnavailable, ModelTimeout) as exc:
            log.error("ASR unavailable: %s", exc)
            self._emit(EventType.ERROR, stage="asr_open", error=str(exc))
            self.state.to(TurnState.IDLE, "asr unavailable")
            return
        self._emit(EventType.ASR_START)
        # Pre-roll: the first syllable lives before the gate opened.
        pre = self.preroll.read_last(self.preroll.capacity)
        if pre.size:
            await self._asr_stream.push(
                AudioFrame(samples=pre, sample_rate=self.config.audio.sample_rate, seq=-1)
            )
            self._utterance_ms += 1000.0 * pre.size / self.config.audio.sample_rate

    async def _required_silence(self, *, stable: bool = False) -> float:
        # Chen vào lúc trợ lý đang nói, "ừ" là người nghe gật đầu chứ không
        # phải ngập ngừng trước một câu dài: đừng giữ 1400 ms trước khi trả
        # lại lượt. Nói tiếp sau đó thì thành một lần ngắt lời mới, vẫn đúng.
        # Đo 25/09: "ừ" chen vào giữ thêm ~1.1 s trước khi máy nói tiếp.
        if self._interrupted is not None and _is_backchannel(self._partial_text):
            return float(self.config.conversation.turn_detection.silence_ms)
        return await self.turn_detector.required_silence_ms(
            text=self._partial_text, utterance_ms=self._utterance_ms, stable=stable
        )

    async def _mark_endpoint(self) -> None:
        self._endpoint_at_ms = self._audio_clock_ms
        stable = self._endpoint_stable()
        self._required_silence_ms = await self._required_silence(stable=stable)
        self._emit(
            EventType.ENDPOINT_CANDIDATE,
            required_silence_ms=round(self._required_silence_ms, 1),
            text=self._partial_text,
            stable=stable,
        )
        if not stable:
            self._start_endpoint_decode()

    def _reset_endpoint_state(self) -> None:
        self._endpoint_seq += 1
        self._endpoint_text = None
        self._spec_attempts = 0
        if self._endpoint_task is not None and not self._endpoint_task.done():
            self._endpoint_task.cancel()
        self._endpoint_task = None

    def _start_endpoint_decode(self) -> None:
        stream = self._asr_stream
        if (
            not self.config.conversation.turn_detection.endpoint_decode
            or stream is None
            or not hasattr(stream, "decode_now")
        ):
            return
        if self._endpoint_task is not None and not self._endpoint_task.done():
            self._endpoint_task.cancel()
        self._endpoint_seq += 1
        self._endpoint_task = asyncio.create_task(
            self._endpoint_decode(stream, self._endpoint_seq, self._speech_ms, self.gen.turn_id),
            name="asr-endpoint-decode",
        )

    async def _endpoint_decode(self, stream, seq: int, speech_ms: float, turn_id: int) -> None:
        started = now_ms()
        try:
            with observing(self._probe("asr", operation="endpoint")):
                transcript = await stream.decode_now()
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            # Only an optimisation: the periodic partial and the final decode
            # still decide the turn exactly as before.
            self._emit(EventType.ERROR, stage="asr_endpoint", error=type(exc).__name__)
            return
        if not self._endpoint_live(stream, seq, turn_id):
            return   # the speaker went on, or the turn is already decided
        text = (transcript.text or "").strip()
        self._endpoint_text = (speech_ms, text, transcript.confidence)
        stable = self._endpoint_stable()
        self._emit(EventType.ASR_ENDPOINT_TRANSCRIPT, text=text, stable=stable,
                   decode_ms=round(now_ms() - started, 1))
        if text:
            self._partial_text = text
            await self.sink.send_control(ControlMessage("transcript", {"text": text, "final": False}))
        # Checked again after every await: a write that yields (backpressure)
        # can let the pause confirm the turn meanwhile, and a speculation
        # opened after that is a second request nobody will adopt.
        if self._endpoint_at_ms is not None and self._endpoint_live(stream, seq, turn_id):
            self._required_silence_ms = await self._required_silence(stable=stable)
        if stable and text and self._endpoint_live(stream, seq, turn_id):
            await self._maybe_speculate(text)

    def _endpoint_live(self, stream, seq: int, turn_id: int) -> bool:
        return (
            seq == self._endpoint_seq
            and stream is self._asr_stream
            and turn_id == self.gen.turn_id
            and self.state.state is TurnState.LISTENING
        )

    # ------------------------------------------------------------------ #
    # shadow-mode answers
    # ------------------------------------------------------------------ #
    def _llm_has_room(self, free_slots: int) -> bool:
        limiter = getattr(self.models.llm, "limiter", None)
        if limiter is None:
            return True
        snap = limiter.snapshot()
        # Queued SPEECH is what a speculation would delay. A search waiting for
        # its own capped slots is not, and counting it turned speculation off
        # in every session while one search was queued.
        waiting = snap.get("speech_waiting", snap.get("waiting", 0))
        return waiting == 0 and snap["active"] + free_slots < snap["parallel"]

    def _tools_for(self, allow_search: bool) -> tuple[list[dict[str, Any]] | None, list[str] | None]:
        has_tools = bool(
            self.executor and self.models.llm.capabilities.tools and len(self.executor.registry)
        )
        tools = self.executor.registry.openai_tools() if has_tools else None
        tool_names = self.executor.registry.names() if has_tools else None
        if not allow_search and tools:
            # Đang đọc kết quả tra cứu thì không được tra tiếp, nếu không mỗi
            # câu trả lời lại đẻ ra một yêu cầu mới.
            tools = [t for t in tools if t["function"]["name"] != "search"]
            tool_names = [n for n in (tool_names or []) if n != "search"]
            if not tools:
                tools, tool_names = None, None
        return tools, tool_names

    async def _maybe_speculate(self, text: str) -> None:
        cfg = self.config.conversation.speculation
        if (not cfg.enabled or self._interrupted is not None or self._closed
                or self.state.state is not TurnState.LISTENING):
            return
        if self._speculation is not None and self._speculation.text == text:
            return
        if self._spec_attempts >= cfg.max_attempts:
            self._count("speculation_skipped_attempts")
            return
        if not self._llm_has_room(cfg.min_free_slots):
            self._count("speculation_skipped_busy")
            return
        await self._discard_speculation("superseded")
        tools, tool_names = self._tools_for(allow_search=True)
        messages = self.context.preview(text, tool_names=tool_names)
        probe = self._probe("llm", None, round=0, speculative=True)
        probe.mark(EventType.LLM_START)
        if not getattr(self.models.llm, "instrumented", False):
            probe.mark(EventType.LLM_REQUEST_SENT)
        llm = self.models.llm
        self._speculation = Speculation(
            text, messages, tools, lambda: llm.stream(messages, tools=tools), probe,
            self.gen.spawn_detached,
        )
        self._spec_attempts += 1
        self._count("speculations")
        self._emit(EventType.SPECULATION_STARTED, text=text, attempt=self._spec_attempts)

    def _take_speculation(self) -> Speculation | None:
        spec, self._speculation = self._speculation, None
        return spec

    async def _discard_speculation(self, reason: str, spec: Speculation | None = None) -> None:
        spec = spec or self._take_speculation()
        if spec is None or spec.adopted or spec.discarded:
            return
        await spec.discard(reason)
        self._count("speculations_discarded")
        self._emit(EventType.SPECULATION_DISCARDED, reason=reason, text=spec.text,
                   lead_ms=round(spec.lead_ms, 1), buffered=len(spec.buffer))

    async def _maybe_confirm_turn(self) -> None:
        detection = self.config.conversation.turn_detection
        if self._utterance_ms >= detection.max_utterance_ms:
            await self._confirm_turn("max utterance")
            return
        if self._endpoint_at_ms is None:
            return
        # The gate already consumed end_frames of silence before it fired.
        consumed = self.gate.end_frames * self.config.audio.frame_ms
        silence_ms = consumed + (self._audio_clock_ms - self._endpoint_at_ms)
        if silence_ms >= self._required_silence_ms:
            if self._speech_ms < detection.min_utterance_ms:
                # Too short to be speech: a door, a cough, a keystroke. Measured
                # on speech frames only — the pre-roll is mostly silence and
                # counting it would let every thump through as a turn.
                self._count("utterance_discarded_short")
                await self._abandon_turn("too short", announce=False)
                await self._resume_or_idle("noise")
                return
            verify = self.config.conversation.barge_in.verify_speech_ms
            if (
                verify and self._interrupted is not None and self.speech_vad is not None
                and self._interjection_speech_ms < verify
            ):
                # Loud enough and long enough for the energy gate, but the
                # speech model heard no voice in it: a cough, a knock, a
                # chair. ASR would still make a word of it ("đây") and the
                # assistant would answer that word.
                self._count("interjections_rejected")
                self._emit(EventType.INTERJECTION_REJECTED,
                           speech_ms=round(self._interjection_speech_ms, 1), text=self._partial_text)
                await self._abandon_turn("not speech", announce=False)
                await self._resume_or_idle("noise")
                return
            await self._confirm_turn(f"silence {silence_ms:.0f}ms")

    async def _confirm_turn(self, reason: str) -> None:
        self._endpoint_at_ms = None
        detection = self.config.conversation.turn_detection
        # Nothing voiced since the endpoint decode: it IS the final transcript.
        early = (
            self._endpoint_text[1]
            if detection.reuse_endpoint_transcript and self._endpoint_stable()
            and reason.startswith("silence")
            else None
        )
        early_confidence = self._endpoint_text[2] if early is not None and self._endpoint_text else None
        self._endpoint_seq += 1
        # What was heard so far, in case this turn is cut before ASR final.
        self._confirmed_text = (self.gen.turn_id, early if early is not None else self._partial_text)
        self.state.to(TurnState.THINKING, reason)
        self._arm_barge_in()
        key = self.gen.begin(self.gen.turn_id)
        commit = detection.commit_silence_ms
        if (
            commit and reason.startswith("silence") and self._last_voice_audio_ms is not None
            and detection.silence_ms <= self._required_silence_ms < commit
        ):
            # Neutral wait: confirmed, but not sure. Work starts now; the
            # first sound waits for `commit` ms of silence.
            self._commit_gate = (key, self._last_voice_audio_ms + commit)
        else:
            self._commit_gate = None
        if self._last_voice_ingress_ms is not None:
            self.trace.record(Event(type=EventType.SPEECH_LAST_FRAME, session_id=self.session_id,
                turn_id=key.turn_id, generation_id=key.generation_id, ts_ms=self._last_voice_ingress_ms,
                data={"audio_clock_ms": self._last_voice_audio_ms, "basis": "server_ingress_vad"}))
        self._emit(EventType.TURN_CONFIRMED, reason=reason, utterance_ms=round(self._utterance_ms, 1))
        await self.sink.send_control(ControlMessage("state", {"state": "thinking"}))
        self.gen.spawn(self._respond(key, early_text=early, early_confidence=early_confidence),
                       key, name=f"respond-{key}")

    def _arm_barge_in(self) -> None:
        """Open the interruption window as soon as the assistant owns the turn.

        Arming only at the first audio frame leaves the whole think window —
        1.5-1.9 s on this CPU stack — deaf: speech there fired no barge-in, was
        never handed to ASR, and was simply lost. `state.py` has always allowed
        THINKING -> LISTENING for exactly this; nothing used to reach it.

        Not armed while the gate is still open, which is the `max utterance`
        confirm: the user is mid-sentence there, and arming would cancel the
        turn that safety valve just created.
        """
        if not self.gate.active:
            self.barge_in.arm(self._audio_clock_ms)

    async def _abandon_turn(self, reason: str, *, announce: bool = True) -> None:
        await self._discard_speculation(reason)
        self._reset_endpoint_state()
        await self._close_asr()
        self.gate.reset()
        self._endpoint_at_ms = None
        self._utterance_ms = 0.0
        self._speech_ms = 0.0
        self._emit(EventType.TURN_END, reason=reason, answered=False)
        self.state.to(TurnState.IDLE, reason)
        if announce:
            await self.sink.send_control(ControlMessage("state", {"state": "idle"}))

    async def _close_asr(self) -> None:
        if self._endpoint_task is not None and not self._endpoint_task.done():
            self._endpoint_task.cancel()
        self._endpoint_task = None
        if self._asr_stream is not None:
            try:
                await self._asr_stream.close()
            except Exception as exc:  # pragma: no cover - backend dependent
                log.warning("ASR stream close failed: %s", exc)
                self._emit(EventType.ERROR, stage="asr_close", error=type(exc).__name__)
            self._asr_stream = None

    # ------------------------------------------------------------------ #
    # text turns (chat beside voice, same engine and history)
    # ------------------------------------------------------------------ #
    async def push_text(self, text: str) -> None:
        text = (text or "").strip()
        if not text or self._closed:
            return
        if self.state.is_assistant_active():
            await self._handle_barge_in(reason="text interrupt")
        self._end_drain(self._interrupted)
        self._interrupted = None
        await self._discard_speculation("text turn")
        turn_id = self.gen.next_turn()
        self.state.to(TurnState.THINKING, "text turn")
        self._arm_barge_in()
        key = self.gen.begin(turn_id)
        self._emit(EventType.TURN_START, turn_id=turn_id, source="text")
        self._emit(EventType.TURN_CONFIRMED, reason="text", source="text")
        self._emit(EventType.ASR_FINAL, text=text, source="text")
        # Same control message a spoken turn sends. Without it the client badge
        # sits on the previous state for the whole think window — measured at
        # 1.48 s on the CPU stack, which reads as a dead UI.
        await self.sink.send_control(ControlMessage("state", {"state": "thinking"}))
        self.context.start_turn(turn_id, text)
        self.gen.spawn(self._answer(key, text), key, name=f"answer-{key}")

    # ------------------------------------------------------------------ #
    # the response pipeline
    # ------------------------------------------------------------------ #
    async def _respond(self, key: GenerationKey, *, early_text: str | None = None,
                       early_confidence: float | None = None) -> None:
        spec = None
        try:
            stream = self._asr_stream
            if early_text is not None:
                # The endpoint decode heard every voiced frame; decoding the
                # same audio plus trailing silence again only costs latency.
                probe = self._probe("asr", key, operation="final")
                probe.mark(EventType.ASR_FINALIZE_START, reused_endpoint_decode=True)
                transcript = Transcript(text=early_text, is_final=True, confidence=early_confidence)
                probe.mark(EventType.ASR_FINALIZE_END, reused_endpoint_decode=True)
                accept = getattr(stream, "accept_final", None)
                if accept is not None:
                    accept(early_text)
                if self._asr_stream is stream:
                    self._asr_stream = None
                if stream is not None:
                    # Off the critical path: close() waits for any partial
                    # decode still running on this stream.
                    self.gen.spawn_detached(self._close_stream(stream), name="asr-close")
            else:
                try:
                    probe = self._probe("asr", key, operation="final")
                    probe.mark(EventType.ASR_FINALIZE_START)
                    with observing(probe):
                        async with asyncio.timeout(self.config.models.operation_timeout_s):
                            transcript = await stream.finish() if stream is not None else Transcript(text="", is_final=True)
                    probe.mark(EventType.ASR_FINALIZE_END)
                finally:
                    if self._asr_stream is stream:
                        self._asr_stream = None
                    if stream is not None:
                        await stream.close()
            spec = self._take_speculation()
            if not self.gen.check(key):
                self._emit(EventType.STALE_DROPPED, stage="asr_final")
                return
            text = (transcript.text or "").strip()
            self._emit(EventType.ASR_FINAL, text=text)
            await self.sink.send_control(
                ControlMessage("transcript", {"text": text, "final": True})
            )
            if not text:
                self._count("empty_transcripts")
                await self._finish_turn(key, answered=False, announce=False, reason="empty transcript")
                await self._resume_or_idle("empty transcript")
                return
            carried = self._interrupted
            self._interrupted = None
            # Bản cuối có thể nghe "ừ" thành "từ" (đo 25/09: partial "ừ", final
            # "từ", và máy trả lời "từ" bằng mười cụm). Một từ duy nhất mà
            # partial đọc ra là lời đệm thì vẫn là lời đệm.
            backchannel = _is_backchannel(text) or (
                len(text.split()) == 1
                and (_is_backchannel(self._partial_text) or self._partial_backchannel)
            ) or (
                # Two words, a periodic partial that read as a backchannel, and
                # nothing that asks for anything: "vâng ạ" heard as "thân ạ",
                # "ừ ừ" as "từ từ" (bộ dev G3 29/09, 4/96 answered as questions).
                len(text.split()) == 2
                and self._partial_backchannel
                and not _commands(text)
            ) or self._short_interjection(carried, text)
            if (
                carried is not None
                and carried.content_heard
                and self._resumable(carried)
                and backchannel
            ):
                # "ừ" nói chen vào giữa câu trả lời: nghe tiếp, đừng trả lời "ừ".
                self._count("backchannels")
                # Handed back BEFORE the awaits: a Stop or a typed turn landing
                # in them must find it to end it, not leave its drain running.
                self._interrupted = carried
                await self._finish_turn(key, answered=False, announce=False, reason="backchannel")
                await self._resume_or_idle("backchannel")
                return
            self._end_drain(carried)
            text = await self._merge_unanswered(carried, text, key.turn_id)
            threshold = self.config.conversation.clarify_confidence_below
            needs_clarification = bool(threshold and transcript.confidence is not None
                                       and transcript.confidence < threshold)
            # A guess about a number or name must not become trusted history.
            self.context.start_turn(key.turn_id, "(ASR không chắc; đã hỏi lại)" if needs_clarification else text)
            prefetched, spec = spec, None
            if needs_clarification:
                if prefetched is not None:
                    await self._discard_speculation("ASR confidence low", prefetched)
                await self._say_clarification(key, text)
            else:
                await self._answer(key, text, prefetched=prefetched)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # never let one turn take the session down
            log.exception("respond failed")
            self._emit(EventType.ERROR, stage="respond", error=repr(exc))
            await self._speak_fallback(key)
        finally:
            if spec is not None:
                # Not adopted: empty transcript, backchannel, stale key...
                await self._discard_speculation("not adopted", spec)

    def _short_interjection(self, carried: _Interrupted | None, text: str) -> bool:
        """Too little voice to be a request, whatever the ASR spelled."""
        limit = self.config.conversation.barge_in.backchannel_max_speech_ms
        return bool(
            limit and carried is not None and self.speech_vad is not None
            and len(text.split()) <= 2 and not _commands(text)
            and self._interjection_speech_ms <= limit
        )

    async def _close_stream(self, stream) -> None:
        try:
            await stream.close()
        except Exception as exc:  # pragma: no cover - backend dependent
            log.warning("ASR stream close failed: %s", exc)
            self._emit(EventType.ERROR, stage="asr_close", error=type(exc).__name__)

    async def _say_clarification(self, key: GenerationKey, heard: str) -> None:
        """Do not ask an LLM to guess a low-confidence identifier."""
        lower = heard.lower()
        if "tài khoản" in lower:
            prompt = "Tôi chưa nghe rõ số tài khoản. Bạn đọc lại số tài khoản giúp tôi nhé?"
        elif any(term in lower for term in ("số điện thoại", "mã otp", "mã xác nhận")):
            prompt = "Tôi chưa nghe rõ dãy số. Bạn đọc lại dãy số giúp tôi nhé?"
        elif any(term in lower for term in ("số tiền", "chuyển tiền", "bao nhiêu tiền")):
            prompt = "Tôi chưa nghe rõ số tiền. Bạn đọc lại số tiền giúp tôi nhé?"
        else:
            prompt = "Tôi chưa nghe rõ ý chính. Bạn nhắc lại giúp tôi nhé?"
        self._count("asr_clarifications")
        await self._say_fixed(key, prompt, role="clarification")

    async def _say_fixed(self, key: GenerationKey, prompt: str, *, role: str = "content") -> None:
        response = self._new_response(key)
        response.generated.append(prompt)
        self._response = response
        await self._send_delta(key, prompt)
        queue: asyncio.Queue[Phrase | None] = asyncio.Queue()
        response.queue = queue
        response.llm_done = True
        audio = self._ack_audio if role == "ack" and prompt == self.config.conversation.search.instant_ack else None
        queue.put_nowait(Phrase(prompt, role=role, audio=audio, filler=role == "ack"))
        queue.put_nowait(None)
        task = self.gen.spawn(self._speak(key, response, queue, use_opener=False), key,
                              name=f"clarify-{key}")
        response.speak_task = task
        await task

    async def _answer(
        self, key: GenerationKey, user_text: str, *, allow_search: bool = True, output_role: str = "content",
        prefetched: Speculation | None = None,
    ) -> None:
        if (allow_search and self.search_agent is not None
                and self.config.conversation.search.force_source_lookup):
            from .search_routing import requires_public_lookup
            if requires_public_lookup(user_text):
                if prefetched is not None:
                    await self._discard_speculation("source lookup required", prefetched)
                if self._dispatch_search(user_text) is not None:
                    ack = self.config.conversation.search.instant_ack
                    self._emit(EventType.FILLER, text=ack, source="search")
                    await self._say_fixed(key, ack, role="ack")
                else:
                    await self._say_fixed(key, "Tôi đang xử lý lượt tra cứu khác. Xin thử lại sau.", role="fallback")
                return
        response = self._new_response(key, output_role=output_role)
        self._response = response
        phrases: asyncio.Queue[Phrase | None] = asyncio.Queue()
        response.queue = phrases
        speak_task = self.gen.spawn(self._speak(key, response, phrases), key, name=f"speak-{key}")
        response.speak_task = speak_task

        # Một chỗ duy nhất dựng bộ chia cụm, dùng chung với bàn thử model.
        make_segmenter = lambda: pipeline_segmenter(self.models.tts.capabilities, self.config.conversation.streaming_first_phrase_chars)
        tools, tool_names = self._tools_for(allow_search)

        try:
            for round_index in range(_MAX_TOOL_ROUNDS + 1):
                segmenter = make_segmenter()
                tool_calls: list[Any] = []
                saw_token = False
                # turn_id pinned: a drained producer outlives its turn's place
                # as "current", and its stamps belong to the turn that asked.
                messages: list[Message] = self.context.messages(tool_names=tool_names)
                adopted = None
                if round_index == 0 and prefetched is not None:
                    spec, prefetched = prefetched, None
                    if spec.matches(user_text, messages, tools):
                        adopted = spec
                        self._count("speculations_adopted")
                        self._emit(EventType.SPECULATION_ADOPTED, lead_ms=round(spec.lead_ms, 1),
                                   buffered=len(spec.buffer), done=spec.done)
                    else:
                        await self._discard_speculation("prompt changed", spec)
                probe = adopted.probe if adopted else self._probe("llm", key, round=round_index)
                if adopted is None:
                    probe.mark(EventType.LLM_START)
                with observing(probe):
                    if adopted is not None:
                        stream = adopted.adopt()
                    else:
                        if not getattr(self.models.llm, "instrumented", False):
                            probe.mark(EventType.LLM_REQUEST_SENT)
                        stream = self.models.llm.stream(messages, tools=tools)
                    if self.config.conversation.streaming_first_phrase_wait_ms:
                        stream = with_first_phrase_deadline(
                            stream, self.config.conversation.streaming_first_phrase_wait_ms, segmenter,
                        )
                    async with contextlib.aclosing(stream):
                        async for delta in stream:
                            if delta is None:
                                for phrase in segmenter.flush_first_boundary():
                                    await self._enqueue_phrase(phrases, Phrase(phrase, role=response.output_role, filler=response.output_role == "ack"), key)
                                continue
                            if not self._alive(key):
                                self._emit(EventType.STALE_DROPPED, stage="llm")
                                return
                            if delta.text:
                                if not saw_token:
                                    saw_token = True
                                    probe.mark(EventType.LLM_FIRST_TOKEN) if not getattr(self.models.llm, "instrumented", False) else None
                                response.generated.append(delta.text)
                                self._progress_ms = now_ms()
                                await self._send_delta(key, delta.text)
                                for phrase in segmenter.push(delta.text):
                                    await self._enqueue_phrase(phrases, Phrase(phrase, role=response.output_role, filler=response.output_role == "ack"), key)
                            if delta.tool_call is not None:
                                # Parallel calls arrive as separate deltas. Keeping only
                                # the last one dropped the others and left the history
                                # with one tool message for two calls.
                                tool_calls.append(delta.tool_call)
                            # Consume the trailing usage chunk before closing the request.
                for phrase in segmenter.flush():
                    await self._enqueue_phrase(phrases, Phrase(phrase, role=response.output_role, filler=response.output_role == "ack"), key)
                probe.mark(EventType.LLM_COMPLETE)

                if not tool_calls or self.executor is None:
                    break
                await self._await_drain_decision(key)
                if not self._alive(key):
                    return
                acknowledged_search=[]
                for index, call in enumerate(tool_calls):
                    # One filler per round, not per call: the point is to cover
                    # a silence, and the second one lands in the middle of it.
                    acknowledged_search.append(await self._run_tool(
                        key, call, phrases, user_text, allow_filler=index == 0
                    ))
                    if not self._alive(key):
                        return
                if (self.config.conversation.search.skip_redundant_ack and
                        acknowledged_search and all(acknowledged_search)):
                    break
            response.llm_done = True
        except asyncio.CancelledError:
            if prefetched is not None:
                await self._discard_speculation("cancelled", prefetched)
            raise
        except (ModelUnavailable, ModelTimeout) as exc:
            log.warning("LLM unavailable: %s", exc)
            self._emit(EventType.ERROR, stage="llm", error=str(exc))
            self._count("fallbacks")
            await self.sink.send_control(ControlMessage("error", {"stage": "llm", "error": "model_unavailable"}))
            await self._enqueue_phrase(phrases, Phrase(_FALLBACK_REPLY, role="fallback"), key)
        except Exception as exc:
            log.exception("answer failed")
            self._emit(EventType.ERROR, stage="answer", error=repr(exc))
            self._count("fallbacks")
            await self.sink.send_control(ControlMessage("error", {"stage": "answer", "error": "generation_failed"}))
            await self._enqueue_phrase(phrases, Phrase(_FALLBACK_REPLY, role="fallback"), key)
        finally:
            await phrases.put(None)

        try:
            await speak_task
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # pragma: no cover - speak logs its own
            log.warning("speak task ended badly: %s", exc)

    async def _run_tool(
        self,
        key: GenerationKey,
        tool_call: Any,
        phrases: asyncio.Queue[Phrase | None],
        user_text: str,
        *,
        allow_filler: bool = True,
    ) -> bool:
        """Slow path. The audio loop keeps running; only this turn waits."""
        assert self.executor is not None
        ctx = TaskContext(key=key, session_id=self.session_id, user_text=user_text)
        with observing(self._probe("tool", key, tool=tool_call.name)):
            # `_alive`, not `is_current`: a cut turn still draining is never
            # current again, and a cough over the filler used to turn the
            # result it was waiting for into "stale".
            task = self.gen.spawn(
                self.executor.run(
                    tool_call.name, tool_call.arguments, ctx, is_current=self._alive
                ), key, name="tool-" + tool_call.name
            )
        try:
            filler = self.config.conversation.filler
            if allow_filler and filler.enabled and filler.phrases:
                done, _ = await asyncio.wait({task}, timeout=filler.after_ms / 1000.0)
                if not done:
                    # Only now is the wait long enough to be worth covering. A
                    # filler spoken unconditionally is just added latency.
                    phrase = filler.phrases[key.turn_id % len(filler.phrases)]
                    if self._filler_allowed(filler.cooldown_ms):
                        self._emit(EventType.FILLER, text=phrase)
                        await self._enqueue_phrase(phrases, Phrase(phrase, filler=True), key)
            result = await task
        finally:
            if not task.done():
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
        if not self._alive(key):
            return False
        # Tool có thể yêu cầu nói ngay một câu cố định, không chờ model soạn.
        # Với tra cứu, đó là khác biệt giữa im lặng 3 giây và trả lời 1 giây.
        speak_now = result.data.get("speak_now") if result.data else None
        if speak_now:
            text = str(speak_now)
            self._emit(EventType.FILLER, text=text, source=tool_call.name)
            # Người nghe phải thấy đúng thứ mình nghe: câu này không đi qua
            # model nên phải tự gửi lên transcript.
            await self._send_delta(key, text + " ")
            response = self._response
            if response is not None and response.key == key:
                response.generated.append(text + " ")
            await self._enqueue_phrase(phrases, Phrase(text, audio=self._ack_audio, filler=True, role="ack"), key)
        if tool_call.name == "search" and result.data.get("accepted"):
            response = self._response
            if response is not None and response.key == key:
                response.output_role = "ack"
        content = result.content if result.ok else (result.content or "không lấy được dữ liệu")
        self.context.add_tool_exchange(tool_call.name, content)
        return bool(tool_call.name=='search' and result.ok and result.data.get('accepted') and speak_now)

    async def _speak(
        self,
        key: GenerationKey,
        response: ResponseState,
        phrases: asyncio.Queue[Phrase | None],
        *,
        use_opener: bool = True,
    ) -> None:
        tts = self.models.tts
        started = False
        opener = self.config.conversation.opener
        use_opener = (use_opener and opener.enabled and bool(opener.text)
                      and self._filler_allowed(opener.cooldown_ms, reserve=False))
        # Tra bộ nhớ đệm theo ĐÚNG giọng đang dùng, và không bao giờ chờ: nếu
        # chưa ấm thì lượt này không có câu mở, và một tác vụ nền làm ấm cho
        # lượt sau. Đổi giọng giữa phiên tự khỏi theo đường này.
        opener_audio = (
            self.models.warm_speech(opener.text, self.voice) if use_opener else None
        )
        if use_opener and opener_audio is None:
            self.gen.spawn_detached(
                self._prewarm(opener.text), name="warm-opener"
            )
        opened = opener_audio is None
        try:
            while True:
                if not opened:
                    opened = True
                    try:
                        # Chạy đua: cụm thật tới trước thì không cần câu mở.
                        item = await asyncio.wait_for(
                            phrases.get(), timeout=opener.after_ms / 1000.0
                        )
                    except asyncio.TimeoutError:
                        # KHÔNG gửi assistant_delta. `_speak` chạy song song với
                        # `_answer` đang stream token, nên một delta phát từ đây
                        # chen vào GIỮA câu model đang viết: đo được
                        # "Về câu bạnVâng.  hỏi lúc nãy". Âm thanh vẫn đúng thứ
                        # tự vì `_speak` là người ghi duy nhất; chỉ chữ hiển thị
                        # hỏng. Câu đệm của đường tool cũng không lên transcript,
                        # cùng lý do và cùng cách xử lý.
                        if self._filler_allowed(opener.cooldown_ms):
                            self._emit(EventType.FILLER, text=opener.text, source="opener")
                            item = Phrase(opener.text, audio=opener_audio, filler=True)
                        else:
                            item = await phrases.get()
                else:
                    item = await phrases.get()
                if item is None:
                    response.queue_ended = True
                    break
                if not self.gen.check(key):
                    self._emit(EventType.STALE_DROPPED, stage="tts_phrase")
                    return
                if not started:
                    started = True
                    self._emit(EventType.TTS_START)
                item = self._prepare_phrase(item, key)
                response.begin_phrase(item)
                probe = self._probe("tts", key, phrase_id=item.phrase_id, role=item.role, cached=item.audio is not None)
                await self.sink.send_control(ControlMessage("audio_segment", {
                    "generation_id": key.generation_id, "turn_id": key.turn_id,
                    "phrase_id": item.phrase_id, "role": item.role, "server_ts_ms": now_ms(),
                }))
                phrase_started = now_ms()
                phrase_samples = 0
                phrase_chunks = 0
                if item.audio is None:
                    speech_text = (normalize_for_speech(item.text, self.config.conversation.pronunciations)
                                   if self.config.conversation.speech_normalization else item.text)
                    source = tts.synthesize(speech_text, voice=self.voice)
                    if self.config.conversation.pauses.enabled:
                        # Not the cached opener/ack: a longer tail there only
                        # delays the content queued behind it.
                        source = shape_pauses(source, item.text, self.config.conversation.pauses)
                elif hasattr(item.audio, "__aiter__"):
                    source = item.audio          # held audio of a cut turn
                else:
                    source = _aiter(item.audio)
                with observing(probe):
                    async with asyncio.timeout(self.config.models.operation_timeout_s), contextlib.aclosing(source):
                        async for chunk in source:
                            phrase_samples += chunk.samples.size
                            phrase_chunks += 1
                            if phrase_chunks == 1 and not getattr(tts, "instrumented", False):
                                probe.mark(EventType.TTS_CHUNK_READY)
                            for frame in self._slice_output(chunk):
                                if response.first_audio_ms is None and not response.holding:
                                    await self._await_commit(key)
                                if response.holding:
                                    # Cut by speech that may be a cough: finish
                                    # the phrase into memory, send nothing.
                                    response.current_frames.append(frame)
                                    response.hold_progress.set()
                                    continue
                                # Re-checked per frame, not per phrase: a real talker
                                # can hand back two seconds at once, and an
                                # interruption inside those two seconds must still
                                # stop the very next frame.
                                if not self.gen.check(key):
                                    self._emit(EventType.STALE_DROPPED, stage="tts_audio")
                                    return
                                await self._send_audio_frame(key, response, frame)
                                response.current_frames.append(frame)
                probe.mark(EventType.TTS_PHRASE_COMPLETE, total_ms=now_ms()-phrase_started,
                           audio_ms=phrase_samples*1000.0/tts.capabilities.native_sample_rate,
                           chunks=phrase_chunks, phrase_wait_ms=phrase_started-item.ready_at_ms)
                if response.holding:
                    return      # the held phrase is complete; a resume takes it from here
                await self.sink.send_control(ControlMessage("audio_end", {
                    "generation_id": key.generation_id, "turn_id": key.turn_id,
                    "phrase_id": item.phrase_id, "role": item.role,
                }))
                if response.holding:
                    # Cut while that write yielded. The phrase stays `current`
                    # with all its frames, as if cut mid-synthesis, and nothing
                    # more is read: the queue is the resume's now, and the
                    # history was settled at the cut.
                    return
                response.end_phrase()
                if response.current_start is not None:
                    response.sent.append(
                        _Sent(item, response.current_start, response.play_end,
                              frames=response.current_frames)
                    )
                response.current = None
                response.spoken.append(item.text)
                if item.role != "filler":
                    response.history_spoken.append(item.text)
            if started:
                self._emit(EventType.TTS_COMPLETE, phrases=len(response.spoken))
            await self.sink.send_control(ControlMessage("audio_generation_end", {
                "generation_id": key.generation_id, "turn_id": key.turn_id,
            }))
            # Hold the turn until the client has PLAYED it, not just received
            # it. The talker runs faster than real time, so the last seconds of
            # an answer are still coming out of the speaker after the last
            # frame was sent — and finishing the turn here disarmed barge-in
            # for exactly those seconds: talking over them started a new turn
            # while the old answer kept playing on top of it.
            if self.playback_feedback_enabled:
                # Nothing sent, nothing to play: no report will come. Otherwise
                # wait for it until the schedule says playback ended, plus a
                # margin; a report that never comes is not a TTS failure.
                if response.first_audio_ms is not None:
                    left = max(0.0, response.play_end - time.monotonic()) + _PLAYBACK_DONE_MARGIN_S
                    try:
                        await asyncio.wait_for(response.playback_done.wait(), left)
                    except TimeoutError:
                        self._count("playback_done_timeouts")
                        self._emit(EventType.ERROR, stage="playback_feedback",
                                   error="generation_end_missing", waited_s=round(left, 2))
            else:
                tail = response.play_end - time.monotonic()
                if tail > 0:
                    await asyncio.sleep(tail)
            await self._finish_turn(key, answered=started)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            log.exception("speak failed")
            self._emit(EventType.ERROR, stage="tts", error=repr(exc))
            if not response.holding:
                await self.sink.send_control(ControlMessage("error", {"stage": "tts", "error": "synthesis_failed"}))
            await self._finish_turn(key, answered=started)
        finally:
            if response.holding:
                response.hold_done.set()
                response.hold_progress.set()

    async def _await_commit(self, key: GenerationKey) -> None:
        gate = self._commit_gate
        if gate is None or gate[0] != key:
            return
        left = gate[1] - self._audio_clock_ms
        started = now_ms()
        if left > 0:
            # Audio clock in a live session; wall time bounds it in case the
            # microphone stream stalls and the audio clock stops.
            while self.gen.is_current(key) and self._audio_clock_ms < gate[1] and now_ms() - started < left:
                await asyncio.sleep(0.01)
        self._emit(EventType.AUDIO_COMMIT, waited_ms=round(now_ms() - started, 1), gated=left > 0)
        self._commit_gate = None

    def _slice_output(self, chunk: Any) -> list[AudioFrame]:
        """Cut a synthesised block into wire-sized frames.

        VieNeu returns a whole phrase in one array. Putting that on the socket
        as one message means playback cannot start until the last byte lands,
        and the client ends up holding a single two-second buffer that a
        barge-in can only stop wholesale.
        """
        size = max(1, int(chunk.sample_rate * self.config.audio.output_frame_ms / 1000))
        samples = chunk.samples
        if samples.size <= size:
            return [AudioFrame(samples=samples, sample_rate=chunk.sample_rate)]
        return [
            AudioFrame(samples=samples[i : i + size], sample_rate=chunk.sample_rate)
            for i in range(0, samples.size, size)
        ]

    async def _send_audio_frame(
        self, key: GenerationKey, response: ResponseState, frame: AudioFrame
    ) -> float | None:
        """Send one frame; return when the client will start playing it
        (None while its startup buffer is still filling)."""
        if response.first_audio_ms is None:
            response.first_audio_ms = now_ms()
            self._emit(EventType.TTS_FIRST_AUDIO, sample_rate=frame.sample_rate)
            if self.state.state is TurnState.THINKING:
                self.state.to(TurnState.SPEAKING, "first audio")
            cfg = self.config.conversation.barge_in
            # The client starts playing only once playback_startup_ms is
            # buffered; until its own report of the start arrives, guard the
            # whole window in which that start can fall.
            startup = cfg.playback_startup_ms if cfg.guard_from_playback else 0
            self.barge_in.arm(self._audio_clock_ms, guard_ms=startup + cfg.guard_ms)
            await self.sink.send_control(
                ControlMessage(
                    "speaking",
                    {
                        "generation_id": key.generation_id,
                        "turn_id": key.turn_id,
                        "sample_rate": frame.sample_rate,
                    },
                )
            )
        await self.sink.send_audio(frame, key)
        if response.current is not None and not response.current_placed:
            item = response.current
            self.trace.record(Event.for_key(EventType.AUDIO_SENT, key, phrase_id=item.phrase_id, role=item.role))
            response.roles_sent.add(item.role)
        self.preprocessor.far_end(frame)
        self._progress_ms = now_ms()
        return response.schedule(frame.samples.size / float(frame.sample_rate))

    def _new_response(self, key: GenerationKey, **fields: Any) -> ResponseState:
        """A response modelling the player this client actually has."""
        startup_s = (
            self.config.conversation.barge_in.playback_startup_ms / 1000.0
            if self.playback_feedback_enabled else 0.0   # no feedback: legacy scheduler
        )
        return ResponseState(key=key, startup_s=startup_s, **fields)

    async def _speak_fallback(self, key: GenerationKey) -> None:
        if not self.gen.is_current(key):
            return
        # `_response`, like every other reply: the client's playback report
        # (and a barge-in's account of what was heard) is matched against it.
        response = self._new_response(key, in_history=self.context.find(key.turn_id) is not None)
        self._response = response
        queue: asyncio.Queue[Phrase | None] = asyncio.Queue()
        await self._enqueue_phrase(queue, Phrase(_FALLBACK_REPLY, role="fallback"), key)
        await queue.put(None)
        await self._speak(key, response, queue)

    async def _finish_turn(
        self, key: GenerationKey, *, answered: bool, announce: bool = True, reason: str | None = None
    ) -> None:
        if not self.gen.is_current(key):
            return
        response = self._response
        if response is not None and response.key == key and response.in_history:
            self.context.commit_assistant(
                response.generated_text, " ".join(response.history_spoken), interrupted=False
            )
        self.barge_in.disarm()
        self._reset_endpoint_state()
        await self._close_asr()
        self._utterance_ms = 0.0
        self._speech_ms = 0.0
        self._partial_text = ""
        self._partial_backchannel = False
        self.gate.reset()
        # `reason` marks a turn left unanswered on purpose ("ừ" heard as a
        # backchannel): a checker must not count it as a turn that failed.
        self._emit(EventType.TURN_END, answered=answered, **({"reason": reason} if reason else {}))
        if self.state.state in {TurnState.THINKING, TurnState.SPEAKING}:
            self.state.to(TurnState.IDLE, "turn finished")
        if announce:
            await self.sink.send_control(ControlMessage("state", {"state": "idle"}))
        self._record_turn_metrics(key.turn_id)

    # ------------------------------------------------------------------ #
    # interruption
    # ------------------------------------------------------------------ #
    async def _handle_barge_in(self, reason: str = "user speech") -> None:
        key = self.gen.current
        if key is None:
            return
        at = time.monotonic()
        fired_frames = self.barge_in.run_frames if reason == "user speech" else 0
        self._emit(EventType.BARGE_IN, reason=reason)
        # A barge-in over a turn that was itself left undecided by an earlier
        # one: that earlier turn is now definitely over.
        previous = self._interrupted
        self._end_drain(previous)
        self._interrupted = None
        response = self._response if self._response and self._response.key == key else None
        drain: list[asyncio.Task] = []
        hold: asyncio.Task | None = None
        draining = False
        cfg = self.config.conversation.barge_in
        if (
            reason == "user speech"
            and response is not None
            and cfg.resume_after_false
        ):
            draining = True
            speak = response.speak_task
            if (
                cfg.resume_from_cut and speak is not None and not speak.done()
                and response.current is not None
            ):
                # Mid-phrase: let the talker finish this phrase into memory
                # (nothing is sent), so a false interruption continues from
                # where playback stopped instead of reading it again.
                response.holding = True
                hold = speak
            # Keep the LLM writing; only the voice stops. See _Interrupted.
            drain = self.gen.detach(key, keep=lambda t: t is not speak)
            if hold is not None:
                self.gen.detach(key, keep=lambda t: t is hold)
            self._drain_targets[key] = None
            self._drain_tasks[key] = drain
            for task in drain:
                # A tool call in flight is drained with its producer and ends
                # first: the drain is over only when ALL of them are.
                task.add_done_callback(lambda _t, k=key, tasks=drain: self._drain_done(k, tasks))
        if not draining:
            # Nothing kept, so nothing may keep feeding this generation either:
            # a resumed answer is still fed by the producer of the turn it resumed.
            self._stop_drains(key)
        cancelled = await self.gen.cancel(key)
        self._emit(EventType.CANCEL, tasks=cancelled, reason=reason)

        # The client may have several hundred ms of audio already scheduled;
        # only it can drop that, so say so explicitly and let it fence on the
        # generation id.
        await self.sink.send_control(
            ControlMessage(
                "playback_reset",
                {"generation_id": key.generation_id, "turn_id": key.turn_id, "reason": reason},
            )
        )
        self._emit(EventType.PLAYBACK_RESET, generation_id=key.generation_id)

        if response is not None:
            heard = response.heard_before + [p.text for p in response.heard_by(at) if p.role != "filler"]
            response.spoken = heard
            response.history_spoken = list(heard)
            if response.in_history:
                self.context.commit_assistant(
                    response.generated_text, " ".join(heard), interrupted=True
                )
        # Only speech can turn out to be a cough; a button or a typed turn is
        # the user meaning it.
        if reason == "user speech":
            turn = self.context.find(key.turn_id)
            user_text = turn.user_text if turn is not None else ""
            if not user_text and self._confirmed_text[0] == key.turn_id:
                user_text = self._confirmed_text[1]
            absorbed: tuple[int, ...] = ()
            if (
                turn is None and previous is not None and previous.user_text
                and not previous.content_heard and cfg.merge_unanswered
            ):
                # Cut before its own final transcript while the turn IT cut was
                # still unanswered: one sentence in three pieces. Ending that
                # earlier turn above used to lose its piece with it.
                user_text = f"{previous.user_text} {user_text}".strip()
                absorbed = (*previous.absorbed, previous.key.turn_id)
            self._interrupted = _Interrupted(
                key=key,
                response=response,
                user_text=user_text,
                at_ms=self._audio_clock_ms,
                replay=response.unheard_by(at) if response else [],
                content_heard=response.content_heard_by(at) if response else False,
                drain=drain,
                at_s=at,
                hold=hold,
                absorbed=absorbed,
            )
        self._response = None
        self.barge_in.disarm()
        await self._discard_speculation("barge-in")
        self._reset_endpoint_state()
        await self._close_asr()
        self._count("barge_ins")
        self._record_turn_metrics(key.turn_id)

        self.state.to(TurnState.LISTENING, "barge-in")
        # Keep the interrupting words: a new turn starts from the pre-roll.
        turn_id = self.gen.next_turn()
        self._utterance_ms = 0.0
        # The frames that fired the barge-in are this turn's speech. Not
        # counting them made a short "dừng" over the answer "too short" (and
        # the answer resumed) while the same word from silence was a turn. The
        # firing frame itself is counted by `_on_frame` right after this.
        self._speech_ms = max(0, fired_frames - 1) * float(self.config.audio.frame_ms)
        self._partial_text = ""
        self._partial_backchannel = False
        self._endpoint_at_ms = None
        self._required_silence_ms = float(self.config.conversation.turn_detection.silence_ms)
        self._emit(EventType.TURN_START, turn_id=turn_id, source="barge_in")
        try:
            with observing(self._probe("asr")):
                self._asr_stream = await self.models.asr.open_stream(
                    sample_rate=self.config.audio.sample_rate
                )
            pre = self._interjection_preroll()
            self._emit(EventType.ASR_START, preroll_ms=round(1000.0 * pre.size / self.config.audio.sample_rate, 1))
            if pre.size:
                await self._asr_stream.push(
                    AudioFrame(samples=pre, sample_rate=self.config.audio.sample_rate, seq=-1)
                )
                self._utterance_ms += 1000.0 * pre.size / self.config.audio.sample_rate
        except Exception as exc:
            log.error("could not reopen ASR after barge-in: %s", exc)
            self._emit(EventType.ERROR, stage="asr_reopen", error=str(exc))
            self.state.to(TurnState.IDLE, "asr reopen failed")
        self.gate.reset(active=True)

    def _interjection_preroll(self) -> np.ndarray:
        """Audio before the barge-in to hand ASR: back to the burst's onset.

        Called from `_on_frame` before the firing frame is written to the
        rings, which is fine: that frame is pushed to ASR right after.
        """
        rate = self.config.audio.sample_rate
        fixed = self.preroll.capacity
        cap = self.config.conversation.barge_in.burst_preroll_max_ms
        if not cap or self._burst_onset_ms is None:
            return self.preroll.read_last(fixed)
        back_ms = self._audio_clock_ms - self._burst_onset_ms + 100.0   # a syllable's run-up
        wanted = int(rate * min(cap, back_ms) / 1000)
        if wanted <= fixed:
            return self.preroll.read_last(fixed)
        return self.burst_ring.read_last(wanted)

    def _alive(self, key: GenerationKey) -> bool:
        """The producer fence: current, or a cut turn still draining."""
        return self.gen.is_current(key) or key in self._drain_targets

    async def _send_delta(self, key: GenerationKey, text: str) -> None:
        target = key if self.gen.is_current(key) else self._drain_targets.get(key)
        if target is None or not self.gen.is_current(target):
            return   # draining, undecided: the client shows nothing for now
        await self.sink.send_control(
            ControlMessage("assistant_delta", {"text": text, "generation_id": target.generation_id})
        )

    def _drain_done(self, key: GenerationKey, tasks: list[asyncio.Task]) -> None:
        if all(t.done() for t in tasks):
            self._drain_targets.pop(key, None)
            self._drain_tasks.pop(key, None)

    def _end_drain(self, carried: _Interrupted | None) -> None:
        """The cut turn will not come back: stop its LLM."""
        if carried is None:
            return
        for task in [*carried.drain, *([carried.hold] if carried.hold else [])]:
            if not task.done():
                task.cancel()
        self._stop_drains(carried.key)

    def _stop_drains(self, key: GenerationKey) -> None:
        """Stop the drain of `key`, and every producer still writing into it.

        A resumed answer is fed through a pipe by the producer of the turn it
        resumed (its target was moved here). Stopping only the pipe left that
        producer streaming — holding an LLM slot, and free to run a tool into
        whatever turn is newest — for an answer nobody would hear.
        """
        for producer, target in list(self._drain_targets.items()):
            if producer == key or target == key:
                self._drain_targets.pop(producer, None)
                for task in self._drain_tasks.pop(producer, []):
                    if not task.done():
                        task.cancel()

    def _drain_undecided(self, key: GenerationKey) -> bool:
        """A cut turn whose interruption is still being heard (cough or not?)."""
        seen: set[GenerationKey] = set()
        while not self.gen.is_current(key) and key not in seen:
            seen.add(key)
            if key not in self._drain_targets:
                return False
            target = self._drain_targets[key]
            if target is None:
                return True
            key = target
        return False

    async def _await_drain_decision(self, key: GenerationKey) -> None:
        """Hold a tool call of a cut turn until the interruption is decided.

        The LLM keeps writing through a barge-in so that a cough costs nothing,
        but writing is all it may do: the interruption may well be "khoan, đừng
        chuyển". A resume lets the call through; a real turn cancels it here.
        """
        while self._drain_undecided(key):
            await asyncio.sleep(0.02)

    @staticmethod
    def _resumable(carried: _Interrupted) -> bool:
        old = carried.response
        if old is None or old.queue is None:
            return False
        return bool(carried.replay) or any(not t.done() for t in carried.drain)

    async def _merge_unanswered(self, carried: _Interrupted | None, text: str, turn_id: int) -> str:
        """Nối nửa đầu câu vào lượt này, nếu người dùng chưa nghe câu trả lời nào.

        Ngừng 480 ms giữa câu là đủ để chốt lượt; nói tiếp khi máy đang nghĩ
        thì thành ngắt lời, và lượt mới chỉ còn nửa sau. Trả lời nửa sau một
        mình là trả lời một câu người dùng chưa từng hỏi.
        """
        if (
            carried is None
            or not self.config.conversation.barge_in.merge_unanswered
            or not carried.user_text
            or carried.content_heard
        ):
            return text
        merged = f"{carried.user_text} {text}"
        self.context.drop_last(carried.key.turn_id)
        for absorbed in reversed(carried.absorbed):
            self.context.drop_last(absorbed)
        # This turn's words are the merged ones now: a barge-in before they
        # reach the history (the await below) must carry all of them.
        self._confirmed_text = (turn_id, merged)
        self._count("turns_merged")
        self._emit(EventType.TURN_MERGED, previous_turn=carried.key.turn_id, text=merged)
        await self.sink.send_control(ControlMessage("transcript", {"text": merged, "final": True}))
        return merged

    async def _resume_or_idle(self, why: str) -> None:
        """After a turn ended without announcing it: continue the cut answer, or say idle.

        The interjection's own turn ("ừ", a cough) ends in IDLE, which is
        what `_resume_interrupted` requires — but telling the client meant an
        `idle` flash in the middle of an answer that carries on: measured
        02/10/2026, conversation_check took the flash for the end of the
        answer and closed the session 412 ms into the resume. A client cannot
        tell that flash from "done".
        """
        if not await self._resume_interrupted(why):
            await self.sink.send_control(ControlMessage("state", {"state": "idle"}))

    async def _resume_interrupted(self, why: str) -> bool:
        """Trả lại lượt cho trợ lý sau một lần ngắt lời hoá ra không có gì.

        Câu người dùng đã nghe trọn thì không đọc lại; câu đang nghe dở bị cắt
        thì đọc lại từ đầu câu đó. Phần model chưa viết xong vẫn đang được
        viết tiếp (drain), nên nói tiếp là nối vào chứ không hỏi lại từ đầu.
        """
        carried = self._interrupted
        self._interrupted = None
        cfg = self.config.conversation.barge_in
        ok = (
            carried is not None
            and cfg.resume_after_false
            and not self._closed
            and self.state.state is TurnState.IDLE
            and self._audio_clock_ms - carried.at_ms <= cfg.resume_window_ms
            and self._resumable(carried)
        )
        if not ok:
            self._end_drain(carried)
            return False
        assert carried is not None and carried.response is not None
        old = carried.response
        turn_id = self.gen.next_turn()
        self.state.to(TurnState.THINKING, f"resume after {why}")
        self._arm_barge_in()
        key = self.gen.begin(turn_id)
        self._count("resumed")
        self._emit(EventType.TURN_START, turn_id=turn_id, source="resume")
        cut_info: dict[str, Any] = {}
        items = self._resume_items(carried, cut_info)
        self._emit(
            EventType.RESUMED,
            why=why,
            interrupted_turn=carried.key.turn_id,
            replay=len(carried.replay),
            draining=any(not t.done() for t in carried.drain),
            **({"cut": cut_info} if cut_info else {}),
        )
        await self.sink.send_control(ControlMessage("state", {"state": "thinking", "source": "resume"}))
        # Text the drained LLM writes from here on belongs to this generation.
        for producer, target in list(self._drain_targets.items()):
            if producer == carried.key or target == carried.key:
                self._drain_targets[producer] = key
        # Same list object: what the drain still writes lands in this turn's history.
        response = self._new_response(key, generated=old.generated, spoken=list(old.spoken),
                                      history_spoken=list(old.history_spoken),
                                      heard_before=list(old.history_spoken), in_history=old.in_history)
        self._response = response
        queue: asyncio.Queue[Phrase | None] = asyncio.Queue()
        response.queue = queue
        for phrase in items:
            await queue.put(phrase)
        if old.generated:
            await self.sink.send_control(
                ControlMessage(
                    "assistant_delta",
                    {"text": old.generated_text, "generation_id": key.generation_id},
                )
            )
        if old.queue_ended:
            response.llm_done = True
            await queue.put(None)
        else:
            self.gen.spawn(self._pipe(old.queue, queue, response), key, name=f"pipe-{key}")
        # Không câu mở: "Vâng." chen vào trước khi nói tiếp nghe như một lượt mới.
        response.speak_task = self.gen.spawn(
            self._speak(key, response, queue, use_opener=False), key, name=f"resume-{key}"
        )
        return True

    def _resume_items(self, carried: _Interrupted, cut_info: dict[str, Any]) -> list[Phrase]:
        """What to say again, and from where.

        Re-reading the cut phrase from its start repeats up to a whole phrase
        the user just heard — seconds of "Ngày xửa ngày xưa, có một con cáo"
        twice over one "ừ". With the phrase's audio kept, playback continues
        from where the client actually stopped (its `playback_stopped` on
        reset), backed off to the quietest point before it.
        """
        old = carried.response
        cfg = self.config.conversation.barge_in
        if old is None or not cfg.resume_from_cut:
            return [Phrase(p.text, audio=p.audio) for p in carried.replay]
        at = carried.at_s
        cut = old.cut_observed_s if old.cut_observed_s is not None else at
        entries = [
            (s.phrase, s.frames, None, old._start(s))
            for s in old.sent if old._end(s) > at and not s.phrase.filler
        ]
        if old.current is not None and not old.current.filler:
            entries.append((old.current, old.current_frames, old.hold_done if old.holding else None,
                            old.current_started()))
        items: list[Phrase] = []
        for index, (phrase, frames, done, started) in enumerate(entries):
            if not frames and done is None:
                items.append(Phrase(phrase.text, audio=phrase.audio))   # nothing kept: read it again
                continue
            skip = 0
            if index == 0 and started is not None and cut > started and frames:
                played = cut - started
                target = int((played - cfg.resume_backoff_ms / 1000.0) * frames[0].sample_rate)
                skip = _quiet_point(frames, target)
                cut_info.update(phrase_id=phrase.phrase_id, played_ms=round(1000 * played, 1),
                                from_ms=round(1000 * skip / frames[0].sample_rate, 1),
                                basis="client_stop" if old.cut_observed_s is not None else "server_estimate")
            items.append(Phrase(phrase.text, audio=_held_audio(frames, skip, done, old.hold_progress)))
        return items

    async def _pipe(
        self,
        source: asyncio.Queue[Phrase | None],
        sink: asyncio.Queue[Phrase | None],
        response: ResponseState,
    ) -> None:
        """Hand what the cut turn still has queued (and is still writing) on.

        Fillers are dropped: "để tôi kiểm tra" after the answer has resumed is
        a line from a moment that is over.
        """
        try:
            while True:
                item = await source.get()
                if item is None:
                    break
                if not item.filler:
                    await sink.put(item)
            response.llm_done = True
        finally:
            await sink.put(None)

    async def interrupt(self, reason: str = "client") -> None:
        """Explicit stop from the user plane (a button, a DTMF, a hangup)."""
        if self.state.is_assistant_active():
            await self._handle_barge_in(reason=reason)
            if self.state.state is TurnState.LISTENING:
                await self._abandon_turn("explicit interrupt")
        elif self._interrupted is not None:
            # A voice barge-in already stopped the audio and is still deciding
            # whether that speech was a cough. The button decides it: ignoring
            # it here let a short "dừng" resume the very answer it stopped.
            self._end_drain(self._interrupted)
            self._interrupted = None
            if self.state.state is TurnState.LISTENING:
                await self._abandon_turn("explicit interrupt")

    # ------------------------------------------------------------------ #
    # tra cứu bất đồng bộ  (Speech agent  <-->  Back end - search)
    # ------------------------------------------------------------------ #
    def _dispatch_search(self, query: str) -> SearchRequest | None:
        """Gửi yêu cầu đi rồi trả lại quyền điều khiển ngay lập tức."""
        if self.search_agent is None:
            return None
        request = self.pending.open(query, self.gen.turn_id)
        if request is None:
            self._emit(EventType.SEARCH_DROPPED, reason="too_many_inflight", query=query)
            return None
        self._emit(
            EventType.SEARCH_REQUESTED,
            request_id=request.id,
            query=query,
            inflight=self.pending.inflight,
        )
        # spawn_detached, không phải spawn: việc này phải sống qua ngắt lời.
        self.gen.spawn_detached(self._run_search(request, self.gen.current), name=f"search-{request.id}")
        return request

    async def _prewarm(self, *texts: str) -> None:
        """Tổng hợp trước những câu cố định, một lần cho cả tiến trình."""
        wanted = list(texts) or [
            self.config.conversation.search.instant_ack,
            self.config.conversation.opener.text
            if self.config.conversation.opener.enabled
            else "",
        ]
        for text in wanted:
            if not text:
                continue
            try:
                audio = await self.models.cached_speech(text, self.voice)
            except Exception as exc:  # pragma: no cover - backend dependent
                log.warning("không tổng hợp trước được %r: %s", text, exc)
                continue
            if text == self.config.conversation.search.instant_ack:
                self._ack_audio = audio

    async def _run_search(self, request: SearchRequest, key=None) -> None:
        assert self.search_agent is not None
        try:
            with observing(self._probe("llm", key, role="search", round=1, search_request_id=request.id)):
                result = await self.search_agent.search(request)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            log.exception("search agent failed")
            result = SearchResult(
                request=request,
                ok=False,
                content="Tôi chưa tra cứu được thông tin này.",
                error=repr(exc),
            )
        if self.pending.complete(result):
            self._emit(
                EventType.SEARCH_RESULT,
                turn_id=request.turn_id,
                request_id=request.id,
                ok=result.ok,
                latency_ms=round(result.latency_ms, 1),
                source=result.source,
                source_title=result.source_title,
                source_url=result.source_url,
            )
        else:
            self._emit(EventType.SEARCH_DROPPED, turn_id=request.turn_id,
                       request_id=request.id, reason="expired")

    async def _delivery_loop(self) -> None:
        """Phát kết quả khi có chỗ trống trong hội thoại.

        Không cắt ngang người dùng và không đè lên chính mình: kết quả chờ tới
        lúc phiên rảnh. Đó là khác biệt giữa một trợ lý và một cái loa thông báo.
        """
        try:
            while not self._closed:
                await asyncio.sleep(0.1)
                for request in self.pending.sweep():
                    self._emit(
                        EventType.SEARCH_DROPPED, turn_id=request.turn_id,
                        request_id=request.id, reason="timeout"
                    )
                if not self.pending.has_ready:
                    continue
                if self.state.state is not TurnState.IDLE or self.gen.live_tasks():
                    continue
                result = self.pending.pop_ready()
                if result is not None:
                    if any(t.turn_id > result.request.turn_id and t.user_text for t in self.context.turns):
                        self.pending.dropped_stale += 1
                        self._emit(EventType.SEARCH_DROPPED, request_id=result.request.id,
                                   turn_id=result.request.turn_id, reason="new_user_turn")
                    else:
                        await self._deliver_search(result)
        except asyncio.CancelledError:
            return

    async def _deliver_search(self, result: SearchResult) -> None:
        """Một lượt nói do HỆ THỐNG chủ động mở, không do người dùng."""
        prefix = self.config.conversation.search.delivery_prefix
        if result.ok:
            label = result.source_title or result.source or "không rõ"
            source = f"Nguồn: {label}" + (f" ({result.source_url})" if result.source_url else "") + ". "
            self.context.add_tool_exchange(
                "search",
                f"Kết quả tra cứu cho “{result.request.query}”: {result.content}. {source}"
                f"Hãy mở đầu bằng “{prefix}”, chỉ nêu dữ kiện có trong đoạn trích, "
                "nói nguồn bằng tên, không đọc URL. Nếu đoạn trích không trả lời câu hỏi thì nói không tra được.",
            )
        turn_id = self.gen.next_turn()
        self.context.start_delivery_turn(turn_id)
        self.state.to(TurnState.THINKING, "search result")
        self._arm_barge_in()
        key = self.gen.begin(turn_id)
        self._emit(EventType.TURN_START, turn_id=turn_id, source="search")
        self._emit(EventType.TURN_CONFIRMED, reason="search result", source="search")
        self._emit(
            EventType.SEARCH_DELIVERED,
            request_id=result.request.id,
            wait_ms=round(result.request.age_ms, 1), ok=result.ok, error=result.error,
        )
        await self.sink.send_control(
            ControlMessage("state", {"state": "thinking", "source": "search"})
        )
        if result.ok and result.source_url:
            await self.sink.send_control(ControlMessage("search_source", {
                "generation_id": key.generation_id,
                "request_id": result.request.id,
                "title": result.source_title,
                "url": result.source_url,
            }))
        self.gen.spawn(
            (self._answer(key, result.request.query, allow_search=False)
             if result.ok else self._say_fixed(key, "Tôi chưa tra được thông tin này từ nguồn đã chọn.", role="fallback")),
            key,
            name=f"deliver-{result.request.id}",
        )

    # ------------------------------------------------------------------ #
    # watchdog
    # ------------------------------------------------------------------ #
    async def _watchdog(self) -> None:
        """Sweeps turns that never finished.

        A turn whose tasks died without reaching TURN_END leaves the machine in
        THINKING forever and every later utterance is silently swallowed. The
        sweep is the difference between a session that recovers and a session
        that is simply over while still looking connected.
        """
        timeout_ms = self.config.conversation.orphan_turn_timeout_ms
        try:
            while not self._closed:
                await asyncio.sleep(0.5)
                if not self.state.is_assistant_active():
                    continue
                # Stalled, not merely long. Since a turn lasts until the client
                # has PLAYED its audio, a 25-second answer spends 25 seconds in
                # SPEAKING; timing the state alone swept it as an orphan in the
                # middle of a sentence (measured 25/09). An orphan is a turn
                # with nothing coming out and nothing left to play.
                response = self._response
                if response is not None and response.play_end > time.monotonic():
                    continue
                stalled_ms = now_ms() - max(self.state.since_ms, self._progress_ms)
                if stalled_ms < timeout_ms:
                    continue
                key = self.gen.current
                self._count("orphan_turns")
                self._emit(
                    EventType.ERROR,
                    stage="orphan_turn",
                    state=self.state.state.value,
                    elapsed_ms=round(self.state.elapsed_ms, 1),
                    live_tasks=self.gen.live_tasks(),
                )
                if key is not None:
                    self._stop_drains(key)
                    await self.gen.cancel(key)
                    await self.sink.send_control(
                        ControlMessage(
                            "playback_reset",
                            {"generation_id": key.generation_id, "reason": "orphan"},
                        )
                    )
                self.barge_in.disarm()
                await self._close_asr()
                self.gate.reset()
                self.state.to(TurnState.IDLE, "orphan sweep")
                await self.sink.send_control(ControlMessage("state", {"state": "idle"}))
        except asyncio.CancelledError:
            return

    # ------------------------------------------------------------------ #
    # bookkeeping
    # ------------------------------------------------------------------ #
    def _on_state_change(self, previous: TurnState, current: TurnState, reason: str) -> None:
        self._emit(
            EventType.STATE_CHANGED, previous=previous.value, current=current.value, reason=reason
        )

    def _emit(self, type: EventType, **data: Any) -> Event:
        if type is EventType.ERROR:
            self._count("errors_" + str(data.get("stage", "unknown")))
        turn_id = data.pop("turn_id", None)
        key = self.gen.current
        event = Event(
            type=type,
            session_id=self.session_id,
            turn_id=turn_id if turn_id is not None else self.gen.turn_id or None,
            generation_id=key.generation_id if key else None,
            data=data,
        )
        self.trace.record(event)
        if self.config.observability.log_events:
            log.debug("%s %s", type.value, data if data else "")
        return event

    def _count(self, name: str, amount: int = 1) -> None:
        self.counters[name] = self.counters.get(name, 0) + amount

    def _record_turn_metrics(self, turn_id: int) -> None:
        turn = self.trace.turn(turn_id)
        if turn is None:
            return
        self.last_turn_metrics = turn.metrics()

    def stats(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "state": self.state.state.value,
            "turn_id": self.gen.turn_id,
            "generation_id": self.gen.current.generation_id if self.gen.current else None,
            "counters": dict(self.counters),
            "stale_drops": self.gen.stale_drops,
            "barge_in": {
                "fired": self.barge_in.stats.fired,
                "suppressed_guard": self.barge_in.stats.suppressed_guard,
                "suppressed_level": self.barge_in.stats.suppressed_level,
            },
            "search": {
                "agent": getattr(self.search_agent, "name", None),
                **self.pending.stats(),
            },
            "turns": self.trace.metrics_rows()[-10:],
        }
