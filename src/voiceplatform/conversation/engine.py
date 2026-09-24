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
from ..observability.trace import SessionTrace
from ..tasks.base import TaskContext
from ..tasks.executor import TaskExecutor
from ..tasks.registry import ToolRegistry
from ..tasks.search import SearchAgent, SearchRequest, SearchResult
from .barge_in import BargeInDetector
from .context import ConversationContext
from .pending import PendingSearches
from .segmenter import PhraseSegmenter
from .sink import AudioSink
from .state import TurnState, TurnStateMachine
from .turn_detector import build_turn_detector

log = get_logger("engine")


async def _aiter(items: list[Any]):
    """Cho audio dựng sẵn đi qua đúng đường mà audio vừa tổng hợp đi."""
    for item in items:
        yield item

_MAX_TOOL_ROUNDS = 2
_FALLBACK_REPLY = "Xin lỗi, tôi đang gặp trục trặc. Bạn nhắc lại giúp tôi nhé."


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


@dataclass(slots=True)
class ResponseState:
    """What the current generation has produced so far."""

    key: GenerationKey
    generated: list[str] = field(default_factory=list)
    spoken: list[str] = field(default_factory=list)
    first_audio_ms: float | None = None

    @property
    def generated_text(self) -> str:
        return "".join(self.generated)

    @property
    def spoken_text(self) -> str:
        return " ".join(self.spoken)


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
            config.conversation.system_prompt, config.conversation.history_turns
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
        self.turn_detector = build_turn_detector(
            config.conversation.turn_detection.backend,
            silence_ms=config.conversation.turn_detection.silence_ms,
            max_silence_ms=config.conversation.turn_detection.max_silence_ms,
        )
        self.preroll = RingBuffer(
            int(audio.sample_rate * config.media.vad.pre_roll_ms / 1000) or 1
        )

        # per-turn scratch
        # Turn timing runs on an audio clock (frames consumed), not wall time.
        # In a live session the two agree; everywhere else — tests, replays,
        # a burst of frames after a jitter stall — only audio time is right.
        self._audio_clock_ms = 0.0
        self._asr_stream: Any = None
        self._utterance_ms = 0.0   # everything handed to ASR, pre-roll included
        self._speech_ms = 0.0      # frames that actually looked like speech
        self._partial_text = ""
        self._endpoint_at_ms: float | None = None
        self._required_silence_ms = float(config.conversation.turn_detection.silence_ms)
        self._response: ResponseState | None = None
        self._watchdog_task: asyncio.Task | None = None
        self._delivery_task: asyncio.Task | None = None
        self._ack_audio: list[Any] | None = None
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
        self._emit(type, **data)

    # ------------------------------------------------------------------ #
    # lifecycle
    # ------------------------------------------------------------------ #
    async def start(self) -> None:
        self._emit(EventType.SESSION_OPEN, models=self.models.describe())
        await self.sink.send_control(
            ControlMessage(
                "ready",
                {
                    "session_id": self.session_id,
                    "input_sample_rate": self.config.audio.sample_rate,
                    "frame_ms": self.config.audio.frame_ms,
                    "output_sample_rate": int(
                        getattr(self.models.tts, "capabilities").native_sample_rate
                    ),
                    "models": self.models.describe(),
                },
            )
        )
        if self._watchdog_task is None:
            self._watchdog_task = asyncio.create_task(self._watchdog(), name="orphan-sweep")
        if self._delivery_task is None and self.search_agent is not None:
            self._delivery_task = asyncio.create_task(
                self._delivery_loop(), name="search-delivery"
            )
            # Nền, không chặn phiên: nếu chưa kịp ấm thì lượt đầu vẫn tổng hợp
            # như thường, chỉ chậm hơn.
            asyncio.create_task(self._prewarm_ack(), name="prewarm-ack")

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for task in (self._watchdog_task, self._delivery_task):
            if task is not None:
                task.cancel()
        self._watchdog_task = None
        self._delivery_task = None
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

        # 1. Interruption first: it must not wait on anything below.
        if self.state.is_assistant_active() and self.barge_in.update(
            probability, frame, self._audio_clock_ms
        ):
            await self._handle_barge_in()
            # The interrupting speech itself belongs to the new turn.

        self.preroll.write(frame.samples)
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
                self._count("endpoint_cancelled")
        elif edge is GateEdge.END:
            self._emit(EventType.VAD_END, utterance_ms=round(self._utterance_ms, 1))
            if self.state.state is TurnState.LISTENING:
                await self._mark_endpoint()

        if self.state.state is TurnState.LISTENING:
            await self._maybe_confirm_turn()

    async def _on_partial(self, transcript: Transcript) -> None:
        self._partial_text = transcript.text
        if EventType.ASR_FIRST_PARTIAL.value not in self._turn_firsts():
            self._emit(EventType.ASR_FIRST_PARTIAL, text=transcript.text)
        self._emit(EventType.ASR_PARTIAL, text=transcript.text)
        await self.sink.send_control(
            ControlMessage("transcript", {"text": transcript.text, "final": False})
        )
        # A longer partial can change the endpointing decision mid-pause.
        if self._endpoint_at_ms is not None:
            self._required_silence_ms = await self.turn_detector.required_silence_ms(
                text=self._partial_text, utterance_ms=self._utterance_ms
            )

    def _turn_firsts(self) -> dict[str, float]:
        turn = self.trace.turn(self.gen.turn_id)
        return turn.firsts if turn else {}

    # ------------------------------------------------------------------ #
    # turn boundaries
    # ------------------------------------------------------------------ #
    async def _begin_listening(self) -> None:
        turn_id = self.gen.next_turn()
        self.state.to(TurnState.LISTENING, "speech started")
        self._utterance_ms = 0.0
        self._speech_ms = 0.0
        self._partial_text = ""
        self._endpoint_at_ms = None
        self._required_silence_ms = float(self.config.conversation.turn_detection.silence_ms)
        self._emit(EventType.TURN_START, turn_id=turn_id)
        try:
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

    async def _mark_endpoint(self) -> None:
        self._endpoint_at_ms = self._audio_clock_ms
        self._required_silence_ms = await self.turn_detector.required_silence_ms(
            text=self._partial_text, utterance_ms=self._utterance_ms
        )
        self._emit(
            EventType.ENDPOINT_CANDIDATE,
            required_silence_ms=round(self._required_silence_ms, 1),
            text=self._partial_text,
        )

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
                await self._abandon_turn("too short")
                return
            await self._confirm_turn(f"silence {silence_ms:.0f}ms")

    async def _confirm_turn(self, reason: str) -> None:
        self._endpoint_at_ms = None
        self.state.to(TurnState.THINKING, reason)
        self._arm_barge_in()
        key = self.gen.begin(self.gen.turn_id)
        self._emit(EventType.TURN_CONFIRMED, reason=reason, utterance_ms=round(self._utterance_ms, 1))
        await self.sink.send_control(ControlMessage("state", {"state": "thinking"}))
        self.gen.spawn(self._respond(key), key, name=f"respond-{key}")

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

    async def _abandon_turn(self, reason: str) -> None:
        await self._close_asr()
        self.gate.reset()
        self._endpoint_at_ms = None
        self._utterance_ms = 0.0
        self._speech_ms = 0.0
        self._emit(EventType.TURN_END, reason=reason, answered=False)
        self.state.to(TurnState.IDLE, reason)
        await self.sink.send_control(ControlMessage("state", {"state": "idle"}))

    async def _close_asr(self) -> None:
        if self._asr_stream is not None:
            try:
                await self._asr_stream.close()
            except Exception:  # pragma: no cover - backend dependent
                pass
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
    async def _respond(self, key: GenerationKey) -> None:
        try:
            stream = self._asr_stream
            transcript = (
                await stream.finish()
                if stream is not None
                else Transcript(text="", is_final=True)
            )
            self._asr_stream = None
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
                await self._finish_turn(key, answered=False)
                return
            self.context.start_turn(key.turn_id, text)
            await self._answer(key, text)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # never let one turn take the session down
            log.exception("respond failed")
            self._emit(EventType.ERROR, stage="respond", error=repr(exc))
            await self._speak_fallback(key)

    async def _answer(
        self, key: GenerationKey, user_text: str, *, allow_search: bool = True
    ) -> None:
        response = ResponseState(key=key)
        self._response = response
        phrases: asyncio.Queue[Phrase | None] = asyncio.Queue()
        speak_task = self.gen.spawn(self._speak(key, response, phrases), key, name=f"speak-{key}")

        segmenter_kwargs = dict(
            max_chars=90,
            min_words=5,
            keep_emotion_cues=self.models.tts.capabilities.emotion_cues,
        )
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

        try:
            for round_index in range(_MAX_TOOL_ROUNDS + 1):
                segmenter = PhraseSegmenter(**segmenter_kwargs)
                tool_calls: list[Any] = []
                saw_token = False
                self._emit(EventType.LLM_START, round=round_index)
                messages: list[Message] = self.context.messages(tool_names=tool_names)
                async for delta in self.models.llm.stream(messages, tools=tools):
                    if not self.gen.check(key):
                        self._emit(EventType.STALE_DROPPED, stage="llm")
                        return
                    if delta.text:
                        if not saw_token:
                            saw_token = True
                            self._emit(EventType.LLM_FIRST_TOKEN)
                        response.generated.append(delta.text)
                        await self.sink.send_control(
                            ControlMessage(
                                "assistant_delta",
                                {"text": delta.text, "generation_id": key.generation_id},
                            )
                        )
                        for phrase in segmenter.push(delta.text):
                            await phrases.put(Phrase(phrase))
                    if delta.tool_call is not None:
                        # Parallel calls arrive as separate deltas. Keeping only
                        # the last one dropped the others and left the history
                        # with one tool message for two calls.
                        tool_calls.append(delta.tool_call)
                    if delta.finish_reason:
                        break
                for phrase in segmenter.flush():
                    await phrases.put(Phrase(phrase))
                self._emit(EventType.LLM_COMPLETE, round=round_index)

                if not tool_calls or self.executor is None:
                    break
                for index, call in enumerate(tool_calls):
                    # One filler per round, not per call: the point is to cover
                    # a silence, and the second one lands in the middle of it.
                    await self._run_tool(
                        key, call, phrases, user_text, allow_filler=index == 0
                    )
                    if not self.gen.check(key):
                        return
        except asyncio.CancelledError:
            raise
        except (ModelUnavailable, ModelTimeout) as exc:
            log.warning("LLM unavailable: %s", exc)
            self._emit(EventType.ERROR, stage="llm", error=str(exc))
            await phrases.put(Phrase(_FALLBACK_REPLY))
        except Exception as exc:
            log.exception("answer failed")
            self._emit(EventType.ERROR, stage="answer", error=repr(exc))
            await phrases.put(Phrase(_FALLBACK_REPLY))
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
    ) -> None:
        """Slow path. The audio loop keeps running; only this turn waits."""
        assert self.executor is not None
        ctx = TaskContext(key=key, session_id=self.session_id, user_text=user_text)
        task = asyncio.ensure_future(
            self.executor.run(
                tool_call.name, tool_call.arguments, ctx, is_current=self.gen.is_current
            )
        )
        filler = self.config.conversation.filler
        if allow_filler and filler.enabled and filler.phrases:
            done, _ = await asyncio.wait({task}, timeout=filler.after_ms / 1000.0)
            if not done:
                # Only now is the wait long enough to be worth covering. A
                # filler spoken unconditionally is just added latency.
                phrase = filler.phrases[key.turn_id % len(filler.phrases)]
                self._emit(EventType.FILLER, text=phrase)
                await phrases.put(Phrase(phrase))
        result = await task
        if not self.gen.check(key):
            return
        # Tool có thể yêu cầu nói ngay một câu cố định, không chờ model soạn.
        # Với tra cứu, đó là khác biệt giữa im lặng 3 giây và trả lời 1 giây.
        speak_now = result.data.get("speak_now") if result.data else None
        if speak_now:
            text = str(speak_now)
            self._emit(EventType.FILLER, text=text, source=tool_call.name)
            # Người nghe phải thấy đúng thứ mình nghe: câu này không đi qua
            # model nên phải tự gửi lên transcript.
            await self.sink.send_control(
                ControlMessage(
                    "assistant_delta", {"text": text + " ", "generation_id": key.generation_id}
                )
            )
            response = self._response
            if response is not None and response.key == key:
                response.generated.append(text + " ")
            await phrases.put(Phrase(text, audio=self._ack_audio))
        content = result.content if result.ok else (result.content or "không lấy được dữ liệu")
        self.context.add_tool_exchange(tool_call.name, content)

    async def _speak(
        self,
        key: GenerationKey,
        response: ResponseState,
        phrases: asyncio.Queue[Phrase | None],
    ) -> None:
        tts = self.models.tts
        started = False
        try:
            while True:
                item = await phrases.get()
                if item is None:
                    break
                if not self.gen.check(key):
                    self._emit(EventType.STALE_DROPPED, stage="tts_phrase")
                    return
                if not started:
                    started = True
                    self._emit(EventType.TTS_START)
                source = (
                    _aiter(item.audio)
                    if item.audio is not None
                    else tts.synthesize(item.text, voice=self.voice)
                )
                async for chunk in source:
                    for frame in self._slice_output(chunk):
                        # Re-checked per frame, not per phrase: a real talker
                        # can hand back two seconds at once, and an
                        # interruption inside those two seconds must still
                        # stop the very next frame.
                        if not self.gen.check(key):
                            self._emit(EventType.STALE_DROPPED, stage="tts_audio")
                            return
                        await self._send_audio_frame(key, response, frame)
                response.spoken.append(item.text)
            if started:
                self._emit(EventType.TTS_COMPLETE, phrases=len(response.spoken))
            await self._finish_turn(key, answered=started)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            log.exception("speak failed")
            self._emit(EventType.ERROR, stage="tts", error=repr(exc))
            await self._finish_turn(key, answered=started)

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
    ) -> None:
        if response.first_audio_ms is None:
            response.first_audio_ms = now_ms()
            self._emit(EventType.TTS_FIRST_AUDIO, sample_rate=frame.sample_rate)
            if self.state.state is TurnState.THINKING:
                self.state.to(TurnState.SPEAKING, "first audio")
            self.barge_in.arm(self._audio_clock_ms)
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
        self.preprocessor.far_end(frame)

    async def _speak_fallback(self, key: GenerationKey) -> None:
        if not self.gen.is_current(key):
            return
        response = ResponseState(key=key)
        queue: asyncio.Queue[Phrase | None] = asyncio.Queue()
        await queue.put(Phrase(_FALLBACK_REPLY))
        await queue.put(None)
        await self._speak(key, response, queue)

    async def _finish_turn(self, key: GenerationKey, *, answered: bool) -> None:
        if not self.gen.is_current(key):
            return
        response = self._response
        if response is not None and response.key == key:
            self.context.commit_assistant(
                response.generated_text, response.spoken_text, interrupted=False
            )
        self.barge_in.disarm()
        await self._close_asr()
        self._utterance_ms = 0.0
        self._speech_ms = 0.0
        self._partial_text = ""
        self.gate.reset()
        self._emit(EventType.TURN_END, answered=answered)
        if self.state.state in {TurnState.THINKING, TurnState.SPEAKING}:
            self.state.to(TurnState.IDLE, "turn finished")
        await self.sink.send_control(ControlMessage("state", {"state": "idle"}))
        self._record_turn_metrics(key.turn_id)

    # ------------------------------------------------------------------ #
    # interruption
    # ------------------------------------------------------------------ #
    async def _handle_barge_in(self, reason: str = "user speech") -> None:
        key = self.gen.current
        if key is None:
            return
        self._emit(EventType.BARGE_IN, reason=reason)
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

        response = self._response
        if response is not None and response.key == key:
            self.context.commit_assistant(
                response.generated_text, response.spoken_text, interrupted=True
            )
        self._response = None
        self.barge_in.disarm()
        await self._close_asr()
        self._count("barge_ins")
        self._record_turn_metrics(key.turn_id)

        self.state.to(TurnState.LISTENING, "barge-in")
        # Keep the interrupting words: a new turn starts from the pre-roll.
        turn_id = self.gen.next_turn()
        self._utterance_ms = 0.0
        self._speech_ms = 0.0
        self._partial_text = ""
        self._endpoint_at_ms = None
        self._required_silence_ms = float(self.config.conversation.turn_detection.silence_ms)
        self._emit(EventType.TURN_START, turn_id=turn_id, source="barge_in")
        try:
            self._asr_stream = await self.models.asr.open_stream(
                sample_rate=self.config.audio.sample_rate
            )
            self._emit(EventType.ASR_START)
            pre = self.preroll.read_last(self.preroll.capacity)
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

    async def interrupt(self, reason: str = "client") -> None:
        """Explicit stop from the user plane (a button, a DTMF, a hangup)."""
        if self.state.is_assistant_active():
            await self._handle_barge_in(reason=reason)
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
        self.gen.spawn_detached(self._run_search(request), name=f"search-{request.id}")
        return request

    async def _prewarm_ack(self) -> None:
        text = self.config.conversation.search.instant_ack
        if not text:
            return
        try:
            self._ack_audio = await self.models.cached_speech(text, self.voice)
        except Exception as exc:  # pragma: no cover - backend dependent
            log.warning("could not pre-synthesise the search acknowledgement: %s", exc)

    async def _run_search(self, request: SearchRequest) -> None:
        assert self.search_agent is not None
        try:
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
                request_id=request.id,
                ok=result.ok,
                latency_ms=round(result.latency_ms, 1),
                source=result.source,
            )
        else:
            self._emit(EventType.SEARCH_DROPPED, request_id=request.id, reason="expired")

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
                        EventType.SEARCH_DROPPED, request_id=request.id, reason="timeout"
                    )
                if not self.pending.has_ready:
                    continue
                if self.state.state is not TurnState.IDLE or self.gen.live_tasks():
                    continue
                result = self.pending.pop_ready()
                if result is not None:
                    await self._deliver_search(result)
        except asyncio.CancelledError:
            return

    async def _deliver_search(self, result: SearchResult) -> None:
        """Một lượt nói do HỆ THỐNG chủ động mở, không do người dùng."""
        prefix = self.config.conversation.search.delivery_prefix
        self.context.add_tool_exchange(
            "search",
            f"Kết quả tra cứu cho “{result.request.query}”: {result.content}. "
            f"Hãy mở đầu bằng “{prefix}” rồi nói kết quả ngắn gọn.",
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
            wait_ms=round(result.request.age_ms, 1),
        )
        await self.sink.send_control(
            ControlMessage("state", {"state": "thinking", "source": "search"})
        )
        self.gen.spawn(
            self._answer(key, result.request.query, allow_search=False),
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
                if self.state.elapsed_ms < timeout_ms:
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
