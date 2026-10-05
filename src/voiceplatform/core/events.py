"""The event vocabulary every plane speaks.

Names match the per-turn timeline in docs/OBSERVABILITY.md one to one; adding a
stage means adding a member here first so the trace keeps its shape.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from .clock import now_ms
from .ids import GenerationKey


class EventType(str, Enum):
    # session
    SESSION_OPEN = "session_open"
    SESSION_CLOSE = "session_close"
    STATE_CHANGED = "state_changed"
    ERROR = "error"

    # media plane
    AUDIO_RECEIVED = "audio_received"
    VAD_START = "vad_start"
    VAD_END = "vad_end"

    # conversation plane
    TURN_START = "turn_start"
    TURN_END = "turn_end"
    ENDPOINT_CANDIDATE = "endpoint_candidate"
    TURN_CONFIRMED = "turn_confirmed"
    BARGE_IN = "barge_in"
    CANCEL = "cancel"
    PLAYBACK_RESET = "playback_reset"
    RESUMED = "resumed"            # a false barge-in handed the turn back
    TURN_MERGED = "turn_merged"    # a paused sentence rejoined its first half
    STALE_DROPPED = "stale_dropped"
    FILLER = "filler"
    # G3: endpointing on a fresh transcript, shadow-mode answers, interjections
    ASR_ENDPOINT_TRANSCRIPT = "asr_endpoint_transcript"
    SPECULATION_STARTED = "speculation_started"
    SPECULATION_ADOPTED = "speculation_adopted"
    SPECULATION_DISCARDED = "speculation_discarded"
    BARGE_IN_GUARD = "barge_in_guard"          # guard re-anchored to playback onset
    INTERJECTION_REJECTED = "interjection_rejected"  # heard as noise, not a request
    AUDIO_COMMIT = "audio_commit"              # first audio released after the commit wait

    # model plane
    ASR_START = "asr_start"
    ASR_FIRST_PARTIAL = "asr_first_partial"
    ASR_PARTIAL = "asr_partial"
    ASR_FINAL = "asr_final"
    LLM_START = "llm_start"
    LLM_FIRST_TOKEN = "llm_first_token"
    LLM_COMPLETE = "llm_complete"
    TTS_START = "tts_start"
    TTS_FIRST_AUDIO = "tts_first_audio"
    TTS_COMPLETE = "tts_complete"

    # Measurement schema v2: server monotonic clock, explicit operation ids.
    SPEECH_LAST_FRAME = "speech_last_frame"
    ASR_FINALIZE_START = "asr_finalize_start"
    ASR_FINALIZE_END = "asr_finalize_end"
    MODEL_QUEUED = "model_queued"
    MODEL_SLOT_ACQUIRED = "model_slot_acquired"
    MODEL_INFERENCE_START = "model_inference_start"
    MODEL_INFERENCE_END = "model_inference_end"
    LLM_REQUEST_SENT = "llm_request_sent"
    LLM_FIRST_TOOL_DELTA = "llm_first_tool_delta"
    LLM_USAGE = "llm_usage"
    LLM_TERMINATED = "llm_terminated"
    PHRASE_READY = "phrase_ready"
    TTS_LOCK_ACQUIRED = "tts_lock_acquired"
    TTS_CHUNK_READY = "tts_chunk_ready"
    TTS_PHRASE_COMPLETE = "tts_phrase_complete"
    AUDIO_SENT = "audio_sent"
    AUDIO_GENERATION_END = "audio_generation_end"
    PLAYBACK_SIGNAL_STARTED = "playback_signal_started"
    PLAYBACK_GENERATION_END = "playback_generation_end"
    PLAYBACK_STARTED = "playback_started"
    PLAYBACK_STOPPED = "playback_stopped"
    PLAYBACK_UNDERRUN = "playback_underrun"
    PLAYBACK_RESUMED = "playback_resumed"
    PLAYBACK_BUFFER = "playback_buffer"
    CLIENT_CLOCK_SYNC = "client_clock_sync"
    EVENT_LOOP_LAG = "event_loop_lag"

    # task plane
    TOOL_START = "tool_start"
    TOOL_COMPLETE = "tool_complete"
    TOOL_FAILED = "tool_failed"
    SEARCH_REQUESTED = "search_requested"
    SEARCH_RESULT = "search_result"
    SEARCH_DELIVERED = "search_delivered"
    SEARCH_DROPPED = "search_dropped"


@dataclass(slots=True)
class Event:
    type: EventType
    session_id: str
    ts_ms: float = field(default_factory=now_ms)
    turn_id: int | None = None
    generation_id: int | None = None
    data: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def for_key(
        cls,
        type: EventType,
        key: GenerationKey,
        **data: Any,
    ) -> "Event":
        return cls(
            type=type,
            session_id=key.session_id,
            turn_id=key.turn_id,
            generation_id=key.generation_id,
            data=data,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "type": self.type.value,
            "session_id": self.session_id,
            "turn_id": self.turn_id,
            "generation_id": self.generation_id,
            "ts_ms": round(self.ts_ms, 3),
            **({"data": self.data} if self.data else {}),
        }
