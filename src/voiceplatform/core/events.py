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
    STALE_DROPPED = "stale_dropped"
    FILLER = "filler"

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
