"""Typed configuration.

Loading is fail-closed: an unknown key is a typo that would otherwise sit in the
file doing nothing, so it raises instead. Every tunable that changes perceived
latency or turn-taking lives here, never inline in the engine.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, get_type_hints

from .errors import ConfigError


@dataclass(slots=True)
class AudioConfig:
    sample_rate: int = 16000
    frame_ms: int = 20          # media-plane granularity (VAD / transport)
    channels: int = 1
    # What the platform asks the talker to produce. The borrowed engines
    # default to 16 kHz on their own, so this has to be handed to them or a
    # 24 kHz voice arrives quietly resampled down.
    output_sample_rate: int = 24000
    # Assistant audio is sliced to this before it goes on the wire. A talker
    # that returns a whole phrase in one blob would otherwise put two seconds
    # of audio in a single message: playback cannot start until the last byte
    # lands, and the client has one enormous buffer to stop on a barge-in.
    output_frame_ms: int = 40


@dataclass(slots=True)
class VadConfig:
    backend: str = "energy"     # energy | silero
    threshold: float = 0.5
    energy_threshold: float = 0.012
    # Frames of speech before START fires; frames of silence before END fires.
    start_frames: int = 2       # ~40 ms at 20 ms frames
    end_frames: int = 12        # ~240 ms: an endpoint candidate, not a decision
    pre_roll_ms: int = 320      # audio kept before START so no onset is lost


@dataclass(slots=True)
class AecConfig:
    backend: str = "client"     # client (getUserMedia) | none | apm
    enabled: bool = True


@dataclass(slots=True)
class MediaConfig:
    vad: VadConfig = field(default_factory=VadConfig)
    aec: AecConfig = field(default_factory=AecConfig)
    denoise: str = "none"       # none | rnnoise | apm
    jitter_target_ms: int = 60


@dataclass(slots=True)
class TurnDetectionConfig:
    backend: str = "heuristic"  # vad_only | heuristic | semantic
    # How long after the VAD endpoint candidate we still wait for more speech.
    # VAD alone ends a turn on any pause; these hold the turn open while the
    # sentence is clearly unfinished.
    silence_ms: int = 480
    max_silence_ms: int = 1400  # hard stop even when the text looks unfinished
    min_utterance_ms: int = 240
    max_utterance_ms: int = 30000


@dataclass(slots=True)
class BargeInConfig:
    enabled: bool = True
    # Consecutive speech frames while the assistant talks. One frame turns every
    # cough into an interruption; the counter must reset on every silent frame.
    speech_frames: int = 6      # ~120 ms at 20 ms frames
    # Ignore the first moments of assistant audio: room echo leaking back in is
    # loudest right at onset when AEC has not converged.
    guard_ms: int = 150
    min_rms: float = 0.02


@dataclass(slots=True)
class FillerConfig:
    enabled: bool = True
    after_ms: int = 700         # only if the task path is actually slow
    phrases: list[str] = field(
        default_factory=lambda: [
            "Để tôi kiểm tra một chút.",
            "Chờ tôi tra cứu nhanh nhé.",
        ]
    )


@dataclass(slots=True)
class SearchConfig:
    """Tra cứu bất đồng bộ: gửi đi, nói tiếp, trả kết quả sau."""

    enabled: bool = False
    # Quá số này thì từ chối yêu cầu mới. Xếp hàng dài chỉ tạo ra một loạt câu
    # trả lời muộn không còn ai nhớ đã hỏi gì.
    max_inflight: int = 2
    # Kết quả già hơn ngưỡng này thì bỏ: nói ra chỉ làm người nghe bối rối.
    ttl_ms: int = 60000
    # Câu mở đầu khi phát kết quả, để người nghe biết nó thuộc về câu hỏi cũ.
    delivery_prefix: str = "Về câu bạn hỏi lúc nãy"
    # Phát NGAY khi yêu cầu tra cứu được gửi đi, không chờ model soạn câu.
    # Đo được: vòng LLM quyết định gọi công cụ tốn 1.66 s, rồi vòng soạn câu
    # tốn thêm 1.5 s nữa trước khi có tiếng. Câu cố định này cắt hẳn quãng đó.
    instant_ack: str = "Để tôi tra cứu nhé."


@dataclass(slots=True)
class ConversationConfig:
    turn_detection: TurnDetectionConfig = field(default_factory=TurnDetectionConfig)
    barge_in: BargeInConfig = field(default_factory=BargeInConfig)
    filler: FillerConfig = field(default_factory=FillerConfig)
    search: SearchConfig = field(default_factory=SearchConfig)
    history_turns: int = 12
    # Deliberately short. A longer voice-style block measured 0/10 on tool
    # calling against Qwen3.5-9B; this wording measured 10/10 with no false
    # calls. See ConversationContext._system and scripts/measure_tool_calling.py
    # before making it more thorough.
    system_prompt: str = (
        "Trả lời sẽ được đọc thành tiếng: viết ngắn, không markdown."
    )
    # A turn whose generation never produced audio and was never cancelled is a
    # leak; the engine sweeps them so state cannot wedge in THINKING.
    orphan_turn_timeout_ms: int = 20000


@dataclass(slots=True)
class EngineSpec:
    """One model-plane engine: a name in the registry plus its options."""

    backend: str = "mock"
    options: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ModelsConfig:
    mode: str = "cascade"       # cascade | half_cascade | s2s
    asr: EngineSpec = field(default_factory=lambda: EngineSpec(backend="mock"))
    llm: EngineSpec = field(default_factory=lambda: EngineSpec(backend="mock"))
    tts: EngineSpec = field(default_factory=lambda: EngineSpec(backend="mock"))
    s2s: EngineSpec = field(default_factory=lambda: EngineSpec(backend="none"))
    # Tác nhân tra cứu, tách hẳn khỏi model hội thoại. "none" = không có.
    search: EngineSpec = field(default_factory=lambda: EngineSpec(backend="none"))


@dataclass(slots=True)
class TasksConfig:
    enabled: bool = True
    tools: list[str] = field(default_factory=lambda: ["clock"])
    default_timeout_ms: int = 8000
    max_parallel: int = 4


@dataclass(slots=True)
class ObservabilityConfig:
    trace_dir: str = "runtime/traces"
    write_traces: bool = True
    log_events: bool = True
    # Turns kept per session trace. Named `keep_sessions` until it was noticed
    # that it never counted sessions.
    keep_turns: int = 200


@dataclass(slots=True)
class ServerConfig:
    host: str = "127.0.0.1"
    port: int = 18100
    web_dir: str = "web"
    cors_origins: list[str] = field(default_factory=lambda: ["*"])
    # TLS is not about secrecy here, it is about having a microphone at all:
    # getUserMedia and AudioWorklet need a secure context and the only
    # exemption is localhost. Served over plain HTTP to another machine, the
    # product loses its entire voice path and keeps only the text box.
    ssl_certfile: str | None = None
    ssl_keyfile: str | None = None
    # /sessions lists every live session id, and an id is the key to that
    # session's transcripts. On a shared link that is an enumeration vector,
    # so both it and /config can be pinned to loopback.
    private_introspection: bool = False


@dataclass(slots=True)
class Config:
    audio: AudioConfig = field(default_factory=AudioConfig)
    media: MediaConfig = field(default_factory=MediaConfig)
    conversation: ConversationConfig = field(default_factory=ConversationConfig)
    models: ModelsConfig = field(default_factory=ModelsConfig)
    tasks: TasksConfig = field(default_factory=TasksConfig)
    observability: ObservabilityConfig = field(default_factory=ObservabilityConfig)
    server: ServerConfig = field(default_factory=ServerConfig)

    # --- frame maths used all over the conversation plane -------------------
    @property
    def frame_samples(self) -> int:
        return int(self.audio.sample_rate * self.audio.frame_ms / 1000)

    def frames_for_ms(self, ms: float) -> int:
        return max(1, int(round(ms / self.audio.frame_ms)))

    @classmethod
    def load(cls, path: str | Path | None) -> "Config":
        if path is None:
            return cls()
        payload = _read_mapping(Path(path))
        return _build(cls, payload, root=str(path))

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def _read_mapping(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise ConfigError(f"config not found: {path}")
    text = path.read_text(encoding="utf-8")
    if path.suffix in {".yaml", ".yml"}:
        try:
            import yaml
        except ImportError as exc:  # pragma: no cover - env specific
            raise ConfigError("PyYAML is required to read YAML configs") from exc
        data = yaml.safe_load(text) or {}
    else:
        import json

        data = json.loads(text)
    if not isinstance(data, dict):
        raise ConfigError(f"config root must be a mapping: {path}")
    return data


def _build(cls: type, payload: dict[str, Any], root: str) -> Any:
    # `from __future__ import annotations` makes field.type a string, so the
    # nested dataclasses only surface through resolved hints.
    hints = get_type_hints(cls)
    known = {f.name: hints.get(f.name, f.type) for f in fields(cls)}
    unknown = sorted(set(payload) - set(known))
    if unknown:
        raise ConfigError(
            f"{root}: unknown key(s) for {cls.__name__}: {', '.join(unknown)}"
        )
    kwargs: dict[str, Any] = {}
    for name, hint in known.items():
        if name not in payload:
            continue
        value = payload[name]
        if isinstance(hint, type) and is_dataclass(hint):
            if not isinstance(value, dict):
                raise ConfigError(f"{root}: {name} must be a mapping")
            kwargs[name] = _build(hint, value, root=f"{root}:{name}")
        else:
            kwargs[name] = value
    return cls(**kwargs)
