"""Typed configuration.

Loading is fail-closed: an unknown key is a typo that would otherwise sit in the
file doing nothing, so it raises instead. Every tunable that changes perceived
latency or turn-taking lives here, never inline in the engine.
"""

from __future__ import annotations

import dataclasses
import math
import types
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, get_args, get_origin, get_type_hints

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
    semantic_model_path: str = ""
    semantic_threshold: float = 0.6
    semantic_probe_timeout_ms: int = 60
    # How long after the VAD endpoint candidate we still wait for more speech.
    # VAD alone ends a turn on any pause; these hold the turn open while the
    # sentence is clearly unfinished.
    silence_ms: int = 480
    max_silence_ms: int = 1400  # hard stop even when the text looks unfinished
    min_utterance_ms: int = 240
    max_utterance_ms: int = 30000
    # Decode the whole utterance the moment the VAD sees a pause, instead of
    # deciding on the last periodic partial. With partials every 500 ms the
    # text judged at the pause was missing its last words ("tôi muốn chuyển
    # tiền" for "... tiền cho"), so a dangling "cho" could never hold the turn.
    endpoint_decode: bool = True
    # That decode covers every speech frame when nothing voiced followed it,
    # and the ASR engine is an offline transducer: decoding the same audio
    # again at the confirm only adds its latency to the answer.
    reuse_endpoint_transcript: bool = True
    # Confident completion ("... không?", "... ạ", "cảm ơn") on a transcript
    # that covers all the speech: end the turn after this much silence instead
    # of silence_ms. 0 = off. An A/B range, not a new default for every turn.
    fast_silence_ms: int = 0
    # Start working at the confirm, start TALKING only when sure. A turn
    # confirmed on the neutral wait (silence_ms, no cue either way) may still
    # be a mid-sentence pause; its first audio waits until the silence has
    # lasted this long. The LLM and TTS are not held — only the speaker.
    # Measured on the G3 confirm set 29/09: with shadow-mode answers ready
    # sooner, a premature confirm became AUDIBLE in 8/100 pauses (2/100
    # before), because the answer now started before the user went on.
    # 0 = off. Confident (fast) and held (max/middle) turns are not gated.
    commit_silence_ms: int = 0


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
    # Speech that barged in, then turned out to be nothing — a cough, a door,
    # or a listener's "ừ" / "vâng" — used to leave the answer cancelled and the
    # assistant silent. With this on, the unspoken rest of the answer resumes
    # from the phrase that was cut, if the barge-in is resolved within the
    # window (audio clock, ms).
    resume_after_false: bool = True
    resume_window_ms: int = 5000
    # Anchor the echo guard to when the client actually starts PLAYING, not
    # when the server sent the first frame. The worklet buffers
    # playback_buffer_ms before it starts, so a guard counted from the send
    # had expired before the loudest echo reached the microphone.
    guard_from_playback: bool = True
    # Sent to the client as `ready.playback_buffer_ms`: the worklet buffers
    # this much before it starts, and again after every underrun.
    playback_startup_ms: int = 160
    # Inside the guard, count only frames this loud instead of none at all: a
    # user who starts talking as the assistant starts still interrupts, and
    # onset echo (quieter than a voice at the microphone) still does not.
    guard_min_rms: float = 1.0          # 1.0 = count nothing inside the guard
    # Resume a false interruption from where playback stopped (backed off to a
    # quiet point) using audio the server already has, instead of re-reading
    # the cut phrase from its start.
    resume_from_cut: bool = True
    resume_backoff_ms: int = 300
    # Which VAD decides "the user is talking over me". "same" = the session
    # VAD (energy by default). "silero" = a Silero model of its own, used for
    # the barge-in run AND to verify the interjection afterwards. Offline A/B
    # on the G3 stimuli (docs/audits/2026-09-29/g3/vad-ab.json): noise fired
    # the energy barge-in 52-69/100, Silero at 3 frames 1-5/100, and Silero
    # fired on real interruptions sooner (p95 120-140 ms vs 160 ms). Neither
    # survives background voices or echo at the barge-in floor.
    vad: str = "same"                   # same | silero   — what TRIGGERS the stop
    # A speech model used only AFTER the stop, to decide what the
    # interjection was (verify_speech_ms, backchannel_max_speech_ms). Kept
    # apart from `vad` because, as the trigger, Silero at 3 frames made
    # simulated loudspeaker echo cut 12/12 answers (energy: 7/12): echo of a
    # voice IS a voice. "silero" here costs 0.08 ms per 32 ms window.
    speech_model: str = "none"          # none | silero (implied by vad: silero)
    # An interjection whose audio holds less than this much Silero speech is
    # noise, whatever ASR makes of it (gipformer turns a cough into "đây").
    # 0 = off. Needs vad: silero.
    verify_speech_ms: int = 0
    # The interjection's ASR starts from the beginning of the current loud
    # burst (after >= 300 ms below min_rms), up to this much before the
    # barge-in fired — not from a fixed pre-roll. A barge-in that fires late
    # (the guard ate "Thôi", the comma broke the run, it fired on "dừng lại"
    # 481 ms after onset) otherwise hands ASR only the tail: bộ dev G3 29/09,
    # "Thôi, dừng lại" heard as "thực sự". 0 = the fixed pre-roll only.
    burst_preroll_max_ms: int = 800
    # An interjection this short (Silero speech ms), of at most two words and
    # no request word, is a backchannel whatever ASR spelled: gipformer writes
    # "ừ" as "từ", "ừ ừ" as "từ từ", "vâng ạ" as "thân ạ". Offline on the G3
    # stimuli, backchannels held <= ~500 ms of Silero speech and real
    # interruptions >= ~680 ms. 0 = off. Needs vad: silero.
    backchannel_max_speech_ms: int = 0
    # The user paused long enough to confirm a turn, then kept talking before
    # anything but a filler was heard. That is one sentence, not two turns:
    # the first half is carried into the new turn instead of being answered
    # (or lost) on its own.
    merge_unanswered: bool = True


@dataclass(slots=True)
class FillerConfig:
    enabled: bool = True
    after_ms: int = 700         # only if the task path is actually slow
    cooldown_ms: int = 0        # 0 keeps the previous per-turn behaviour
    phrases: list[str] = field(
        default_factory=lambda: [
            "Để tôi kiểm tra một chút.",
            "Chờ tôi tra cứu nhanh nhé.",
        ]
    )


@dataclass(slots=True)
class OpenerConfig:
    """Câu mở đã tổng hợp sẵn, phát khi lượt trả lời chưa kịp ra tiếng.

    Vì sao cần: vòng LLM 0 của một lượt gọi công cụ KHÔNG sinh chữ nào — nó chỉ
    sinh lời gọi — nên không có gì để nói cho tới khi vòng đó xong. Đo được
    TTFA p50 2203 ms ở lượt quyết định đi tra và 1234 ms ở lượt trả kết quả,
    trong khi câu báo "Để tôi tra cứu nhé." đã nằm sẵn trong bộ nhớ từ đầu.

    KHÔNG phát vô điều kiện: một câu đệm luôn phát chỉ là độ trễ tự thêm vào
    (xem ARCHITECTURE §5). Nó chỉ chạy khi sau `after_ms` vẫn chưa có cụm nào
    sẵn sàng — tức đúng những lượt mà nếu không có nó, người dùng ngồi nghe im
    lặng hai, ba giây.

    Đánh đổi: ở lượt chậm, sau câu mở sẽ có một quãng lặng trước câu thật. Đó
    là cái giá của việc có tiếng dưới một giây, và nó rẻ hơn im lặng hoàn toàn.
    """

    enabled: bool = True
    text: str = "Vâng."
    # 350 ms thắng cuộc đua ở MỌI lượt, kể cả những lượt vốn đã ra tiếng trong
    # 560 ms — tức là thêm một tiếng đệm vào chỗ không cần. 550 ms để lượt
    # nhanh tự về đích, mà lượt chậm vẫn có tiếng trước một giây: 550 cộng
    # thời gian phát một câu đã dựng sẵn vẫn còn xa 1000.
    after_ms: int = 550
    cooldown_ms: int = 0


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
    # Once an accepted asynchronous search has a spoken ACK, do not spend
    # another native LLM slot repeating that wait message.
    skip_redundant_ack: bool = False
    # Require a source for explicit lookup and place-location questions. The
    # route is intentionally narrow; other questions still use LLM tool calls.
    force_source_lookup: bool = False


@dataclass(slots=True)
class SpeculationConfig:
    """Shadow-mode answer: start the LLM on a stable transcript before the turn is confirmed.

    Nothing leaves the process until the turn is confirmed AND the final
    transcript equals the one speculated on: no text delta, no audio, no tool
    execution (a tool call in the stream only runs after adoption). A changed
    transcript or resumed speech cancels it.
    """

    enabled: bool = False
    max_attempts: int = 2           # per user turn
    # Never take the last free LLM slot: speculation must not delay another
    # session's confirmed request.
    min_free_slots: int = 1


@dataclass(slots=True)
class PauseConfig:
    """Quãng nghỉ ở chỗ nối hai cụm và quãng im dài trong một cụm.

    Talker đọc mỗi cụm như một câu riêng: có im ở đầu, im ở cuối (ZeroTTS giữ
    80 ms sau `<eoa>`), và tự lấy mẫu ra những quãng im 450–530 ms giữa câu.
    Phát nối đuôi nhau, hai đoạn im đó cộng lại thành 100–180 ms ở MỌI chỗ nối,
    bất kể đó là cuối câu hay giữa mệnh đề (đo 02/10/2026) — nghe như ngập ngừng.
    Nay cắt bỏ im đầu/cuối của từng cụm rồi đặt lại quãng nghỉ theo dấu câu
    kết thúc cụm, và nén quãng im dài trong cụm xuống `inner_max_ms`.
    """

    # Tắt mặc định: talker giả lập của test không có im để chuẩn hoá, và đổi
    # độ dài audio của nó chỉ làm sai lệch các test lịch phát. Bật trong config
    # triển khai (configs/local-cpu.yaml).
    enabled: bool = False
    # Dưới mức này (RMS mỗi 10 ms, dBFS) là im. Âm xát yếu đầu cụm vẫn trên mức
    # này; `lead_ms` giữ thêm một đoạn trước điểm bắt đầu để khỏi xén mất.
    threshold_db: float = -45.0
    lead_ms: int = 30
    sentence_ms: int = 300      # cụm kết thúc bằng . ! ? … (ZeroTTS đọc liền: 300–530 ms)
    clause_ms: int = 150        # cụm kết thúc bằng , ; :
    cut_ms: int = 40            # cụm bị cắt không có dấu câu (giữa mệnh đề)
    # Dấu phẩy của ZeroTTS dài 300–500 ms; quá ngưỡng này là "khoảng chết".
    inner_max_ms: int = 280


@dataclass(slots=True)
class ConversationConfig:
    turn_detection: TurnDetectionConfig = field(default_factory=TurnDetectionConfig)
    barge_in: BargeInConfig = field(default_factory=BargeInConfig)
    filler: FillerConfig = field(default_factory=FillerConfig)
    opener: OpenerConfig = field(default_factory=OpenerConfig)
    search: SearchConfig = field(default_factory=SearchConfig)
    speculation: SpeculationConfig = field(default_factory=SpeculationConfig)
    pauses: PauseConfig = field(default_factory=PauseConfig)
    history_turns: int = 12
    streaming_first_phrase_chars: int = 48
    streaming_first_phrase_wait_ms: int = 0
    # Empty preserves the original leading tool rule. {tools} is replaced by
    # sorted available names; voice style still follows this stable prefix.
    tool_instruction: str = ""
    # Deliberately short. A longer voice-style block measured 0/10 on tool
    # calling against Qwen3.5-9B; this wording measured 10/10 with no false
    # calls. See ConversationContext._system and scripts/measure_tool_calling.py
    # before making it more thorough.
    system_prompt: str = (
        "Trả lời sẽ được đọc thành tiếng: viết ngắn, không markdown."
    )
    speech_normalization: bool = False
    pronunciations: dict[str, str] = field(default_factory=dict)
    # 0 disables this guard. Some ASR backends do not report confidence; in
    # that case the model is asked to clarify through the system prompt.
    clarify_confidence_below: float = 0.0
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
    # Run a tiny ASR + TTS pass every N seconds while idle (0 = off), and once
    # whenever a client connects. On a shared box under memory pressure the
    # CPU engines' pages get evicted between visitors: measured 25/09, the
    # first TTS call after an idle stretch took 1537 ms at RTF 1.2 against
    # 60-80 ms warm — the first caller heard a stutter nobody else did.
    keep_warm_s: int = 0
    prewarm_before_sessions: bool = False
    startup_timeout_s: float = 180.0
    operation_timeout_s: float = 30.0


@dataclass(slots=True)
class TasksConfig:
    enabled: bool = True
    tools: list[str] = field(default_factory=lambda: ["clock"])
    default_timeout_ms: int = 8000
    max_parallel: int = 4
    max_queue: int = 16


@dataclass(slots=True)
class ObservabilityConfig:
    trace_dir: str = "runtime/traces"
    write_traces: bool = True
    log_events: bool = True
    # Turns kept per session trace. Named `keep_sessions` until it was noticed
    # that it never counted sessions.
    keep_turns: int = 200
    # 0 disables pruning. Applies only to completed session JSONL traces.
    trace_retention_days: int = 0


@dataclass(slots=True)
class ServerConfig:
    host: str = "127.0.0.1"
    port: int = 18100
    web_dir: str = "web"
    # The pages are served by this same server, so they need no CORS at all.
    # "*" here opens only the public routes; a loopback-only route answers
    # only an Origin listed by name (app/access.py).
    cors_origins: list[str] = field(default_factory=list)
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
    max_sessions: int = 3
    max_audio_bytes: int = 131072
    max_text_chars: int = 4000
    idle_timeout_s: float = 120.0
    max_session_s: float = 3600.0
    readiness_timeout_s: float = 2.0
    readiness_ttl_s: float = 2.0


@dataclass(slots=True)
class CollectConfig:
    """Trang /collect: đồng nghiệp tự thu clip lượt lời có nhãn cho G3.

    Tắt mặc định: bật lên là mở một route GHI file giọng nói người thật cho
    cả LAN. Ghi dưới `dir` với quyền 0600 (app/collect.py); không bao giờ đặt
    `dir` trong docs/.
    """

    enabled: bool = False
    dir: str = "runtime/collect"
    max_clip_s: float = 20.0
    # 20 s mono int16 16 kHz = 640 000 byte + 44 byte header.
    max_upload_bytes: int = 1048576
    # Rỗng = không hỏi mã. Có giá trị thì trang hỏi, và mọi POST/DELETE phải
    # mang đúng mã trong header X-Collect-Code.
    access_code: str = ""


@dataclass(slots=True)
class Config:
    audio: AudioConfig = field(default_factory=AudioConfig)
    media: MediaConfig = field(default_factory=MediaConfig)
    conversation: ConversationConfig = field(default_factory=ConversationConfig)
    models: ModelsConfig = field(default_factory=ModelsConfig)
    tasks: TasksConfig = field(default_factory=TasksConfig)
    observability: ObservabilityConfig = field(default_factory=ObservabilityConfig)
    server: ServerConfig = field(default_factory=ServerConfig)
    collect: CollectConfig = field(default_factory=CollectConfig)

    # --- frame maths used all over the conversation plane -------------------
    @property
    def frame_samples(self) -> int:
        return int(self.audio.sample_rate * self.audio.frame_ms / 1000)

    def frames_for_ms(self, ms: float) -> int:
        return max(1, int(round(ms / self.audio.frame_ms)))

    @classmethod
    def load(cls, path: str | Path | None) -> "Config":
        config = cls() if path is None else _build(cls, _read_mapping(Path(path)), root=str(path))
        config.validate()
        return config

    def validate(self) -> None:
        """Reject wrong types and unsafe limits before any model is loaded."""
        _validate_types(self, "config")
        positive = {
            "audio.sample_rate": self.audio.sample_rate,
            "audio.output_sample_rate": self.audio.output_sample_rate,
            "audio.frame_ms": self.audio.frame_ms,
            "audio.output_frame_ms": self.audio.output_frame_ms,
            "vad.start_frames": self.media.vad.start_frames,
            "vad.end_frames": self.media.vad.end_frames,
            "conversation.history_turns": self.conversation.history_turns,
            "conversation.streaming_first_phrase_chars": self.conversation.streaming_first_phrase_chars,
            "conversation.orphan_turn_timeout_ms": self.conversation.orphan_turn_timeout_ms,
            "turn.max_utterance_ms": self.conversation.turn_detection.max_utterance_ms,
            "barge_in.speech_frames": self.conversation.barge_in.speech_frames,
            "search.max_inflight": self.conversation.search.max_inflight,
            "search.ttl_ms": self.conversation.search.ttl_ms,
            "tasks.default_timeout_ms": self.tasks.default_timeout_ms,
            "tasks.max_parallel": self.tasks.max_parallel,
            "observability.keep_turns": self.observability.keep_turns,
        }
        positive.update({f"server.{name}": getattr(self.server, name) for name in (
            "max_sessions", "max_audio_bytes", "max_text_chars", "idle_timeout_s",
            "max_session_s", "readiness_timeout_s", "readiness_ttl_s",
        )})
        positive.update({f"models.{name}": getattr(self.models, name) for name in (
            "startup_timeout_s", "operation_timeout_s",
        )})
        for name, value in positive.items():
            if value <= 0:
                raise ConfigError(f"{name} must be positive")
        for name, value in {
            "tasks.max_queue": self.tasks.max_queue, "models.keep_warm_s": self.models.keep_warm_s,
            "vad.pre_roll_ms": self.media.vad.pre_roll_ms,
            "media.jitter_target_ms": self.media.jitter_target_ms,
            "turn.silence_ms": self.conversation.turn_detection.silence_ms,
            "turn.min_utterance_ms": self.conversation.turn_detection.min_utterance_ms,
            "barge_in.guard_ms": self.conversation.barge_in.guard_ms,
            "barge_in.playback_startup_ms": self.conversation.barge_in.playback_startup_ms,
            "barge_in.resume_backoff_ms": self.conversation.barge_in.resume_backoff_ms,
            "barge_in.verify_speech_ms": self.conversation.barge_in.verify_speech_ms,
            "barge_in.burst_preroll_max_ms": self.conversation.barge_in.burst_preroll_max_ms,
            "barge_in.backchannel_max_speech_ms": self.conversation.barge_in.backchannel_max_speech_ms,
            "turn.fast_silence_ms": self.conversation.turn_detection.fast_silence_ms,
            "turn.commit_silence_ms": self.conversation.turn_detection.commit_silence_ms,
            "turn.semantic_probe_timeout_ms": self.conversation.turn_detection.semantic_probe_timeout_ms,
            "observability.trace_retention_days": self.observability.trace_retention_days,
            "speculation.max_attempts": self.conversation.speculation.max_attempts,
            "speculation.min_free_slots": self.conversation.speculation.min_free_slots,
            "barge_in.resume_window_ms": self.conversation.barge_in.resume_window_ms,
            "filler.after_ms": self.conversation.filler.after_ms,
            "filler.cooldown_ms": self.conversation.filler.cooldown_ms,
            "opener.after_ms": self.conversation.opener.after_ms,
            "opener.cooldown_ms": self.conversation.opener.cooldown_ms,
            "conversation.streaming_first_phrase_wait_ms": self.conversation.streaming_first_phrase_wait_ms,
            **{f"pauses.{name}": getattr(self.conversation.pauses, name) for name in (
                "lead_ms", "sentence_ms", "clause_ms", "cut_ms", "inner_max_ms",
            )},
        }.items():
            if value < 0:
                raise ConfigError(f"{name} must be non-negative")
        if not -90 < self.conversation.pauses.threshold_db < 0:
            raise ConfigError("pauses.threshold_db must be in (-90, 0) dBFS")
        turn = self.conversation.turn_detection
        bi = self.conversation.barge_in
        if (bi.verify_speech_ms or bi.backchannel_max_speech_ms) and "silero" not in (bi.vad, bi.speech_model):
            raise ConfigError("barge_in.verify_speech_ms / backchannel_max_speech_ms need a speech model "
                              "(barge_in.speech_model: silero or barge_in.vad: silero)")
        if turn.fast_silence_ms > turn.silence_ms:
            raise ConfigError("turn.fast_silence_ms must not exceed turn.silence_ms")
        if turn.max_silence_ms < turn.silence_ms or turn.max_utterance_ms < turn.min_utterance_ms:
            raise ConfigError("turn detection maximum must be >= minimum")
        if not 0 < turn.semantic_threshold <= 1:
            raise ConfigError("turn.semantic_threshold must be in (0, 1]")
        if turn.backend == "semantic":
            if not turn.semantic_model_path:
                raise ConfigError("turn.semantic_model_path is required for semantic backend")
            from ..conversation.turn_detector.text_model import load_text_turn_model
            try:
                load_text_turn_model(turn.semantic_model_path)
            except (OSError, ValueError, TypeError) as exc:
                raise ConfigError(f"turn.semantic_model_path is invalid: {exc}") from exc
        if self.audio.channels != 1:
            raise ConfigError("audio.channels must be 1 (mono)")
        if not 1 <= self.server.port <= 65535:
            raise ConfigError("server.port must be between 1 and 65535")
        if bool(self.server.ssl_certfile) != bool(self.server.ssl_keyfile):
            raise ConfigError("TLS certificate and key must be configured together")
        if self.models.mode != "cascade":
            raise ConfigError("only models.mode=cascade is implemented")
        for name, value, choices in (
            ("vad.backend", self.media.vad.backend, ("energy", "silero")),
            ("aec.backend", self.media.aec.backend, ("client", "none", "apm")),
            ("denoise", self.media.denoise, ("none", "rnnoise", "apm")),
            ("turn.backend", turn.backend, ("vad_only", "heuristic", "semantic")),
            ("barge_in.vad", self.conversation.barge_in.vad, ("same", "silero")),
            ("barge_in.speech_model", self.conversation.barge_in.speech_model, ("none", "silero")),
        ):
            if value not in choices:
                raise ConfigError(f"{name}: unsupported value {value!r}")
        for value in (self.media.vad.threshold, self.media.vad.energy_threshold,
                      self.conversation.barge_in.min_rms, self.conversation.barge_in.guard_min_rms):
            if not 0 <= value <= 1:
                raise ConfigError("VAD and RMS thresholds must be between 0 and 1")
        if not 0 <= self.conversation.clarify_confidence_below <= 1:
            raise ConfigError("conversation.clarify_confidence_below must be between 0 and 1")
        if any(not key.strip() or not value.strip() for key, value in self.conversation.pronunciations.items()):
            raise ConfigError("conversation.pronunciations needs non-empty words")
        if not all(8000 <= rate <= 192000 for rate in (self.audio.sample_rate, self.audio.output_sample_rate)):
            raise ConfigError("sample rates must be between 8000 and 192000")
        collect = self.collect
        if not 1 <= collect.max_clip_s <= 120:
            raise ConfigError("collect.max_clip_s must be between 1 and 120 seconds")
        if collect.max_upload_bytes < 44 + int(collect.max_clip_s * 16000) * 2:
            # Otherwise a clip under max_clip_s is refused as "too big".
            raise ConfigError("collect.max_upload_bytes must hold collect.max_clip_s of 16 kHz mono int16 WAV")
        if collect.enabled and not collect.dir.strip():
            raise ConfigError("collect.dir is required when collect is enabled")
        if any(not "!" <= ch <= "~" for ch in collect.access_code):
            # It travels in an HTTP header, and fetch() refuses non-Latin-1 there.
            raise ConfigError("collect.access_code must be printable ASCII without spaces")

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def _validate_types(obj: Any, path: str) -> None:
    def matches(value: Any, hint: Any) -> bool:
        origin, args = get_origin(hint), get_args(hint)
        if hint is Any:
            return True
        if origin is types.UnionType:
            return any(matches(value, item) for item in args)
        if origin is list:
            return isinstance(value, list) and all(matches(item, args[0]) for item in value)
        if origin is dict:
            return isinstance(value, dict) and all(matches(k, args[0]) and matches(v, args[1]) for k, v in value.items())
        if hint is float:
            return type(value) in (int, float) and math.isfinite(value)
        if hint in (int, bool):
            return type(value) is hint
        return isinstance(value, hint)

    for name, hint in get_type_hints(type(obj)).items():
        value = getattr(obj, name)
        if not matches(value, hint):
            raise ConfigError(f"{path}.{name}: invalid type or non-finite value")
        if is_dataclass(value):
            _validate_types(value, f"{path}.{name}")


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
