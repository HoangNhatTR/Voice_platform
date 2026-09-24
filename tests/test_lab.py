"""Bàn thử model: các con số phải là của engine đang chạy, không phải bản sao."""

from __future__ import annotations

import numpy as np
import pytest

from voiceplatform.app.lab import Busy, LabService, read_wav, wav_bytes, word_errors
from voiceplatform.core.config import Config
from voiceplatform.core.errors import VoicePlatformError
from voiceplatform.models.registry import ModelPlane


class _Session:
    """Chỗ đứng cho một phiên đang mở; chỉ cần mang được giọng."""

    voice: str | None = None


class _Platform:
    """Đúng ba thứ LabService cần, không hơn."""

    def __init__(self, config: Config) -> None:
        self.models = ModelPlane(
            config.models, output_sample_rate=config.audio.output_sample_rate
        )
        self.sessions: dict[str, _Session] = {}
        self.voice: str | None = None


@pytest.fixture
async def lab(config):
    config.models.tts.options = {"first_audio_delay_ms": 5, "rtf": 0.01}
    platform = _Platform(config)
    await platform.models.start()
    yield LabService(platform, config), platform
    await platform.models.close()


# ------------------------------------------------------------------ âm thanh

def test_wav_survives_a_round_trip():
    rate = 16000
    tone = (0.5 * np.sin(2 * np.pi * 220 * np.arange(rate) / rate)).astype(np.float32)
    back, back_rate = read_wav(wav_bytes(tone, rate))
    assert back_rate == rate
    assert back.size == tone.size
    assert float(np.abs(back - tone).max()) < 1e-3


def test_an_unsupported_wav_width_says_how_to_fix_it():
    import io
    import wave

    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(1)
        handle.setframerate(16000)
        handle.writeframes(b"\x00" * 100)
    with pytest.raises(VoicePlatformError) as excinfo:
        read_wav(buffer.getvalue())
    assert "16-bit" in str(excinfo.value)


# ------------------------------------------------------------------ WER

def test_word_errors_returns_a_count_not_a_rate():
    """Gộp nhiều file phải cộng tử và mẫu riêng.

    Trung bình các tỉ lệ sẽ cho một clip ba từ đúng trọng số với một clip ba
    phút, và con số đó không so được với bất cứ báo cáo nào.
    """
    short = word_errors("một hai ba", "một hai bốn")          # 1/3
    long = word_errors(" ".join(["từ"] * 300), " ".join(["từ"] * 300))  # 0/300
    assert short == (1, 3)
    assert long == (0, 300)
    pooled = (short[0] + long[0]) / (short[1] + long[1])
    naive = (short[0] / short[1] + long[0] / long[1]) / 2
    assert pooled < naive / 20   # gộp đúng: 0.0033 chứ không phải 0.167


def test_word_errors_counts_insertions_and_deletions():
    assert word_errors("a b c", "a c")[0] == 1
    assert word_errors("a c", "a b c")[0] == 1


# ------------------------------------------------------------------ bài thử

async def test_asr_runs_the_loaded_engine_and_reports_rtf(lab):
    service, platform = lab
    audio = np.zeros(16000, dtype=np.float32)
    out = await service.try_asr(audio, 16000, reference="")
    assert out["engine"] == "mock"
    assert out["text"]
    assert out["audio_ms"] == 1000.0
    assert out["rtf"] is not None
    assert out["contended"] is False


async def test_asr_scores_against_a_reference_when_given_one(lab):
    service, _ = lab
    audio = np.zeros(8000, dtype=np.float32)
    first = await service.try_asr(audio, 16000, reference="")
    out = await service.try_asr(audio, 16000, reference=first["text"])
    assert out["wer_words"] > 0
    assert out["wer"] is not None


async def test_tts_shows_the_text_the_talker_actually_receives(lab):
    """Cột `prepared` là chỗ duy nhất nhìn thấy cue bị bỏ.

    Mock TTS khai `emotion_cues=False`, nên "[cười]" phải BIẾN MẤT trước khi
    tới talker. Engine nào đọc nó thành chữ là lỗi; trước đây lỗi đó không có
    chỗ nào để lộ ra.
    """
    service, platform = lab
    assert platform.models.tts.capabilities.emotion_cues is False
    out = await service.try_tts("Chào bạn [cười] rất vui được gặp bạn.", None)
    assert out["emotion_cues"] is False
    joined = " ".join(out["prepared"])
    assert "[cười]" not in joined
    assert "Chào bạn" in joined


async def test_tts_returns_audio_and_a_real_time_factor(lab):
    service, _ = lab
    out = await service.try_tts("Một câu ngắn để đo.", None)
    assert out["wav_base64"]
    assert out["audio_ms"] > 0
    assert out["rtf"] is not None
    assert out["phrases"] and out["phrases"][0]["first_chunk_ms"] is not None


async def test_llm_is_prompted_through_the_real_context(lab):
    """Thứ tự khối công cụ so với khối giọng nói đo được là 0/10 với 10/10."""
    service, _ = lab
    out = await service.try_llm("mấy giờ rồi", with_tools=True)
    assert "công cụ" in out["system_prompt"]
    assert out["total_ms"] > 0
    plain = await service.try_llm("mấy giờ rồi", with_tools=False)
    assert "công cụ" not in plain["system_prompt"]


async def test_a_live_session_marks_the_number_instead_of_refusing_it(lab):
    """Từ chối thì phiền, im lặng trả số nhiễu thì tệ hơn. Đánh dấu là đúng."""
    service, platform = lab
    platform.sessions["s1"] = _Session()
    out = await service.try_tts("Một câu.", None)
    assert out["contended"] is True
    assert out["live_sessions"] == 1


# ------------------------------------------------------------------ đổi engine

async def test_swapping_is_refused_while_a_session_is_live(lab):
    service, platform = lab
    platform.sessions["s1"] = _Session()
    with pytest.raises(Busy):
        await service.swap("tts", "mock", {})


async def test_a_failed_swap_leaves_the_old_engine_in_place(lab):
    """Đóng engine cũ trước là mất cả hai khi bản mới không nạp được."""
    service, platform = lab
    before = platform.models.asr
    with pytest.raises(Exception):
        await service.swap("asr", "khong-co-backend-nay", {})
    assert platform.models.asr is before
    out = await service.try_asr(np.zeros(1600, dtype=np.float32), 16000, "")
    assert out["text"]


async def test_swapping_the_talker_drops_the_pre_synthesised_line(lab):
    """Câu "Để tôi tra cứu nhé." đã dựng bằng talker CŨ, ở tốc độ lấy mẫu CŨ."""
    service, platform = lab
    await platform.models.cached_speech("Để tôi tra cứu nhé.", None)
    assert platform.models._speech_cache
    await service.swap("tts", "mock", {"sample_rate": 16000})
    assert platform.models._speech_cache == {}
    assert platform.models.tts.capabilities.native_sample_rate == 16000


async def test_swapping_updates_what_config_reports(lab):
    service, platform = lab
    await service.swap("search", "mock", {"delay_ms": 1})
    described = service.describe()
    assert described["kinds"]["search"]["backend"] == "mock"
    assert described["kinds"]["search"]["loaded"]["name"] == "mock"
    out = await service.try_search("giá vàng")
    assert out["ok"] is True
    assert out["latency_ms"] >= 0


# ------------------------------------------------------------------ giọng

def test_preset_voices_reads_display_name_pairs():
    """`list_preset_voices()` trả (hiển thị, tên); phải lấy TÊN, không lấy nhãn."""
    from voiceplatform.models.tts.bridge_viet_s2s import preset_voices

    class _Pairs:
        @staticmethod
        def list_preset_voices():
            return [("Minh Quân", "minh_quan"), ("Mai Chi", "mai_chi")]

    class _Plain:
        @staticmethod
        def list_preset_voices():
            return ["a", "b"]

    class _Nested:
        _tts = _Pairs()

    class _Silent:
        pass

    assert preset_voices(_Pairs()) == ("minh_quan", "mai_chi")
    assert preset_voices(_Plain()) == ("a", "b")
    assert preset_voices(_Nested()) == ("minh_quan", "mai_chi")
    # Talker chạy qua subprocess không khai được — rỗng là sự thật, không phải lỗi.
    assert preset_voices(_Silent()) == ()


async def test_a_voice_the_engine_does_not_have_is_refused(lab):
    service, _ = lab
    with pytest.raises(VoicePlatformError) as excinfo:
        service.set_voice("giong-khong-co")
    # Lỗi phải nói ra engine có những giọng nào, không chỉ nói là sai.
    assert "mock-a" in str(excinfo.value)


async def test_setting_a_voice_does_not_reload_the_engine(lab):
    """Dựng lại engine có thể mất vài chục giây; đổi giọng thì không được phép."""
    service, platform = lab
    before = platform.models.tts
    out = service.set_voice("mock-b")
    assert platform.models.tts is before
    assert out["voice"] == "mock-b"
    assert service.config.models.tts.options["voice"] == "mock-b"


async def test_clearing_the_voice_removes_it_from_config(lab):
    service, platform = lab
    service.set_voice("mock-b")
    service.set_voice("")
    assert platform.voice is None
    assert "voice" not in service.config.models.tts.options


async def test_changing_the_voice_reaches_a_session_already_open(lab):
    """Chỉ đặt cho phiên MỚI thì người dùng đổi giọng rồi nghe tiếp vẫn giọng cũ."""
    service, platform = lab
    live = _Session()
    platform.sessions["s1"] = live
    service.set_voice("mock-b")
    assert live.voice == "mock-b"


async def test_the_test_uses_the_session_voice_when_none_is_given(lab):
    service, platform = lab
    service.set_voice("mock-b")
    out = await service.try_tts("Một câu.", None)
    assert out["voice"] == "mock-b"
    explicit = await service.try_tts("Một câu.", "mock-a")
    assert explicit["voice"] == "mock-a"


async def test_swapping_the_talker_drops_a_voice_the_new_one_lacks(lab):
    """Giữ lại thì mỗi lần tổng hợp là một cảnh báo rồi âm thầm đổi giọng."""
    service, platform = lab
    service.set_voice("mock-b")
    await service.swap("tts", "mock", {"sample_rate": 16000})
    assert platform.voice == "mock-b", "giọng còn hợp lệ thì phải giữ nguyên"

    # Giọng của talker cũ gần như không bao giờ tồn tại ở talker mới.
    platform.voice = "giong-cua-talker-cu"
    await service.swap("tts", "mock", {})
    assert platform.voice is None
    assert "voice" not in service.config.models.tts.options


def test_zerotts_is_a_registered_choice():
    from voiceplatform.core.config import EngineSpec
    from voiceplatform.models.registry import TTS_BACKENDS, build_tts

    assert "zerotts" in TTS_BACKENDS
    engine = build_tts(EngineSpec(backend="zerotts"), output_sample_rate=24000)
    assert engine.name == "zerotts"
    # Model sinh 48 kHz; nút vặn của nền tảng phải có nghĩa ở đây như mọi nơi.
    assert engine.source_sample_rate == 48000
    assert engine.output_sample_rate == 24000
    # ZeroTTS không khai token cảm xúc nào: cue phải bị bỏ trước khi tới nó.
    assert engine.capabilities.emotion_cues is False
