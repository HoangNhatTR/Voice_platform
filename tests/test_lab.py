"""Bàn thử model: các con số phải là của engine đang chạy, không phải bản sao."""

from __future__ import annotations

import numpy as np
import pytest

from voiceplatform.app.lab import Busy, LabService, read_wav, wav_bytes, word_errors
from voiceplatform.core.config import Config
from voiceplatform.core.errors import VoicePlatformError
from voiceplatform.models.registry import ModelPlane


class _Platform:
    """Đúng ba thứ LabService cần, không hơn."""

    def __init__(self, config: Config) -> None:
        self.models = ModelPlane(
            config.models, output_sample_rate=config.audio.output_sample_rate
        )
        self.sessions: dict[str, object] = {}


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
    platform.sessions["s1"] = object()
    out = await service.try_tts("Một câu.", None)
    assert out["contended"] is True
    assert out["live_sessions"] == 1


# ------------------------------------------------------------------ đổi engine

async def test_swapping_is_refused_while_a_session_is_live(lab):
    service, platform = lab
    platform.sessions["s1"] = object()
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
