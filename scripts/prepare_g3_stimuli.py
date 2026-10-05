"""Cố định bộ input cho nghiệm thu G3 (chờ lượt, ngắt lời, nói tiếp).

Giọng "người dùng" là TTS của chính server (/try/tts) ở bảy giọng KHÁC giọng
trợ lý; tiếng động là tín hiệu tổng hợp có seed. Mọi thứ ghi một lần vào JSON
kèm hash, để lần đo trước và sau khi sửa code nghe đúng cùng một thứ.

Đây là proxy: TTS đọc đều, không ngập ngừng, không có phòng, không có micro.
Nó đo được cơ chế (luật giữ lượt, ngắt lời, nối câu) chứ không thay được
giọng người thật. Nhãn "phải là một lượt" là nhãn thiết kế của bộ câu, chưa
có người nghe xác nhận.

  PYTHONPATH=src python scripts/prepare_g3_stimuli.py \
      --base https://127.0.0.1:19101 --output docs/audits/2026-09-29/g3/stimuli.json
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import ssl
import urllib.request
import wave
from pathlib import Path

import numpy as np

RATE = 16000
ASSISTANT_VOICE = "kimoanh"
USER_VOICES = ["baotrang", "giahuy", "hamy", "huuduc", "maichi", "quangminh", "tiendat"]

# (phần 1, phần 2, từ khoá phải còn trong transcript cuối). Phần 1 là câu CHƯA
# trọn theo thiết kế: từ nối, câu mở, dãy số đang đọc, tiếng ngập ngừng.
HOLD = [
    ("Tôi muốn chuyển tiền cho", "mẹ tôi ở Đà Nẵng", ["chuyển", "đà nẵng"]),
    ("Cho tôi hỏi", "lãi suất tiết kiệm mười hai tháng là bao nhiêu", ["hỏi", "lãi suất"]),
    ("Số tài khoản của tôi là không chín một", "hai ba bốn năm sáu bảy", ["tài khoản", "bảy"]),
    ("Chuyển năm trăm nghìn đến", "số tài khoản không một hai ba bốn năm sáu", ["chuyển", "tài khoản"]),
    ("Tôi cần", "khoá thẻ tín dụng ngay bây giờ", ["cần", "thẻ"]),
    ("Nếu tôi gửi tiết kiệm", "thì bao lâu mới được rút", ["tiết kiệm", "rút"]),
    ("Ừm", "cho tôi hỏi giờ mở cửa chi nhánh", ["chi nhánh"]),
    ("Tôi định", "mở thêm một tài khoản nữa", ["định", "tài khoản"]),
    ("Mã khách hàng là bốn năm", "sáu bảy tám chín", ["bốn", "chín"]),
    ("Tôi muốn hỏi về", "phí chuyển tiền quốc tế", ["hỏi", "quốc tế"]),
    ("Làm ơn", "đọc lại số dư giúp tôi", ["làm ơn", "số dư"]),
    ("Chuyển khoản cho số điện thoại không chín tám", "bảy sáu năm bốn ba hai một", ["điện thoại", "một"]),
    ("Tôi muốn đặt lịch hẹn vào", "sáng thứ hai tuần sau", ["lịch", "thứ hai"]),
    ("Lãi suất hiện tại là bao nhiêu và", "có ưu đãi gì không", ["lãi suất", "ưu đãi"]),
    ("Cho mình", "xem lịch sử giao dịch tháng này", ["cho mình", "giao dịch"]),
    ("Mình cần hỏi", "về thẻ ghi nợ quốc tế", ["hỏi", "ghi nợ"]),
    ("Tài khoản nhận là của", "công ty An Phát", ["tài khoản", "an phát"]),
    ("Tôi muốn rút tiền nhưng", "không nhớ mật khẩu", ["rút", "mật khẩu"]),
    ("Bạn ơi", "hôm nay tỷ giá đô la là bao nhiêu", ["tỷ giá"]),
    ("Gửi tiền vào tài khoản của", "chị Lan ở chi nhánh Cầu Giấy", ["gửi", "cầu giấy"]),
]
HOLD_PAUSES_MS = [500, 600, 700, 800, 1000]

COMPLETE = [
    "Bây giờ là mấy giờ rồi?",
    "Hai cộng hai bằng mấy?",
    "Hôm nay là thứ mấy?",
    "Cảm ơn bạn nhiều nhé.",
    "Thủ đô của Pháp là gì?",
    "Tôi muốn khoá thẻ.",
    "Kiểm tra số dư giúp tôi.",
    "Chi nhánh gần nhất ở đâu?",
    "Phí chuyển tiền là bao nhiêu?",
    "Bạn tên là gì?",
    "Tôi quên mật khẩu rồi.",
    "Lãi suất tháng này có tăng không?",
    "Mở tài khoản cần giấy tờ gì?",
    "Cho tôi xem lịch sử giao dịch.",
    "Thời tiết Hà Nội hôm nay thế nào?",
    "Làm sao để đổi mã PIN?",
    "Ngân hàng làm việc đến mấy giờ?",
    "Chuyển cho mẹ tôi hai triệu.",
    "Số tài khoản là không chín một hai ba bốn năm sáu bảy tám.",
    "Được rồi, cảm ơn nhé.",
]

# Câu nói tiếp SAU khi lượt đã được chốt: phần 1 trông như đã trọn.
CONTINUE = [
    ("Bây giờ là mấy giờ", "ở Tokyo vậy", ["giờ", "tokyo"]),
    ("Cho tôi xem số dư", "tài khoản tiết kiệm", ["số dư", "tiết kiệm"]),
    ("Tôi muốn khoá thẻ", "thẻ tín dụng đuôi bốn năm", ["khoá", "tín dụng"]),
    ("Chuyển hai triệu", "cho mẹ tôi nhé", ["hai triệu", "mẹ"]),
    ("Thời tiết hôm nay", "ở Đà Lạt thế nào", ["thời tiết", "đà lạt"]),
    ("Lãi suất tiết kiệm", "kỳ hạn sáu tháng là bao nhiêu", ["lãi suất", "sáu tháng"]),
    ("Tôi muốn mở tài khoản", "cho con gái tôi", ["tài khoản", "con gái"]),
    ("Hôm nay là ngày mấy", "theo âm lịch", ["ngày", "âm lịch"]),
    ("Đọc lại giúp tôi", "số điện thoại vừa rồi", ["đọc lại", "điện thoại"]),
    ("Phí rút tiền", "ở cây ATM khác ngân hàng", ["phí", "atm"]),
]
CONTINUE_PAUSES_MS = [650, 850, 1100, 1400, 1800]

BACKCHANNEL = ["Ừ.", "Ừm.", "Vâng.", "Dạ.", "Ờ.", "Vâng ạ.", "Dạ vâng.", "Đúng rồi.", "Ok.", "Ừ ừ."]

INTERRUPT = [
    ("Thôi, dừng lại.", ["dừng"]),
    ("Khoan đã, bây giờ là mấy giờ?", ["giờ"]),
    ("Không phải, tôi hỏi lãi suất cơ.", ["lãi suất"]),
    ("Chờ chút, cho tôi hỏi cái khác.", ["hỏi"]),
    ("Thôi được rồi, cảm ơn bạn.", ["cảm ơn"]),
    ("Dừng lại đi.", ["dừng"]),
    ("Ý tôi là chuyển cho mẹ tôi.", ["mẹ"]),
    ("Không, tôi muốn khoá thẻ.", ["khoá", "thẻ"]),
    ("Nói ngắn thôi.", ["ngắn"]),
    ("Đợi đã, hai cộng ba bằng mấy?", ["hai", "ba"]),
]

LONG_QUESTION = "Bạn kể cho tôi nghe một câu chuyện cổ tích ngắn về con cáo và con quạ đi."

# ---------------------------------------------------------------- confirmation set
# Viết và khoá TRƯỚC khi chỉnh luật turn detection theo kết quả bộ dev: câu
# khác, người nói khác thứ tự, seed khác. Kết quả nghiệm thu G3 lấy từ bộ này;
# bộ dev ở trên chỉ để phát triển.
CONFIRM = {
    "hold": [
        ("Tôi muốn gửi tiết kiệm", "kỳ hạn mười hai tháng", ["gửi", "kỳ hạn"]),
        ("Cho em hỏi", "phí thường niên của thẻ này", ["hỏi", "thường niên"]),
        ("Số điện thoại của tôi là không chín tám", "bảy sáu năm bốn ba hai", ["điện thoại", "hai"]),
        ("Khi tôi chuyển tiền ra nước ngoài", "thì mất bao lâu", ["nước ngoài", "bao lâu"]),
        ("Mã giao dịch là ba bảy", "hai tám một năm", ["mã", "năm"]),
        ("Anh ơi", "tài khoản của tôi bị khoá rồi", ["khoá"]),
        ("Tôi cần mở", "một thẻ tín dụng mới", ["mở", "tín dụng"]),
        ("Chuyển giúp tôi hai triệu sang", "tài khoản của vợ tôi", ["chuyển", "vợ"]),
        ("Ờ", "cho tôi xem sao kê tháng trước", ["sao kê"]),
        ("Nếu tôi rút trước hạn", "thì có mất lãi không", ["rút", "lãi"]),
        ("Làm sao để", "đổi mật khẩu đăng nhập", ["làm sao", "mật khẩu"]),
        ("Tài khoản của tôi có", "bao nhiêu tiền vậy", ["tài khoản", "bao nhiêu"]),
        ("Tôi đang ở", "chi nhánh Hoàn Kiếm", ["đang ở", "hoàn kiếm"]),
        ("Vì tôi làm mất thẻ", "nên muốn khoá ngay", ["mất thẻ", "khoá"]),
        ("Mình muốn biết", "hạn mức chuyển khoản mỗi ngày", ["biết", "hạn mức"]),
        ("Người nhận là", "anh Tuấn ở Hải Phòng", ["người nhận", "hải phòng"]),
        ("Tôi định vay", "mua nhà trả góp hai mươi năm", ["vay", "trả góp"]),
        ("Làm ơn kiểm tra giúp tôi", "giao dịch hôm qua", ["kiểm tra", "hôm qua"]),
        ("Tôi hỏi", "về gói bảo hiểm sức khoẻ", ["hỏi", "bảo hiểm"]),
        ("Số thẻ của tôi bắt đầu bằng bốn", "năm sáu bảy", ["số thẻ", "bảy"]),
    ],
    "complete": [
        "Tỷ giá đô la hôm nay là bao nhiêu?",
        "Tôi muốn rút tiền mặt.",
        "Hạn mức thẻ của tôi là bao nhiêu?",
        "Ngân hàng có mở cửa chủ nhật không?",
        "Mật khẩu của tôi bị khoá rồi.",
        "Bạn giúp tôi đặt lịch hẹn nhé.",
        "Hôm nay là ngày bao nhiêu?",
        "Phí chuyển khoản liên ngân hàng là bao nhiêu?",
        "Cảm ơn, tôi hiểu rồi.",
        "Tôi cần tư vấn về khoản vay.",
        "Lãi suất vay mua nhà là bao nhiêu?",
        "Cho tôi số tổng đài.",
        "Tài khoản của tôi còn bao nhiêu tiền?",
        "Tôi muốn huỷ thẻ tín dụng.",
        "Mấy giờ thì ngân hàng đóng cửa?",
        "Làm thế nào để mở tài khoản online?",
        "Thẻ của tôi bị nuốt ở cây ATM.",
        "Giao dịch vừa rồi có thành công không?",
        "Tôi không nhận được mã OTP.",
        "Được rồi, tạm biệt nhé.",
    ],
    "continue": [
        ("Chuyển năm trăm nghìn", "cho anh Minh", ["năm trăm", "minh"]),
        ("Tôi muốn hỏi giờ làm việc", "của chi nhánh Đống Đa", ["giờ làm việc", "đống đa"]),
        ("Cho tôi xem lịch sử giao dịch", "tuần trước", ["lịch sử", "tuần trước"]),
        ("Khoá thẻ ghi nợ", "đuôi một hai ba bốn", ["khoá", "đuôi"]),
        ("Lãi suất gửi tiết kiệm", "ba tháng là bao nhiêu", ["lãi suất", "ba tháng"]),
        ("Tôi muốn đổi số điện thoại", "đăng ký dịch vụ", ["số điện thoại", "đăng ký"]),
        ("Phí rút tiền mặt", "bằng thẻ tín dụng", ["phí", "tín dụng"]),
        ("Hôm nay tỷ giá euro", "là bao nhiêu", ["tỷ giá", "bao nhiêu"]),
        ("Tôi cần vay tiền", "để mua xe", ["vay", "mua xe"]),
        ("Mở tài khoản cho con", "dưới mười tám tuổi", ["tài khoản", "mười tám"]),
    ],
    "interrupt": [
        ("Khoan, tôi muốn hỏi chuyện khác.", ["hỏi"]),
        ("Dừng lại, không phải cái đó.", ["dừng"]),
        ("Thôi, cho tôi biết số dư.", ["số dư"]),
        ("Chờ đã, ba nhân bốn bằng mấy?", ["ba", "bốn"]),
        ("Không cần nữa, cảm ơn.", ["cảm ơn"]),
        ("Ý tôi là thẻ tín dụng.", ["tín dụng"]),
        ("Nói chậm lại một chút.", ["chậm"]),
        ("Khoan đã, bây giờ mấy giờ rồi?", ["giờ"]),
        ("Thôi được, dừng ở đây.", ["dừng"]),
        ("Không, tôi hỏi về khoản vay.", ["vay"]),
    ],
}


def _ssl() -> ssl.SSLContext:
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    return ctx


def synth(base: str, text: str, voice: str) -> np.ndarray:
    body = json.dumps({"text": text, "voice": voice}).encode()
    request = urllib.request.Request(base + "/try/tts", data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, context=_ssl(), timeout=120) as response:
        reply = json.loads(response.read())
    if "wav_base64" not in reply:
        raise RuntimeError(f"/try/tts failed for {text!r}: {reply}")
    with wave.open(io.BytesIO(base64.b64decode(reply["wav_base64"]))) as handle:
        rate = handle.getframerate()
        pcm = np.frombuffer(handle.readframes(handle.getnframes()), dtype="<i2").astype(np.float32) / 32768
    if rate != RATE:
        # Tuyến tính là đủ cho mục đích này và giống hệt đường vào của server.
        n = int(round(pcm.size * RATE / rate))
        pcm = np.interp(np.linspace(0, pcm.size - 1, n), np.arange(pcm.size), pcm).astype(np.float32)
    return trim(pcm)


def trim(pcm: np.ndarray, floor: float = 0.004, keep_ms: int = 40) -> np.ndarray:
    """Cắt lặng hai đầu, giữ 40 ms: quãng nghỉ phải là quãng nghỉ ta CHÈN vào."""
    frame = RATE // 100
    energy = np.array([np.sqrt(np.mean(pcm[i:i + frame] ** 2)) for i in range(0, pcm.size, frame)] or [0])
    voiced = np.nonzero(energy >= floor)[0]
    if voiced.size == 0:
        return pcm
    keep = keep_ms * RATE // 1000
    start = max(0, voiced[0] * frame - keep)
    end = min(pcm.size, (voiced[-1] + 1) * frame + keep)
    return pcm[start:end]


def silence(ms: int) -> np.ndarray:
    return np.zeros(int(RATE * ms / 1000), np.float32)


# ------------------------------------------------------------------ noise

def _envelope(n: int, attack: float, decay: float) -> np.ndarray:
    t = np.arange(n) / RATE
    return (1 - np.exp(-t / max(attack, 1e-4))) * np.exp(-t / max(decay, 1e-4))


def _band(noise: np.ndarray, lo: float, hi: float) -> np.ndarray:
    spectrum = np.fft.rfft(noise)
    freqs = np.fft.rfftfreq(noise.size, 1 / RATE)
    spectrum[(freqs < lo) | (freqs > hi)] = 0
    return np.fft.irfft(spectrum, noise.size).astype(np.float32)


def noise_clip(kind: str, rng: np.random.Generator) -> np.ndarray:
    def burst(ms, lo, hi, attack, decay, peak):
        n = int(RATE * ms / 1000)
        x = _band(rng.normal(0, 1, n), lo, hi) * _envelope(n, attack, decay)
        return (peak * x / (np.abs(x).max() + 1e-9)).astype(np.float32)

    if kind == "cough":
        return burst(rng.integers(180, 320), 200, 3500, 0.01, 0.09, rng.uniform(0.25, 0.5))
    if kind == "cough_double":
        a = burst(rng.integers(160, 240), 200, 3500, 0.01, 0.07, rng.uniform(0.25, 0.45))
        b = burst(rng.integers(160, 240), 200, 3500, 0.01, 0.07, rng.uniform(0.2, 0.4))
        return np.concatenate([a, silence(int(rng.integers(90, 180))), b])
    if kind == "throat":
        return burst(rng.integers(350, 550), 100, 1200, 0.05, 0.25, rng.uniform(0.15, 0.3))
    if kind == "knock":
        parts = []
        for _ in range(int(rng.integers(2, 4))):
            parts += [burst(40, 60, 900, 0.001, 0.012, rng.uniform(0.4, 0.7)), silence(int(rng.integers(120, 220)))]
        return np.concatenate(parts)
    if kind == "clap":
        return burst(60, 400, 6000, 0.0005, 0.01, rng.uniform(0.4, 0.7))
    if kind == "keyboard":
        parts = []
        for _ in range(int(rng.integers(6, 11))):
            parts += [burst(15, 1500, 7000, 0.0005, 0.003, rng.uniform(0.15, 0.35)), silence(int(rng.integers(60, 160)))]
        return np.concatenate(parts)
    if kind == "rustle":
        n = int(RATE * rng.integers(600, 1000) / 1000)
        x = _band(rng.normal(0, 1, n), 1500, 7500) * np.hanning(n)
        return (rng.uniform(0.05, 0.12) * x / (np.abs(x).max() + 1e-9)).astype(np.float32)
    if kind == "breath":
        n = int(RATE * rng.integers(300, 500) / 1000)
        x = _band(rng.normal(0, 1, n), 300, 2500) * np.hanning(n)
        return (rng.uniform(0.04, 0.1) * x / (np.abs(x).max() + 1e-9)).astype(np.float32)
    raise ValueError(kind)


NOISE_KINDS = ["cough", "cough_double", "throat", "knock", "clap", "keyboard", "rustle", "breath"]


def encode(pcm: np.ndarray) -> str:
    return base64.b64encode((np.clip(pcm, -1, 1) * 32767).astype("<i2").tobytes()).decode()


_CHAT_ABBREVIATIONS = {
    "mk", "t", "ck", "gd", "đt", "dt", "tks", "thanks", "tkiem", "ko", "k", "kh", "dc", "đc", "j", "e",
    "a", "vs", "nx", "ntn", "bn", "sdt", "stk", "tk", "nh", "list", "file", "ok", "oke", "lệnhs",
}


def real_sentences(root: str, rng: np.random.Generator, n: int) -> list[str]:
    """Banking sentences NOT written for this test: sampled from the intent data."""
    import re
    pool = set()
    for path in sorted(Path(root).rglob("*.json")):
        if path.name == "intents.json" or "fallback" in path.parts:
            continue
        try:
            rows = json.loads(path.read_text())
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        for row in rows if isinstance(rows, list) else []:
            text = (row.get("text") or "").strip() if isinstance(row, dict) else ""
            words = text.split()
            # Diacritics present, no digits/abbreviations the talker may read
            # oddly, a length where a mid-sentence pause is plausible.
            lowered = [w.strip(",.?").lower() for w in words]
            marked = sum(any(ord(c) > 127 for c in w) for w in lowered)
            if (6 <= len(words) <= 14 and re.fullmatch(r"[^\W\d_][\w\s,.?]*", text)
                    and not re.search(r"\d|[A-Z]{2,}", text)
                    # Typed chat is not speech: no "mk", "ck", "tks", no text
                    # typed without diacritics.
                    and not set(lowered) & _CHAT_ABBREVIATIONS and marked >= 0.5 * len(words)):
                pool.add(text)
    pool = sorted(pool)
    picks = rng.choice(len(pool), size=n, replace=False)
    return [pool[i] for i in picks]


def real_set(args) -> None:
    base = args.base.rstrip("/")
    rng = np.random.default_rng(args.seed)
    texts = real_sentences(args.source, rng, 40)
    cases: dict[str, list[dict]] = {"hold": [], "complete": []}
    cache: dict[tuple[str, str], np.ndarray] = {}

    def say(text: str, voice: str) -> np.ndarray:
        if (text, voice) not in cache:
            cache[(text, voice)] = synth(base, text, voice)
        return cache[(text, voice)]

    for i, text in enumerate(texts[:20]):
        words = text.rstrip(" .?,").split()
        # The split is drawn, not chosen: 2 words at least on either side.
        k = int(rng.integers(2, len(words) - 1))
        part1, part2 = " ".join(words[:k]), " ".join(words[k:])
        for j, pause in enumerate(HOLD_PAUSES_MS):
            voice = USER_VOICES[(3 * i + j) % len(USER_VOICES)]
            a, b = say(part1, voice), say(part2, voice)
            cases["hold"].append({
                "id": f"real-hold-{i:02d}-{pause}", "voice": voice, "text": f"{part1} … {part2}",
                "parts": [part1, part2], "pause_ms": pause, "keys": [words[0], words[-1]],
                "pcm": encode(np.concatenate([a, silence(pause), b])),
                "pause_at_ms": round(1000 * a.size / RATE, 1), "expect": "one_turn",
                "source_sentence": text,
            })
    for i, text in enumerate(texts[20:]):
        for j in range(5):
            voice = USER_VOICES[(2 * i + j) % len(USER_VOICES)]
            cases["complete"].append({
                "id": f"real-complete-{i:02d}-{voice}", "voice": voice, "text": text,
                "pcm": encode(say(text, voice)), "expect": "respond",
            })
    payload = {
        "schema": 1, "set": "real", "sample_rate": RATE, "seed": args.seed, "source": args.source,
        "assistant_voice": ASSISTANT_VOICE, "user_voices": USER_VOICES,
        "basis": ("sentences sampled (seeded) from an intent dataset, split at a drawn word; "
                  "server TTS voices; a split may leave a prefix a human would also call complete"),
        "long_question": {"text": LONG_QUESTION, "pcm_by_voice": {v: encode(say(LONG_QUESTION, v)) for v in USER_VOICES[:3]}},
        "cases": cases,
    }
    blob = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode()
    payload["sha256"] = hashlib.sha256(blob).hexdigest()
    out = Path(args.output)
    out.write_text(json.dumps(payload, ensure_ascii=False))
    print(json.dumps({k: len(v) for k, v in cases.items()}), payload["sha256"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base", default="https://127.0.0.1:19101")
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=20260929)
    parser.add_argument("--set", choices=("dev", "confirm", "real"), default="dev")
    parser.add_argument("--source", default="/home/ai01/AIHoang/intent_data_v5.4.0",
                        help="--set real: intent dataset whose sentences are sampled")
    args = parser.parse_args()
    if args.set == "real":
        return real_set(args)
    global HOLD, COMPLETE, CONTINUE, INTERRUPT, USER_VOICES
    if args.set == "confirm":
        HOLD, COMPLETE, CONTINUE, INTERRUPT = (CONFIRM["hold"], CONFIRM["complete"],
                                               CONFIRM["continue"], CONFIRM["interrupt"])
        USER_VOICES = list(reversed(USER_VOICES))   # other speaker order
    base = args.base.rstrip("/")
    rng = np.random.default_rng(args.seed)
    cache: dict[tuple[str, str], np.ndarray] = {}

    def say(text: str, voice: str) -> np.ndarray:
        if (text, voice) not in cache:
            cache[(text, voice)] = synth(base, text, voice)
        return cache[(text, voice)]

    cases: dict[str, list[dict]] = {k: [] for k in ("hold", "complete", "continue", "backchannel", "noise", "interrupt")}
    for i, (part1, part2, keys) in enumerate(HOLD):
        for j, pause in enumerate(HOLD_PAUSES_MS):
            voice = USER_VOICES[(i + j) % len(USER_VOICES)]
            a, b = say(part1, voice), say(part2, voice)
            cases["hold"].append({
                "id": f"hold-{i:02d}-{pause}", "voice": voice, "text": f"{part1} … {part2}",
                "parts": [part1, part2], "pause_ms": pause, "keys": keys,
                "pcm": encode(np.concatenate([a, silence(pause), b])),
                "pause_at_ms": round(1000 * a.size / RATE, 1),
                "expect": "one_turn",
            })
    for i, text in enumerate(COMPLETE):
        for j in range(5):
            voice = USER_VOICES[(i * 5 + j) % len(USER_VOICES)]
            cases["complete"].append({
                "id": f"complete-{i:02d}-{voice}", "voice": voice, "text": text,
                "pcm": encode(say(text, voice)), "expect": "respond",
            })
    for i, (part1, part2, keys) in enumerate(CONTINUE):
        for j, pause in enumerate(CONTINUE_PAUSES_MS):
            voice = USER_VOICES[(i + 2 * j) % len(USER_VOICES)]
            a, b = say(part1, voice), say(part2, voice)
            cases["continue"].append({
                "id": f"continue-{i:02d}-{pause}", "voice": voice, "text": f"{part1} … {part2}",
                "parts": [part1, part2], "pause_ms": pause, "keys": keys,
                "pcm": encode(np.concatenate([a, silence(pause), b])),
                "pause_at_ms": round(1000 * a.size / RATE, 1),
                "expect": "both_halves",
            })
    for n in range(100):
        text = BACKCHANNEL[n % len(BACKCHANNEL)]
        voice = USER_VOICES[(n // len(BACKCHANNEL) + n) % len(USER_VOICES)]
        cases["backchannel"].append({
            "id": f"backchannel-{n:03d}", "voice": voice, "text": text,
            "pcm": encode(say(text, voice)), "expect": "resume",
            "offset_s": round(float(rng.uniform(0.8, 3.0)), 3),
        })
    for n in range(100):
        kind = NOISE_KINDS[n % len(NOISE_KINDS)]
        cases["noise"].append({
            "id": f"noise-{n:03d}-{kind}", "kind": kind, "text": f"<{kind}>",
            "pcm": encode(noise_clip(kind, rng)), "expect": "resume",
            "offset_s": round(float(rng.uniform(0.8, 3.0)), 3),
        })
    for n in range(100):
        text, keys = INTERRUPT[n % len(INTERRUPT)]
        voice = USER_VOICES[(n // len(INTERRUPT) + n) % len(USER_VOICES)]
        # Một phần ba chen ngay lúc trợ lý vừa cất tiếng: đó là chỗ guard
        # window có thể nuốt mất lời chen thật.
        onset = n % 3 == 0
        cases["interrupt"].append({
            "id": f"interrupt-{n:03d}{'-onset' if onset else ''}", "voice": voice, "text": text,
            "keys": keys, "pcm": encode(say(text, voice)), "expect": "stop_and_answer",
            "offset_s": round(float(rng.uniform(0.0, 0.25) if onset else rng.uniform(0.8, 3.0)), 3),
            "onset_window": onset,
        })

    long_question = {voice: encode(say(LONG_QUESTION, voice)) for voice in USER_VOICES[:3]}
    payload = {
        "schema": 1, "set": args.set, "sample_rate": RATE, "seed": args.seed,
        "assistant_voice": ASSISTANT_VOICE, "user_voices": USER_VOICES,
        "basis": "server TTS (ZeroTTS) in non-assistant voices + seeded synthetic noise; not human speech",
        "long_question": {"text": LONG_QUESTION, "pcm_by_voice": long_question},
        "cases": cases,
    }
    blob = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode()
    payload["sha256"] = hashlib.sha256(blob).hexdigest()
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False))
    print(json.dumps({k: len(v) for k, v in cases.items()}), payload["sha256"])


if __name__ == "__main__":
    main()
