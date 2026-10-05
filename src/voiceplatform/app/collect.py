"""Trang /collect: đồng nghiệp tự thu clip lượt lời có nhãn cho G3, qua LAN.

G3 cần ≥500 lượt người thật từ ≥10 người nói, và trước trang này là 0. Trang
cho mỗi người một kế hoạch ~50 câu (app/collect_bank.py), thu bằng ĐÚNG đường
micro của sản phẩm (web/capture.js: cùng ràng buộc getUserMedia, cùng cách hạ
mẫu về 16 kHz int16), cho người nói nghe lại và tự xác nhận rồi mới lưu.

Quyền truy cập:

- `GET /collect`, `/collect/info`, `/collect/prompts`, `POST /collect/consent`,
  `POST /collect/clip`, `DELETE /collect/...`: mở cho LAN — người nói dùng máy
  của chính họ. Mọi route ghi/xoá từ chối request trình duyệt mang `Origin`
  của trang khác (`access.foreign_origin`), và khi `collect.access_code` có
  giá trị thì phải mang đúng mã trong header `X-Collect-Code` (header riêng
  cũng buộc trình duyệt hỏi preflight).
- `GET /collect/progress`: chỉ từ chính máy chủ (loopback), BẤT KỂ
  `server.private_introspection` — nó liệt kê mọi người nói.

Không có gì ở đây tin chữ client gửi: server dựng lại câu từ (speaker_id,
prompt_id), tự đo lại quãng dừng trên WAV đã nhận, và từ chối khi hai bên lệch
quá `PAUSE_TOLERANCE_MS`. Clip lưu 0600 dưới `collect.dir/<người>/<phiên>/`,
mỗi thay đổi là một dòng JSON nối vào `manifest.jsonl` (có khoá), không sửa
dòng cũ: rút clip là thêm một dòng `withdraw` và xoá file WAV.
"""

from __future__ import annotations

import asyncio
import fcntl
import hashlib
import hmac
import io
import json
import os
import re
import secrets
import threading
import unicodedata
import wave
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse

from ..core.config import Config
from ..observability.logging import get_logger
from .access import foreign_origin, is_loopback
from .collect_bank import ACOUSTIC_FAMILIES, FAMILIES, PAUSE_FAMILIES, Bank, load_bank

log = get_logger("collect")

SAMPLE_RATE = 16000
MIN_CLIP_S = 0.3
# G3 cần ~500 lượt; 20.000 clip ≤ 1 MB mỗi cái là trần ổ đĩa, không phải mục tiêu.
MAX_LIVE_CLIPS = 20000
# Phát hiện quãng dừng — giữ ĐỒNG BỘ với web/pause.js (có test so hai bản).
FRAME_MS = 10
MIN_PAUSE_MS = 300
MIN_VOICED_MS = 50          # cụm "có tiếng" ngắn hơn là tiếng click, bỏ
SILENT_PEAK_DB = -55.0      # frame to nhất dưới mức này: không thu được gì
PAUSE_TOLERANCE_MS = 250
# Mục tiêu G3 (docs/audits/2026-09-30/g3/HUMAN_DATA.md).
TARGETS = (
    ("clips", "Tổng số clip", 500),
    ("speakers", "Số người nói có clip", 10),
    ("pause", "Có quãng dừng (hold + continue)", 100),
    ("backchannel_noise", "Lời đệm + tiếng động", 100),
    ("interrupt", "Ngắt lời thật", 100),
)
CONDITIONS: dict[str, dict[str, str]] = {
    "device": {"laptop_mic": "Micro laptop", "headset_mic": "Tai nghe có mic",
               "phone": "Điện thoại", "external_mic": "Micro rời"},
    "playback": {"speaker": "Loa ngoài", "headphones": "Tai nghe"},
    "room": {"quiet": "Yên tĩnh", "noisy": "Có tiếng ồn"},
    "accent": {"": "Không khai", "north": "Bắc", "central": "Trung", "south": "Nam"},
}
WAV_TYPES = {"audio/wav", "audio/x-wav", "audio/wave", "audio/vnd.wave"}
_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{2,63}")
_CLIP_ID = re.compile(r"c-[0-9a-f]{16}")
_CAPTURE_KEYS = ("sampleRate", "contextRate", "echoCancellation", "noiseSuppression",
                 "autoGainControl", "channelCount", "latency")
_REPO_DOCS = Path(__file__).resolve().parents[3] / "docs"


class ClipError(Exception):
    def __init__(self, status: int, detail: str, code: str = "invalid") -> None:
        super().__init__(detail)
        self.status = status
        self.detail = detail
        self.code = code

    def response(self) -> JSONResponse:
        return JSONResponse({"detail": self.detail, "code": self.code}, status_code=self.status)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def valid_id(value: str | None) -> bool:
    return bool(value) and _ID.fullmatch(value) is not None


# ------------------------------------------------------------------ âm thanh

def read_wav_pcm16(payload: bytes) -> np.ndarray:
    """WAV mono 16-bit 16 kHz → int16. Mọi dạng khác bị từ chối, không chuyển."""
    if payload[:4] != b"RIFF" or payload[8:12] != b"WAVE":
        raise ClipError(400, "không phải file WAV", "format")
    try:
        with wave.open(io.BytesIO(payload), "rb") as handle:
            shape = (handle.getnchannels(), handle.getsampwidth(), handle.getframerate())
            compression = handle.getcomptype()
            frames = handle.getnframes()
            raw = handle.readframes(frames)
    except (wave.Error, EOFError, ValueError) as exc:
        raise ClipError(400, f"WAV hỏng hoặc không phải PCM: {exc}", "format") from exc
    if shape != (1, 2, SAMPLE_RATE) or compression != "NONE":
        raise ClipError(400, "cần WAV mono, PCM 16-bit, 16 kHz", "format")
    if len(raw) != frames * 2:
        raise ClipError(400, "WAV bị cắt cụt", "format")
    return np.frombuffer(raw, dtype="<i2")


def wav_from_pcm16(pcm: np.ndarray) -> bytes:
    """WAV chuẩn từ int16: bỏ mọi chunk phụ (LIST, metadata) client gửi kèm."""
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(SAMPLE_RATE)
        handle.writeframes(pcm.astype("<i2").tobytes())
    return buffer.getvalue()


def analyze_pcm(pcm: np.ndarray, rate: int = SAMPLE_RATE) -> dict[str, Any]:
    """Quãng im dài nhất NẰM GIỮA hai vùng có tiếng, theo năng lượng.

    Frame 10 ms; ngưỡng thích nghi max(nền + 10 dB, đỉnh − 35 dB) với nền là
    phân vị 10% và đỉnh là phân vị 99% của dB từng frame, để micro laptop có
    AGC và điện thoại trong phòng ồn cùng dùng được. Cụm có tiếng ngắn hơn
    50 ms (tiếng click chuột lúc bấm ghi, chạm môi) bị bỏ trước khi tìm khoảng
    trống. Bản JS ở web/pause.js phải cho đúng cùng con số.
    """
    hop = rate * FRAME_MS // 1000
    n = int(pcm.size) // hop
    out: dict[str, Any] = {
        "frame_ms": FRAME_MS, "duration_ms": round(1000.0 * pcm.size / rate, 1), "sound": False,
        "floor_db": None, "peak_db": None, "threshold_db": None, "voiced_ms": 0,
        "speech_start_ms": None, "speech_end_ms": None,
        "longest_gap_ms": 0, "pause_at_ms": None, "pause_ms": None,
    }
    if n == 0:
        return out
    frames = pcm[: n * hop].astype(np.float64).reshape(n, hop) / 32768.0
    rms = np.sqrt((frames * frames).sum(axis=1) / hop)
    db = 20.0 * np.log10(np.maximum(rms, 1e-5))
    ordered = np.sort(db)
    floor = float(ordered[int(0.10 * (n - 1))])
    peak = float(ordered[int(0.99 * (n - 1))])
    threshold = max(floor + 10.0, peak - 35.0)
    raw = db >= threshold
    out.update(floor_db=round(floor, 1), peak_db=round(peak, 1), threshold_db=round(threshold, 1))
    if float(ordered[-1]) < SILENT_PEAK_DB or not raw.any():
        return out
    out["sound"] = True
    voiced = raw.copy()
    min_run = MIN_VOICED_MS // FRAME_MS
    i = 0
    while i < n:
        if voiced[i]:
            j = i
            while j < n and voiced[j]:
                j += 1
            if j - i < min_run:
                voiced[i:j] = False
            i = j
        else:
            i += 1
    idx = np.flatnonzero(voiced)
    if idx.size == 0:
        # Chỉ có tiếng rất ngắn (gõ bàn): vẫn là "có tiếng", nhưng không có
        # vùng nói nào để tìm quãng dừng giữa.
        loud = np.flatnonzero(raw)
        out.update(speech_start_ms=int(loud[0]) * FRAME_MS, speech_end_ms=(int(loud[-1]) + 1) * FRAME_MS)
        return out
    first, last = int(idx[0]), int(idx[-1])
    best_at, best_len = -1, 0
    i = first
    while i <= last:
        if not voiced[i]:
            j = i
            while j <= last and not voiced[j]:
                j += 1
            if j - i > best_len:
                best_at, best_len = i, j - i
            i = j
        else:
            i += 1
    out.update(voiced_ms=int(idx.size) * FRAME_MS, speech_start_ms=first * FRAME_MS,
               speech_end_ms=(last + 1) * FRAME_MS, longest_gap_ms=best_len * FRAME_MS)
    if best_len * FRAME_MS >= MIN_PAUSE_MS:
        out.update(pause_at_ms=best_at * FRAME_MS, pause_ms=best_len * FRAME_MS)
    return out


# ------------------------------------------------------------------ manifest

def _private_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    if (path.stat().st_mode & 0o777) != 0o700:
        os.chmod(path, 0o700)


def _write_private(path: Path, data: bytes) -> None:
    """Như trace: tạo 0600 ngay từ đầu, không bao giờ đè file có sẵn."""
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    try:
        os.fchmod(descriptor, 0o600)
        view = memoryview(data)
        while view:
            view = view[os.write(descriptor, view):]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def read_manifest(path: Path) -> list[dict[str, Any]]:
    """Mọi bản ghi của manifest, đọc dưới khoá chia sẻ. Dòng hỏng bị bỏ."""
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        fcntl.flock(handle, fcntl.LOCK_SH)
        try:
            lines = handle.read().splitlines()
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)
    records = []
    for number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except ValueError:
            log.warning("manifest %s: dòng %d hỏng, bỏ qua", path, number)
            continue
        if isinstance(record, dict) and record.get("type"):
            records.append(record)
    return records


class ManifestState:
    """Trạng thái gập lại từ chuỗi bản ghi; dùng chung cho server và script xuất."""

    def __init__(self) -> None:
        self.speakers: dict[str, dict[str, Any]] = {}     # đồng ý còn hiệu lực
        self.clips: dict[str, dict[str, Any]] = {}        # clip còn sống
        self.withdrawn: dict[str, dict[str, Any]] = {}
        self.sessions: dict[str, dict[str, Any]] = {}     # phiên → người nói + điều kiện

    def apply(self, record: dict[str, Any]) -> None:
        kind = record["type"]
        if kind == "consent":
            self.speakers[record["speaker_id"]] = record
        elif kind == "consent_withdrawn":
            speaker = record["speaker_id"]
            self.speakers.pop(speaker, None)
            for clip_id in [cid for cid, clip in self.clips.items() if clip["speaker_id"] == speaker]:
                self.withdrawn[clip_id] = self.clips.pop(clip_id)
        elif kind == "clip":
            self.clips[record["id"]] = record
            self.sessions.setdefault(record["session_id"], {
                "speaker_id": record["speaker_id"], "conditions": record["conditions"]})
        elif kind == "withdraw":
            clip = self.clips.pop(record["id"], None)
            if clip is not None:
                self.withdrawn[record["id"]] = clip

    @classmethod
    def load(cls, path: Path) -> "ManifestState":
        state = cls()
        for record in read_manifest(path):
            try:
                state.apply(record)
            except (KeyError, TypeError):
                log.warning("manifest %s: bản ghi thiếu trường, bỏ qua: %.120s", path, record)
        return state


class CollectStore:
    def __init__(self, root: Path, bank: Bank, max_clip_s: float) -> None:
        resolved = root.resolve()
        if resolved == _REPO_DOCS or _REPO_DOCS in resolved.parents:
            raise ValueError(f"collect.dir {root} nằm trong docs/: không ghi giọng người thật vào docs")
        self.root = root
        self.bank = bank
        self.max_clip_s = max_clip_s
        self.manifest = root / "manifest.jsonl"
        self._lock = threading.Lock()
        _private_dir(root)
        self.state = ManifestState.load(self.manifest)

    # --- ghi ------------------------------------------------------------
    def _append(self, record: dict[str, Any]) -> None:
        line = (json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n").encode()
        descriptor = os.open(self.manifest, os.O_WRONLY | os.O_APPEND | os.O_CREAT | os.O_NOFOLLOW, 0o600)
        try:
            os.fchmod(descriptor, 0o600)
            fcntl.flock(descriptor, fcntl.LOCK_EX)
            try:
                os.write(descriptor, line)
                os.fsync(descriptor)
            finally:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)
        self.state.apply(record)

    def _clip_path(self, speaker: str, session: str, clip_id: str) -> Path:
        path = self.root / speaker / session / f"{clip_id}.wav"
        # Regex đã chặn "/" và "..", đây chỉ là chốt thứ hai.
        if self.root.resolve() not in path.resolve().parents:
            raise ClipError(400, "đường dẫn không hợp lệ", "path")
        return path

    # --- đồng ý ---------------------------------------------------------
    def consent(self, speaker_id: str, pseudonym: str, user_agent: str) -> dict[str, Any]:
        record = {"type": "consent", "speaker_id": speaker_id, "pseudonym": pseudonym,
                  "consent_version": self.bank.consent["version"],
                  "consent_sha256": self.bank.consent_sha256, "at": _now(), "user_agent": user_agent}
        with self._lock:
            self._append(record)
        return record

    def has_consent(self, speaker_id: str) -> bool:
        record = self.state.speakers.get(speaker_id)
        return record is not None and record.get("consent_version") == self.bank.consent["version"]

    def withdraw_consent(self, speaker_id: str) -> int:
        with self._lock:
            clips = [c for c in self.state.clips.values() if c["speaker_id"] == speaker_id]
            if speaker_id not in self.state.speakers and not clips:
                return -1
            for clip in clips:
                self._unlink(clip)
            self._append({"type": "consent_withdrawn", "speaker_id": speaker_id, "at": _now(),
                          "clips": sorted(c["id"] for c in clips)})
        return len(clips)

    # --- clip -----------------------------------------------------------
    def add_clip(self, meta: dict[str, Any], payload: bytes) -> dict[str, Any]:
        speaker, session, prompt_id = meta["speaker_id"], meta["session_id"], meta["prompt_id"]
        prompt = self.bank.render(speaker, prompt_id)
        pcm = read_wav_pcm16(payload)
        duration_s = pcm.size / SAMPLE_RATE
        if duration_s < MIN_CLIP_S:
            raise ClipError(400, f"clip quá ngắn ({duration_s:.2f} s, cần ≥ {MIN_CLIP_S} s)", "duration")
        if duration_s > self.max_clip_s:
            raise ClipError(400, f"clip quá dài ({duration_s:.1f} s, tối đa {self.max_clip_s:g} s)", "duration")
        analysis = analyze_pcm(pcm)
        if not analysis["sound"]:
            raise ClipError(422, "không nghe thấy tiếng nào — kiểm tra micro rồi ghi lại", "silent")
        client_pause = meta.get("client_pause")
        record: dict[str, Any] = {
            "type": "clip", "id": "c-" + secrets.token_hex(8), "speaker_id": speaker,
            "session_id": session, "prompt_id": prompt_id, "bank_version": self.bank.version,
            "family": prompt.family, "digits": prompt.digits, "text": prompt.text,
            "part1": prompt.part1, "part2": prompt.part2, "keys": list(prompt.keys),
            "kind": prompt.kind, "instruction": prompt.instruction,
            "conditions": meta["conditions"], "duration_ms": round(duration_s * 1000, 1),
            "audio_sha256": hashlib.sha256(pcm.tobytes()).hexdigest(),
            "analysis_server": analysis, "analysis_client": client_pause,
            "pause_at_ms": None, "pause_ms": None,
            # Người nói tự nghe lại và xác nhận. Một người khác soát lại sau
            # thì ghi verified_by khác — trường này để phân biệt hai loại.
            "human_verified": "yes", "labeler_id": speaker, "verified_by": "self",
            "capture": meta.get("capture") or {}, "user_agent": meta.get("user_agent", ""),
        }
        if prompt.family in PAUSE_FAMILIES:
            at, length = analysis["pause_at_ms"], analysis["pause_ms"]
            if at is None:
                raise ClipError(422, f"không thấy quãng dừng ≥ {MIN_PAUSE_MS} ms giữa hai phần — ghi lại "
                                     "và dừng rõ khoảng một giây", "no_pause")
            if not client_pause or client_pause.get("pause_at_ms") is None:
                raise ClipError(400, "thiếu quãng dừng client đo (client_pause_at_ms, client_pause_ms)", "pause")
            c_at, c_len = client_pause["pause_at_ms"], client_pause["pause_ms"]
            if (abs(c_at - at) > PAUSE_TOLERANCE_MS
                    or abs((c_at + c_len) - (at + length)) > PAUSE_TOLERANCE_MS):
                raise ClipError(422, f"quãng dừng trình duyệt đo ({c_at:.0f}+{c_len:.0f} ms) lệch server "
                                     f"({at}+{length} ms) quá {PAUSE_TOLERANCE_MS} ms — ghi lại", "pause_mismatch")
            if at <= 0 or at + length >= record["duration_ms"]:
                raise ClipError(422, "quãng dừng nằm ngoài clip — ghi lại", "pause_outside")
            record.update(pause_at_ms=at, pause_ms=length)
        with self._lock:
            if not self.has_consent(speaker):
                raise ClipError(403, "người nói chưa đồng ý (hoặc nội dung đồng ý đã đổi)", "consent_required")
            bound = self.state.sessions.get(session)
            if bound is not None and bound["speaker_id"] != speaker:
                raise ClipError(409, "phiên này thuộc người nói khác — mở phiên mới", "session_owner")
            if bound is not None and bound["conditions"] != meta["conditions"]:
                raise ClipError(409, "điều kiện thu đã đổi — mỗi điều kiện là một phiên mới", "session_conditions")
            if any(c["speaker_id"] == speaker and c["prompt_id"] == prompt_id for c in self.state.clips.values()):
                raise ClipError(409, "câu này đã có clip — rút clip cũ nếu muốn thu lại", "duplicate")
            if len(self.state.clips) >= MAX_LIVE_CLIPS:
                # Mỗi người một clip mỗi câu, nhưng speaker_id thì ai cũng tự
                # sinh được: không có trần thì một client hỏng ghi đầy ổ đĩa.
                raise ClipError(507, f"kho đã đủ {MAX_LIVE_CLIPS} clip — báo người vận hành", "store_full")
            path = self._clip_path(speaker, session, record["id"])
            _private_dir(self.root / speaker)
            _private_dir(path.parent)
            record["wav"] = path.relative_to(self.root).as_posix()
            record["at"] = _now()
            _write_private(path, wav_from_pcm16(pcm))
            try:
                self._append(record)
            except BaseException:
                path.unlink(missing_ok=True)
                raise
        return record

    def _unlink(self, clip: dict[str, Any]) -> None:
        try:
            (self.root / clip["wav"]).unlink()
        except FileNotFoundError:
            pass

    def withdraw_clip(self, speaker_id: str, clip_id: str) -> bool:
        with self._lock:
            clip = self.state.clips.get(clip_id)
            # Không phân biệt "không có" với "của người khác": đoán id là vô ích.
            if clip is None or clip["speaker_id"] != speaker_id:
                return False
            self._unlink(clip)
            self._append({"type": "withdraw", "id": clip_id, "speaker_id": speaker_id, "at": _now()})
        return True

    # --- đọc ------------------------------------------------------------
    def speaker_view(self, speaker_id: str) -> dict[str, Any]:
        with self._lock:
            consent = self.state.speakers.get(speaker_id)
            clips = sorted((c for c in self.state.clips.values() if c["speaker_id"] == speaker_id),
                           key=lambda c: c["at"])
        recorded = {c["prompt_id"]: c["id"] for c in clips}
        prompts = []
        for block, prompt_id in self.bank.plan(speaker_id):
            item = self.bank.render(speaker_id, prompt_id).public()
            item.update(block=block, clip_id=recorded.get(prompt_id))
            prompts.append(item)
        return {
            "speaker_id": speaker_id, "bank_version": self.bank.version,
            "block_size": self.bank.block_size,
            "consent": None if consent is None else {
                "version": consent["consent_version"], "at": consent["at"],
                "pseudonym": consent["pseudonym"],
                "current": consent["consent_version"] == self.bank.consent["version"]},
            "prompts": prompts,
            "clips": [{"id": c["id"], "prompt_id": c["prompt_id"], "family": c["family"],
                       "text": c["text"] or c["instruction"], "session_id": c["session_id"],
                       "duration_ms": c["duration_ms"], "at": c["at"]} for c in clips],
            "recorded": len(clips),
        }

    def progress(self) -> dict[str, Any]:
        with self._lock:
            clips = list(self.state.clips.values())
            speakers = dict(self.state.speakers)
            withdrawn = len(self.state.withdrawn)
        families = Counter(c["family"] for c in clips)
        have = {
            "clips": len(clips),
            "speakers": len({c["speaker_id"] for c in clips}),
            "pause": families["hold"] + families["continue"],
            "backchannel_noise": families["backchannel"] + families["noise"],
            "interrupt": families["interrupt"],
        }
        per_speaker: dict[str, dict[str, Any]] = {}
        for speaker_id, consent in speakers.items():
            per_speaker[speaker_id] = {"speaker_id": speaker_id, "pseudonym": consent["pseudonym"],
                                       "consented_at": consent["at"], "consent_version": consent["consent_version"],
                                       "clips": 0, "minutes": 0.0, "families": Counter(), "sessions": set()}
        for clip in clips:
            row = per_speaker.setdefault(clip["speaker_id"], {
                "speaker_id": clip["speaker_id"], "pseudonym": None, "consented_at": None,
                "consent_version": None, "clips": 0, "minutes": 0.0, "families": Counter(), "sessions": set()})
            row["clips"] += 1
            row["minutes"] += clip["duration_ms"] / 60000
            row["families"][clip["family"]] += 1
            row["sessions"].add(clip["session_id"])
        speaker_rows = []
        for row in sorted(per_speaker.values(), key=lambda r: (-r["clips"], r["speaker_id"])):
            speaker_rows.append({**row, "minutes": round(row["minutes"], 2),
                                 "families": {f: row["families"][f] for f in FAMILIES},
                                 "sessions": len(row["sessions"])})
        conditions = {name: dict(Counter(c["conditions"].get(name, "") for c in clips)) for name in CONDITIONS}
        combos = Counter(" / ".join(c["conditions"].get(k, "") for k in ("device", "playback", "room")) for c in clips)
        return {
            "bank_version": self.bank.version,
            "totals": {"clips": len(clips), "withdrawn": withdrawn, "speakers_consented": len(speakers),
                       "sessions": len({c["session_id"] for c in clips}),
                       "minutes": round(sum(c["duration_ms"] for c in clips) / 60000, 2)},
            "families": {f: families[f] for f in FAMILIES},
            "digit_sequence": sum(1 for c in clips if c["digits"] and c["family"] in PAUSE_FAMILIES),
            "targets": [{"name": name, "label": label, "need": need, "have": have[name], "ok": have[name] >= need}
                        for name, label, need in TARGETS],
            "speakers": speaker_rows,
            "conditions": conditions,
            "device_playback_room": dict(combos.most_common()),
            "note": "Tổng mọi clip còn hiệu lực. Lúc xuất, người nói được chia train/heldout "
                    "(scripts/export_collect_labels.py), nên số heldout nhỏ hơn số này.",
        }


# ------------------------------------------------------------------ định tuyến

def _client_pause(request: Request) -> dict[str, float] | None:
    at = request.query_params.get("client_pause_at_ms")
    length = request.query_params.get("client_pause_ms")
    if at in (None, "") and length in (None, ""):
        return None
    try:
        values = {"pause_at_ms": float(at), "pause_ms": float(length)}
    except (TypeError, ValueError) as exc:
        raise ClipError(400, "client_pause_at_ms / client_pause_ms phải là số", "pause") from exc
    if not all(np.isfinite(v) and 0 <= v <= 600000 for v in values.values()):
        raise ClipError(400, "quãng dừng client ngoài khoảng hợp lệ", "pause")
    return values


def _capture(raw: str | None) -> dict[str, Any]:
    """Thiết lập micro trình duyệt báo (AEC/NS/AGC có thật sự bật không)."""
    if not raw or len(raw) > 600:
        return {}
    try:
        data = json.loads(raw)
    except ValueError:
        return {}
    if not isinstance(data, dict):
        return {}
    kept: dict[str, Any] = {}
    for key in _CAPTURE_KEYS:
        value = data.get(key)
        if isinstance(value, bool) or (isinstance(value, (int, float)) and np.isfinite(value)):
            kept[key] = value
        elif isinstance(value, str) and len(value) <= 40:
            kept[key] = value
    return kept


def _pseudonym(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    text = " ".join(value.split())
    if not 1 <= len(text) <= 40 or any(unicodedata.category(ch).startswith("C") for ch in text):
        return None
    return text


def install(app: FastAPI, config: Config) -> CollectStore | None:
    """Gắn các route /collect vào app. Tắt (mặc định) thì không gắn gì: 404."""
    settings = config.collect
    if not settings.enabled:
        return None
    bank = load_bank()
    store = CollectStore(Path(settings.dir), bank, settings.max_clip_s)
    app.state.collect = store
    web_dir = Path(config.server.web_dir)
    code = settings.access_code.encode()
    log.info("collect: bật, ghi vào %s (%d clip, %d người đã đồng ý)%s", settings.dir,
             len(store.state.clips), len(store.state.speakers), ", cần mã" if code else "")

    def deny(request: Request) -> JSONResponse | None:
        if foreign_origin(request, config):
            return JSONResponse({"detail": "yêu cầu từ trang khác bị từ chối", "code": "origin"}, status_code=403)
        if code:
            supplied = request.headers.get("x-collect-code", "").encode()
            if not hmac.compare_digest(supplied, code):
                return JSONResponse({"detail": "sai hoặc thiếu mã truy cập", "code": "access_code"}, status_code=403)
        return None

    def speaker_of(request: Request) -> str:
        speaker = request.query_params.get("speaker_id")
        if not valid_id(speaker):
            raise ClipError(400, "speaker_id không hợp lệ", "speaker_id")
        return speaker

    @app.get("/collect")
    async def collect_page() -> Any:
        page = web_dir / "collect.html"
        if page.exists():
            return FileResponse(str(page), headers={"Cache-Control": "no-cache"})
        return JSONResponse({"detail": "chưa có web/collect.html"}, status_code=404)

    @app.get("/collect/info")
    async def collect_info() -> Any:
        return {**bank.public(), "needs_code": bool(code), "max_clip_s": settings.max_clip_s,
                "min_clip_s": MIN_CLIP_S, "min_pause_ms": MIN_PAUSE_MS,
                "pause_tolerance_ms": PAUSE_TOLERANCE_MS,
                "conditions": {name: list(options.items()) for name, options in CONDITIONS.items()}}

    @app.post("/collect/consent")
    async def give_consent(request: Request) -> Any:
        if (denied := deny(request)) is not None:
            return denied
        kind = request.headers.get("content-type", "").split(";")[0].strip().lower()
        if kind != "application/json":
            return JSONResponse({"detail": "cần Content-Type: application/json"}, status_code=415)
        try:
            body = await request.json()
        except ValueError:
            return JSONResponse({"detail": "JSON không hợp lệ"}, status_code=400)
        if not isinstance(body, dict):
            return JSONResponse({"detail": "body phải là một object JSON"}, status_code=400)
        speaker = body.get("speaker_id")
        if not isinstance(speaker, str) or not valid_id(speaker):
            return JSONResponse({"detail": "speaker_id không hợp lệ", "code": "speaker_id"}, status_code=400)
        if body.get("agree") is not True:
            return JSONResponse({"detail": "cần đánh dấu đồng ý", "code": "consent_required"}, status_code=400)
        if body.get("consent_version") != bank.consent["version"]:
            return JSONResponse({"detail": "nội dung đồng ý đã đổi — tải lại trang", "code": "consent_version"},
                                status_code=409)
        pseudonym = _pseudonym(body.get("pseudonym"))
        if pseudonym is None:
            return JSONResponse({"detail": "bí danh cần 1–40 ký tự in được", "code": "pseudonym"}, status_code=400)
        user_agent = request.headers.get("user-agent", "")[:200]
        record = await asyncio.to_thread(store.consent, speaker, pseudonym, user_agent)
        return {"ok": True, "consent": {"version": record["consent_version"], "at": record["at"],
                                        "pseudonym": pseudonym, "current": True}}

    @app.delete("/collect/consent")
    async def withdraw_consent(request: Request) -> Any:
        if (denied := deny(request)) is not None:
            return denied
        try:
            speaker = speaker_of(request)
        except ClipError as exc:
            return exc.response()
        removed = await asyncio.to_thread(store.withdraw_consent, speaker)
        if removed < 0:
            return JSONResponse({"detail": "không có người nói này"}, status_code=404)
        return {"ok": True, "withdrawn": removed}

    @app.get("/collect/prompts")
    async def prompts(request: Request) -> Any:
        if (denied := deny(request)) is not None:
            return denied
        try:
            speaker = speaker_of(request)
        except ClipError as exc:
            return exc.response()
        return await asyncio.to_thread(store.speaker_view, speaker)

    @app.post("/collect/clip")
    async def upload_clip(request: Request) -> Any:
        if (denied := deny(request)) is not None:
            return denied
        kind = request.headers.get("content-type", "").split(";")[0].strip().lower()
        if kind not in WAV_TYPES:
            return JSONResponse({"detail": "cần Content-Type: audio/wav"}, status_code=415)
        query = request.query_params
        try:
            speaker = speaker_of(request)
            session = query.get("session_id")
            if not valid_id(session):
                raise ClipError(400, "session_id không hợp lệ", "session_id")
            prompt_id = query.get("prompt_id", "")
            if prompt_id not in bank.templates:
                raise ClipError(400, "prompt_id không có trong bộ câu", "prompt_id")
            if query.get("confirmed") != "yes":
                raise ClipError(400, "người nói chưa xác nhận clip (confirmed=yes)", "confirmed")
            conditions = {}
            for name, options in CONDITIONS.items():
                value = query.get(name, "")
                if value not in options:
                    raise ClipError(400, f"{name} không hợp lệ", "conditions")
                conditions[name] = value
            meta = {"speaker_id": speaker, "session_id": session, "prompt_id": prompt_id,
                    "conditions": conditions, "client_pause": _client_pause(request),
                    "capture": _capture(query.get("capture")),
                    "user_agent": request.headers.get("user-agent", "")[:200]}
            if not store.has_consent(speaker):
                raise ClipError(403, "người nói chưa đồng ý (hoặc nội dung đồng ý đã đổi)", "consent_required")
        except ClipError as exc:
            return exc.response()
        chunks = bytearray()
        async for chunk in request.stream():
            chunks.extend(chunk)
            if len(chunks) > settings.max_upload_bytes:
                return JSONResponse({"detail": f"file quá lớn (tối đa {settings.max_upload_bytes} byte)",
                                     "code": "size"}, status_code=413)
        if not chunks:
            return JSONResponse({"detail": "không có dữ liệu âm thanh", "code": "format"}, status_code=400)
        try:
            record = await asyncio.to_thread(store.add_clip, meta, bytes(chunks))
        except ClipError as exc:
            return exc.response()
        return {"ok": True, "clip": {
            "id": record["id"], "prompt_id": prompt_id, "family": record["family"],
            "duration_ms": record["duration_ms"], "pause_at_ms": record["pause_at_ms"],
            "pause_ms": record["pause_ms"], "analysis": record["analysis_server"]}}

    @app.delete("/collect/clip/{clip_id}")
    async def withdraw_clip(clip_id: str, request: Request) -> Any:
        if (denied := deny(request)) is not None:
            return denied
        try:
            speaker = speaker_of(request)
        except ClipError as exc:
            return exc.response()
        if not _CLIP_ID.fullmatch(clip_id):
            return JSONResponse({"detail": "clip id không hợp lệ", "code": "clip_id"}, status_code=400)
        if not await asyncio.to_thread(store.withdraw_clip, speaker, clip_id):
            return JSONResponse({"detail": "không có clip này của bạn"}, status_code=404)
        return {"ok": True, "withdrawn": clip_id}

    @app.get("/collect/progress")
    async def progress(request: Request) -> Any:
        # Luôn chỉ loopback, kể cả khi private_introspection tắt: trang này
        # liệt kê bí danh và số clip của mọi người nói.
        host = request.client.host if request.client else ""
        if foreign_origin(request, config) or not is_loopback(host):
            return JSONResponse({"detail": "chỉ xem được từ chính máy chạy server"}, status_code=403)
        return await asyncio.to_thread(store.progress)

    return store


__all__ = ["ACOUSTIC_FAMILIES", "CollectStore", "ManifestState", "analyze_pcm", "install",
           "read_manifest", "read_wav_pcm16"]
