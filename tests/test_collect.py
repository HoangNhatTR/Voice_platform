"""Trang /collect: người thật tự thu clip lượt lời cho G3.

Các test này giữ ba lời hứa của trang: chỉ ghi khi người nói đã đồng ý, chỉ
ghi đúng định dạng benchmark nhận và đúng chỗ (0600, không thoát khỏi
collect.dir), và bộ xuất cho ra thứ prepare_g3_human_stimuli.py chấp nhận mà
không người nói hay phiên nào nằm ở cả train lẫn heldout.

TestClient mặc định đến từ host "testclient" — với các route này, đó là một máy
khác trong LAN. `client=("127.0.0.1", …)` là chính máy chủ.
"""

from __future__ import annotations

import csv
import io
import json
import os
import shutil
import stat
import subprocess
import sys
import wave
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from scripts.export_collect_labels import ExportError, export
from scripts.prepare_g3_human_stimuli import FIELDS as IMPORT_FIELDS
from scripts.prepare_g3_human_stimuli import prepare
from voiceplatform.app import collect
from voiceplatform.app.collect import PAUSE_TOLERANCE_MS, analyze_pcm, read_manifest
from voiceplatform.app.collect_bank import BankError, load_bank, validate
from voiceplatform.app.server import create_app
from voiceplatform.core.config import Config
from voiceplatform.core.errors import ConfigError

REPO = Path(__file__).resolve().parents[1]
RATE = 16000
SAME = {"Origin": "http://testserver"}
EVIL = {"Origin": "https://evil.example"}
ASSISTANT = REPO / "docs/audits/2026-09-29/g3/stimuli-confirm.json"


# ------------------------------------------------------------------ tín hiệu giả

def signal(*segments: tuple[str, float], seed: int = 0) -> np.ndarray:
    """("tone"|"burst"|"quiet", giây) nối lại, cộng nhiễu nền -66 dBFS."""
    rng = np.random.default_rng(seed)
    parts = []
    for kind, seconds in segments:
        n = int(round(seconds * RATE))
        t = np.arange(n) / RATE
        if kind == "tone":
            parts.append(0.3 * np.sin(2 * np.pi * 220 * t) * (1 + 0.3 * np.sin(2 * np.pi * 3 * t)))
        elif kind == "burst":
            parts.append(rng.normal(0, 0.2, n))
        else:
            parts.append(np.zeros(n))
    x = np.concatenate(parts)
    x = x + rng.normal(0, 0.0005, x.size)
    return np.clip(x * 32767, -32768, 32767).astype("<i2")


def wav(pcm: np.ndarray, rate: int = RATE, channels: int = 1, width: int = 2) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(channels)
        handle.setsampwidth(width)
        handle.setframerate(rate)
        if width == 1:
            handle.writeframes(((pcm.astype(np.int32) >> 8) + 128).astype(np.uint8).tobytes())
        else:
            handle.writeframes(np.repeat(pcm, channels).astype("<i2").tobytes())
    return buffer.getvalue()


HOLD = signal(("quiet", 0.3), ("tone", 0.8), ("quiet", 1.0), ("tone", 0.8), ("quiet", 0.3))
STEADY = signal(("quiet", 0.3), ("tone", 1.5), ("quiet", 0.3), seed=1)
SHORT = signal(("quiet", 0.6), ("tone", 0.35), ("quiet", 0.6), seed=2)
KNOCK = signal(("quiet", 0.5), ("burst", 0.15), ("quiet", 0.6), seed=3)


# ------------------------------------------------------------------ dựng app

@pytest.fixture
def collect_config(config, tmp_path):
    config.collect.enabled = True
    config.collect.dir = str(tmp_path / "collect")
    config.server.web_dir = str(REPO / "web")
    return config


@pytest.fixture
def app(collect_config):
    # Không chạy lifespan: các route /collect không cần model nào.
    return create_app(collect_config)


@pytest.fixture
def lan(app):
    return TestClient(app)


@pytest.fixture
def local(app):
    return TestClient(app, client=("127.0.0.1", 40000))


def give_consent(client, speaker: str, headers: dict | None = None):
    version = client.get("/collect/info").json()["consent"]["version"]
    return client.post("/collect/consent", headers=headers or {}, json={
        "speaker_id": speaker, "pseudonym": "Mèo Mướp", "agree": True, "consent_version": version})


def prompt_of(client, speaker: str, family: str, *, digits: bool | None = None, nth: int = 0,
              headers: dict | None = None) -> dict:
    view = client.get("/collect/prompts", params={"speaker_id": speaker}, headers=headers or {}).json()
    found = [p for p in view["prompts"] if p["family"] == family and not p["clip_id"]
             and (digits is None or p["digits"] == digits)]
    return found[nth]


def upload(client, speaker: str, prompt_id: str, pcm: np.ndarray, *, session: str = "ss-test-0001",
           pause: dict | None | bool = True, headers: dict | None = None, body: bytes | None = None,
           **params):
    query = {"speaker_id": speaker, "session_id": session, "prompt_id": prompt_id,
             "device": "laptop_mic", "playback": "speaker", "room": "quiet", "accent": "",
             "confirmed": "yes", **params}
    if pause is True:
        found = analyze_pcm(pcm)
        if found["pause_at_ms"] is not None:
            query.update(client_pause_at_ms=found["pause_at_ms"], client_pause_ms=found["pause_ms"])
    elif isinstance(pause, dict):
        query.update(pause)
    return client.post("/collect/clip", params=query, content=wav(pcm) if body is None else body,
                       headers={"Content-Type": "audio/wav", **(headers or {})})


def wavs_under(root: Path) -> list[Path]:
    return sorted(root.rglob("*.wav")) if root.exists() else []


# ------------------------------------------------------------------ cấu hình

def test_disabled_by_default_mounts_nothing(config):
    assert Config().collect.enabled is False
    client = TestClient(create_app(config), client=("127.0.0.1", 40000))
    for method, path in (("get", "/collect"), ("get", "/collect/info"), ("get", "/collect/prompts"),
                         ("post", "/collect/clip"), ("get", "/collect/progress")):
        assert getattr(client, method)(path).status_code == 404


def test_the_deployed_config_enables_it_without_a_code():
    loaded = Config.load(REPO / "configs/local-cpu.yaml")
    assert loaded.collect.enabled is True
    assert loaded.collect.access_code == ""
    assert loaded.collect.dir == "runtime/collect"


@pytest.mark.parametrize(("field", "value"), [
    ("max_upload_bytes", 1000),         # không chứa nổi max_clip_s
    ("max_clip_s", 0.5),
    ("access_code", "mã có dấu"),       # header HTTP không mang được
])
def test_unsafe_collect_settings_fail_closed(field, value):
    config = Config()
    setattr(config.collect, field, value)
    with pytest.raises(ConfigError):
        config.validate()


def test_recordings_never_go_under_docs(collect_config):
    collect_config.collect.dir = str(REPO / "docs" / "collect-test")
    with pytest.raises(ValueError, match="docs"):
        create_app(collect_config)
    assert not (REPO / "docs" / "collect-test").exists()


# ------------------------------------------------------------------ câu mẫu

def test_each_block_is_balanced_and_digit_heavy():
    bank = load_bank()
    plan = bank.plan("sp-balance01")
    assert bank.blocks() >= 2 and len(plan) == bank.blocks() * bank.block_size == bank.blocks() * 50
    for block in range(bank.blocks()):
        ids = [pid for b, pid in plan if b == block]
        families = [bank.templates[pid]["family"] for pid in ids]
        assert {f: families.count(f) for f in set(families)} == {
            "hold": 12, "continue": 10, "complete": 8, "backchannel": 6, "noise": 4, "interrupt": 10}
        holds = [bank.render("sp-balance01", pid) for pid in ids if bank.templates[pid]["family"] == "hold"]
        assert sum(p.digits for p in holds) / len(holds) >= 0.4
    assert len({pid for _, pid in plan}) == len(plan)          # không câu nào lặp
    assert bank.templates[plan[0][1]]["family"] == "complete"  # làm quen trước


def test_plans_are_deterministic_per_speaker_and_differ_between_speakers():
    bank = load_bank()
    assert bank.plan("sp-aaaa") == bank.plan("sp-aaaa")
    assert bank.plan("sp-aaaa") != bank.plan("sp-bbbb")
    one, two = bank.render("sp-aaaa", "h-d01"), bank.render("sp-bbbb", "h-d01")
    assert one == bank.render("sp-aaaa", "h-d01")
    assert one.part1 != two.part1 or one.part2 != two.part2     # số giả riêng từng người


def test_prompts_are_new_sentences_not_the_tuning_stimuli():
    """Câu đã dùng để chỉnh luật mà vào heldout là nhiễm. Chỉ lời đệm trùng chữ."""
    def plain(text: str) -> str:
        return " ".join("".join(ch if ch.isalnum() or ch.isspace() else " " for ch in text.lower()).split())

    old: set[str] = set()
    sources = sorted((REPO / "docs/audits/2026-09-29/g3").glob("stimuli*.json"))
    if not sources:
        pytest.skip("không có bộ stimuli cũ để so")
    for source in sources:
        for family, cases in json.loads(source.read_text(encoding="utf-8"))["cases"].items():
            if family in ("backchannel", "noise"):
                continue
            for case in cases:
                old.add(plain(case.get("text", "").replace("…", " ")))
                old.update(plain(part) for part in case.get("parts") or ())
    bank = load_bank()
    for speaker in ("sp-x1", "sp-x2"):
        for pid, item in bank.templates.items():
            if item["family"] in ("backchannel", "noise"):
                continue
            prompt = bank.render(speaker, pid)
            mine = {plain(prompt.text), plain(prompt.part1), plain(prompt.part2)} - {""}
            assert not mine & old, f"{pid} trùng câu cũ: {mine & old}"


@pytest.mark.parametrize("mutate", [
    lambda d: d["prompts"].append(dict(d["prompts"][0])),                       # id trùng
    lambda d: d["prompts"][0].update(part2="{khong.co} nữa"),                   # placeholder lạ
    lambda d: d["prompts"][-1].update(keys=["không hề có"]),                    # khoá ngoài câu
    lambda d: d["quotas"].update(hold_digits=1, hold_plain=11),                 # < 40% dãy số
])
def test_a_broken_prompt_bank_refuses_to_load(mutate):
    data = json.loads((REPO / "src/voiceplatform/app/collect_prompts.json").read_text(encoding="utf-8"))
    mutate(data)
    with pytest.raises(BankError):
        validate(data)


# ------------------------------------------------------------------ truy cập

def test_page_and_info_are_open_to_the_lan_and_progress_is_not(lan, local):
    page = lan.get("/collect")
    assert page.status_code == 200 and "text/html" in page.headers["content-type"]
    for asset in ("collect.js", "capture.js", "pause.js", "collect.css", "capture-worklet.js"):
        assert lan.get(f"/static/{asset}").status_code == 200
    info = lan.get("/collect/info").json()
    assert info["needs_code"] is False and info["consent"]["version"] and info["block_size"] == 50
    assert lan.get("/collect/prompts", params={"speaker_id": "sp-lan00001"}).status_code == 200
    # Tiến độ liệt kê mọi người nói: chỉ từ chính máy chủ, kể cả khi
    # private_introspection đang tắt như ở đây.
    assert lan.app.state.platform.config.server.private_introspection is False
    assert lan.get("/collect/progress").status_code == 403
    assert local.get("/collect/progress", headers=EVIL).status_code == 403
    assert local.get("/collect/progress").status_code == 200


def test_a_foreign_page_cannot_write_or_delete(lan):
    assert give_consent(lan, "sp-origin01", headers=EVIL).status_code == 403
    assert give_consent(lan, "sp-origin01", headers=SAME).status_code == 200
    prompt = prompt_of(lan, "sp-origin01", "complete")
    assert upload(lan, "sp-origin01", prompt["id"], STEADY, headers=EVIL).status_code == 403
    assert lan.get("/collect/prompts", params={"speaker_id": "sp-origin01"}, headers=EVIL).status_code == 403
    saved = upload(lan, "sp-origin01", prompt["id"], STEADY, headers=SAME)
    assert saved.status_code == 200
    clip_id = saved.json()["clip"]["id"]
    assert lan.delete(f"/collect/clip/{clip_id}", params={"speaker_id": "sp-origin01"},
                      headers=EVIL).status_code == 403
    assert lan.delete("/collect/consent", params={"speaker_id": "sp-origin01"}, headers=EVIL).status_code == 403
    assert len(wavs_under(Path(lan.app.state.collect.root))) == 1


def test_the_access_code_guards_every_write_when_set(collect_config):
    collect_config.collect.access_code = "s3cret-LAN"
    client = TestClient(create_app(collect_config))
    info = client.get("/collect/info")
    assert info.json()["needs_code"] is True and "s3cret" not in info.text
    good, bad = {"X-Collect-Code": "s3cret-LAN"}, {"X-Collect-Code": "nope"}
    assert give_consent(client, "sp-code0001").status_code == 403
    assert give_consent(client, "sp-code0001", headers=bad).status_code == 403
    assert give_consent(client, "sp-code0001", headers=good).status_code == 200
    assert client.get("/collect/prompts", params={"speaker_id": "sp-code0001"}).status_code == 403
    prompt = prompt_of(client, "sp-code0001", "complete", headers=good)
    assert upload(client, "sp-code0001", prompt["id"], STEADY).status_code == 403
    saved = upload(client, "sp-code0001", prompt["id"], STEADY, headers=good)
    assert saved.status_code == 200
    clip_id = saved.json()["clip"]["id"]
    assert client.delete(f"/collect/clip/{clip_id}", params={"speaker_id": "sp-code0001"}).status_code == 403
    assert client.delete(f"/collect/clip/{clip_id}", params={"speaker_id": "sp-code0001"},
                         headers=good).status_code == 200


def test_ids_that_could_escape_the_directory_are_refused(lan, tmp_path):
    for bad in ("../x", "..", "a/b", "a\\b", ".hidden", "a b", "x" * 80, "", "sp-%2e%2e"):
        assert give_consent(lan, bad).status_code == 400
        assert lan.get("/collect/prompts", params={"speaker_id": bad}).status_code == 400
    assert give_consent(lan, "sp-safe0001").status_code == 200
    prompt = prompt_of(lan, "sp-safe0001", "complete")
    for session in ("../../escape", "a/b", "..", "ss x"):
        assert upload(lan, "sp-safe0001", prompt["id"], STEADY, session=session).status_code == 400
    assert upload(lan, "../sp", prompt["id"], STEADY).status_code == 400
    assert upload(lan, "sp-safe0001", "../../etc/passwd", STEADY).status_code == 400
    assert lan.delete("/collect/clip/c-not-hex", params={"speaker_id": "sp-safe0001"}).status_code == 400
    assert lan.delete("/collect/clip/c-0123456789abcdef", params={"speaker_id": "../x"}).status_code == 400
    assert wavs_under(tmp_path) == []
    assert sorted(p.name for p in tmp_path.iterdir()) == ["collect"]


# ------------------------------------------------------------------ đồng ý

def test_no_clip_is_stored_without_consent(lan):
    root = Path(lan.app.state.collect.root)
    prompt = prompt_of(lan, "sp-noconsent", "complete")
    refused = upload(lan, "sp-noconsent", prompt["id"], STEADY)
    assert refused.status_code == 403 and refused.json()["code"] == "consent_required"
    version = lan.get("/collect/info").json()["consent"]["version"]
    base = {"speaker_id": "sp-noconsent", "pseudonym": "Mèo", "consent_version": version}
    assert lan.post("/collect/consent", json={**base, "agree": False}).status_code == 400
    assert lan.post("/collect/consent", json={**base, "agree": True, "consent_version": "cũ"}).status_code == 409
    assert lan.post("/collect/consent", json={**base, "agree": True, "pseudonym": " "}).status_code == 400
    assert lan.post("/collect/consent", json={**base, "agree": True, "pseudonym": "a\x00b"}).status_code == 400
    assert lan.post("/collect/consent", content=json.dumps({**base, "agree": True}),
                    headers={"Content-Type": "text/plain"}).status_code == 415
    assert wavs_under(root) == []
    assert lan.post("/collect/consent", json={**base, "agree": True}).status_code == 200
    view = lan.get("/collect/prompts", params={"speaker_id": "sp-noconsent"}).json()
    assert view["consent"]["current"] is True and view["consent"]["pseudonym"] == "Mèo"
    assert upload(lan, "sp-noconsent", prompt["id"], STEADY).status_code == 200
    consent = [r for r in read_manifest(root / "manifest.jsonl") if r["type"] == "consent"][0]
    assert consent["consent_version"] == version and consent["at"] and consent["consent_sha256"]


# ------------------------------------------------------------------ định dạng WAV

def test_only_mono_16_bit_16_khz_wavs_of_sane_length_are_kept(collect_config):
    collect_config.collect.max_clip_s = 2.0
    collect_config.collect.max_upload_bytes = 80000
    client = TestClient(create_app(collect_config))
    root = Path(client.app.state.collect.root)
    assert give_consent(client, "sp-format01").status_code == 200
    pid = prompt_of(client, "sp-format01", "complete")["id"]
    tone = signal(("quiet", 0.2), ("tone", 0.6), ("quiet", 0.2))   # stereo still < 80 000 byte
    cases = {
        "stereo": (wav(tone, channels=2), 400),
        "8 kHz": (wav(tone[::2], rate=8000), 400),
        "8-bit": (wav(tone, width=1), 400),
        "not wav": (b"OggS" + bytes(4000), 400),
        "truncated": (wav(tone)[:30], 400),
        "too short": (wav(signal(("tone", 0.2))), 400),
        "too long": (wav(signal(("tone", 2.4))), 400),          # 76 844 byte: dưới giới hạn byte
        "too big": (wav(signal(("tone", 3.0))), 413),
        "empty": (b"", 400),
    }
    for name, (body, status) in cases.items():
        response = upload(client, "sp-format01", pid, tone, body=body)
        assert response.status_code == status, name
    wrong_type = client.post("/collect/clip", content=wav(tone), headers={"Content-Type": "text/plain"}, params={
        "speaker_id": "sp-format01", "session_id": "ss-1", "prompt_id": pid, "device": "phone",
        "playback": "speaker", "room": "quiet", "confirmed": "yes"})
    assert wrong_type.status_code == 415
    silent = upload(client, "sp-format01", pid, np.zeros(RATE, dtype="<i2"))
    assert silent.status_code == 422 and silent.json()["code"] == "silent"
    unconfirmed = upload(client, "sp-format01", pid, tone, confirmed="no")
    assert unconfirmed.status_code == 400
    assert upload(client, "sp-format01", pid, tone, room="garden").status_code == 400
    assert wavs_under(root) == []
    assert upload(client, "sp-format01", pid, tone).status_code == 200
    assert len(wavs_under(root)) == 1


# ------------------------------------------------------------------ quãng dừng

def test_a_hold_clip_needs_a_pause_both_sides_agree_on(lan):
    root = Path(lan.app.state.collect.root)
    speaker = "sp-pause001"
    assert give_consent(lan, speaker).status_code == 200
    hold = prompt_of(lan, speaker, "hold", digits=True)
    no_pause = upload(lan, speaker, hold["id"], STEADY)
    assert no_pause.status_code == 422 and no_pause.json()["code"] == "no_pause"
    assert upload(lan, speaker, hold["id"], HOLD, pause=False).status_code == 400
    found = analyze_pcm(HOLD)
    assert found["pause_at_ms"] == pytest.approx(1100, abs=30) and found["pause_ms"] == pytest.approx(1000, abs=40)
    skew = PAUSE_TOLERANCE_MS + 60
    shifted = upload(lan, speaker, hold["id"], HOLD, pause={
        "client_pause_at_ms": found["pause_at_ms"] + skew, "client_pause_ms": found["pause_ms"]})
    assert shifted.status_code == 422 and shifted.json()["code"] == "pause_mismatch"
    close = upload(lan, speaker, hold["id"], HOLD, pause={
        "client_pause_at_ms": found["pause_at_ms"] + 40, "client_pause_ms": found["pause_ms"] - 30})
    assert close.status_code == 200
    record = [r for r in read_manifest(root / "manifest.jsonl") if r["type"] == "clip"][0]
    assert (record["pause_at_ms"], record["pause_ms"]) == (found["pause_at_ms"], found["pause_ms"])
    assert record["analysis_client"] == {"pause_at_ms": found["pause_at_ms"] + 40,
                                         "pause_ms": found["pause_ms"] - 30}
    assert record["human_verified"] == "yes" and record["verified_by"] == "self"
    assert record["labeler_id"] == speaker
    # Chữ do server dựng lại, không lấy từ request.
    rendered = lan.app.state.collect.bank.render(speaker, hold["id"])
    assert record["text"] == rendered.text and record["keys"] == list(rendered.keys)
    assert record["part1"] == hold["part1"] and record["digits"] is True


def test_short_clicks_do_not_count_as_speech():
    click = signal(("quiet", 0.4), ("burst", 0.02), ("quiet", 0.6), ("tone", 1.0), ("quiet", 0.4), seed=5)
    found = analyze_pcm(click)
    assert found["pause_at_ms"] is None                         # click → giọng không phải quãng dừng
    assert found["speech_start_ms"] == pytest.approx(1020, abs=20)
    knock = analyze_pcm(KNOCK)
    assert knock["sound"] is True and knock["pause_at_ms"] is None


# ------------------------------------------------------------------ phiên, rút clip, quyền file

def test_a_session_belongs_to_one_speaker_and_one_condition(lan):
    for speaker in ("sp-sess-a01", "sp-sess-b01"):
        assert give_consent(lan, speaker).status_code == 200
    first = prompt_of(lan, "sp-sess-a01", "complete")
    assert upload(lan, "sp-sess-a01", first["id"], STEADY, session="ss-shared-1").status_code == 200
    other = prompt_of(lan, "sp-sess-b01", "complete")
    stolen = upload(lan, "sp-sess-b01", other["id"], STEADY, session="ss-shared-1")
    assert stolen.status_code == 409 and stolen.json()["code"] == "session_owner"
    second = prompt_of(lan, "sp-sess-a01", "complete")
    moved = upload(lan, "sp-sess-a01", second["id"], STEADY, session="ss-shared-1", device="phone")
    assert moved.status_code == 409 and moved.json()["code"] == "session_conditions"
    again = upload(lan, "sp-sess-a01", first["id"], STEADY, session="ss-shared-1")
    assert again.status_code == 409 and again.json()["code"] == "duplicate"


def test_a_full_store_refuses_more_clips_instead_of_filling_the_disk(lan, monkeypatch):
    # speaker_id is self-minted, so the per-prompt duplicate rule alone does not bound the disk.
    monkeypatch.setattr(collect, "MAX_LIVE_CLIPS", 1)
    for speaker in ("sp-full-a01", "sp-full-b01"):
        assert give_consent(lan, speaker).status_code == 200
    first = prompt_of(lan, "sp-full-a01", "complete")
    assert upload(lan, "sp-full-a01", first["id"], STEADY, session="ss-full-a1").status_code == 200
    second = prompt_of(lan, "sp-full-b01", "complete")
    full = upload(lan, "sp-full-b01", second["id"], STEADY, session="ss-full-b1")
    assert full.status_code == 507 and full.json()["code"] == "store_full"


def test_files_are_owner_only(lan):
    root = Path(lan.app.state.collect.root)
    assert give_consent(lan, "sp-perm0001").status_code == 200
    assert upload(lan, "sp-perm0001", prompt_of(lan, "sp-perm0001", "complete")["id"], STEADY).status_code == 200
    (clip,) = wavs_under(root)
    assert stat.S_IMODE(clip.stat().st_mode) == 0o600
    assert stat.S_IMODE((root / "manifest.jsonl").stat().st_mode) == 0o600
    for directory in (root, clip.parent, clip.parent.parent):
        assert stat.S_IMODE(directory.stat().st_mode) == 0o700
    assert clip.relative_to(root).parts[:2] == ("sp-perm0001", "ss-test-0001")


def test_a_speaker_can_withdraw_their_own_clips_and_only_theirs(lan, local):
    root = Path(lan.app.state.collect.root)
    for speaker in ("sp-wd-a0001", "sp-wd-b0001"):
        assert give_consent(lan, speaker).status_code == 200
    prompt = prompt_of(lan, "sp-wd-a0001", "complete")
    clip_id = upload(lan, "sp-wd-a0001", prompt["id"], STEADY).json()["clip"]["id"]
    assert local.get("/collect/progress").json()["totals"]["clips"] == 1
    assert lan.delete(f"/collect/clip/{clip_id}", params={"speaker_id": "sp-wd-b0001"}).status_code == 404
    assert len(wavs_under(root)) == 1
    assert lan.delete(f"/collect/clip/{clip_id}", params={"speaker_id": "sp-wd-a0001"}).status_code == 200
    assert wavs_under(root) == []
    assert lan.delete(f"/collect/clip/{clip_id}", params={"speaker_id": "sp-wd-a0001"}).status_code == 404
    view = lan.get("/collect/prompts", params={"speaker_id": "sp-wd-a0001"}).json()
    assert view["recorded"] == 0 and not next(p for p in view["prompts"] if p["id"] == prompt["id"])["clip_id"]
    progress = local.get("/collect/progress").json()
    assert progress["totals"]["clips"] == 0 and progress["totals"]["withdrawn"] == 1
    # Rút toàn bộ: xoá mọi file, và không thu tiếp được nếu chưa đồng ý lại.
    for family in ("complete", "interrupt"):
        assert upload(lan, "sp-wd-a0001", prompt_of(lan, "sp-wd-a0001", family)["id"], STEADY).status_code == 200
    assert len(wavs_under(root)) == 2
    gone = lan.delete("/collect/consent", params={"speaker_id": "sp-wd-a0001"})
    assert gone.status_code == 200 and gone.json()["withdrawn"] == 2
    assert wavs_under(root) == []
    after = upload(lan, "sp-wd-a0001", prompt_of(lan, "sp-wd-a0001", "complete")["id"], STEADY)
    assert after.status_code == 403 and after.json()["code"] == "consent_required"


def test_state_survives_a_restart_and_progress_counts_targets(collect_config):
    first = TestClient(create_app(collect_config))
    assert give_consent(first, "sp-restart01").status_code == 200
    for family, pcm in (("hold", HOLD), ("continue", HOLD), ("backchannel", SHORT), ("noise", KNOCK),
                        ("interrupt", STEADY)):
        assert upload(first, "sp-restart01", prompt_of(first, "sp-restart01", family)["id"], pcm,
                      device="headset_mic", playback="headphones", room="noisy", accent="south").status_code == 200
    second = TestClient(create_app(collect_config), client=("127.0.0.1", 40000))
    view = second.get("/collect/prompts", params={"speaker_id": "sp-restart01"}).json()
    assert view["recorded"] == 5 and view["consent"]["current"] is True
    progress = second.get("/collect/progress").json()
    assert progress["families"] == {"hold": 1, "continue": 1, "complete": 0, "backchannel": 1,
                                    "noise": 1, "interrupt": 1}
    targets = {t["name"]: t for t in progress["targets"]}
    assert targets["clips"]["need"] == 500 and targets["speakers"]["need"] == 10
    assert targets["pause"]["have"] == 2 and targets["backchannel_noise"]["have"] == 2
    assert targets["interrupt"]["have"] == 1 and not targets["interrupt"]["ok"]
    assert progress["conditions"]["device"] == {"headset_mic": 5}
    assert progress["conditions"]["accent"] == {"south": 5}
    assert progress["speakers"][0]["pseudonym"] == "Mèo Mướp" and progress["speakers"][0]["sessions"] == 1


# ------------------------------------------------------------------ xuất → prepare()

def test_export_feeds_prepare_without_leaking_speakers(lan, tmp_path):
    root = Path(lan.app.state.collect.root)
    speakers = ("sp-e2e-a001", "sp-e2e-b001", "sp-e2e-c001")
    clips = {
        "hold": HOLD, "continue": signal(("quiet", 0.2), ("tone", 1.0), ("quiet", 0.8), ("tone", 0.6),
                                         ("quiet", 0.2), seed=7),
        "complete": STEADY, "backchannel": SHORT, "noise": KNOCK, "interrupt": STEADY,
    }
    withdrawn = None
    for n, speaker in enumerate(speakers):
        assert give_consent(lan, speaker).status_code == 200
        for family, pcm in clips.items():
            for nth in range(5 if family == "interrupt" else 1):
                prompt = prompt_of(lan, speaker, family)
                response = upload(lan, speaker, prompt["id"], pcm, session=f"ss-{speaker}-{nth % 2}",
                                  device="phone" if nth % 2 else "laptop_mic")
                assert response.status_code == 200, response.text
        extra = upload(lan, speaker, prompt_of(lan, speaker, "complete")["id"], STEADY, session=f"ss-{speaker}-0")
        withdrawn = withdrawn or extra.json()["clip"]["id"]
        if n == 0:
            assert lan.delete(f"/collect/clip/{withdrawn}", params={"speaker_id": speaker}).status_code == 200

    out = tmp_path / "export"
    summary = export(root, out, heldout_speakers=["sp-e2e-b001"], seed=11)
    assert summary["speakers"] == {"train": ["sp-e2e-a001", "sp-e2e-c001"], "heldout": ["sp-e2e-b001"]}
    with (out / "labels.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert set(IMPORT_FIELDS) <= set(rows[0]) and len(rows) == summary["exported"] == 3 * 10 + 2
    assert withdrawn not in {r["id"] for r in rows}
    for key in ("speaker_id", "session_id"):
        sides: dict[str, set] = {}
        for row in rows:
            sides.setdefault(row[key], set()).add(row["split"])
        assert all(len(s) == 1 for s in sides.values()), key
    for row in rows:
        assert row["human_verified"] == "yes" and row["verified_by"] == "self"
        assert row["labeler_id"] == row["speaker_id"]
        assert stat.S_IMODE((out / row["wav"]).stat().st_mode) == 0o600
        if row["family"] in ("backchannel", "noise", "interrupt"):
            offset = float(row["offset_s"])
            assert (0.1 <= offset <= 0.6) if row["onset_window"] == "yes" else (0.5 <= offset <= 2.5)
            assert int(row["trim_start_ms"]) > 0                  # khoảng lặng đầu đã cắt
        if row["family"] in ("hold", "continue"):
            assert row["part1"] and row["part2"] and row["keys"]
            assert float(row["pause_at_ms"]) > 0 and float(row["pause_ms"]) >= 300
    interrupts = [r for r in rows if r["family"] == "interrupt" and r["split"] == "heldout"]
    assert sum(r["onset_window"] == "yes" for r in interrupts) == 1   # 20% của 5
    assert next(r for r in rows if r["family"] == "noise")["text"] == ""

    if not ASSISTANT.exists():
        pytest.skip("không có stimuli-confirm.json để lấy giọng trợ lý")
    counts = prepare(out / "labels.csv", tmp_path / "stimuli.json", ASSISTANT)
    # Người nói heldout giữ clip "complete" thứ hai (chỉ người nói đầu rút clip đó).
    assert counts == {"hold": 1, "continue": 1, "complete": 2, "backchannel": 1, "noise": 1, "interrupt": 5}
    payload = json.loads((tmp_path / "stimuli.json").read_text(encoding="utf-8"))
    assert payload["speakers_heldout"] == ["sp-e2e-b001"]
    hold = payload["cases"]["hold"][0]
    assert hold["pause_at_ms"] == pytest.approx(1100, abs=30) and hold["parts"][1]


def test_export_split_by_fraction_is_deterministic_and_refuses_unsafe_outputs(lan, tmp_path):
    root = Path(lan.app.state.collect.root)
    for n in range(5):
        speaker = f"sp-frac-{n:04d}"
        assert give_consent(lan, speaker).status_code == 200
        assert upload(lan, speaker, prompt_of(lan, speaker, "complete")["id"], STEADY,
                      session=f"ss-frac-{n}").status_code == 200
    one = export(root, tmp_path / "one", heldout_fraction=0.4, seed=3)
    two = export(root, tmp_path / "two", heldout_fraction=0.4, seed=3)
    assert one["speakers"] == two["speakers"] and len(one["speakers"]["heldout"]) == 2
    with pytest.raises(ExportError, match="thư mục mới|đã có"):
        export(root, tmp_path / "one", heldout_fraction=0.4, seed=3)
    with pytest.raises(ExportError, match="docs"):
        export(root, REPO / "docs" / "never-here", heldout_fraction=0.4)
    assert not (REPO / "docs" / "never-here").exists()
    with pytest.raises(ExportError, match="không có clip"):
        export(root, tmp_path / "three", heldout_speakers=["sp-nobody01"])


def test_the_export_cli_runs(lan, tmp_path):
    root = Path(lan.app.state.collect.root)
    assert give_consent(lan, "sp-cli00001").status_code == 200
    assert upload(lan, "sp-cli00001", prompt_of(lan, "sp-cli00001", "complete")["id"], STEADY).status_code == 200
    env = {**os.environ, "PYTHONPATH": str(REPO / "src")}
    done = subprocess.run(
        [sys.executable, str(REPO / "scripts/export_collect_labels.py"), str(root), str(tmp_path / "cli"),
         "--heldout-speakers", "sp-cli00001"],
        capture_output=True, text=True, env=env, timeout=120)
    assert done.returncode == 0, done.stderr
    summary = json.loads(done.stdout)
    assert summary["exported"] == 1 and "prepare_g3_human_stimuli.py" in summary["next"]


# ------------------------------------------------------------------ JS == Python

@pytest.mark.skipif(shutil.which("node") is None, reason="cần node")
def test_browser_and_server_find_the_same_pause(tmp_path):
    """web/pause.js chạy trong trình duyệt; collect.py chạy trên WAV đã nhận.

    Server từ chối khi hai bên lệch > 250 ms, nên chúng phải cho cùng số trên
    cùng mẫu — kể cả tín hiệu khó: click, tiếng thở trong quãng dừng, nhiễu.
    """
    rng = np.random.default_rng(42)
    cases = [HOLD, STEADY, SHORT, KNOCK, np.zeros(RATE, dtype="<i2"),
             signal(("quiet", 0.4), ("burst", 0.02), ("quiet", 0.6), ("tone", 1.0), seed=8),
             signal(("tone", 0.7), ("quiet", 0.45), ("burst", 0.03), ("quiet", 0.5), ("tone", 0.9), seed=9),
             np.clip(rng.normal(0, 3000, RATE * 2), -32768, 32767).astype("<i2")]
    for n in range(6):
        seg = [("quiet", float(rng.uniform(0.1, 0.5)))]
        for _ in range(int(rng.integers(1, 4))):
            seg += [("tone", float(rng.uniform(0.2, 1.0))), ("quiet", float(rng.uniform(0.05, 1.2)))]
        cases.append(signal(*seg, seed=100 + n))
    for n, pcm in enumerate(cases):
        (tmp_path / f"{n}.pcm").write_bytes(pcm.astype("<i2").tobytes())
    script = tmp_path / "run.cjs"
    script.write_text(
        "const fs=require('node:fs'),vm=require('node:vm');\n"
        f"const src=fs.readFileSync({json.dumps(str(REPO / 'web/pause.js'))},'utf8').replace(/^export /gm,'');\n"
        "const env={};vm.createContext(env);vm.runInContext(src+'\\nthis.analyzePcm=analyzePcm;',env);\n"
        "const out=process.argv.slice(2).map(p=>{const b=fs.readFileSync(p);"
        "return env.analyzePcm(new Int16Array(b.buffer,b.byteOffset,b.length/2),16000);});\n"
        "console.log(JSON.stringify(out));\n")
    done = subprocess.run(["node", str(script), *[str(tmp_path / f"{n}.pcm") for n in range(len(cases))]],
                          capture_output=True, text=True, timeout=60)
    assert done.returncode == 0, done.stderr
    keys = ("sound", "voiced_ms", "speech_start_ms", "speech_end_ms", "longest_gap_ms", "pause_at_ms", "pause_ms")
    for n, (pcm, js) in enumerate(zip(cases, json.loads(done.stdout))):
        py = analyze_pcm(pcm)
        assert {k: js[k] for k in keys} == {k: py[k] for k in keys}, n
        assert js["threshold_db"] == pytest.approx(py["threshold_db"], abs=0.11), n
