"""Chấm một lần chạy benchmark_g3.py theo đúng các gate G3.

Mỗi con số truy về được từng ca: summary.json liệt kê id của mọi ca lỗi.

  python scripts/summarize_g3.py /tmp/g3-run --output /tmp/g3-run/summary.json
  python scripts/summarize_g3.py RUN_A RUN_B --compare    # bảng so hai lần chạy
"""

from __future__ import annotations

import argparse
import json
import math
import unicodedata
from pathlib import Path
from typing import Any

SYSTEM_SOURCES = {"resume", "search", "text"}
HESITATIONS = {"um", "u", "o", "a", "uh", "hm", "hmm", "e"}
# id -> the whole clip decoded offline by the same ASR (asr_reference_g3.py).
REFERENCE: dict[str, str] = {}


def fold(text: str) -> str:
    text = unicodedata.normalize("NFD", (text or "").lower())
    return "".join(c for c in text if unicodedata.category(c) != "Mn").replace("đ", "d")


def wilson(k: int, n: int, z: float = 1.96) -> list[float] | None:
    if n == 0:
        return None
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [round(max(0.0, centre - half), 4), round(min(1.0, centre + half), 4)]


def rate(k: int, n: int) -> dict:
    return {"k": k, "n": n, "rate": round(k / n, 4) if n else None, "wilson95": wilson(k, n)}


def dist(values: list[float]) -> dict:
    vals = sorted(v for v in values if v is not None)
    if not vals:
        return {"n": 0}

    def rank(q: float) -> float:
        # Nearest rank: p95 là giá trị mà ít nhất 95% mẫu không vượt quá.
        return round(vals[min(len(vals) - 1, max(0, math.ceil(q * len(vals)) - 1))], 1)

    return {"n": len(vals), "p50": rank(0.5), "p95": rank(0.95), "p99": rank(0.99), "max": round(vals[-1], 1)}


def events(turn: dict, kind: str) -> list[dict]:
    return [e for e in turn.get("events", []) if e["type"] == kind]


def first(turn: dict, kind: str) -> dict | None:
    rows = events(turn, kind)
    return rows[0] if rows else None


def source(turn: dict) -> str:
    start = first(turn, "turn_start")
    return ((start or {}).get("data") or {}).get("source", "") or ""


def user_text(turn: dict) -> str:
    merged = first(turn, "turn_merged")
    if merged:
        return merged["data"].get("text", "")
    final = first(turn, "asr_final")
    return (final or {}).get("data", {}).get("text", "") if final else ""


def half_lost(part: str, text: str) -> bool:
    """A half of the sentence is LOST when most of its words never reached the model.

    Keyword checks conflate turn-taking with ASR: "tokyo" heard as "tokyu"
    is a recognition error, not a cut. A half counts as lost when fewer than
    half of its words (diacritics folded) appear in what the LLM was given.
    """
    words = [w for w in fold(part).replace(",", " ").replace(".", " ").split() if w]
    if words and all(w in HESITATIONS for w in words):
        return False        # "ừm" is not content; dropping it loses nothing
    have = set(fold(text).replace(",", " ").replace(".", " ").split())
    return bool(words) and sum(w in have for w in words) < len(words) / 2


def lost_to_turn_taking(case: dict, text: str) -> list[str]:
    """Halves the pipeline lost that the whole-clip decode did hear."""
    reference = REFERENCE.get(case["id"])
    parts = case.get("parts") or [case["text"]]
    return [h for h in parts if half_lost(h, text) and (reference is None or not half_lost(h, reference))]


def unconfirmed_audio(turns: list[dict]) -> list[dict]:
    """Audio của một lượt người dùng phát ra TRƯỚC khi lượt đó được chốt."""
    bad = []
    for turn in turns:
        sent = events(turn, "audio_sent")
        if not sent or source(turn) in SYSTEM_SOURCES:
            continue
        confirmed = first(turn, "turn_confirmed")
        final = first(turn, "asr_final")
        if confirmed is None or sent[0]["ts_ms"] < confirmed["ts_ms"] or (
            final is not None and sent[0]["ts_ms"] < final["ts_ms"]
        ):
            bad.append({"turn_id": turn["turn_id"], "first_audio": sent[0]["ts_ms"],
                        "confirmed": confirmed and confirmed["ts_ms"]})
    return bad


def load(run: Path) -> dict[str, list[dict]]:
    out = {}
    for path in sorted(run.glob("*.json")):
        if path.name in ("manifest.json", "summary.json"):
            continue
        data = json.loads(path.read_text())
        if "family" in data:
            out[data["family"]] = [r for r in data["rows"] if "case" in r]
    return out


# ---------------------------------------------------------------- families

def score_hold(rows: list[dict]) -> dict:
    n = cut = visible = lost = merged = errors = lost_half = user_visible = 0
    failures = []
    waits = []
    for row in rows:
        if "error" in row:
            errors += 1
            failures.append({"id": row["case"]["id"], "why": row["error"]})
            continue
        n += 1
        onset2 = row.get("part2_onset_ms")
        confirms = [first(t, "turn_confirmed") for t in row["turns"] if source(t) not in SYSTEM_SOURCES]
        confirms = [c for c in confirms if c]
        early = [c for c in confirms if onset2 is not None and c["ts_ms"] < onset2]
        # Tiếng máy ra loa giữa quãng nghỉ: người dùng nghe thấy mình bị cắt.
        heard = [p for p in row.get("playback", []) if p["event"] == "playback_started"
                 and onset2 is not None and p["client_ms"] < onset2]
        texts = " ".join(user_text(t) for t in row["turns"] if first(t, "llm_start") or first(t, "turn_merged"))
        missing = [k for k in row["case"]["keys"] if fold(k) not in fold(texts)]
        halves = lost_to_turn_taking(row["case"], texts)
        lost_half += bool(halves)
        was_merged = any(first(t, "turn_merged") for t in row["turns"])
        # What the user notices: the assistant spoke in the pause, or a half is gone.
        user_visible += bool(heard or halves)
        cut += bool(early)
        visible += bool(heard)
        lost += bool(missing)
        merged += was_merged
        for t in row["turns"]:
            for c in events(t, "endpoint_candidate"):
                if onset2 is not None and c["ts_ms"] < onset2:
                    waits.append(c["data"].get("required_silence_ms"))
        if early or heard or missing:
            failures.append({"id": row["case"]["id"], "cut": bool(early), "heard_during_pause": [p.get("role") for p in heard],
                             "missing_keys": missing, "lost_halves": halves, "merged": was_merged, "texts": texts,
                             "asr_reference": REFERENCE.get(row["case"]["id"]),
                             "pause_candidates": [c["data"].get("text") for t in row["turns"] for c in events(t, "endpoint_candidate")]})
    return {"premature_confirm": rate(cut, n), "audio_during_pause": rate(visible, n),
            "lost_half": rate(lost_half, n), "user_visible_cut": rate(user_visible, n),
            "lost_words": rate(lost, n), "merged": rate(merged, n), "errors": errors,
            "required_silence_at_pause_ms": dist(waits), "failures": failures}


def score_complete(rows: list[dict]) -> dict:
    n = errors = double = 0
    endpoint, content, content_tool, confirm_to_final, held = [], [], [], [], 0
    failures = []
    for row in rows:
        if "error" in row:
            errors += 1
            failures.append({"id": row["case"]["id"], "why": row["error"]})
            continue
        n += 1
        end = row["clip"]["end_ms"]
        speech_turns = [t for t in row["turns"] if source(t) not in SYSTEM_SOURCES and first(t, "turn_confirmed")]
        if len(speech_turns) > 1:
            double += 1
            failures.append({"id": row["case"]["id"], "why": "more than one turn",
                             "texts": [user_text(t) for t in speech_turns]})
        if not speech_turns or end is None:
            failures.append({"id": row["case"]["id"], "why": "no confirmed turn"})
            continue
        last = speech_turns[-1]
        wait = first(last, "turn_confirmed")["ts_ms"] - end
        endpoint.append(wait)
        held += wait > 1000
        final = first(last, "asr_final")
        if final:
            confirm_to_final.append(final["ts_ms"] - first(last, "turn_confirmed")["ts_ms"])
        starts = [p["client_ms"] for p in row.get("playback", []) if p["event"] == "playback_started"
                  and p.get("role") == "content" and p["client_ms"] >= end]
        tool = bool(events(last, "tool_start"))
        if starts:
            (content_tool if tool else content).append(min(starts) - end)
    return {"speech_end_to_confirm_ms": dist(endpoint), "confirm_to_asr_final_ms": dist(confirm_to_final),
            "speech_end_to_content_playback_ms": dist(content),
            "speech_end_to_content_playback_ms_tool_turns": dist(content_tool), "held_over_1s": rate(held, n),
            "split_into_two_turns": rate(double, n), "errors": errors, "failures": failures}


def score_continue(rows: list[dict]) -> dict:
    n = lost = merged = separate = errors = lost_half = 0
    failures = []
    for row in rows:
        if "error" in row:
            errors += 1
            continue
        n += 1
        onset2 = row.get("part2_onset_ms")
        texts = " ".join(user_text(t) for t in row["turns"] if first(t, "llm_start") or first(t, "turn_merged"))
        missing = [k for k in row["case"]["keys"] if fold(k) not in fold(texts)]
        was_merged = any(first(t, "turn_merged") for t in row["turns"])
        answered_alone = any(p["event"] == "playback_started" and p.get("role") == "content"
                             and onset2 is not None and p["client_ms"] < onset2 for p in row.get("playback", []))
        halves = lost_to_turn_taking(row["case"], texts)
        lost_half += bool(halves)
        lost += bool(missing)
        merged += was_merged
        separate += answered_alone
        if missing:
            failures.append({"id": row["case"]["id"], "missing_keys": missing, "texts": texts, "merged": was_merged})
    return {"lost_half": rate(lost_half, n), "lost_words": rate(lost, n), "merged": rate(merged, n),
            "part1_answered_before_part2": rate(separate, n), "errors": errors, "failures": failures}


def _event_turn(row: dict) -> dict | None:
    """The user turn the interjection opened (barge-in), if any."""
    onset = row["clip"]["onset_ms"] or row["clip"]["start_ms"]
    for turn in row.get("turns", []):
        start = first(turn, "turn_start")
        if start and source(turn) == "barge_in" and onset - 50 <= start["ts_ms"] <= onset + row["clip"]["duration_ms"] + 800:
            return turn
    return None


def _barge_ts(row: dict) -> float | None:
    onset = row["clip"]["onset_ms"] or row["clip"]["start_ms"]
    for turn in row.get("turns", []):
        for e in events(turn, "barge_in"):
            if e["data"].get("reason") == "user speech" and onset - 50 <= e["ts_ms"] <= onset + row["clip"]["duration_ms"] + 800:
                return e["ts_ms"]
    return None


def score_false(rows: list[dict]) -> dict:
    """noise / backchannel: nothing here is the user taking the turn."""
    n = fired = wrong = errors = skipped = 0
    resume, gap, detect = [], [], []
    failures = []
    by_kind: dict[str, list[int]] = {}
    for row in rows:
        if "error" in row:
            errors += 1
            continue
        if "skipped" in row:
            skipped += 1
            continue
        n += 1
        outcome = row["outcome"]
        is_wrong = outcome in ("answered", "abandoned")
        fired += outcome != "not_fired"
        wrong += is_wrong
        kind = row["case"].get("kind") or fold(row["case"]["text"]).strip(".")
        tally = by_kind.setdefault(kind, [0, 0])
        tally[0] += is_wrong
        tally[1] += 1
        if row.get("resume_control_ms") and row.get("new_playback_ms"):
            resume.append(row["new_playback_ms"] - row["resume_control_ms"])
        if row.get("emulated_stop_ms") and row.get("new_playback_ms"):
            gap.append(row["new_playback_ms"] - row["emulated_stop_ms"])
        barge = _barge_ts(row)
        if barge is not None and row["clip"]["onset_ms"]:
            detect.append(barge - row["clip"]["onset_ms"])
        if is_wrong:
            turn = _event_turn(row)
            failures.append({"id": row["case"]["id"], "outcome": outcome,
                             "heard_as": user_text(turn) if turn else None})
    return {"false_interruption": rate(wrong, n), "barge_in_fired": rate(fired, n),
            "resume_after_decision_ms": dist(resume), "silence_stop_to_resumed_audio_ms": dist(gap),
            "onset_to_barge_in_ms": dist(detect), "by_kind_wrong": {k: rate(*v) for k, v in sorted(by_kind.items())},
            "errors": errors, "skipped_not_playing": skipped, "failures": failures}


def score_interrupt(rows: list[dict]) -> dict:
    n = missed = answered = lost = leaked = errors = skipped = lost_half = 0
    stop, server, stop_onset_window = [], [], []
    failures = []
    for row in rows:
        if "error" in row:
            errors += 1
            continue
        if "skipped" in row:
            skipped += 1
            continue
        n += 1
        onset = row["clip"]["onset_ms"]
        ok = row["outcome"] == "answered"
        answered += ok
        missed += row["outcome"] == "not_fired"
        leaked += bool(row.get("old_frames_after_reset"))
        if row.get("emulated_stop_ms") and onset:
            value = row["emulated_stop_ms"] - onset
            stop.append(value)
            if row["case"].get("onset_window"):
                stop_onset_window.append(value)
        barge = _barge_ts(row)
        if barge is not None and onset:
            server.append(barge - onset)
        turn = _event_turn(row)
        text = user_text(turn) if turn else ""
        missing = [k for k in row["case"]["keys"] if fold(k) not in fold(text)]
        lost += bool(missing)
        lost_half += bool(lost_to_turn_taking(row["case"], text))
        if not ok or missing:
            failures.append({"id": row["case"]["id"], "outcome": row["outcome"], "heard_as": text, "missing_keys": missing})
    return {"answered_new_request": rate(answered, n), "missed_interruption": rate(missed, n),
            "lost_half": rate(lost_half, n), "lost_words": rate(lost, n), "old_audio_after_reset": rate(leaked, n),
            "onset_to_stop_ms": dist(stop), "onset_to_stop_ms_at_playback_onset": dist(stop_onset_window),
            "onset_to_server_barge_in_ms": dist(server), "errors": errors, "skipped_not_playing": skipped,
            "failures": failures}


def score_echo(rows: list[dict]) -> dict:
    """Only the assistant's own voice came back through the mic."""
    n = hit = 0
    minutes = 0.0
    failures = []
    for row in rows:
        if "error" in row:
            continue
        n += 1
        resets = [t for t in row.get("resets", []) if t <= row["play_start_ms"] + row["listened_ms"]]
        hit += bool(resets)
        minutes += row["listened_ms"] / 60000
        if resets:
            failures.append({"id": row["case"]["id"], "resets_after_play_ms": [round(t - row["play_start_ms"]) for t in resets]})
    return {"answers_cut_by_echo": rate(hit, n), "listened_minutes": round(minutes, 2), "failures": failures}


SCORERS = {"hold": score_hold, "complete": score_complete, "continue": score_continue,
           "backchannel": score_false, "noise": score_false, "interrupt": score_interrupt, "echo": score_echo}


def summarize(run: Path) -> dict:
    families = load(run)
    manifest = json.loads((run / "manifest.json").read_text())
    out: dict[str, Any] = {"run": str(run), "asr_reference": bool(REFERENCE), "manifest": {k: manifest.get(k) for k in (
        "started_at", "finished_at", "stimuli_sha256", "echo_db", "echo_delay_ms", "harness_sha256", "basis")},
        "source_sha256": manifest.get("readyz", {}).get("runtime", {}).get("source_sha256"),
        "config_sha256": manifest.get("readyz", {}).get("runtime", {}).get("config_sha256"),
        "families": {}}
    all_turns: dict[tuple, dict] = {}
    for family, rows in families.items():
        out["families"][family] = SCORERS[family](rows)
        for row in rows:
            for turn in row.get("turns", []):
                all_turns[(turn.get("session_id"), turn["turn_id"])] = turn
    out["unconfirmed_audio"] = {"violations": unconfirmed_audio(list(all_turns.values())), "turns": len(all_turns)}
    f = out["families"]
    gate = {}
    if "hold" in f:
        gate["premature_cut_lt_2pct"] = f["hold"]["premature_confirm"]
    if "noise" in f or "backchannel" in f:
        k = sum(f[x]["false_interruption"]["k"] for x in ("noise", "backchannel") if x in f)
        n = sum(f[x]["false_interruption"]["n"] for x in ("noise", "backchannel") if x in f)
        gate["false_interruption_lt_3pct"] = rate(k, n)
        resume = []
        for x in ("noise", "backchannel"):
            if x in f:
                resume.append(f[x]["resume_after_decision_ms"])
        gate["resume_p95_le_500ms"] = resume
    if "interrupt" in f:
        gate["stop_p95_le_300ms"] = f["interrupt"]["onset_to_stop_ms"]
    gate["user_visible_cut"] = f["hold"]["user_visible_cut"] if "hold" in f else None
    gate["no_lost_half"] = {x: f[x]["lost_half"] for x in ("hold", "continue", "interrupt") if x in f}
    gate["no_unconfirmed_audio"] = len(out["unconfirmed_audio"]["violations"])
    out["gate"] = gate
    return out


def compare(a: dict, b: dict) -> str:
    lines = [f"| chỉ số | {Path(a['run']).name} | {Path(b['run']).name} |", "|---|---:|---:|"]

    def show(x):
        if isinstance(x, dict) and "rate" in x:
            return f"{x['k']}/{x['n']} ({100 * (x['rate'] or 0):.1f}%)"
        if isinstance(x, dict) and "p95" in x:
            return f"p50 {x['p50']} · p95 {x['p95']} · max {x['max']} (n={x['n']})"
        return str(x)

    for family in sorted(set(a["families"]) | set(b["families"])):
        fa, fb = a["families"].get(family, {}), b["families"].get(family, {})
        for key in sorted(set(fa) | set(fb)):
            if key in ("failures", "by_kind_wrong"):
                continue
            lines.append(f"| {family}.{key} | {show(fa.get(key))} | {show(fb.get(key))} |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("runs", nargs="+")
    parser.add_argument("--output")
    parser.add_argument("--compare", action="store_true")
    parser.add_argument("--asr-reference", help="asr_reference_g3.py output: separates ASR errors from cuts")
    args = parser.parse_args()
    if args.asr_reference:
        REFERENCE.update(json.loads(Path(args.asr_reference).read_text())["text"])
    results = [summarize(Path(r)) for r in args.runs]
    if args.compare and len(results) == 2:
        print(compare(*results))
    else:
        for result in results:
            printable = json.loads(json.dumps(result))
            for fam in printable["families"].values():
                fam["failures"] = len(fam.get("failures", []))
            print(json.dumps(printable, ensure_ascii=False, indent=1))
    if args.output:
        Path(args.output).write_text(json.dumps(results[0] if len(results) == 1 else results, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
