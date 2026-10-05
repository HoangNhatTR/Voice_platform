"""Fail-closed G3/G4/G5 release decision from independently produced evidence.

Missing human, listening, LAN or soak evidence is BLOCKED. A failing measured
threshold is FAIL. This script never labels a synthetic proxy as a human gate.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

try:
    from .g4_listening import summarize as listening_summary
    from .summarize_g1 import summarize as load_summary
except ImportError:  # direct `python scripts/g5_release_gate.py`
    from g4_listening import summarize as listening_summary
    from summarize_g1 import summarize as load_summary


def check(result: dict, name: str, passed: bool | None, detail: object) -> None:
    result["checks"][name] = {"status": "blocked" if passed is None else "pass" if passed else "fail",
                              "detail": detail}


def evaluate(*, g3: Path | None = None, labels: Path | None = None,
             g3_stimuli: Path | None = None,
             listening: Path | None = None, load_one: Path | None = None,
             load_three: Path | None = None, soak_8h: Path | None = None,
             soak_24h: Path | None = None, fault: Path | None = None,
             lan: Path | None = None) -> dict:
    result: dict = {"schema": 1, "checks": {}}
    source_hashes: dict[str, str] = {}
    config_hashes: dict[str, str] = {}
    load_stimuli_hashes: dict[str, str] = {}
    label_digest = None
    if labels and labels.is_file():
        label_digest = hashlib.sha256(labels.read_bytes()).hexdigest()
        with labels.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        speakers = {row.get("speaker_id", "").strip() for row in rows}
        speech_speakers = {row.get("speaker_id", "").strip() for row in rows
                           if row.get("family") in ("hold", "continue", "complete")}
        verified = all(row.get("human_verified", "").strip().lower() == "yes" for row in rows)
        count = {family: sum(row.get("family") == family for row in rows)
                 for family in ("hold", "continue", "complete", "backchannel", "noise", "interrupt")}
        speech_count = count["hold"] + count["continue"] + count["complete"]
        check(result, "g3_human_corpus", speech_count >= 500 and len(speech_speakers - {""}) >= 10 and verified
              and count["hold"] + count["continue"] >= 100,
              {"speech_clips": speech_count, "all_clips": len(rows), "speech_speakers": len(speech_speakers - {""}),
               "all_speakers": len(speakers - {""}),
               "families": count, "human_verified": verified})
    else:
        check(result, "g3_human_corpus", None, "human labels CSV missing")

    if g3_stimuli and g3_stimuli.is_file():
        stimuli = json.loads(g3_stimuli.read_text())
        same_labels = label_digest is not None and stimuli.get("source_labels_sha256") == label_digest
        payload = {key: value for key, value in stimuli.items() if key != "sha256"}
        valid_digest = stimuli.get("sha256") == hashlib.sha256(
            json.dumps(payload, ensure_ascii=False, sort_keys=True).encode()
        ).hexdigest()
        check(result, "g3_labeled_wav_integrity", same_labels and "human microphone WAVs" in stimuli.get("basis", "")
              and valid_digest and bool(stimuli.get("speakers_heldout")),
              {"same_labels": same_labels, "valid_digest": valid_digest, "sha256": stimuli.get("sha256"),
               "heldout_speakers": len(stimuli.get("speakers_heldout", []))})
    else:
        stimuli = None
        check(result, "g3_labeled_wav_integrity", None, "validated heldout WAV stimuli missing")

    if g3 and g3.is_file():
        data = json.loads(g3.read_text())
        source_hashes["g3"] = data.get("source_sha256") or ""
        config_hashes["g3"] = data.get("config_sha256") or ""
        families = data.get("families", {})
        basis = data.get("manifest", {}).get("basis", "")
        check(result, "g3_same_heldout_wavs", data.get("manifest", {}).get("stimuli_sha256") == stimuli.get("sha256")
              if stimuli else None,
              {"run": data.get("manifest", {}).get("stimuli_sha256"),
               "labels": stimuli.get("sha256") if stimuli else None})
        check(result, "g3_human_basis", "human microphone WAVs" in basis and bool(data.get("asr_reference")),
              {"basis": basis, "asr_reference": data.get("asr_reference")})
        hold = families.get("hold", {})
        rate = hold.get("premature_confirm", {})
        check(result, "g3_turn_cut", rate.get("n", 0) >= 100 and rate.get("rate") is not None
              and rate["rate"] < 0.02, rate)
        for family in ("hold", "continue", "interrupt"):
            lost = families.get(family, {}).get("lost_half")
            check(result, f"g3_{family}_no_lost_half", lost is not None and lost.get("n", 0) > 0
                  and lost.get("k") == 0, lost or "family missing")
        false = data.get("gate", {}).get("false_interruption_lt_3pct")
        check(result, "g3_false_interrupt", false is not None and false.get("n", 0) >= 100
              and false.get("rate") is not None and false["rate"] < 0.03, false or "noise/backchannel missing")
        stop = data.get("gate", {}).get("stop_p95_le_300ms")
        check(result, "g3_stop", stop is not None and stop.get("n", 0) >= 100
              and stop.get("p95", 1e9) <= 300, stop or "interrupt family missing")
        resumes = data.get("gate", {}).get("resume_p95_le_500ms", [])
        check(result, "g3_resume", bool(resumes) and sum(v.get("n", 0) for v in resumes) >= 100
              and all(v.get("p95", 1e9) <= 500 for v in resumes if v.get("n", 0)), resumes)
        leaked = data.get("unconfirmed_audio", {}).get("violations", [])
        check(result, "g3_no_unconfirmed_audio", not leaked and data.get("unconfirmed_audio", {}).get("turns", 0) > 0,
              {"violations": len(leaked), "turns": data.get("unconfirmed_audio", {}).get("turns", 0)})
    else:
        for name in ("same_heldout_wavs", "human_basis", "turn_cut", "hold_no_lost_half", "continue_no_lost_half",
                     "interrupt_no_lost_half", "false_interrupt", "stop", "resume", "no_unconfirmed_audio"):
            check(result, f"g3_{name}", None, "G3 summary missing")

    if listening and listening.is_dir():
        provenance_file = listening / "provenance.json"
        if provenance_file.is_file():
            provenance = json.loads(provenance_file.read_text())
            source_hashes["listening"] = provenance.get("candidate_source_sha256") or ""
            config_hashes["listening"] = provenance.get("candidate_config_sha256") or ""
            check(result, "g4_current_listening_pack", len(provenance.get("clips", [])) >= 12
                  and all(row.get("candidate_sha256") and row.get("baseline_sha256") for row in provenance.get("clips", [])),
                  {"clips": len(provenance.get("clips", [])), "source": source_hashes["listening"]})
        else:
            check(result, "g4_current_listening_pack", None, "listening pack provenance missing")
        try:
            scores = listening_summary(listening)
        except (OSError, ValueError, KeyError, ZeroDivisionError) as exc:
            check(result, "g4_blind_listening", None, str(exc))
        else:
            check(result, "g4_blind_listening", scores["listening_threshold_met"], scores)
    else:
        check(result, "g4_current_listening_pack", None, "fresh listening pack missing")
        check(result, "g4_blind_listening", None, "five completed score sheets missing")

    for name, directory, expected_load in (("one", load_one, 1), ("three", load_three, 3)):
        if directory and (directory / "manifest.json").is_file():
            summary = load_summary(directory)
            source_hashes[name] = summary["manifest"]["health"]["runtime"].get("source_sha256", "")
            config_hashes[name] = summary["manifest"]["health"]["runtime"].get("config_sha256", "")
            load_stimuli_hashes[name] = summary["manifest"].get("stimuli_sha256", "")
            direct = summary["by_case"].get("direct", {})
            attempted, valid = direct.get("attempted", 0), direct.get("valid", 0)
            check(result, f"load_{name}_sample_success", summary["manifest"].get("load") == expected_load
                  and attempted >= 200 and valid / attempted >= 0.99 if attempted else False,
                  {"load": summary["manifest"].get("load"), "attempted": attempted, "valid": valid})
            playback = direct.get("synthetic_signal_last_speech_to_content_playback_ms", {})
            check(result, f"load_{name}_playback", playback.get("n", 0) >= 200
                  and playback.get("p95", 1e9) < 1000, playback)
            ttft = direct.get("stages", {}).get("llm_queued_to_content_delta_ms", {})
            check(result, f"load_{name}_llm_ttft", ttft.get("n", 0) >= 200
                  and ttft.get("p95", 1e9) < 1000, ttft)
            check(result, f"load_{name}_cleanup", not summary["manifest"].get("sessions_after_close"),
                  summary["manifest"].get("sessions_after_close"))
        else:
            for suffix in ("sample_success", "playback", "llm_ttft", "cleanup"):
                check(result, f"load_{name}_{suffix}", None, "200-turn browser benchmark missing")

    for name, path, hours in (("8h", soak_8h, 8), ("24h", soak_24h, 24)):
        if path and path.is_file():
            data = json.loads(path.read_text())
            source_hashes[name] = data.get("source_sha256") or ""
            config_hashes[name] = data.get("config_sha256") or ""
            resource = data.get("idle_resource_change", {})
            minimum_cycles = int(hours * 3600 / 300 * 0.9)
            check(result, f"soak_{name}", data.get("elapsed_seconds", 0) >= hours * 3600
                  and data.get("sessions") == 3 and data.get("all_checks_passed")
                  and data.get("duration_met") and data.get("interval_s", 1e9) <= 300
                  and data.get("rounds_per_cycle", 0) >= 10 and len(data.get("cycles", [])) >= minimum_cycles
                  and abs(resource.get("rss_mb", 1e9)) <= 256
                  and abs(resource.get("swap_mb", 1e9)) <= 64
                  and abs(resource.get("threads", 1e9)) <= 2,
                  {"elapsed_seconds": data.get("elapsed_seconds"), "cycles": len(data.get("cycles", [])),
                   "minimum_cycles": minimum_cycles, "all_checks_passed": data.get("all_checks_passed"),
                   "idle_resource_change": resource})
        else:
            check(result, f"soak_{name}", None, "soak artifact missing")

    if fault and fault.is_file():
        data = json.loads(fault.read_text())
        source_hashes["fault"] = data.get("source_sha256") or ""
        config_hashes["fault"] = data.get("config_sha256") or ""
        check(result, "fault_probe", data.get("passed") is True and all(data.get("checks", {}).values()),
              data.get("checks", {}))
    else:
        check(result, "fault_probe", None, "live disconnect/interrupt/admission probe missing")

    if lan and lan.is_file():
        device = json.loads(lan.read_text())
        required = ("microphone", "aec", "speaker", "headphones", "browser_playback", "dependency_failure")
        check(result, "lan_and_fault_matrix", all(device.get(key) is True for key in required)
              and bool(device.get("device_name")) and bool(device.get("tested_at")), device)
    else:
        check(result, "lan_and_fault_matrix", None, "real LAN browser/device and fault report missing")
    hashes = {value for value in source_hashes.values() if value}
    check(result, "same_load_stimuli", (len(set(load_stimuli_hashes.values())) == 1
          and all(load_stimuli_hashes.values())) if len(load_stimuli_hashes) == 2 else None,
          load_stimuli_hashes)
    same_source = False if len(hashes) > 1 else (bool(hashes) and all(source_hashes.values())) if len(source_hashes) >= 7 else None
    check(result, "same_source", same_source, source_hashes)
    configs = {value for value in config_hashes.values() if value}
    same_config = False if len(configs) > 1 else (bool(configs) and all(config_hashes.values())) if len(config_hashes) >= 7 else None
    check(result, "same_config", same_config, config_hashes)
    states = [item["status"] for item in result["checks"].values()]
    result["decision"] = "fail" if "fail" in states else "blocked" if "blocked" in states else "pass"
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("g3", "labels", "g3_stimuli", "listening", "load_one", "load_three", "soak_8h", "soak_24h", "fault", "lan"):
        parser.add_argument("--" + key.replace("_", "-"), type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    inputs = {key: getattr(args, key) for key in ("g3", "labels", "g3_stimuli", "listening", "load_one", "load_three",
                                                  "soak_8h", "soak_24h", "fault", "lan")}
    result = evaluate(**inputs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"decision": result["decision"],
                      "checks": {name: item["status"] for name, item in result["checks"].items()}}, indent=2))
    raise SystemExit(0 if result["decision"] == "pass" else 1)


if __name__ == "__main__":
    main()
