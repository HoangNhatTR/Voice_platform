"""Repeat real browser/ASR/LLM/TTS turns and audit cleanup between cycles.

Run against an isolated G4 candidate server. This harness does not simulate a
human microphone or certify LAN acoustics. For G5, run separate 8h and 24h
artifacts; a short run is only a harness smoke test.
"""

from __future__ import annotations

import argparse
import json
import ssl
import subprocess
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


def fetch(base: str, path: str) -> dict:
    context = ssl.create_default_context()
    if base.startswith("https://127.0.0.1:") or base.startswith("https://localhost:"):
        context.check_hostname = False
        context.verify_mode = ssl.CERT_NONE
    with urllib.request.urlopen(base + path, context=context, timeout=10) as response:
        return json.load(response)


def idle_checks(sessions: dict, metrics: dict, baseline: dict | None = None) -> dict[str, bool]:
    gauges = metrics.get("gauges", {})
    checks = {"sessions_released": not sessions and gauges.get("sessions") == 0}
    for name in ("asr", "llm", "tts", "search", "tool"):
        limiter = gauges.get(name)
        if limiter is not None:
            checks[f"{name}_queue_empty"] = limiter.get("active", 0) == 0 and limiter.get("waiting", 0) == 0
    if baseline is not None:
        for name in ("asr_workers", "tts_workers"):
            if name in gauges and name in baseline:
                checks[f"{name}_returned"] = gauges[name] <= baseline[name]
    return checks


def run(base: str, stimuli: Path, output: Path, seconds: int, interval_s: int,
        sessions: int, rounds: int = 11, max_cycles: int = 0) -> dict:
    if seconds <= 0 or interval_s < 0 or sessions not in (1, 3) or rounds <= 0:
        raise ValueError("seconds > 0, interval >= 0, rounds > 0 and sessions 1 or 3 are required")
    stimuli = stimuli.resolve()
    output = output.resolve()
    if not stimuli.is_file():
        raise FileNotFoundError(stimuli)
    output.mkdir(parents=True, exist_ok=False)
    initial_health = fetch(base, "/readyz")
    if not initial_health.get("ok") or fetch(base, "/sessions"):
        raise RuntimeError("isolated server must be ready and have no active sessions")
    initial_metrics = fetch(base, "/metrics")
    baseline = initial_metrics.get("gauges", {})
    started = time.monotonic()
    deadline = started + seconds
    report = {"schema": 1, "started_at": datetime.now(timezone.utc).isoformat(),
              "requested_seconds": seconds, "base": base, "sessions": sessions,
              "rounds_per_cycle": rounds, "interval_s": interval_s,
              "source_sha256": initial_health.get("runtime", {}).get("source_sha256"),
              "config_sha256": initial_health.get("runtime", {}).get("config_sha256"),
              "stimuli": str(stimuli), "baseline_gauges": baseline, "cycles": []}
    try:
        while time.monotonic() < deadline and (not max_cycles or len(report["cycles"]) < max_cycles):
            index = len(report["cycles"]) + 1
            cycle_dir = output / f"cycle-{index:04d}"
            command = ["node", "scripts/benchmark_g1.cjs", "--base", base, "--sessions", str(sessions),
                       "--rounds", str(rounds), "--cases", "direct", "--warmup", "no",
                       "--stimuli", str(stimuli), "--output", str(cycle_dir)]
            task = subprocess.run(command, capture_output=True, text=True, timeout=240,
                                  cwd=Path(__file__).resolve().parents[1])
            (output / f"cycle-{index:04d}.log").write_text(task.stdout + "\n" + task.stderr)
            if task.returncode:
                report["cycles"].append({"index": index, "benchmark_exit": task.returncode,
                                         "checks": {"benchmark_success": False}})
                break
            # The browser must close the WebSocket; give the server a bounded
            # grace period to join cancellation/tool/native workers.
            for _ in range(30):
                active = fetch(base, "/sessions")
                if not active:
                    break
                time.sleep(1)
            metrics = fetch(base, "/metrics")
            checks = {"benchmark_success": True, **idle_checks(active, metrics, baseline)}
            ready = fetch(base, "/readyz")
            checks["still_ready"] = bool(ready.get("ok"))
            checks["same_source"] = ready.get("runtime", {}).get("source_sha256") == report["source_sha256"]
            checks["same_config"] = ready.get("runtime", {}).get("config_sha256") == report["config_sha256"]
            report["cycles"].append({"index": index, "benchmark_exit": task.returncode,
                                     "checks": checks, "gauges_after_close": metrics.get("gauges", {})})
            if not all(checks.values()):
                break
            next_at = started + index * interval_s
            if interval_s:
                time.sleep(max(0, min(deadline, next_at) - time.monotonic()))
    finally:
        report["finished_at"] = datetime.now(timezone.utc).isoformat()
        report["elapsed_seconds"] = round(time.monotonic() - started, 3)
        report["passed_cycles"] = sum(all(c["checks"].values()) for c in report["cycles"])
        report["all_checks_passed"] = bool(report["cycles"]) and report["passed_cycles"] == len(report["cycles"])
        report["duration_met"] = report["elapsed_seconds"] >= seconds
        idle = [cycle.get("gauges_after_close", {}) for cycle in report["cycles"] if cycle.get("gauges_after_close")]
        if idle:
            first, last = idle[0], idle[-1]
            report["idle_resource_change"] = {
                "rss_mb": round(last.get("process_rss_mb", 0) - first.get("process_rss_mb", 0), 3),
                "swap_mb": round(last.get("process_swap_mb", 0) - first.get("process_swap_mb", 0), 3),
                "threads": last.get("process_threads", 0) - first.get("process_threads", 0),
            }
        (output / "soak-report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True)
    parser.add_argument("--stimuli", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seconds", required=True, type=int)
    parser.add_argument("--interval-s", type=int, default=300)
    parser.add_argument("--sessions", type=int, default=3)
    parser.add_argument("--rounds", type=int, default=11)
    parser.add_argument("--max-cycles", type=int, default=0)
    args = parser.parse_args()
    result = run(args.base, args.stimuli, args.output, args.seconds, args.interval_s,
                 args.sessions, args.rounds, args.max_cycles)
    print(json.dumps({key: result[key] for key in ("elapsed_seconds", "passed_cycles", "all_checks_passed", "duration_met")}, indent=2))
    raise SystemExit(0 if result["all_checks_passed"] and result["duration_met"] else 1)


if __name__ == "__main__":
    main()
