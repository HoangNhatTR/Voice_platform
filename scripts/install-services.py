#!/usr/bin/env python3
"""Install only voice-platform units; --start also enables and starts the stack."""
import argparse
import os
import subprocess
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--start", action="store_true")
parser.add_argument("--api-cpus", default="none",
                    help="CPUAffinity for the API: 'none' (default), 'auto' = the fastest cores, or a list like '5-9 15-19'")
args = parser.parse_args()


def fastest_cpus():
    """CPUs whose max clock is the highest: the X925 cluster on the GB10.

    Only for a host whose fast cores are kept for this service. Pinning does
    not reserve them: on the shared host (02/10/2026, load 12-17, other users'
    processes at 100-150% on cores 5, 8, 15) ZeroTTS measured RTF 0.85
    unpinned vs 0.90 pinned, 8 interleaved runs each — the scheduler moving
    threads to idle slow cores beats being confined to busy fast ones. One
    earlier single sample (0.80 vs 0.63) said the opposite; do not trust one.
    """
    freqs = {}
    for path in Path("/sys/devices/system/cpu").glob("cpu[0-9]*/cpufreq/cpuinfo_max_freq"):
        try:
            freqs[int(path.parent.parent.name[3:])] = int(path.read_text())
        except (OSError, ValueError):
            pass
    if not freqs or len(set(freqs.values())) < 2:
        return ""                       # one cluster: nothing to choose
    top = max(freqs.values())
    return " ".join(str(cpu) for cpu in sorted(freqs) if freqs[cpu] == top)


api_cpus = fastest_cpus() if args.api_cpus == "auto" else ("" if args.api_cpus == "none" else args.api_cpus)
root = Path(__file__).resolve().parents[1]
unit_dir = Path.home() / ".config/systemd/user"
unit_dir.mkdir(parents=True, exist_ok=True)
# systemd quotes are not shell quotes; paths are passed as one argument.
def quote(path):
    return '"' + str(path).replace("\\", "\\\\").replace('"', '\\"').replace("%", "%%") + '"'
for role in ("llm", "api"):
    dependency = "Wants=voice-platform-llm.service\nAfter=voice-platform-llm.service\n" if role == "api" else ""
    affinity = f"CPUAffinity={api_cpus}\n" if role == "api" and api_cpus else ""
    unit = f"""[Unit]
Description=Voice platform {role}
{dependency}StartLimitIntervalSec=600
StartLimitBurst=5

[Service]
Type=simple
WorkingDirectory={str(root).replace(chr(32), chr(92) + "x20").replace("%", "%%")}
ExecStart={quote(root / 'scripts/start-local.sh')} {role}
Environment=VOICEPLATFORM_PYTHON=/home/ai01/AIHoang/speech2speech/.venv/bin/python
Environment=VOICEPLATFORM_S2S_ROOT=/home/ai01/AIHoang/speech2speech
Restart=on-failure
RestartSec=5
TimeoutStopSec=60
KillMode=control-group
{affinity}
[Install]
WantedBy=default.target
"""
    destination = unit_dir / f"voice-platform-{role}.service"
    if destination.exists() and destination.read_text() != unit:
        destination.with_suffix(".service.previous").write_text(destination.read_text())
    destination.write_text(unit)
    print(destination)
subprocess.run(["systemctl", "--user", "daemon-reload"], check=True)
if args.start:
    subprocess.run(["systemctl", "--user", "enable", "--now", "voice-platform-llm.service", "voice-platform-api.service"], check=True)
