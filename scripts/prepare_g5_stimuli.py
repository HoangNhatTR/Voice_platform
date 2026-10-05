"""Separate frozen G4 speech clips into direct and lookup G5 workloads.

Street-location questions need lookup even if the listening pack categorizes
them as pronunciation/name cases. They must not inflate direct success counts.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def prepare(folder: Path, output: Path, listening_output: Path | None = None) -> dict:
    stimuli = json.loads((folder / "stimuli.json").read_text(encoding="utf-8"))["direct"]
    cases = json.loads((folder / "cases.json").read_text(encoding="utf-8"))
    if len(stimuli) != len(cases):
        raise ValueError("stimuli and cases length differ")
    direct, lookups, listening = [], [], []
    for stimulus, case in zip(stimuli, cases):
        if stimulus.get("reference") != case.get("user_text"):
            raise ValueError(f"stimulus order differs at {case.get('item_id')}")
        if case.get("category") in ("search_found", "name"):
            routed = {**stimulus, "search": True}
            lookups.append(routed)
        else:
            routed = {**stimulus, "search": False}
            direct.append(routed)
        listening.append(routed)
    if not direct or not lookups:
        raise ValueError("need both direct and lookup questions")
    payload = {"direct": direct, "search": lookups}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False) + "\n", encoding="utf-8")
    if listening_output is not None:
        listening_output.parent.mkdir(parents=True, exist_ok=True)
        listening_output.write_text(json.dumps({"direct": listening}, ensure_ascii=False) + "\n", encoding="utf-8")
    return {"direct": len(direct), "search": len(lookups)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--listening-output", type=Path)
    args = parser.parse_args()
    print(json.dumps(prepare(args.folder, args.output, args.listening_output)))


if __name__ == "__main__":
    main()
