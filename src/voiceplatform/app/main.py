"""CLI: serve, demo, doctor."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from ..core.config import Config
from ..core.errors import VoicePlatformError
from ..observability.logging import setup_logging


def _load(path: str | None) -> Config:
    return Config.load(path) if path else Config()


def _config_of(args: argparse.Namespace) -> str | None:
    """--config may be absent from the namespace; see build_parser()."""
    return getattr(args, "config", None)


def cmd_serve(args: argparse.Namespace) -> int:
    from .server import run

    config = _load(_config_of(args))
    if args.port:
        config.server.port = args.port
    if args.host:
        config.server.host = args.host
    run(config)
    return 0


def cmd_demo(args: argparse.Namespace) -> int:
    from .simulate import demo

    config = _load(_config_of(args))
    stats = asyncio.run(demo(config, realtime=not args.fast))
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    return 0


def cmd_doctor(args: argparse.Namespace) -> int:
    from ..models.registry import ModelPlane

    config = _load(_config_of(args))
    report: dict[str, object] = {"config": _config_of(args) or "(defaults)"}
    try:
        plane = ModelPlane(config.models, output_sample_rate=config.audio.output_sample_rate)
        report["models"] = plane.describe()
        report["models_ok"] = True
    except VoicePlatformError as exc:
        report["models_ok"] = False
        report["models_error"] = str(exc)
    try:
        from ..tasks.registry import build_registry

        report["tools"] = build_registry(config.tasks.tools).names()
    except VoicePlatformError as exc:
        report["tools_error"] = str(exc)
    report["web_client"] = Path(config.server.web_dir).is_dir()
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report.get("models_ok") else 1


def build_parser() -> argparse.ArgumentParser:
    # --config is accepted on both sides of the subcommand. Argparse puts
    # global options before it, which is not where anyone types them.
    common = argparse.ArgumentParser(add_help=False)
    # SUPPRESS matters: a shared option through parents= otherwise writes its
    # default into the namespace *after* the main parser ran, so
    # `--config x serve` silently became `--config None` and the whole run fell
    # back to the mock stack while claiming to have read the file.
    common.add_argument(
        "--config", "-c", default=argparse.SUPPRESS, help="YAML config path"
    )

    parser = argparse.ArgumentParser(prog="voiceplatform", parents=[common])
    sub = parser.add_subparsers(dest="command", required=True)

    serve = sub.add_parser("serve", help="run the realtime server", parents=[common])
    serve.add_argument("--host")
    serve.add_argument("--port", type=int)
    serve.set_defaults(func=cmd_serve)

    demo = sub.add_parser("demo", help="run a scripted session with no browser", parents=[common])
    # Audio is fed at wall-clock speed by default. --fast runs the same
    # script as quickly as the machine allows: turn-taking still behaves
    # identically (it is timed on the audio clock), but every latency in the
    # report is then measured against a wall clock that barely moved.
    demo.add_argument("--fast", action="store_true", help="feed audio as fast as possible")
    demo.set_defaults(func=cmd_demo)

    doctor = sub.add_parser("doctor", help="check config and engines", parents=[common])
    doctor.set_defaults(func=cmd_doctor)
    return parser


def main(argv: list[str] | None = None) -> int:
    setup_logging()
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args))
    except VoicePlatformError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
