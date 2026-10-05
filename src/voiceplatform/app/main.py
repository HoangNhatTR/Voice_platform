"""CLI: serve, demo, doctor."""

from __future__ import annotations

import argparse
import asyncio
import json
import socket
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


_LAN_TLS_DIR = Path(".tls")


def _check_port_free(host: str, port: int) -> None:
    """Thử bind TRƯỚC khi nạp model.

    uvicorn chạy lifespan — tức là nạp ASR và dựng tiến trình con TTS, khoảng
    40 giây trên máy này — RỒI mới bind cổng. Nên một cổng trùng bắt người ta
    trả đủ giá nạp model rồi mới báo lỗi, và thứ báo ra là `address already in
    use` chứ không phải "server cũ của bạn vẫn đang chạy". Đã cắn hai lần.
    """
    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # Cùng cờ với uvicorn, nếu không một cổng đang TIME_WAIT sẽ báo động giả.
    probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        probe.bind((host if host != "0.0.0.0" else "", port))
    except OSError as exc:
        raise VoicePlatformError(
            f"cổng {port} đang bị chiếm ({exc.strerror}). Xem ai giữ nó:\n"
            f"  ss -ltnp | grep :{port}\n"
            "rồi `kill <pid>`. Một server cũ còn chạy sẽ phục vụ code CŨ — "
            "đối chiếu bằng /healthz trước khi tin những gì nó trả về."
        ) from exc
    finally:
        probe.close()


def cmd_serve(args: argparse.Namespace) -> int:
    from .server import run

    config = _load(_config_of(args))
    if args.port:
        config.server.port = args.port
    if args.host:
        config.server.host = args.host
    if args.lan:
        # Everything --lan turns on is a consequence of the page leaving this
        # machine, not a preference: reachable address, a secure context so the
        # microphone exists at all, and the session list off the open network.
        config.server.host = args.host or "0.0.0.0"
        config.server.private_introspection = True
        args.cert = args.cert or str(_LAN_TLS_DIR / "server.crt")
        args.key = args.key or str(_LAN_TLS_DIR / "server.key")
    if args.cert or args.key:
        if not (args.cert and args.key):
            raise VoicePlatformError("--cert và --key phải đi cùng nhau")
        for path in (args.cert, args.key):
            if not Path(path).exists():
                raise VoicePlatformError(
                    f"không có {path}. Sinh chứng chỉ trước: ./scripts/make-lan-cert.sh"
                )
        config.server.ssl_certfile = args.cert
        config.server.ssl_keyfile = args.key
    elif config.server.host not in {"127.0.0.1", "localhost", "::1"}:
        # Not fatal — a tester can still type — but it is the single most
        # common reason a LAN demo has no voice, so it is said out loud.
        print(
            f"cảnh báo: phục vụ {config.server.host} qua HTTP thuần. Trình duyệt "
            "ở máy khác sẽ KHÔNG mở được micro (getUserMedia cần secure context). "
            "Dùng ./scripts/lan.sh để chạy kèm TLS.",
            file=sys.stderr,
        )
    # Sau khi chốt host/port, TRƯỚC khi run() kéo theo cả việc nạp model.
    _check_port_free(config.server.host, config.server.port)
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
        async def dependencies():
            try:
                return await plane.check_dependencies(config.server.readiness_timeout_s)
            finally:
                await plane.close()
        report["dependencies"] = asyncio.run(dependencies())
        report["models_ok"] = all(v["ok"] for v in report["dependencies"].values())
        report["inference_tested"] = False
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
    serve.add_argument("--cert", help="chứng chỉ TLS (bắt buộc nếu muốn có micro ngoài localhost)")
    serve.add_argument("--key", help="khoá riêng TLS")
    serve.add_argument(
        "--lan", action="store_true",
        help="mở cho máy khác: bind 0.0.0.0, bật TLS ở .tls/, khoá /sessions và /config về loopback",
    )
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
