"""The CLI must read the config wherever it is typed."""

from __future__ import annotations

from voiceplatform.app.main import _config_of, build_parser


def test_config_before_the_subcommand():
    args = build_parser().parse_args(["--config", "configs/local-cpu.yaml", "serve"])
    assert _config_of(args) == "configs/local-cpu.yaml"


def test_config_after_the_subcommand():
    args = build_parser().parse_args(["serve", "--config", "configs/local-cpu.yaml"])
    assert _config_of(args) == "configs/local-cpu.yaml"


def test_no_config_means_defaults():
    assert _config_of(build_parser().parse_args(["serve"])) is None


def test_demo_and_doctor_take_it_too():
    for command in ("demo", "doctor"):
        args = build_parser().parse_args([command, "--config", "x.yaml"])
        assert _config_of(args) == "x.yaml"
