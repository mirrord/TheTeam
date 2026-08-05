"""Tests for the talos CLI --trace-flowcharts / --trace-flowcharts-path flags."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from talos.__main__ import _build_parser, _resolve_trace_path, main
from talos.config import TalosConfig


class TestArgParsing:
    def test_trace_flowcharts_defaults_off(self):
        args = _build_parser().parse_args(["shell"])
        assert args.trace_flowcharts is False
        assert args.trace_flowcharts_path is None

    def test_trace_flowcharts_flag_does_not_swallow_subcommand(self):
        """Regression: a bare boolean flag must not consume 'shell' as a value."""
        args = _build_parser().parse_args(["--trace-flowcharts", "shell"])
        assert args.trace_flowcharts is True
        assert args.interface == "shell"

    def test_trace_flowcharts_path_override(self):
        args = _build_parser().parse_args(
            ["--trace-flowcharts", "--trace-flowcharts-path", "C:/tmp/x.log", "shell"]
        )
        assert args.trace_flowcharts is True
        assert args.trace_flowcharts_path == Path("C:/tmp/x.log")
        assert args.interface == "shell"

    def test_trace_flowcharts_path_without_toggle_is_harmless(self):
        args = _build_parser().parse_args(
            ["--trace-flowcharts-path", "C:/tmp/x.log", "shell"]
        )
        assert args.trace_flowcharts is False
        assert args.trace_flowcharts_path == Path("C:/tmp/x.log")


class TestResolveTracePath:
    def test_explicit_path_used_as_is(self, tmp_path):
        explicit = tmp_path / "custom.log"
        result = _resolve_trace_path(explicit, tmp_path / "config.yaml")
        assert result == explicit

    def test_default_path_under_config_dir_traces(self, tmp_path):
        config_path = tmp_path / "config.yaml"
        result = _resolve_trace_path(None, config_path)
        assert result.parent == tmp_path / "traces"
        assert result.name.startswith("flowchart-trace-")
        assert result.suffix == ".log"


class TestMainWiring:
    @patch("talos.__main__.build_agent")
    @patch("talos.__main__.ensure_config")
    @patch("pithos.flowchart.enable_global_trace")
    def test_main_enables_trace_before_building_agent(
        self, mock_enable, mock_ensure_config, mock_build_agent, tmp_path
    ):
        config_path = tmp_path / "config.yaml"
        mock_ensure_config.return_value = (TalosConfig(), config_path)
        mock_build_agent.return_value = MagicMock()

        with patch("talos.interfaces.shell.ShellInterface") as mock_shell_cls:
            rc = main(
                [
                    "--config",
                    str(config_path),
                    "--trace-flowcharts",
                    "shell",
                ]
            )

        assert rc == 0
        mock_enable.assert_called_once()
        (called_path,), _ = mock_enable.call_args
        assert called_path.parent == tmp_path / "traces"
        mock_shell_cls.assert_called_once()

    @patch("talos.__main__.build_agent")
    @patch("talos.__main__.ensure_config")
    @patch("pithos.flowchart.enable_global_trace")
    def test_main_uses_explicit_trace_path(
        self, mock_enable, mock_ensure_config, mock_build_agent, tmp_path
    ):
        config_path = tmp_path / "config.yaml"
        custom_trace = tmp_path / "custom-trace.log"
        mock_ensure_config.return_value = (TalosConfig(), config_path)
        mock_build_agent.return_value = MagicMock()

        with patch("talos.interfaces.shell.ShellInterface"):
            main(
                [
                    "--config",
                    str(config_path),
                    "--trace-flowcharts",
                    "--trace-flowcharts-path",
                    str(custom_trace),
                    "shell",
                ]
            )

        mock_enable.assert_called_once_with(custom_trace)

    @patch("talos.__main__.build_agent")
    @patch("talos.__main__.ensure_config")
    @patch("pithos.flowchart.enable_global_trace")
    def test_main_does_not_enable_trace_when_flag_absent(
        self, mock_enable, mock_ensure_config, mock_build_agent, tmp_path
    ):
        config_path = tmp_path / "config.yaml"
        mock_ensure_config.return_value = (TalosConfig(), config_path)
        mock_build_agent.return_value = MagicMock()

        with patch("talos.interfaces.shell.ShellInterface"):
            main(["--config", str(config_path), "shell"])

        mock_enable.assert_not_called()
