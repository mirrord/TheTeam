"""Tests for the per-tool ``no_timeout`` configuration item."""

import subprocess
import threading
from unittest.mock import MagicMock, Mock, patch

import pytest

from pithos.config_manager import ConfigManager
from pithos.tools.cli_provider import CLIToolProvider
from pithos.tools.flowchart_tool import FlowchartToolExecutor
from pithos.tools.models import ToolMetadata
from pithos.tools.provider import parse_no_timeout

# ---------------------------------------------------------------------------
# parse_no_timeout
# ---------------------------------------------------------------------------


class TestParseNoTimeout:
    def test_missing_key_is_empty(self):
        assert parse_no_timeout({}) == frozenset()

    def test_none_value_is_empty(self):
        assert parse_no_timeout({"no_timeout": None}) == frozenset()

    def test_list_of_names(self):
        assert parse_no_timeout({"no_timeout": ["python", "flowchart:x"]}) == {
            "python",
            "flowchart:x",
        }

    def test_single_string_is_accepted(self):
        assert parse_no_timeout({"no_timeout": "python"}) == {"python"}

    def test_blank_entries_are_ignored(self):
        assert parse_no_timeout({"no_timeout": ["", "  ", "python "]}) == {"python"}

    @pytest.mark.parametrize("bad", [True, 5, {"python": True}])
    def test_malformed_value_is_empty(self, bad):
        assert parse_no_timeout({"no_timeout": bad}) == frozenset()

    def test_non_dict_config_is_empty(self):
        assert parse_no_timeout(None) == frozenset()
        assert parse_no_timeout(Mock()) == frozenset()


# ---------------------------------------------------------------------------
# CLIToolProvider
# ---------------------------------------------------------------------------


def _make_cli(no_timeout=None) -> CLIToolProvider:
    config = {"timeout": 7}
    if no_timeout is not None:
        config["no_timeout"] = no_timeout
    provider = CLIToolProvider(config=config)
    for name in ("python", "echo"):
        provider._discovered[name] = ToolMetadata(
            name=name,
            path=f"/usr/bin/{name}",
            description=name,
            platform="unix",
            source="system",
        )
    return provider


def _completed(returncode=0):
    return subprocess.CompletedProcess(args=[], returncode=returncode, stdout="ok", stderr="")


class TestCLINoTimeout:
    def test_default_timeout_applies_without_config(self):
        provider = _make_cli()
        assert provider.get_timeout("python") == 7

    def test_listed_tool_has_no_timeout(self):
        provider = _make_cli(["python"])
        assert provider.get_timeout("python") is None
        assert provider.get_timeout("echo") == 7

    def test_subprocess_called_with_none_for_listed_tool(self):
        provider = _make_cli(["python"])
        with patch(
            "pithos.tools.cli_provider.subprocess.run", return_value=_completed()
        ) as run:
            result = provider.execute("python --version")
        assert result.success is True
        assert run.call_args.kwargs["timeout"] is None

    def test_subprocess_called_with_timeout_for_unlisted_tool(self):
        provider = _make_cli(["python"])
        with patch(
            "pithos.tools.cli_provider.subprocess.run", return_value=_completed()
        ) as run:
            provider.execute("echo hi")
        assert run.call_args.kwargs["timeout"] == 7

    def test_timeout_message_reports_effective_timeout(self):
        provider = _make_cli()
        with patch(
            "pithos.tools.cli_provider.subprocess.run",
            side_effect=subprocess.TimeoutExpired(cmd="echo", timeout=7),
        ):
            result = provider.execute("echo hi")
        assert result.success is False
        assert "timed out after 7 seconds" in result.stderr


# ---------------------------------------------------------------------------
# FlowchartToolExecutor
# ---------------------------------------------------------------------------


def _make_cm(no_timeout=None) -> Mock:
    cm = Mock(spec=ConfigManager)
    cm.get_registered_flowchart_names.return_value = iter(["slow", "other"])
    cfg: dict = {"flowcharts": {"enabled": True, "timeout": 0.05}}
    if no_timeout is not None:
        cfg["no_timeout"] = no_timeout
    cm.get_config.return_value = cfg
    return cm


class TestFlowchartNoTimeout:
    def test_default_timeout_applies(self):
        executor = FlowchartToolExecutor(_make_cm(), timeout=0.05)
        assert executor.get_timeout("slow") == 0.05

    def test_dispatcher_entry_disables_all(self):
        executor = FlowchartToolExecutor(_make_cm(["flowchart"]), timeout=0.05)
        assert executor.get_timeout("slow") is None
        assert executor.get_timeout("other") is None

    def test_named_entry_disables_only_that_flowchart(self):
        executor = FlowchartToolExecutor(_make_cm(["flowchart:slow"]), timeout=0.05)
        assert executor.get_timeout("slow") is None
        assert executor.get_timeout("other") == 0.05

    def test_config_read_failure_keeps_timeout(self):
        cm = _make_cm()
        cm.get_config.side_effect = RuntimeError("boom")
        executor = FlowchartToolExecutor(cm, timeout=0.05)
        assert executor.get_timeout("slow") == 0.05

    @patch("pithos.flowchart.Flowchart")
    def test_listed_flowchart_runs_past_configured_timeout(self, MockFlowchart):
        """A run longer than the configured timeout must complete successfully
        when the flowchart is exempted via no_timeout."""
        release = threading.Event()

        def _slow_run(**kwargs):
            # Comfortably exceed the 0.05s timeout, then finish.
            release.wait(timeout=0.3)
            return "finished"

        fc_instance = MagicMock()
        fc_instance.run.side_effect = _slow_run
        MockFlowchart.from_registered.return_value = fc_instance

        executor = FlowchartToolExecutor(_make_cm(["flowchart:slow"]), timeout=0.05)
        result = executor.run("slow", "hi", agents={})

        assert result.success is True
        assert result.stdout == "finished"

    @patch("pithos.flowchart.Flowchart")
    def test_unlisted_flowchart_still_times_out(self, MockFlowchart):
        release = threading.Event()

        def _slow_run(**kwargs):
            release.wait(timeout=5)
            return "too late"

        fc_instance = MagicMock()
        fc_instance.run.side_effect = _slow_run
        MockFlowchart.from_registered.return_value = fc_instance

        executor = FlowchartToolExecutor(_make_cm(["flowchart:slow"]), timeout=0.05)
        result = executor.run("other", "hi", agents={})
        release.set()

        assert result.success is False
        assert "timed out after 0.05 seconds" in result.stderr
