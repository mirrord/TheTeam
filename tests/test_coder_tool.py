"""Tests for the CoderToolExecutor virtual-tool wrapper."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pithos.tools.coder.coder import Coder, CoderToolExecutor
from pithos.tools.coder.models import CodeHarnessConfig


class _ScriptedAgent:
    def __init__(self, replies: list[str]) -> None:
        self._replies = list(replies)
        self.sent: list[str] = []

    def send(self, prompt: str, **kw: Any) -> str:
        self.sent.append(prompt)
        return self._replies.pop(0) if self._replies else "no action"

    def set_system_prompt(self, p: str) -> None:
        pass


class _FakeCM:
    def get_config(self, name: str, category: str = "") -> Any:
        return None


def _executor(replies: list[str], tmp_path: Path) -> CoderToolExecutor:
    cfg = CodeHarnessConfig(write_trajectory=False, include_file_listing=False)
    agent = _ScriptedAgent(replies)
    coder = Coder(_FakeCM(), config=cfg, agent_factory=lambda: agent)
    return CoderToolExecutor(_FakeCM(), coder=coder)


class TestDiscovery:
    def test_discover_exposes_code(self) -> None:
        ex = CoderToolExecutor(_FakeCM())
        tools = ex.discover()
        assert "code" in tools
        assert tools["code"].tool_type == "code"

    def test_can_execute(self) -> None:
        ex = CoderToolExecutor(_FakeCM())
        assert ex.can_execute("code")
        assert not ex.can_execute("python")


class TestExecute:
    def test_runs_task_and_returns_result(self, tmp_path: Path) -> None:
        replies = [
            "[EDIT path=z.txt mode=create]\nhi\n[/EDIT]",
            "[SUBMIT]\nMade z.txt.\n[/SUBMIT]",
        ]
        ex = _executor(replies, tmp_path)
        result = ex.execute(f"code --root {tmp_path} create z.txt")
        assert result.success
        assert "Made z.txt" in result.stdout
        assert (tmp_path / "z.txt").read_text() == "hi"

    def test_empty_task_fails(self, tmp_path: Path) -> None:
        ex = _executor([], tmp_path)
        result = ex.execute("code")
        assert not result.success
        assert "empty task" in result.stderr

    def test_non_submit_marks_failure(self, tmp_path: Path) -> None:
        # Agent never submits; the step limit trips -> not a success.
        cfg = CodeHarnessConfig(
            write_trajectory=False, include_file_listing=False, step_limit=1
        )
        agent = _ScriptedAgent(["[EDIT path=a.txt mode=create]\nx\n[/EDIT]"])
        coder = Coder(_FakeCM(), config=cfg, agent_factory=lambda: agent)
        ex = CoderToolExecutor(_FakeCM(), coder=coder)
        result = ex.execute(f"code --root {tmp_path} do a thing")
        assert not result.success
        assert result.exit_code == 1

    def test_report_surfaced_as_attachment_not_inline(self, tmp_path: Path) -> None:
        out_dir = tmp_path / "traj"
        cfg = CodeHarnessConfig(
            write_trajectory=True,
            include_file_listing=False,
            output_dir=str(out_dir),
        )
        replies = ["[SUBMIT]\nAll done.\n[/SUBMIT]"]
        agent = _ScriptedAgent(replies)
        coder = Coder(_FakeCM(), config=cfg, agent_factory=lambda: agent)
        ex = CoderToolExecutor(_FakeCM(), coder=coder)
        result = ex.execute(f"code --root {tmp_path} do a thing")
        # The report file is surfaced as an attachment, never inlined.
        assert result.attachment_paths
        assert result.report_paths == []
        assert Path(result.attachment_paths[0]).exists()
        # stdout is a concise summary, not the full report/diff markdown.
        assert "# Coding task result" not in result.stdout
        assert "All done." in result.stdout


class TestParseRemainder:
    def test_no_root(self) -> None:
        root, task = CoderToolExecutor._parse_remainder("fix the bug")
        assert root is None
        assert task == "fix the bug"

    def test_root_space(self) -> None:
        root, task = CoderToolExecutor._parse_remainder("--root /tmp/ws fix the bug")
        assert root == "/tmp/ws"
        assert task == "fix the bug"

    def test_root_equals(self) -> None:
        root, task = CoderToolExecutor._parse_remainder("--root=/tmp/ws do it")
        assert root == "/tmp/ws"
        assert task == "do it"
