"""Tests for the CoderToolExecutor virtual-tool wrapper."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pithos.tools.coder.coder import Coder, CoderToolExecutor
from pithos.tools.coder.models import CodeHarnessConfig


def _call(name: str, **arguments: Any) -> str:
    """Render a single JSON tool call inside a fenced ``tool_call`` block."""
    payload = {"name": name, "arguments": arguments}
    return f"```tool_call\n{json.dumps(payload)}\n```"


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
            _call("edit", path="z.txt", mode="create", content="hi"),
            _call("submit", summary="Made z.txt."),
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
        agent = _ScriptedAgent([_call("edit", path="a.txt", mode="create", content="x")])
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
        replies = [_call("submit", summary="All done.")]
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

    def test_workspace_location_reported_to_caller(self, tmp_path: Path) -> None:
        ex = _executor([_call("submit", summary="done")], tmp_path)
        result = ex.execute(f"code --root {tmp_path} do a thing")
        # The caller is told which workspace the run used.
        assert "Workspace:" in result.stdout
        assert str(Path(tmp_path).resolve()) in result.stdout


class TestCallerPolicy:
    """The coding subagent inherits the caller's tool policy + confirm channel."""

    def test_caller_policy_extracted_from_context(self, tmp_path: Path) -> None:
        captured: dict[str, Any] = {}

        class _RecordingCoder(Coder):
            def code(self, request_or_task, tool_config=None, confirm_callback=None, **kw):
                captured["tool_config"] = tool_config
                captured["confirm_callback"] = confirm_callback
                return super().code(
                    request_or_task,
                    tool_config=tool_config,
                    confirm_callback=confirm_callback,
                    **kw,
                )

        cfg = CodeHarnessConfig(write_trajectory=False, include_file_listing=False)
        coder = _RecordingCoder(
            _FakeCM(), config=cfg, agent_factory=lambda: _ScriptedAgent([_call("submit", summary="x")])
        )
        ex = CoderToolExecutor(_FakeCM(), coder=coder)

        def my_confirm(cmd: str) -> bool:
            return True

        class _Registry:
            config = {"mode": "strict", "include": ["python"], "confirm": ["bash"]}

        class _Caller:
            tool_registry = _Registry()
            tool_confirm_callback = staticmethod(my_confirm)

        ex.execute(f"code --root {tmp_path} do it", context={"agent": _Caller()})
        assert captured["tool_config"]["include"] == ["python"]
        assert captured["tool_config"]["confirm"] == ["bash"]
        assert captured["confirm_callback"] is my_confirm

    def test_no_caller_agent_uses_defaults(self, tmp_path: Path) -> None:
        root, cb = CoderToolExecutor._caller_policy(None)
        assert root is None and cb is None
        root, cb = CoderToolExecutor._caller_policy({"agent": None})
        assert root is None and cb is None


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
