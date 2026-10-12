"""Tests for the CodeHarness loop (pithos.tools.coder.harness)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pithos.tools.coder.harness import CodeHarness
from pithos.tools.coder.models import (
    CodeHarnessConfig,
    CodeTaskRequest,
    ExitStatus,
)


class _ScriptedAgent:
    """Agent stub that returns a queue of canned replies."""

    def __init__(self, replies: list[str], default: str = "no action here") -> None:
        self._replies = list(replies)
        self._default = default
        self.sent: list[str] = []
        self.system_prompts: list[str] = []

    def send(self, prompt: str, **kw: Any) -> str:
        self.sent.append(prompt)
        return self._replies.pop(0) if self._replies else self._default

    def set_system_prompt(self, p: str) -> None:
        self.system_prompts.append(p)


class _FakeCM:
    """Minimal config manager: returns None so defaults apply everywhere."""

    def get_config(self, name: str, category: str = "") -> Any:
        return None


def _harness(replies: list[str], tmp_path: Path, **cfg_overrides) -> CodeHarness:
    cfg = CodeHarnessConfig(
        write_trajectory=False,
        include_file_listing=False,
        **cfg_overrides,
    )
    agent = _ScriptedAgent(replies)
    harness = CodeHarness(_FakeCM(), config=cfg, agent_factory=lambda: agent)
    harness._last_agent = agent  # type: ignore[attr-defined]
    return harness


class TestHarnessLoop:
    def test_submit_terminates(self, tmp_path: Path) -> None:
        replies = [
            "[EDIT path=foo.txt mode=create]\nhello\n[/EDIT]",
            "[SUBMIT]\nCreated foo.txt as requested.\n[/SUBMIT]",
        ]
        harness = _harness(replies, tmp_path)
        result = harness.run(CodeTaskRequest(task="make foo", root=str(tmp_path)))
        assert result.exit_status == ExitStatus.SUBMITTED
        assert result.success
        assert "Created foo.txt" in result.summary
        assert result.steps == 2
        assert (tmp_path / "foo.txt").read_text() == "hello"

    def test_system_prompt_installed(self, tmp_path: Path) -> None:
        agent = _ScriptedAgent(["[SUBMIT]\ndone\n[/SUBMIT]"])
        cfg = CodeHarnessConfig(write_trajectory=False, include_file_listing=False)
        harness = CodeHarness(_FakeCM(), config=cfg, agent_factory=lambda: agent)
        harness.run(CodeTaskRequest(task="t", root=str(tmp_path)))
        assert agent.system_prompts  # set_system_prompt was called
        assert "autonomous software engineering agent" in agent.system_prompts[0]

    def test_step_limit(self, tmp_path: Path) -> None:
        # Always emits a valid (non-submit) edit, so only the step limit stops it.
        replies = ["[EDIT path=a.txt mode=create]\nx\n[/EDIT]"] * 5
        harness = _harness(replies, tmp_path, step_limit=2)
        # create then subsequent creates fail, but each reply still parses as an
        # action, so the loop advances until the step limit.
        result = harness.run(CodeTaskRequest(task="loop", root=str(tmp_path)))
        assert result.exit_status == ExitStatus.STEP_LIMIT
        assert result.steps == 2

    def test_format_error_limit(self, tmp_path: Path) -> None:
        replies = ["just prose, no action blocks"] * 5
        harness = _harness(
            replies, tmp_path, max_consecutive_format_errors=2
        )
        result = harness.run(CodeTaskRequest(task="t", root=str(tmp_path)))
        assert result.exit_status == ExitStatus.FORMAT_ERROR_LIMIT
        assert result.errors

    def test_format_error_then_recovery(self, tmp_path: Path) -> None:
        replies = [
            "no action yet",  # format error 1
            "[EDIT path=b.txt mode=create]\nok\n[/EDIT]",  # recovers
            "[SUBMIT]\ndone\n[/SUBMIT]",
        ]
        harness = _harness(replies, tmp_path, max_consecutive_format_errors=2)
        result = harness.run(CodeTaskRequest(task="t", root=str(tmp_path)))
        assert result.exit_status == ExitStatus.SUBMITTED
        assert (tmp_path / "b.txt").exists()

    def test_empty_task_errors(self, tmp_path: Path) -> None:
        harness = _harness(["[SUBMIT]\nx\n[/SUBMIT]"], tmp_path)
        result = harness.run(CodeTaskRequest(task="  ", root=str(tmp_path)))
        assert result.exit_status == ExitStatus.ERROR
        assert "empty task" in result.errors

    def test_observation_fed_back(self, tmp_path: Path) -> None:
        replies = [
            "[EDIT path=c.txt mode=create]\nhi\n[/EDIT]",
            "[SUBMIT]\ndone\n[/SUBMIT]",
        ]
        agent = _ScriptedAgent(replies)
        cfg = CodeHarnessConfig(write_trajectory=False, include_file_listing=False)
        harness = CodeHarness(_FakeCM(), config=cfg, agent_factory=lambda: agent)
        harness.run(CodeTaskRequest(task="t", root=str(tmp_path)))
        # Second prompt sent to the agent should contain the observation block.
        assert any("## Observation" in p for p in agent.sent)

    def test_messages_captured(self, tmp_path: Path) -> None:
        replies = ["[SUBMIT]\ndone\n[/SUBMIT]"]
        harness = _harness(replies, tmp_path)
        result = harness.run(CodeTaskRequest(task="t", root=str(tmp_path)))
        assert len(result.messages) == 1
        assert result.messages[0]["reply"].startswith("[SUBMIT]")

    def test_trajectory_written(self, tmp_path: Path) -> None:
        out_dir = tmp_path / "traj"
        agent = _ScriptedAgent(["[SUBMIT]\ndone\n[/SUBMIT]"])
        cfg = CodeHarnessConfig(
            write_trajectory=True,
            include_file_listing=False,
            output_dir=str(out_dir),
        )
        harness = CodeHarness(_FakeCM(), config=cfg, agent_factory=lambda: agent)
        result = harness.run(CodeTaskRequest(task="my task", root=str(tmp_path)))
        assert result.trajectory_path is not None
        assert Path(result.trajectory_path).exists()


class TestWorkspaceAllocation:
    def test_allocates_unique_workspace_under_base(self, tmp_path: Path) -> None:
        base = tmp_path / "wsbase"
        cfg = CodeHarnessConfig(
            write_trajectory=False,
            include_file_listing=False,
            workspace_base=str(base),
        )
        harness = CodeHarness(
            _FakeCM(), config=cfg, agent_factory=lambda: _ScriptedAgent([])
        )
        r1 = harness._resolve_root(CodeTaskRequest(task="t"))
        r2 = harness._resolve_root(CodeTaskRequest(task="t"))
        assert r1 != r2
        assert Path(r1).is_dir() and Path(r2).is_dir()
        assert Path(r1).parent.resolve() == base.resolve()
        assert Path(r2).parent.resolve() == base.resolve()

    def test_explicit_root_overrides_allocation(self, tmp_path: Path) -> None:
        base = tmp_path / "wsbase"
        cfg = CodeHarnessConfig(workspace_base=str(base))
        harness = CodeHarness(
            _FakeCM(), config=cfg, agent_factory=lambda: _ScriptedAgent([])
        )
        root = harness._resolve_root(CodeTaskRequest(task="t", root=str(tmp_path)))
        assert root == str(tmp_path)
        assert not base.exists()  # no workspace allocated

    def test_configured_allowed_root_overrides_allocation(self, tmp_path: Path) -> None:
        fixed = tmp_path / "fixed"
        base = tmp_path / "wsbase"
        cfg = CodeHarnessConfig(allowed_root=str(fixed), workspace_base=str(base))
        harness = CodeHarness(
            _FakeCM(), config=cfg, agent_factory=lambda: _ScriptedAgent([])
        )
        root = harness._resolve_root(CodeTaskRequest(task="t"))
        assert root == str(fixed)
        assert not base.exists()

    def test_run_without_root_uses_allocated_workspace(self, tmp_path: Path) -> None:
        base = tmp_path / "wsbase"
        out = tmp_path / "out"
        cfg = CodeHarnessConfig(
            include_file_listing=False,
            write_trajectory=True,
            output_dir=str(out),
            workspace_base=str(base),
        )
        replies = [
            "[EDIT path=made.txt mode=create]\nhi\n[/EDIT]",
            "[SUBMIT]\ndone\n[/SUBMIT]",
        ]
        harness = CodeHarness(
            _FakeCM(), config=cfg, agent_factory=lambda: _ScriptedAgent(replies)
        )
        result = harness.run(CodeTaskRequest(task="make it"))  # no root
        data = json.loads(Path(result.trajectory_path).read_text(encoding="utf-8"))
        root = Path(data["root"])
        assert root.parent.resolve() == base.resolve()
        assert (root / "made.txt").read_text() == "hi"
