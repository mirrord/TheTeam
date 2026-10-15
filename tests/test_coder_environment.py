"""Tests for the WorkspaceEnvironment used by the coding harness."""

from __future__ import annotations

from pathlib import Path

from pithos.tools.coder.actions import Action
from pithos.tools.coder.environment import WorkspaceEnvironment
from pithos.tools.coder.models import CodeHarnessConfig

# A bounded (strict) tool config: only the listed tools are discoverable, and
# manual descriptions avoid probing executables with --help. This keeps
# discovery fast and deterministic (permissive mode would probe all of PATH).
_TOOL_CONFIG = {
    "mode": "strict",
    "include": ["python", "git", "echo"],
    "safety": {
        "enabled": True,
        "block_shell_injection": True,
        "block_dangerous_patterns": True,
        "review_destructive_flags": True,
    },
    "descriptions": {
        "python": "Python interpreter",
        "git": "Version control",
        "echo": "Echo",
    },
}


def _env(tmp_path: Path, **overrides) -> WorkspaceEnvironment:
    config = CodeHarnessConfig(**overrides)
    return WorkspaceEnvironment(str(tmp_path), config, _TOOL_CONFIG)


class TestConfirmation:
    """Would-be-denied tool calls in the include/confirm groups prompt the user."""

    def test_confirm_group_tool_approved_runs(self, tmp_path: Path) -> None:
        cfg = {**_TOOL_CONFIG, "confirm": ["python"]}
        calls: list[str] = []

        def approve(command: str) -> bool:
            calls.append(command)
            return True

        env = WorkspaceEnvironment(
            str(tmp_path), CodeHarnessConfig(), cfg, confirm_callback=approve
        )
        obs = env.execute_action(Action(kind="run", command="python --version"))
        assert calls == ["python --version"]  # user was asked
        assert "denied by user" not in obs
        assert "<returncode>0</returncode>" in obs

    def test_confirm_group_tool_denied(self, tmp_path: Path) -> None:
        cfg = {**_TOOL_CONFIG, "confirm": ["python"]}

        def deny(command: str) -> bool:
            return False

        env = WorkspaceEnvironment(
            str(tmp_path), CodeHarnessConfig(), cfg, confirm_callback=deny
        )
        obs = env.execute_action(Action(kind="run", command="python --version"))
        assert "denied by user" in obs

    def test_would_be_denied_include_tool_prompts(self, tmp_path: Path) -> None:
        # A tool in the include group that isn't discoverable (not on PATH) is
        # escalated to a prompt rather than denied outright.
        cfg = {**_TOOL_CONFIG, "include": ["nonexistent_tool_abc"]}
        asked: list[str] = []

        def approve(command: str) -> bool:
            asked.append(command)
            return True

        env = WorkspaceEnvironment(
            str(tmp_path), CodeHarnessConfig(), cfg, confirm_callback=approve
        )
        obs = env.execute_action(
            Action(kind="run", command="nonexistent_tool_abc foo")
        )
        assert asked  # the user was prompted
        # Approved, so it is attempted (and fails to exec) — not denied/unavailable.
        assert "not available" not in obs
        assert "denied by user" not in obs

    def test_unknown_tool_not_in_groups_denied_without_prompt(
        self, tmp_path: Path
    ) -> None:
        asked: list[str] = []

        def cb(command: str) -> bool:
            asked.append(command)
            return True

        env = _env(tmp_path)
        env._confirm_callback = cb  # type: ignore[attr-defined]
        obs = env.execute_action(
            Action(kind="run", command="totally_unknown_xyz foo")
        )
        assert not asked  # never prompted — not in include/confirm
        assert "not available" in obs

    def test_no_callback_denies_noninteractive(self, tmp_path: Path) -> None:
        # With a confirm-group tool and no callback + no interactive stdin,
        # the call is denied rather than hanging.
        cfg = {**_TOOL_CONFIG, "confirm": ["python"]}
        env = WorkspaceEnvironment(str(tmp_path), CodeHarnessConfig(), cfg)
        obs = env.execute_action(Action(kind="run", command="python --version"))
        assert "denied by user" in obs


class TestRunAction:
    def test_runs_git(self, tmp_path: Path) -> None:
        import subprocess

        subprocess.run(["git", "init"], cwd=str(tmp_path), capture_output=True)
        env = _env(tmp_path)
        obs = env.execute_action(Action(kind="run", command="git status"))
        assert "<returncode>0</returncode>" in obs
        assert "unsupported" not in obs and "not available" not in obs

    def test_blocks_shell_injection(self, tmp_path: Path) -> None:
        env = _env(tmp_path)
        obs = env.execute_action(
            Action(kind="run", command="git status && rm -rf /")
        )
        assert "blocked by safety checker" in obs

    def test_unknown_tool_reported(self, tmp_path: Path) -> None:
        env = _env(tmp_path)
        obs = env.execute_action(
            Action(kind="run", command="definitely_not_a_real_tool_xyz foo")
        )
        assert "not available" in obs

    def test_git_destructive_blocked_by_default(self, tmp_path: Path) -> None:
        env = _env(tmp_path)
        obs = env.execute_action(Action(kind="run", command="git reset --hard"))
        assert "destructive" in obs

    def test_git_destructive_allowed_when_enabled(self, tmp_path: Path) -> None:
        env = _env(tmp_path, allow_destructive_git=True)
        obs = env.execute_action(Action(kind="run", command="git reset"))
        # It will run (and likely fail - not a repo), but is not pre-blocked.
        assert "destructive" not in obs

    def test_git_unsupported_subcommand(self, tmp_path: Path) -> None:
        env = _env(tmp_path)
        obs = env.execute_action(Action(kind="run", command="git frobnicate"))
        assert "unsupported git subcommand" in obs

    def test_git_disabled(self, tmp_path: Path) -> None:
        env = _env(tmp_path, allow_git=False)
        obs = env.execute_action(Action(kind="run", command="git status"))
        assert "git is disabled" in obs

    def test_output_truncation(self, tmp_path: Path) -> None:
        env = _env(
            tmp_path, max_output_chars=50, output_head_chars=20, output_tail_chars=20
        )
        obs = env._fmt_run("some cmd", 0, "x" * 500, "")
        assert "characters elided" in obs
        assert "<returncode>0</returncode>" in obs


class TestEditAction:
    def test_edit_creates_file(self, tmp_path: Path) -> None:
        env = _env(tmp_path)
        obs = env.execute_action(
            Action(kind="edit", path="new.py", mode="create", body="x = 1")
        )
        assert "applied create" in obs
        assert (tmp_path / "new.py").read_text() == "x = 1"

    def test_edit_escape_rejected(self, tmp_path: Path) -> None:
        env = _env(tmp_path)
        obs = env.execute_action(
            Action(kind="edit", path="../evil.py", mode="create", body="x")
        )
        assert "failed" in obs
        assert not (tmp_path.parent / "evil.py").exists()

    def test_edit_missing_path(self, tmp_path: Path) -> None:
        env = _env(tmp_path)
        obs = env.execute_action(Action(kind="edit", path="", mode="create", body="x"))
        assert "missing path" in obs


class TestWorkspaceHelpers:
    def test_list_files(self, tmp_path: Path) -> None:
        (tmp_path / "a.py").write_text("x")
        (tmp_path / "sub").mkdir()
        (tmp_path / "sub" / "b.py").write_text("y")
        env = _env(tmp_path)
        listing = env.list_files(100)
        assert "a.py" in listing
        assert "sub/b.py" in listing

    def test_list_files_excludes_git(self, tmp_path: Path) -> None:
        (tmp_path / ".git").mkdir()
        (tmp_path / ".git" / "config").write_text("x")
        (tmp_path / "a.py").write_text("x")
        env = _env(tmp_path)
        listing = env.list_files(100)
        assert ".git" not in listing

    def test_system_info(self, tmp_path: Path) -> None:
        env = _env(tmp_path)
        assert env.system_info()
