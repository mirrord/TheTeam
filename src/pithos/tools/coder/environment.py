"""Workspace environment for the coding harness.

:class:`WorkspaceEnvironment` executes the subagent's actions, confined to a
single ``allowed_root``:

* **run**  - a shell/CLI command, executed (no shell, list-args) with the
  working directory pinned to the root.  Reuses
  :class:`~pithos.tools.safety.CommandSafetyChecker` (BLOCK/REVIEW) and the
  :class:`~pithos.tools.cli_provider.CLIToolProvider` allowlist.  ``git`` is
  additionally restricted to a whitelist of subcommands.
* **edit** - a root-jailed file mutation via
  :mod:`pithos.tools.coder.file_ops`.

Every action returns a human-readable observation string that is fed back to
the agent.  Output is truncated to keep the context bounded.
"""

from __future__ import annotations

import logging
import os
import platform
import shlex
import subprocess
import time
from pathlib import Path
from typing import Optional

from ..cli_provider import CLIToolProvider
from ..models import RiskLevel
from ..safety import CommandSafetyChecker
from .actions import Action, edit_action_to_spec
from .file_ops import apply_edit
from .models import CodeHarnessConfig
from .prompts import truncate_output

logger = logging.getLogger(__name__)

#: git subcommands the harness permits (read/compose history; no network/history
#: rewriting by default).
_GIT_SAFE_SUBCOMMANDS = frozenset(
    {
        "status",
        "diff",
        "add",
        "commit",
        "log",
        "branch",
        "checkout",
        "restore",
        "show",
        "rev-parse",
    }
)
_GIT_DESTRUCTIVE_SUBCOMMANDS = frozenset(
    {"reset", "clean", "push", "rebase", "merge"}
)

#: Environment overrides that keep child tools non-interactive / un-paged.
_CHILD_ENV = {
    "PAGER": "cat",
    "MANPAGER": "cat",
    "GIT_PAGER": "cat",
    "PIP_PROGRESS_BAR": "off",
    "TQDM_DISABLE": "1",
}

_DEFAULT_EXCLUDES = (
    ".git",
    "__pycache__",
    ".venv",
    "venv",
    "node_modules",
    ".mypy_cache",
    ".pytest_cache",
)


class WorkspaceEnvironment:
    """Executes harness actions confined to a workspace root."""

    def __init__(
        self,
        root: str,
        config: CodeHarnessConfig,
        tool_config: dict,
        safety_checker: Optional[CommandSafetyChecker] = None,
    ) -> None:
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.config = config
        self.tool_config = tool_config or {}
        self.safety = safety_checker or CommandSafetyChecker(
            self.tool_config.get("safety", {})
        )
        self.cli = CLIToolProvider(
            config=self.tool_config,
            timeout=config.command_timeout,
            max_output_size=config.max_output_chars,
            safety_checker=self.safety,
        )
        try:
            self.cli.discover()
        except Exception as exc:  # pragma: no cover - discovery is best-effort
            logger.debug("CLI discovery failed: %s", exc)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def execute_action(self, action: Action) -> str:
        """Execute *action* and return an observation block."""
        if action.kind == "run":
            return self._run(action.command)
        if action.kind == "edit":
            return self._edit(action)
        return f"[unknown action: {action.kind}]"

    def get_diff(self) -> str:
        """Return the combined working-tree + staged diff, or '' if unavailable."""
        if not self._git_available():
            return ""
        parts = []
        for args in (["diff"], ["diff", "--staged"]):
            proc = self._raw_git(args)
            if proc is not None and proc.stdout.strip():
                parts.append(proc.stdout)
        return "\n".join(parts).strip()

    def list_files(self, max_files: int) -> str:
        """Return a newline-separated, root-relative listing of workspace files."""
        found: list[str] = []
        for dirpath, dirnames, filenames in os.walk(self.root):
            dirnames[:] = [d for d in dirnames if d not in _DEFAULT_EXCLUDES]
            for name in filenames:
                full = Path(dirpath) / name
                try:
                    rel = full.relative_to(self.root).as_posix()
                except ValueError:
                    continue
                found.append(rel)
                if len(found) >= max_files:
                    found.sort()
                    return "\n".join(found) + f"\n... (truncated at {max_files} files)"
        found.sort()
        return "\n".join(found) if found else "(workspace is empty)"

    def system_info(self) -> str:
        u = platform.uname()
        return f"{u.system} {u.release} ({u.machine})"

    def tools_summary(self, limit: int = 25) -> str:
        """Return a short comma-separated sample of allowed command names."""
        names = sorted(getattr(self.cli, "_discovered", {}).keys())
        if not names:
            return ""
        sample = names[:limit]
        suffix = ", ..." if len(names) > limit else ""
        return ", ".join(sample) + suffix

    # ------------------------------------------------------------------
    # Run action
    # ------------------------------------------------------------------

    def _run(self, command: str) -> str:
        command = (command or "").strip()
        if not command:
            return self._fmt_run(command, -1, "", "empty command")

        verdict = self.safety.check(command)
        if verdict.level == RiskLevel.BLOCK:
            return self._fmt_run(
                command, -1, "", f"blocked by safety checker: {verdict.reason}"
            )
        if verdict.level == RiskLevel.REVIEW and self.config.review_policy == "block":
            return self._fmt_run(
                command,
                -1,
                "",
                f"requires review and review_policy=block: {verdict.reason}",
            )

        try:
            argv = shlex.split(command, posix=os.name != "nt")
        except ValueError as exc:
            return self._fmt_run(command, -1, "", f"could not parse command: {exc}")
        if not argv:
            return self._fmt_run(command, -1, "", "empty command")

        tool_name = argv[0]
        if tool_name == "git":
            return self._run_git(command, argv)

        if not self.cli.can_execute(tool_name):
            return self._fmt_run(
                command,
                -1,
                "",
                f"tool {tool_name!r} is not available or not allowed",
            )
        return self._exec(command, argv)

    def _run_git(self, command: str, argv: list[str]) -> str:
        if not self.config.allow_git:
            return self._fmt_run(command, -1, "", "git is disabled for this run")
        if len(argv) < 2:
            return self._fmt_run(command, -1, "", "no git subcommand given")
        sub = argv[1]
        if sub in _GIT_DESTRUCTIVE_SUBCOMMANDS and not self.config.allow_destructive_git:
            return self._fmt_run(
                command, -1, "", f"git {sub!r} is destructive and not enabled"
            )
        if sub not in _GIT_SAFE_SUBCOMMANDS and sub not in _GIT_DESTRUCTIVE_SUBCOMMANDS:
            return self._fmt_run(command, -1, "", f"unsupported git subcommand {sub!r}")
        return self._exec(command, argv)

    def _exec(self, command: str, argv: list[str]) -> str:
        env = dict(os.environ)
        env.update(_CHILD_ENV)
        start = time.time()
        try:
            proc = subprocess.run(
                argv,
                cwd=str(self.root),
                capture_output=True,
                text=True,
                timeout=self.config.command_timeout,
                env=env,
                errors="replace",
            )
        except subprocess.TimeoutExpired:
            return self._fmt_run(
                command,
                -1,
                "",
                f"timed out after {self.config.command_timeout}s",
            )
        except (OSError, FileNotFoundError) as exc:
            return self._fmt_run(command, -1, "", f"failed to execute: {exc}")
        elapsed = time.time() - start
        combined = proc.stdout or ""
        if proc.stderr:
            combined += ("\n" if combined else "") + proc.stderr
        return self._fmt_run(
            command, proc.returncode, combined, "", elapsed=elapsed
        )

    # ------------------------------------------------------------------
    # Edit action
    # ------------------------------------------------------------------

    def _edit(self, action: Action) -> str:
        if not action.path:
            return "[EDIT] failed: missing path attribute"
        spec = edit_action_to_spec(action)
        try:
            result = apply_edit(spec, self.root, backup=self.config.backup_edits)
        except (
            PermissionError,
            FileExistsError,
            FileNotFoundError,
            ValueError,
        ) as exc:
            return f"[EDIT {action.path}] failed: {exc}"
        except Exception as exc:  # pragma: no cover - defensive
            return f"[EDIT {action.path}] failed: {exc}"
        return f"[EDIT] applied {result['mode']} to {action.path}"

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _fmt_run(
        self,
        command: str,
        returncode: int,
        output: str,
        error: str,
        elapsed: float = 0.0,
    ) -> str:
        shaped, elided = truncate_output(
            output,
            max_chars=self.config.max_output_chars,
            head_chars=self.config.output_head_chars,
            tail_chars=self.config.output_tail_chars,
        )
        lines = [f"$ {command}"]
        if error:
            lines.append(f"<error>{error}</error>")
        lines.append(f"<returncode>{returncode}</returncode>")
        if elided:
            lines.append(
                "<note>output was long; showing head and tail "
                f"({elided} characters elided)</note>"
            )
        lines.append("<output>")
        lines.append(shaped.rstrip("\n") if shaped else "(no output)")
        lines.append("</output>")
        return "\n".join(lines)

    def _git_available(self) -> bool:
        import shutil

        return shutil.which("git") is not None

    def _raw_git(self, args: list[str]) -> Optional[subprocess.CompletedProcess]:
        import shutil

        git = shutil.which("git")
        if git is None:
            return None
        try:
            return subprocess.run(
                [git, *args],
                cwd=str(self.root),
                capture_output=True,
                text=True,
                timeout=self.config.command_timeout,
                errors="replace",
            )
        except (subprocess.TimeoutExpired, OSError):
            return None
