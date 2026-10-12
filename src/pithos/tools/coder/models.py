"""Data models for the coding harness (``code`` tool).

These dataclasses describe the configuration, the per-run request, the
step-by-step trajectory, and the final result of a coding-harness run.  They
have no third-party dependencies so importing this module never fails.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class ExitStatus(str, Enum):
    """Why a harness run terminated."""

    SUBMITTED = "submitted"  # agent issued a submit action
    STEP_LIMIT = "step_limit"  # exhausted the step budget
    TIME_LIMIT = "time_limit"  # exceeded the wall-clock budget
    FORMAT_ERROR_LIMIT = "format_error_limit"  # too many unparseable replies
    NO_ACTION = "no_action"  # agent stopped producing actions
    ERROR = "error"  # an unexpected error aborted the run


@dataclass
class CodeHarnessConfig:
    """Runtime configuration for a coding-harness run.

    Mirrors the ``code_config.yaml`` structure.  Every field has a sensible
    default so a bare :class:`CodeHarnessConfig` is usable out of the box.
    """

    enabled: bool = True

    # --- Subagent --------------------------------------------------------
    subagent_config_name: str = "code_harness"
    subagent_model: Optional[str] = None
    subagent_timeout: Optional[float] = None  # per-generation wall-clock cap

    # --- Loop limits -----------------------------------------------------
    step_limit: int = 40  # max agent steps (0 = unlimited)
    wall_time_limit_seconds: int = 0  # 0 = no limit
    max_consecutive_format_errors: int = 3  # 0 = no limit

    # --- Command execution ----------------------------------------------
    command_timeout: int = 60  # per-command subprocess timeout (seconds)
    review_policy: str = "allow"  # how to treat REVIEW safety verdicts: allow | block

    # --- Observation shaping --------------------------------------------
    max_output_chars: int = 10000  # observation truncation threshold
    output_head_chars: int = 5000
    output_tail_chars: int = 5000

    # --- Workspace / editing --------------------------------------------
    allowed_root: Optional[str] = None  # None = allocate a unique workspace per run
    workspace_base: str = "~/.talos/code"  # parent dir for auto-allocated workspaces
    backup_edits: bool = True
    include_file_listing: bool = True
    max_listed_files: int = 200

    # --- Git -------------------------------------------------------------
    allow_git: bool = True
    allow_destructive_git: bool = False
    auto_commit: bool = False

    # --- Output ----------------------------------------------------------
    output_dir: str = "./data/coding"
    write_trajectory: bool = True

    @classmethod
    def from_dict(cls, data: Optional[dict[str, Any]]) -> "CodeHarnessConfig":
        """Build a config from a (possibly partial) dict, applying defaults."""
        if not data:
            return cls()
        known = set(cls.__dataclass_fields__)
        kwargs = {k: v for k, v in data.items() if k in known}
        return cls(**kwargs)


@dataclass
class CodeTaskRequest:
    """A single coding task submitted to the harness."""

    task: str
    root: Optional[str] = None  # workspace root; None = config/cwd default
    test_command: Optional[str] = None  # optional hint shown to the agent


@dataclass
class StepRecord:
    """One step of the harness loop: the reply and what it produced."""

    index: int
    reply: str = ""
    actions: list[str] = field(default_factory=list)
    observations: list[str] = field(default_factory=list)
    format_error: bool = False


@dataclass
class CodeTaskResult:
    """Outcome of a coding-harness run."""

    task: str
    exit_status: ExitStatus = ExitStatus.ERROR
    summary: str = ""
    diff: str = ""
    steps: int = 0
    trajectory_path: Optional[str] = None
    messages: list[dict[str, Any]] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def success(self) -> bool:
        """True when the agent submitted a result (as opposed to a limit/error)."""
        return self.exit_status == ExitStatus.SUBMITTED

    def to_summary(self) -> str:
        """Render a concise, user-facing summary (no full diff).

        This is what the tool surfaces as ``stdout`` to the calling agent, so
        the agent's own reply is based on a short summary rather than the full
        report + diff. The complete report/diff lives in the trajectory file,
        which is surfaced separately as an attachment.
        """
        lines = [
            f"Coding task {self.exit_status.value} after {self.steps} step(s)."
        ]
        if self.summary.strip():
            lines.append("")
            lines.append(self.summary.strip())
        if self.errors:
            lines.append("")
            lines.append("Errors: " + "; ".join(self.errors))
        if self.trajectory_path:
            lines.append("")
            lines.append(f"Full report and diff saved to: {self.trajectory_path}")
        return "\n".join(lines)

    def to_markdown(self) -> str:
        """Render the result as a Markdown report."""
        parts = [
            "# Coding task result",
            "",
            f"**Task:** {self.task}",
            "",
            f"- **Exit status:** {self.exit_status.value}",
            f"- **Steps:** {self.steps}",
        ]
        if self.trajectory_path:
            parts.append(f"- **Trajectory:** {self.trajectory_path}")
        parts.append("")
        parts.append("## Summary")
        parts.append("")
        parts.append(self.summary.strip() or "_No summary was produced._")
        parts.append("")
        if self.diff.strip():
            parts.append("## Diff")
            parts.append("")
            parts.append("```diff")
            parts.append(self.diff.rstrip())
            parts.append("```")
            parts.append("")
        if self.errors:
            parts.append("## Errors")
            parts.append("")
            for err in self.errors:
                parts.append(f"- {err}")
            parts.append("")
        return "\n".join(parts)
