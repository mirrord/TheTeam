"""Top-level :class:`Coder` facade + virtual-tool executor wrapper.

:class:`CoderToolExecutor` adapts the coding harness for use as the ``code``
virtual tool, mirroring the other subagent-driven tools (``web-research``,
``craft-write``).  A calling agent invokes it as::

    [RUN]code fix the failing test in src/foo.py[/RUN]

An optional ``--root <path>`` prefix pins the workspace root; otherwise the
configured ``allowed_root`` (or the current working directory) is used.
"""

from __future__ import annotations

import logging
import re
import time
from typing import Any, Optional

from ...config_manager import ConfigManager
from ..models import ToolMetadata, ToolResult
from ..provider import ToolProvider
from .harness import CodeHarness
from .models import CodeHarnessConfig, CodeTaskRequest, CodeTaskResult

logger = logging.getLogger(__name__)


class Coder:
    """High-level entry point: build a harness and run a task.

    Construction is cheap; the heavy subagent/environment are created per
    :meth:`code` call so concurrent runs stay isolated.
    """

    def __init__(
        self,
        config_manager: ConfigManager,
        config: Optional[CodeHarnessConfig] = None,
        agent_factory: Optional[Any] = None,
    ) -> None:
        self.config_manager = config_manager
        self._config = config
        self._agent_factory = agent_factory

    def code(self, request_or_task, **kwargs: Any) -> CodeTaskResult:
        """Run a coding task and return a :class:`CodeTaskResult`."""
        harness = CodeHarness(
            self.config_manager,
            config=self._config,
            agent_factory=self._agent_factory,
        )
        return harness.run(request_or_task, **kwargs)


class CoderToolExecutor(ToolProvider):
    """Adapts :class:`Coder` for use as a virtual ``code`` tool."""

    TOOL_NAME = "code"

    def __init__(
        self,
        config_manager: ConfigManager,
        coder: Optional[Coder] = None,
    ) -> None:
        self.config_manager = config_manager
        self._coder = coder

    @property
    def coder(self) -> Coder:
        if self._coder is None:
            self._coder = Coder(self.config_manager)
        return self._coder

    # ------------------------------------------------------------------
    # ToolProvider interface
    # ------------------------------------------------------------------

    def discover(self, platform: str = "cross-platform") -> dict[str, ToolMetadata]:
        """Return the metadata entry for this virtual tool."""
        return {
            self.TOOL_NAME: ToolMetadata(
                name=self.TOOL_NAME,
                path="",
                description=(
                    "Subagent-driven coding harness. Plans, edits files, runs "
                    "tests, and self-corrects in a workspace-jailed loop. "
                    "Usage: code [--root <path>] <task description>"
                ),
                platform=platform,
                source="virtual",
                tool_type="code",
            )
        }

    def can_execute(self, tool_name: str) -> bool:
        """Return True for the ``code`` tool name."""
        return tool_name == self.TOOL_NAME

    def execute(
        self,
        command: str,
        context: Optional[dict[str, Any]] = None,
    ) -> ToolResult:
        """Execute the ``code`` tool call extracted from *command*."""
        parts = command.strip().split(None, 1)
        remainder = parts[1].strip() if len(parts) > 1 else ""
        root, task = self._parse_remainder(remainder)
        return self.run(task, root=root)

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def run(self, task: str, root: Optional[str] = None) -> ToolResult:
        """Run a coding task and wrap the result as a :class:`ToolResult`."""
        start = time.time()
        command = f"code {task}" if task else "code"
        if not task or not task.strip():
            return ToolResult(
                success=False,
                stdout="",
                stderr="empty task",
                exit_code=-1,
                execution_time=0.0,
                command=command,
                error_hint="Usage: code [--root <path>] <task description>",
            )
        try:
            result = self.coder.code(CodeTaskRequest(task=task, root=root))
        except Exception as exc:
            logger.exception("coding harness failed")
            return ToolResult(
                success=False,
                stdout="",
                stderr=f"coding harness failed: {exc}",
                exit_code=-1,
                execution_time=time.time() - start,
                command=command,
                error_hint="Check code_config.yaml and the code_harness agent config.",
            )
        return ToolResult(
            success=result.success,
            stdout=result.to_summary(),
            stderr="\n".join(result.errors),
            exit_code=0 if result.success else 1,
            execution_time=time.time() - start,
            command=command,
            attachment_paths=(
                [result.trajectory_path] if result.trajectory_path else []
            ),
        )

    @staticmethod
    def _parse_remainder(remainder: str) -> tuple[Optional[str], str]:
        """Split an optional ``--root <path>`` prefix from the task text.

        Uses a regex (not shlex) so Windows paths containing backslashes are
        preserved. The root value may be quoted to include spaces.
        """
        remainder = (remainder or "").strip()
        if not remainder.startswith("--root"):
            return None, remainder
        m = re.match(
            r"--root(?:=|\s+)(\"[^\"]*\"|'[^']*'|\S+)\s*(.*)",
            remainder,
            re.DOTALL,
        )
        if not m:
            return None, remainder
        root = m.group(1)
        if root and root[0] in "\"'" and root[-1] == root[0]:
            root = root[1:-1]
        task = m.group(2).strip()
        return root, task
