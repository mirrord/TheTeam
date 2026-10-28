"""Top-level :class:`Coder` facade + virtual-tool executor wrapper.

:class:`CoderToolExecutor` adapts the coding harness for use as the ``code``
virtual tool, mirroring the other subagent-driven tools (``web-research``,
``craft-write``).  A calling agent invokes it as::

    ```tool_call
    {"name": "run", "arguments": {"command": "code fix the failing test in src/foo.py"}}
    ```

An optional ``--root <path>`` prefix pins the workspace root; otherwise a fresh
unique workspace is allocated under ``~/.talos/code`` for each invocation
(unless a fixed ``allowed_root`` is configured).
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

    def code(
        self,
        request_or_task,
        tool_config: Optional[dict] = None,
        confirm_callback: Optional[Any] = None,
        **kwargs: Any,
    ) -> CodeTaskResult:
        """Run a coding task and return a :class:`CodeTaskResult`.

        ``tool_config`` and ``confirm_callback`` let the caller hand the coding
        subagent its own tool policy and user-confirmation channel so the
        subagent has the same tool access as the caller.
        """
        harness = CodeHarness(
            self.config_manager,
            config=self._config,
            agent_factory=self._agent_factory,
            tool_config=tool_config,
            confirm_callback=confirm_callback,
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
        """Execute the ``code`` tool call extracted from *command*.

        The calling agent is read from ``context['agent']`` so the coding
        subagent inherits the caller's tool policy (``tool_registry.config``)
        and its user-confirmation channel (``tool_confirm_callback``).
        """
        parts = command.strip().split(None, 1)
        remainder = parts[1].strip() if len(parts) > 1 else ""
        root, task = self._parse_remainder(remainder)

        tool_config, confirm_callback = self._caller_policy(context)
        return self.run(
            task,
            root=root,
            tool_config=tool_config,
            confirm_callback=confirm_callback,
        )

    @staticmethod
    def _caller_policy(
        context: Optional[dict[str, Any]],
    ) -> tuple[Optional[dict], Optional[Any]]:
        """Extract (tool_config, confirm_callback) from the calling agent."""
        agent = (context or {}).get("agent")
        if agent is None:
            return None, None
        tool_config = None
        registry = getattr(agent, "tool_registry", None)
        if registry is not None:
            cfg = getattr(registry, "config", None)
            if isinstance(cfg, dict):
                tool_config = dict(cfg)
        confirm_callback = getattr(agent, "tool_confirm_callback", None)
        return tool_config, confirm_callback

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def run(
        self,
        task: str,
        root: Optional[str] = None,
        tool_config: Optional[dict] = None,
        confirm_callback: Optional[Any] = None,
    ) -> ToolResult:
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
            result = self.coder.code(
                CodeTaskRequest(task=task, root=root),
                tool_config=tool_config,
                confirm_callback=confirm_callback,
            )
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
