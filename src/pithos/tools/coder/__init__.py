"""Coder tool - a full agentic coding harness exposed as the ``code`` virtual tool.

The harness drives a single coding subagent through a tight
**query -> act -> observe** loop (inspired by mini-swe-agent) using pithos's
native structured tool-calling.  Each step the subagent emits ``[RUN] ... [/RUN]``
tool calls which the harness executes through a workspace-jailed
:class:`~pithos.tools.registry.ToolRegistry` (shell commands via
``CLIToolProvider`` + safe file-edit/git actions), feeding truncated
observations back until the agent submits a result or a limit is reached.

This module has no optional third-party dependencies beyond what pithos already
requires; :data:`CODING_HARNESS_AVAILABLE` is always ``True`` and exists purely
for interface parity with other virtual tools that gate on optional extras.
"""

from .models import (
    CodeHarnessConfig,
    CodeTaskRequest,
    CodeTaskResult,
    ExitStatus,
    StepRecord,
)

CODING_HARNESS_AVAILABLE = True


def __getattr__(name):  # pragma: no cover - thin lazy-import shim
    if name == "CodeHarness":
        from .harness import CodeHarness

        return CodeHarness
    if name == "WorkspaceEnvironment":
        from .environment import WorkspaceEnvironment

        return WorkspaceEnvironment
    if name == "Coder":
        from .coder import Coder

        return Coder
    if name == "CoderToolExecutor":
        from .coder import CoderToolExecutor

        return CoderToolExecutor
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "CodeHarnessConfig",
    "CodeTaskRequest",
    "CodeTaskResult",
    "ExitStatus",
    "StepRecord",
    "CODING_HARNESS_AVAILABLE",
    "CodeHarness",
    "WorkspaceEnvironment",
    "Coder",
    "CoderToolExecutor",
]
