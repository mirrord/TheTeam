"""Tool-use task — graded by :class:`ToolTraceGrader`."""

from __future__ import annotations

from typing import Any, Optional

from ..models import GradeResult, TaskCase
from .base import Task


class ToolUseTask(Task):
    """Task verifying that the subject invoked the expected tool(s).

    The :class:`~pithos.eval.runner.EvalRunner` pre-populates
    ``ctx.extras["trace"]`` so the grader can inspect tool calls.
    """

    def grade(
        self, case: TaskCase, output: str, ctx: Optional[Any] = None
    ) -> GradeResult:
        return self.grader.grade(output, case.expected, case=case, ctx=ctx)
