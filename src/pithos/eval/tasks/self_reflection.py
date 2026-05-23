"""Self-reflection task.

A planted-error scenario: the prompt asserts a falsehood and asks the
subject to verify / correct it. Grading delegates to whatever grader
was configured (typically :class:`RegexMatchGrader` looking for the
correct value, or :class:`OllamaJudge` for free-form responses).
"""

from __future__ import annotations

from typing import Any, Optional

from ..models import GradeResult, TaskCase
from .base import Task


class SelfReflectionTask(Task):
    def grade(
        self, case: TaskCase, output: str, ctx: Optional[Any] = None
    ) -> GradeResult:
        return self.grader.grade(output, case.expected, case=case, ctx=ctx)
