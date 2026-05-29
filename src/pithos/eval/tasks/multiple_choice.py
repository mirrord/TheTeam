"""Multiple-choice task — wraps a :class:`MultipleChoiceDataset` and
delegates grading to a letter-match (or compatible) grader."""

from __future__ import annotations

from typing import Any, Optional

from ..models import GradeResult, TaskCase
from .base import Task


class MultipleChoiceTask(Task):
    """Multiple-choice task."""

    def grade(
        self, case: TaskCase, output: str, ctx: Optional[Any] = None
    ) -> GradeResult:
        return self.grader.grade(output, case.expected, case=case, ctx=ctx)
