"""Free-form task — wraps a :class:`FreeFormDataset` plus any grader."""

from __future__ import annotations

from typing import Any, Optional

from ..models import GradeResult, TaskCase
from .base import Task


class FreeFormTask(Task):
    """Free-form task."""

    def grade(
        self, case: TaskCase, output: str, ctx: Optional[Any] = None
    ) -> GradeResult:
        return self.grader.grade(output, case.expected, case=case, ctx=ctx)
