"""Exact-match grader."""

from __future__ import annotations

from typing import Any, Optional

from ..models import GradeResult, TaskCase
from .base import Grader


class ExactMatchGrader(Grader):
    """Score 100 if ``output.strip() == expected.strip()``, else 0.

    Config keys:

    * ``case_sensitive`` *(bool, default True)*.
    """

    grader_name = "exact_match"

    def grade(
        self,
        output: str,
        expected: Any,
        *,
        case: Optional[TaskCase] = None,
        ctx: Optional[Any] = None,
    ) -> GradeResult:
        case_sensitive = bool(self.config.get("case_sensitive", True))
        a = (output or "").strip()
        b = ("" if expected is None else str(expected)).strip()
        if not case_sensitive:
            a = a.lower()
            b = b.lower()
        passed = a == b
        return GradeResult(
            grader=self.grader_name,
            score=100.0 if passed else 0.0,
            passed=passed,
            detail={"normalized_output": a, "normalized_expected": b},
        )
