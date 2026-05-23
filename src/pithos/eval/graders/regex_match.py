"""Regex-match grader."""

from __future__ import annotations

import re
from typing import Any, Optional

from ..models import GradeResult, TaskCase
from .base import Grader


class RegexMatchGrader(Grader):
    """Score 100 if ``pattern`` matches *output*, else 0.

    Config keys:

    * ``pattern`` *(str, required)* — Python regex pattern. If the spec
      omits ``pattern`` the grader falls back to *expected* as the pattern.
    * ``flags`` *(int, default 0)*.
    """

    grader_name = "regex_match"

    def grade(
        self,
        output: str,
        expected: Any,
        *,
        case: Optional[TaskCase] = None,
        ctx: Optional[Any] = None,
    ) -> GradeResult:
        pattern = self.config.get("pattern")
        if pattern is None:
            pattern = expected
        if pattern is None:
            return GradeResult(
                grader=self.grader_name,
                score=0.0,
                passed=False,
                detail={"error": "no pattern provided"},
            )
        flags = int(self.config.get("flags", 0))
        try:
            compiled = re.compile(pattern, flags)
        except re.error as exc:
            return GradeResult(
                grader=self.grader_name,
                score=0.0,
                passed=False,
                detail={"error": f"invalid regex: {exc}"},
            )
        m = compiled.search(output or "")
        passed = m is not None
        return GradeResult(
            grader=self.grader_name,
            score=100.0 if passed else 0.0,
            passed=passed,
            detail={"match": m.group(0) if m else None, "pattern": pattern},
        )
