"""Memory recall grader — checks the subject's final answer surfaced the
seeded fact, optionally cross-checking the metrics memory snapshot."""

from __future__ import annotations

from typing import Any, Optional

from ..models import GradeResult, TaskCase
from .base import Grader


class MemoryRecallGrader(Grader):
    grader_name = "memory_recall"

    def grade(
        self,
        output: str,
        expected: Any,
        *,
        case: Optional[TaskCase] = None,
        ctx: Optional[Any] = None,
    ) -> GradeResult:
        expected_text = ""
        if isinstance(expected, str):
            expected_text = expected
        elif isinstance(expected, dict):
            expected_text = str(expected.get("recall", ""))
        if not expected_text and case is not None:
            expected_text = str(case.metadata.get("expected_recall", ""))

        haystack = (output or "").lower()
        needle = expected_text.strip().lower()
        contains = bool(needle) and needle in haystack

        memory_hits = 0
        if ctx is not None and hasattr(ctx, "extras"):
            run = ctx.extras.get("subject_run")
            if run is not None and run.trace is not None:
                snap = run.trace.metrics_snapshot or {}
                mem = snap.get("memory") or {}
                memory_hits = int(mem.get("hits", 0) or 0)

        score = 100.0 if contains else 0.0
        return GradeResult(
            grader=self.grader_name,
            score=score,
            passed=contains,
            detail={
                "expected_recall": expected_text,
                "memory_hits": memory_hits,
                "matched_in_output": contains,
            },
        )
