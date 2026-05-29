"""Tool-trace grader — checks recorded tool calls against expectations.

Given a trace (looked up from ``ctx.extras["trace"]``) and an
``expected_tools`` list (from :attr:`TaskCase.expected` or
``case.metadata["expected_tools"]``), score how well the subject's tool
usage matched:

* If ``ordered=True`` (default ``False``) — the recorded tool sequence
  must match the expected sequence exactly.
* Otherwise — score by the fraction of expected tools that appear in
  the recorded calls (set overlap).

Unexpected extra tools are reported in :attr:`GradeResult.detail` but do
not lower the score unless ``penalize_extras=True``.
"""

from __future__ import annotations

from typing import Any, Optional

from ..models import GradeResult, TaskCase
from .base import Grader


class ToolTraceGrader(Grader):
    grader_name = "tool_trace"

    def _expected_tools(self, expected: Any, case: Optional[TaskCase]) -> list[str]:
        if isinstance(expected, dict) and "tools" in expected:
            return [str(t) for t in expected["tools"]]
        if isinstance(expected, (list, tuple)):
            return [str(t) for t in expected]
        if case is not None:
            tools = case.metadata.get("expected_tools")
            if tools:
                return [str(t) for t in tools]
        return []

    def grade(
        self,
        output: str,
        expected: Any,
        *,
        case: Optional[TaskCase] = None,
        ctx: Optional[Any] = None,
    ) -> GradeResult:
        expected_tools = self._expected_tools(expected, case)
        trace = None
        if ctx is not None and hasattr(ctx, "extras"):
            trace = ctx.extras.get("trace")

        recorded: list[str] = []
        if trace is not None:
            for node in trace.nodes:
                for call in node.tool_calls:
                    name = call.get("tool") if isinstance(call, dict) else None
                    if name:
                        recorded.append(str(name))

        ordered = bool(self.config.get("ordered", False))
        penalize_extras = bool(self.config.get("penalize_extras", False))

        if not expected_tools:
            # No expectation — pass if no tools were unexpectedly called
            # (when penalize_extras), otherwise grant full credit.
            score = 0.0 if (penalize_extras and recorded) else 100.0
            return GradeResult(
                grader=self.grader_name,
                score=score,
                passed=score == 100.0,
                detail={"recorded": recorded, "expected": []},
            )

        if ordered:
            matches = recorded[: len(expected_tools)] == expected_tools
            score = 100.0 if matches else 0.0
            missing = [t for t in expected_tools if t not in recorded]
            extras = [t for t in recorded if t not in expected_tools]
        else:
            recorded_set = set(recorded)
            expected_set = set(expected_tools)
            hit = expected_set & recorded_set
            score = 100.0 * len(hit) / len(expected_set)
            missing = sorted(expected_set - recorded_set)
            extras = sorted(recorded_set - expected_set)
            if penalize_extras and extras:
                score = max(0.0, score - 25.0 * len(extras))

        return GradeResult(
            grader=self.grader_name,
            score=round(score, 2),
            passed=score >= 100.0,
            detail={
                "recorded": recorded,
                "expected": expected_tools,
                "missing": missing,
                "unexpected": extras,
                "ordered": ordered,
            },
        )
