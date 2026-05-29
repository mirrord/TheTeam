"""Composite grader — weighted combination of child graders."""

from __future__ import annotations

from typing import Any, Optional

from ..config import GraderSpec
from ..models import GradeResult, TaskCase
from .base import Grader


class CompositeGrader(Grader):
    """Combine multiple graders via a weighted sum of their scores.

    Config keys:

    * ``components`` *(list[dict], required)* — each entry is a grader
      spec with optional ``weight`` (default 1.0). Example::

          components:
            - type: letter_match
              weight: 0.6
            - type: llm_judge
              model: glm-4.7-flash
              weight: 0.4

    * ``pass_threshold`` *(float, default 60)* — composite ``passed`` is
      ``True`` iff the aggregated score is at or above the threshold.
    """

    grader_name = "composite"

    def __init__(self, config: Optional[dict] = None) -> None:
        super().__init__(config)
        # Build child graders lazily on first grade() to avoid import
        # cycles with the registry helper.
        self._children: Optional[list[tuple[Grader, float]]] = None

    def _build_children(self) -> list[tuple[Grader, float]]:
        from .base import build_grader

        children: list[tuple[Grader, float]] = []
        for entry in self.config.get("components", []) or []:
            entry = dict(entry)
            weight = float(entry.pop("weight", 1.0))
            spec = GraderSpec.from_dict(entry)
            children.append((build_grader(spec), weight))
        return children

    def grade(
        self,
        output: str,
        expected: Any,
        *,
        case: Optional[TaskCase] = None,
        ctx: Optional[Any] = None,
    ) -> GradeResult:
        if self._children is None:
            self._children = self._build_children()

        if not self._children:
            return GradeResult(
                grader=self.grader_name,
                score=0.0,
                passed=False,
                detail={"error": "no components configured"},
            )

        total_weight = sum(w for _, w in self._children)
        if total_weight <= 0:
            return GradeResult(
                grader=self.grader_name,
                score=0.0,
                passed=False,
                detail={"error": "component weights sum to zero"},
            )

        results: list[dict[str, Any]] = []
        weighted_sum = 0.0
        for child, weight in self._children:
            r = child.grade(output, expected, case=case, ctx=ctx)
            weighted_sum += r.score * weight
            results.append(
                {
                    "grader": r.grader,
                    "weight": weight,
                    "score": r.score,
                    "passed": r.passed,
                    "detail": r.detail,
                }
            )

        score = weighted_sum / total_weight
        passed = score >= float(self.config.get("pass_threshold", 60.0))
        return GradeResult(
            grader=self.grader_name,
            score=score,
            passed=passed,
            detail={"components": results},
        )
