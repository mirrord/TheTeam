"""Stability — post-aggregation cross-round variance analyzer.

Unlike per-trace analyzers, :func:`stability_for_subject` operates on
the list of grade scores collected across rounds for a single subject.
A thin :class:`StabilityAnalyzer` is provided so it can also be invoked
through the standard analyzer pipeline on a synthetic
:class:`EvalTrace` whose ``metrics_snapshot['_round_scores']`` carries
per-round scores; this is primarily used by the reporter.
"""

from __future__ import annotations

import math
from typing import Optional

from ...models import EvalTrace, TrajectoryIssue, TrajectoryIssueSeverity
from .base import Analyzer, AnalyzerContext


def stability_for_subject(
    scores: list[float], min_rounds: int = 3
) -> tuple[float, float, list[TrajectoryIssue]]:
    """Compute ``(mean, std_dev, issues)`` for a series of round scores.

    Emits a warning when fewer than *min_rounds* samples are available
    (CI estimates require at least 3) and another when the coefficient
    of variation exceeds 25%.
    """
    issues: list[TrajectoryIssue] = []
    if not scores:
        return 0.0, 0.0, issues

    mean = sum(scores) / len(scores)
    if len(scores) > 1:
        var = sum((s - mean) ** 2 for s in scores) / (len(scores) - 1)
        std = math.sqrt(var)
    else:
        std = 0.0

    if len(scores) < min_rounds:
        issues.append(
            TrajectoryIssue(
                analyzer="stability",
                code="insufficient_rounds",
                message=(
                    f"Stability estimate based on {len(scores)} round(s); "
                    f"≥{min_rounds} recommended."
                ),
                severity=TrajectoryIssueSeverity.WARNING,
                detail={"rounds": len(scores), "min_rounds": min_rounds},
            )
        )
    if mean > 0:
        cov = std / mean
        if cov > 0.25:
            issues.append(
                TrajectoryIssue(
                    analyzer="stability",
                    code="high_variance",
                    message=(
                        f"Score CoV={cov:.2%} across {len(scores)} rounds " "(>25%)."
                    ),
                    severity=TrajectoryIssueSeverity.WARNING,
                    detail={"mean": mean, "std_dev": std, "cov": cov},
                )
            )
    return mean, std, issues


class StabilityAnalyzer(Analyzer):
    """Analyzer wrapper that reads ``_round_scores`` from the trace snapshot."""

    analyzer_name = "stability"

    def analyze(
        self, trace: EvalTrace, ctx: Optional[AnalyzerContext] = None
    ) -> list[TrajectoryIssue]:
        snapshot = trace.metrics_snapshot or {}
        scores = list(snapshot.get("_round_scores") or [])
        min_rounds = int(self.config.get("min_rounds", 3))
        _, _, issues = stability_for_subject(scores, min_rounds=min_rounds)
        return issues
