"""Latency analyzer — end-to-end + per-step p50/p95 timing summary."""

from __future__ import annotations

from typing import Optional

from ...models import EvalTrace, TrajectoryIssue, TrajectoryIssueSeverity
from .base import Analyzer, AnalyzerContext


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    if len(s) == 1:
        return s[0]
    k = (len(s) - 1) * (pct / 100.0)
    lo = int(k)
    hi = min(lo + 1, len(s) - 1)
    frac = k - lo
    return s[lo] + (s[hi] - s[lo]) * frac


class LatencyAnalyzer(Analyzer):
    """Emit a single info-level issue summarising latency.

    Config keys:

    * ``slow_step_ms`` *(float, optional)* — any single step longer than
      this triggers an additional warning per offending step.
    * ``slow_total_ms`` *(float, optional)* — total end-to-end exceeding
      this raises the summary issue's severity to WARNING.
    """

    analyzer_name = "latency"

    def analyze(
        self, trace: EvalTrace, ctx: Optional[AnalyzerContext] = None
    ) -> list[TrajectoryIssue]:
        per_step = [n.duration_ms for n in trace.nodes if n.duration_ms]
        total = trace.end_to_end_ms

        slow_step_ms = self.config.get("slow_step_ms")
        slow_total_ms = self.config.get("slow_total_ms")

        severity = TrajectoryIssueSeverity.INFO
        if slow_total_ms is not None and total > float(slow_total_ms):
            severity = TrajectoryIssueSeverity.WARNING

        issues: list[TrajectoryIssue] = [
            TrajectoryIssue(
                analyzer=self.analyzer_name,
                code="latency_summary",
                message=(
                    f"End-to-end {total:.1f} ms across " f"{trace.total_steps} step(s)"
                ),
                severity=severity,
                detail={
                    "end_to_end_ms": total,
                    "per_step_p50_ms": _percentile(per_step, 50),
                    "per_step_p95_ms": _percentile(per_step, 95),
                    "max_step_ms": max(per_step) if per_step else 0.0,
                    "step_count": trace.total_steps,
                },
            )
        ]

        if slow_step_ms is not None:
            for node in trace.nodes:
                if node.duration_ms > float(slow_step_ms):
                    issues.append(
                        TrajectoryIssue(
                            analyzer=self.analyzer_name,
                            code="slow_step",
                            message=(
                                f"Step {node.step} ({node.node_id}) took "
                                f"{node.duration_ms:.1f} ms "
                                f"(threshold {slow_step_ms} ms)"
                            ),
                            severity=TrajectoryIssueSeverity.WARNING,
                            step=node.step,
                            detail={
                                "duration_ms": node.duration_ms,
                                "node_id": node.node_id,
                                "node_type": node.node_type,
                            },
                        )
                    )

        return issues
