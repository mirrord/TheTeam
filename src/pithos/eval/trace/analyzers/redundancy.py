"""Redundancy analyzer — flags consecutive near-duplicate node outputs."""

from __future__ import annotations

import hashlib
from typing import Optional

from ...models import EvalTrace, TrajectoryIssue, TrajectoryIssueSeverity
from .base import Analyzer, AnalyzerContext


def _hash_outputs(outputs: list) -> str:
    blob = "\n".join("" if o is None else str(o) for o in outputs)
    return hashlib.sha1(blob.encode("utf-8", errors="replace")).hexdigest()[:12]


class RedundancyAnalyzer(Analyzer):
    """Flag adjacent steps producing identical outputs.

    Config keys:

    * ``min_run_length`` *(int, default 2)* — consecutive runs strictly
      longer than this many identical outputs trigger a warning. With
      the default ``2``, three or more identical-output steps in a row
      will be flagged.
    """

    analyzer_name = "redundancy"

    def analyze(
        self, trace: EvalTrace, ctx: Optional[AnalyzerContext] = None
    ) -> list[TrajectoryIssue]:
        min_run_length = int(self.config.get("min_run_length", 2))
        issues: list[TrajectoryIssue] = []
        run_start: Optional[int] = None
        run_hash: Optional[str] = None
        run_len = 0

        def _flush(end_step: int) -> None:
            if run_hash is None or run_len <= min_run_length:
                return
            issues.append(
                TrajectoryIssue(
                    analyzer=self.analyzer_name,
                    code="redundant_run",
                    message=(
                        f"{run_len} consecutive steps produced identical "
                        f"outputs starting at step {run_start}."
                    ),
                    severity=TrajectoryIssueSeverity.WARNING,
                    step=run_start,
                    detail={"length": run_len, "end_step": end_step},
                )
            )

        for node in trace.nodes:
            if not node.outputs:
                _flush(node.step)
                run_hash = None
                run_len = 0
                run_start = None
                continue
            h = _hash_outputs(node.outputs)
            if h == run_hash:
                run_len += 1
            else:
                _flush(node.step)
                run_hash = h
                run_len = 1
                run_start = node.step
        _flush(trace.nodes[-1].step if trace.nodes else 0)
        return issues
