"""Security stub — placeholder analyzer for v1; performs no checks."""

from __future__ import annotations

from typing import Optional

from ...models import EvalTrace, TrajectoryIssue
from .base import Analyzer, AnalyzerContext


class SecurityStubAnalyzer(Analyzer):
    """No-op analyzer reserving the ``security`` lane in C.L.A.S.S."""

    analyzer_name = "security_stub"

    def analyze(
        self, trace: EvalTrace, ctx: Optional[AnalyzerContext] = None
    ) -> list[TrajectoryIssue]:
        return []
