"""Cost analyzer — estimates USD cost from token usage + price map."""

from __future__ import annotations

from typing import Optional

from ...models import EvalTrace, TrajectoryIssue, TrajectoryIssueSeverity
from .base import Analyzer, AnalyzerContext


class CostAnalyzer(Analyzer):
    """Estimate USD cost from per-model token usage.

    The analyzer reads ``trace.metrics_snapshot['token_usage']`` and the
    price map from ``ctx.price_map`` (or its own ``prices`` config key).
    Each entry of the price map is keyed by model name and contains
    ``prompt_per_1k`` / ``completion_per_1k`` USD prices. Models missing
    from the map default to zero cost (treated as free / local).

    A single info-level issue is emitted with the estimated cost in
    ``detail.estimated_cost_usd``.

    Config keys:

    * ``prices`` *(dict, optional)* — same shape as ``ctx.price_map``;
      provides defaults for analyzers run without an explicit context.
    * ``warn_threshold_usd`` *(float, optional)* — if total cost exceeds
      this value, emit a WARNING-level issue instead of INFO.
    """

    analyzer_name = "cost"

    def analyze(
        self, trace: EvalTrace, ctx: Optional[AnalyzerContext] = None
    ) -> list[TrajectoryIssue]:
        snapshot = trace.metrics_snapshot or {}
        token_usage = snapshot.get("token_usage") or {}

        prices = {}
        if ctx is not None and ctx.price_map:
            prices.update(ctx.price_map)
        if self.config.get("prices"):
            prices.update(self.config["prices"])

        total = 0.0
        per_model: dict[str, float] = {}
        for model, data in token_usage.items():
            entry = prices.get(model, {})
            prompt_cost = float(entry.get("prompt_per_1k", 0.0)) * (
                float(data.get("prompt_tokens", 0) or 0) / 1000.0
            )
            completion_cost = float(entry.get("completion_per_1k", 0.0)) * (
                float(data.get("completion_tokens", 0) or 0) / 1000.0
            )
            cost = prompt_cost + completion_cost
            per_model[model] = cost
            total += cost

        severity = TrajectoryIssueSeverity.INFO
        threshold = self.config.get("warn_threshold_usd")
        if threshold is not None and total > float(threshold):
            severity = TrajectoryIssueSeverity.WARNING

        return [
            TrajectoryIssue(
                analyzer=self.analyzer_name,
                code="cost_estimate",
                message=f"Estimated cost: ${total:.6f}",
                severity=severity,
                detail={
                    "estimated_cost_usd": total,
                    "per_model_usd": per_model,
                },
            )
        ]
