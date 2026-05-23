"""Tool-hallucination analyzer — flags tool calls not in the registry."""

from __future__ import annotations

from typing import Iterable, Optional

from ...models import EvalTrace, TrajectoryIssue, TrajectoryIssueSeverity
from .base import Analyzer, AnalyzerContext


def _registry_names(registry) -> set[str]:
    if registry is None:
        return set()
    for attr in ("list_tool_names", "tool_names", "names"):
        method = getattr(registry, attr, None)
        if callable(method):
            try:
                return {str(n) for n in method()}
            except Exception:
                continue
        elif isinstance(method, Iterable):
            return {str(n) for n in method}
    # Last resort: try iterating the registry directly.
    try:
        return {str(n) for n in registry}
    except TypeError:
        return set()


class ToolHallucinationAnalyzer(Analyzer):
    """Flag tool calls whose name is not registered.

    Config keys:

    * ``allow_empty_registry`` *(bool, default True)* — if the registry
      is missing or empty, skip the check silently rather than flagging
      every call.
    """

    analyzer_name = "tool_hallucination"

    def analyze(
        self, trace: EvalTrace, ctx: Optional[AnalyzerContext] = None
    ) -> list[TrajectoryIssue]:
        registry = ctx.tool_registry if ctx is not None else None
        names = _registry_names(registry)
        if not names and bool(self.config.get("allow_empty_registry", True)):
            return []

        issues: list[TrajectoryIssue] = []
        for node in trace.nodes:
            for call in node.tool_calls or []:
                tool_name = str(call.get("tool", ""))
                if not tool_name:
                    continue
                if tool_name not in names:
                    issues.append(
                        TrajectoryIssue(
                            analyzer=self.analyzer_name,
                            code="unknown_tool",
                            message=(
                                f"Tool {tool_name!r} invoked from step "
                                f"{node.step} is not in the registry."
                            ),
                            severity=TrajectoryIssueSeverity.ERROR,
                            step=node.step,
                            detail={"tool": tool_name, "call": call},
                        )
                    )
        return issues
