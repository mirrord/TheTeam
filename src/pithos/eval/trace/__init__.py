"""Trace ingestion + analyzer package.

The :mod:`pithos.eval.trace.ingest` module maps runtime traces produced
by :class:`pithos.flowchart.ExecutionTrace` and
:class:`pithos.metrics.MetricsCollector` into the unified
:class:`pithos.eval.models.EvalTrace` shape consumed by analyzers,
graders, and the reporter.
"""

from .ingest import (
    build_eval_trace_from_agent,
    build_eval_trace_from_flowchart,
    extract_tool_calls_from_snapshot,
)

__all__ = [
    "build_eval_trace_from_agent",
    "build_eval_trace_from_flowchart",
    "extract_tool_calls_from_snapshot",
]
