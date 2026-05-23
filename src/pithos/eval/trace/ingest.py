"""Map runtime traces + metrics snapshots into a unified :class:`EvalTrace`."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from ..models import EvalTrace, TraceNode


def _sum_token_usage(snapshot: dict) -> tuple[int, int, float]:
    """Return ``(prompt_tokens, completion_tokens, total_response_time_ms)``."""
    prompt = 0
    completion = 0
    total_ms = 0.0
    for model_data in (snapshot.get("token_usage") or {}).values():
        prompt += int(model_data.get("prompt_tokens", 0) or 0)
        completion += int(model_data.get("completion_tokens", 0) or 0)
        total_ms += float(model_data.get("total_response_time_ms", 0.0) or 0.0)
    return prompt, completion, total_ms


def extract_tool_calls_from_snapshot(snapshot: dict) -> list[dict[str, Any]]:
    """Flatten the per-tool metrics into a list of call summaries.

    The :class:`MetricsCollector` aggregates per-tool, so individual call
    timing is lost; we surface one entry per tool name with totals so
    downstream analyzers (e.g. tool hallucination) can iterate over the
    tools that were actually invoked.
    """
    out: list[dict[str, Any]] = []
    for tool_name, data in (snapshot.get("tool_calls") or {}).items():
        out.append(
            {
                "tool": tool_name,
                "successes": int(data.get("successes", 0) or 0),
                "failures": int(data.get("failures", 0) or 0),
                "total_calls": int(data.get("total_calls", 0) or 0),
                "execution_time_ms": float(
                    data.get("total_execution_time_ms", 0.0) or 0.0
                ),
            }
        )
    return out


def build_eval_trace_from_agent(
    *,
    subject_name: str,
    case_id: str,
    collector: Any,
    started_at: datetime,
    ended_at: datetime,
    output: str,
) -> EvalTrace:
    """Build an :class:`EvalTrace` for a single-shot agent run.

    Synthesizes one trace node representing the entire agent turn,
    populated with the aggregated token usage and tool calls captured
    by the supplied :class:`MetricsCollector`.
    """
    snapshot = collector.get_snapshot() if collector is not None else {}
    prompt_tokens, completion_tokens, total_ms = _sum_token_usage(snapshot)
    tool_calls = extract_tool_calls_from_snapshot(snapshot)
    duration_ms = (ended_at - started_at).total_seconds() * 1000.0

    node = TraceNode(
        step=0,
        node_id=subject_name,
        node_type="Agent",
        inputs={"prompt_present": True},
        outputs=[output],
        started_at=started_at,
        ended_at=ended_at,
        duration_ms=duration_ms,
        from_node=None,
        tool_calls=tool_calls,
    )

    return EvalTrace(
        subject_name=subject_name,
        case_id=case_id,
        nodes=[node],
        completed=True,
        started_at=started_at,
        ended_at=ended_at,
        total_prompt_tokens=prompt_tokens,
        total_completion_tokens=completion_tokens,
        total_response_time_ms=total_ms,
        metrics_snapshot=snapshot,
    )


def build_eval_trace_from_flowchart(
    *,
    subject_name: str,
    case_id: str,
    collector: Any,
    runtime_trace: Optional[Any],
    started_at: datetime,
    ended_at: datetime,
) -> EvalTrace:
    """Build an :class:`EvalTrace` for a flowchart run.

    Args:
        runtime_trace: A :class:`pithos.flowchart.ExecutionTrace` from
            :meth:`Flowchart.get_execution_trace`, or ``None`` if
            tracing was not enabled.
    """
    snapshot = collector.get_snapshot() if collector is not None else {}
    prompt_tokens, completion_tokens, total_ms = _sum_token_usage(snapshot)
    aggregate_tool_calls = extract_tool_calls_from_snapshot(snapshot)

    nodes: list[TraceNode] = []
    completed = True
    if runtime_trace is not None and getattr(runtime_trace, "entries", None):
        completed = bool(getattr(runtime_trace, "completed", True))
        for entry in runtime_trace.entries:
            from_node = None
            edge = getattr(entry, "edge", None)
            if edge is not None:
                from_node = getattr(edge, "from_node", None)

            nodes.append(
                TraceNode(
                    step=int(getattr(entry, "step", 0)),
                    node_id=str(getattr(entry, "node_id", "")),
                    node_type=str(getattr(entry, "node_type", "")),
                    inputs=dict(getattr(entry, "inputs", {}) or {}),
                    outputs=list(getattr(entry, "outputs", []) or []),
                    started_at=getattr(entry, "timestamp_start", None),
                    ended_at=getattr(entry, "timestamp_end", None),
                    duration_ms=float(getattr(entry, "duration_ms", 0.0) or 0.0),
                    from_node=from_node,
                    tool_calls=[],
                )
            )
        # Attach aggregate tool calls to the first node so analyzers
        # have somewhere to read them from; per-step attribution is not
        # possible from the current MetricsCollector granularity.
        if nodes and aggregate_tool_calls:
            nodes[0].tool_calls = list(aggregate_tool_calls)
    else:
        # No runtime trace — fall back to the flowchart_paths recorded
        # on the metrics collector.
        for idx, path_entry in enumerate(snapshot.get("flowchart_paths") or []):
            nodes.append(
                TraceNode(
                    step=idx,
                    node_id=str(path_entry.get("node_id", "")),
                    node_type=str(path_entry.get("node_type", "")),
                    duration_ms=float(path_entry.get("duration_ms", 0.0) or 0.0),
                    from_node=path_entry.get("from_node"),
                )
            )
        if nodes and aggregate_tool_calls:
            nodes[0].tool_calls = list(aggregate_tool_calls)

    return EvalTrace(
        subject_name=subject_name,
        case_id=case_id,
        nodes=nodes,
        completed=completed,
        started_at=started_at,
        ended_at=ended_at,
        total_prompt_tokens=prompt_tokens,
        total_completion_tokens=completion_tokens,
        total_response_time_ms=total_ms,
        metrics_snapshot=snapshot,
    )
