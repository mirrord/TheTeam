"""Tests for trace ingestion (snapshot + ExecutionTrace -> EvalTrace)."""

from datetime import datetime, timedelta
from types import SimpleNamespace
from unittest.mock import MagicMock

from pithos.eval.trace.ingest import (
    build_eval_trace_from_agent,
    build_eval_trace_from_flowchart,
    extract_tool_calls_from_snapshot,
)


def _snapshot(token_usage=None, tool_calls=None, flowchart_paths=None):
    return {
        "token_usage": token_usage or {},
        "tool_calls": tool_calls or {},
        "memory": {},
        "flowchart_paths": flowchart_paths or [],
    }


def test_extract_tool_calls_flattens_per_tool_aggregates():
    snap = _snapshot(
        tool_calls={
            "shell": {
                "successes": 2,
                "failures": 1,
                "total_calls": 3,
                "total_execution_time_ms": 12.5,
            }
        }
    )
    out = extract_tool_calls_from_snapshot(snap)
    assert len(out) == 1
    assert out[0]["tool"] == "shell"
    assert out[0]["successes"] == 2
    assert out[0]["failures"] == 1
    assert out[0]["execution_time_ms"] == 12.5


def test_build_eval_trace_from_agent_sums_tokens_and_synthesizes_node():
    collector = MagicMock()
    collector.get_snapshot.return_value = _snapshot(
        token_usage={
            "m1": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_response_time_ms": 100.0,
            },
            "m2": {
                "prompt_tokens": 3,
                "completion_tokens": 2,
                "total_response_time_ms": 25.0,
            },
        },
        tool_calls={
            "echo": {
                "successes": 1,
                "failures": 0,
                "total_calls": 1,
                "total_execution_time_ms": 1.0,
            }
        },
    )
    start = datetime(2026, 1, 1, 12, 0, 0)
    end = start + timedelta(milliseconds=250)

    trace = build_eval_trace_from_agent(
        subject_name="agentA",
        case_id="q1",
        collector=collector,
        started_at=start,
        ended_at=end,
        output="hello",
    )

    assert trace.subject_name == "agentA"
    assert trace.case_id == "q1"
    assert trace.total_prompt_tokens == 13
    assert trace.total_completion_tokens == 7
    assert trace.total_response_time_ms == 125.0
    assert trace.completed is True
    assert len(trace.nodes) == 1
    node = trace.nodes[0]
    assert node.node_type == "Agent"
    assert node.outputs == ["hello"]
    assert node.duration_ms == 250.0
    assert len(node.tool_calls) == 1
    assert node.tool_calls[0]["tool"] == "echo"


def test_build_eval_trace_from_flowchart_uses_execution_trace_entries():
    start = datetime(2026, 1, 1, 12, 0, 0)
    e1_start = start
    e1_end = start + timedelta(milliseconds=10)
    e2_start = e1_end
    e2_end = e2_start + timedelta(milliseconds=20)

    entry1 = SimpleNamespace(
        step=0,
        node_id="n1",
        node_type="PromptNode",
        timestamp_start=e1_start,
        timestamp_end=e1_end,
        duration_ms=10.0,
        inputs={"in": "x"},
        outputs=["a"],
        edge=None,
    )
    entry2 = SimpleNamespace(
        step=1,
        node_id="n2",
        node_type="RouterNode",
        timestamp_start=e2_start,
        timestamp_end=e2_end,
        duration_ms=20.0,
        inputs={},
        outputs=["b"],
        edge=SimpleNamespace(from_node="n1", to_node="n2"),
    )
    runtime_trace = SimpleNamespace(entries=[entry1, entry2], completed=True)

    collector = MagicMock()
    collector.get_snapshot.return_value = _snapshot(
        token_usage={
            "m": {
                "prompt_tokens": 1,
                "completion_tokens": 1,
                "total_response_time_ms": 50.0,
            }
        },
        tool_calls={
            "shell": {
                "successes": 1,
                "failures": 0,
                "total_calls": 1,
                "total_execution_time_ms": 5.0,
            }
        },
    )

    trace = build_eval_trace_from_flowchart(
        subject_name="flowA",
        case_id="q1",
        collector=collector,
        runtime_trace=runtime_trace,
        started_at=start,
        ended_at=e2_end,
    )

    assert trace.total_steps == 2
    assert trace.nodes[0].node_id == "n1"
    assert trace.nodes[0].from_node is None
    assert trace.nodes[1].from_node == "n1"
    assert trace.nodes[0].tool_calls[0]["tool"] == "shell"
    assert trace.completed is True


def test_build_eval_trace_from_flowchart_falls_back_to_flowchart_paths():
    collector = MagicMock()
    collector.get_snapshot.return_value = _snapshot(
        flowchart_paths=[
            {
                "node_id": "n1",
                "node_type": "PromptNode",
                "duration_ms": 5.0,
                "from_node": None,
            },
            {
                "node_id": "n2",
                "node_type": "PromptNode",
                "duration_ms": 7.0,
                "from_node": "n1",
            },
        ]
    )
    start = datetime(2026, 1, 1, 12, 0, 0)
    end = start + timedelta(milliseconds=12)

    trace = build_eval_trace_from_flowchart(
        subject_name="flowA",
        case_id="q1",
        collector=collector,
        runtime_trace=None,
        started_at=start,
        ended_at=end,
    )

    assert trace.total_steps == 2
    assert trace.nodes[0].node_id == "n1"
    assert trace.nodes[1].from_node == "n1"
