"""Tests for trajectory analyzers + C.L.A.S.S. report."""

from datetime import datetime, timedelta

import pytest

from pithos.eval import (
    AnalyzerSpec,
    CaseRecord,
    EvalReport,
    EvalTrace,
    GradeResult,
    TraceNode,
    TrajectoryIssueSeverity,
)
from pithos.eval.metrics_view import build_class_report, to_dataframe
from pithos.eval.trace.analyzers import (
    AnalyzerContext,
    CostAnalyzer,
    LatencyAnalyzer,
    LoopDetector,
    RedundancyAnalyzer,
    StabilityAnalyzer,
    ToolHallucinationAnalyzer,
    build_analyzer,
    stability_for_subject,
)


def _trace(nodes, metrics=None):
    return EvalTrace(
        subject_name="s",
        case_id="q1",
        nodes=nodes,
        completed=True,
        metrics_snapshot=metrics or {},
    )


# --------------------------------------------------------------------------
# LoopDetector
# --------------------------------------------------------------------------


def test_loop_detector_flags_repeated_inputs():
    nodes = [
        TraceNode(step=i, node_id="n1", node_type="X", inputs={"a": 1})
        for i in range(4)
    ]
    issues = LoopDetector().analyze(_trace(nodes))
    cycles = [i for i in issues if i.code == "cycle"]
    assert len(cycles) == 1
    assert cycles[0].severity is TrajectoryIssueSeverity.ERROR
    assert cycles[0].detail["count"] == 4


def test_loop_detector_ignores_differing_inputs():
    nodes = [
        TraceNode(step=i, node_id="n1", node_type="X", inputs={"a": i})
        for i in range(5)
    ]
    # Inputs differ each step → no cycle, but node_id repeated > 5? It's 5, not >5.
    issues = LoopDetector().analyze(_trace(nodes))
    assert issues == []


def test_loop_detector_repeated_node_warning():
    nodes = [
        TraceNode(step=i, node_id="n1", node_type="X", inputs={"a": i})
        for i in range(7)
    ]
    issues = LoopDetector({"max_node_repeats": 5}).analyze(_trace(nodes))
    repeats = [i for i in issues if i.code == "repeated_node"]
    assert len(repeats) == 1
    assert repeats[0].severity is TrajectoryIssueSeverity.WARNING


# --------------------------------------------------------------------------
# RedundancyAnalyzer
# --------------------------------------------------------------------------


def test_redundancy_flags_run_of_identical_outputs():
    nodes = [
        TraceNode(step=i, node_id=f"n{i}", node_type="X", outputs=["same"])
        for i in range(4)
    ]
    issues = RedundancyAnalyzer().analyze(_trace(nodes))
    assert len(issues) == 1
    assert issues[0].code == "redundant_run"
    assert issues[0].detail["length"] == 4


def test_redundancy_clean_when_outputs_differ():
    nodes = [
        TraceNode(step=i, node_id=f"n{i}", node_type="X", outputs=[f"out{i}"])
        for i in range(3)
    ]
    assert RedundancyAnalyzer().analyze(_trace(nodes)) == []


# --------------------------------------------------------------------------
# ToolHallucinationAnalyzer
# --------------------------------------------------------------------------


class _FakeRegistry:
    def __init__(self, names):
        self._names = list(names)

    def list_tool_names(self):
        return self._names


def test_tool_hallucination_flags_unknown_tool():
    nodes = [
        TraceNode(
            step=0,
            node_id="n1",
            node_type="X",
            tool_calls=[{"tool": "shell"}, {"tool": "ghost"}],
        )
    ]
    ctx = AnalyzerContext(tool_registry=_FakeRegistry(["shell"]))
    issues = ToolHallucinationAnalyzer().analyze(_trace(nodes), ctx)
    assert len(issues) == 1
    assert issues[0].detail["tool"] == "ghost"
    assert issues[0].severity is TrajectoryIssueSeverity.ERROR


def test_tool_hallucination_silent_without_registry():
    nodes = [TraceNode(step=0, node_id="n1", node_type="X", tool_calls=[{"tool": "x"}])]
    assert ToolHallucinationAnalyzer().analyze(_trace(nodes)) == []


def test_tool_hallucination_strict_empty_registry_flags_all():
    nodes = [TraceNode(step=0, node_id="n1", node_type="X", tool_calls=[{"tool": "x"}])]
    issues = ToolHallucinationAnalyzer({"allow_empty_registry": False}).analyze(
        _trace(nodes)
    )
    assert len(issues) == 1


# --------------------------------------------------------------------------
# CostAnalyzer
# --------------------------------------------------------------------------


def test_cost_analyzer_with_price_map():
    trace = _trace(
        [],
        metrics={
            "token_usage": {
                "model-a": {"prompt_tokens": 1000, "completion_tokens": 500},
                "model-b": {"prompt_tokens": 2000, "completion_tokens": 0},
            }
        },
    )
    ctx = AnalyzerContext(
        price_map={
            "model-a": {"prompt_per_1k": 0.01, "completion_per_1k": 0.03},
            "model-b": {"prompt_per_1k": 0.002, "completion_per_1k": 0.006},
        }
    )
    issues = CostAnalyzer().analyze(trace, ctx)
    assert len(issues) == 1
    detail = issues[0].detail
    # model-a: 1*0.01 + 0.5*0.03 = 0.025; model-b: 2*0.002 = 0.004 → 0.029
    assert detail["estimated_cost_usd"] == pytest.approx(0.029)
    assert detail["per_model_usd"]["model-a"] == pytest.approx(0.025)


def test_cost_analyzer_warn_threshold():
    trace = _trace(
        [],
        metrics={"token_usage": {"m": {"prompt_tokens": 1000, "completion_tokens": 0}}},
    )
    ctx = AnalyzerContext(price_map={"m": {"prompt_per_1k": 1.0}})
    issues = CostAnalyzer({"warn_threshold_usd": 0.5}).analyze(trace, ctx)
    assert issues[0].severity is TrajectoryIssueSeverity.WARNING


# --------------------------------------------------------------------------
# LatencyAnalyzer
# --------------------------------------------------------------------------


def test_latency_analyzer_summary_and_slow_steps():
    nodes = [
        TraceNode(step=0, node_id="n1", node_type="X", duration_ms=10.0),
        TraceNode(step=1, node_id="n2", node_type="X", duration_ms=500.0),
    ]
    trace = _trace(nodes)
    trace.total_response_time_ms = 510.0
    issues = LatencyAnalyzer({"slow_step_ms": 100, "slow_total_ms": 1000}).analyze(
        trace
    )
    # 1 summary + 1 slow_step
    assert len(issues) == 2
    codes = {i.code for i in issues}
    assert codes == {"latency_summary", "slow_step"}


def test_latency_summary_severity_escalates():
    trace = _trace([TraceNode(step=0, node_id="n1", node_type="X", duration_ms=2000)])
    trace.total_response_time_ms = 2000.0
    issues = LatencyAnalyzer({"slow_total_ms": 100}).analyze(trace)
    assert issues[0].severity is TrajectoryIssueSeverity.WARNING


# --------------------------------------------------------------------------
# Stability
# --------------------------------------------------------------------------


def test_stability_warns_on_few_rounds():
    mean, std, issues = stability_for_subject([90, 95])
    codes = {i.code for i in issues}
    assert "insufficient_rounds" in codes
    assert mean == pytest.approx(92.5)
    assert std > 0


def test_stability_warns_on_high_variance():
    mean, std, issues = stability_for_subject([90, 30, 70])
    assert any(i.code == "high_variance" for i in issues)


def test_stability_analyzer_wrapper():
    trace = _trace([], metrics={"_round_scores": [50, 50, 50]})
    issues = StabilityAnalyzer().analyze(trace)
    assert issues == []


# --------------------------------------------------------------------------
# Build dispatcher
# --------------------------------------------------------------------------


def test_build_analyzer_dispatches():
    assert isinstance(build_analyzer(AnalyzerSpec(type="loop_detector")), LoopDetector)
    assert isinstance(
        build_analyzer(AnalyzerSpec(type="redundancy")), RedundancyAnalyzer
    )
    assert isinstance(
        build_analyzer(AnalyzerSpec(type="tool_hallucination")),
        ToolHallucinationAnalyzer,
    )
    assert isinstance(build_analyzer(AnalyzerSpec(type="cost")), CostAnalyzer)
    assert isinstance(build_analyzer(AnalyzerSpec(type="latency")), LatencyAnalyzer)


def test_build_analyzer_rejects_unknown():
    with pytest.raises(ValueError, match="Unknown analyzer type"):
        build_analyzer(AnalyzerSpec(type="bogus"))


# --------------------------------------------------------------------------
# C.L.A.S.S. report
# --------------------------------------------------------------------------


def _make_record(subject, round_num, score, end_ms=100.0, tokens=None):
    snap = (
        {"token_usage": {"m": {"prompt_tokens": tokens or 0, "completion_tokens": 0}}}
        if tokens
        else {}
    )
    nodes = [TraceNode(step=0, node_id="n", node_type="X", duration_ms=end_ms)]
    trace = EvalTrace(
        subject_name=subject,
        case_id=f"c{round_num}",
        nodes=nodes,
        total_response_time_ms=end_ms,
        metrics_snapshot=snap,
    )
    return CaseRecord(
        subject_name=subject,
        case_id=f"c{round_num}",
        round_num=round_num,
        task_type="free_form",
        output="o",
        grade=GradeResult(grader="g", score=score, passed=score >= 50),
        trace=trace,
        metrics_snapshot=snap,
    )


def test_build_class_report_basic_shape():
    records = [
        _make_record("alpha", 1, 100, end_ms=50.0, tokens=1000),
        _make_record("alpha", 2, 80, end_ms=70.0, tokens=2000),
        _make_record("beta", 1, 40, end_ms=200.0),
        _make_record("beta", 2, 60, end_ms=300.0),
    ]
    report = EvalReport(config_name="X", rounds=2, case_records=records)
    rows = build_class_report(
        report,
        price_map={"m": {"prompt_per_1k": 0.001, "completion_per_1k": 0.0}},
    )
    by_subject = {r["subject"]: r for r in rows}
    assert by_subject["alpha"]["accuracy_mean"] == 90.0
    assert by_subject["alpha"]["cost_usd"] == pytest.approx(0.003)
    assert by_subject["alpha"]["case_count"] == 2
    assert by_subject["beta"]["accuracy_mean"] == 50.0
    assert by_subject["alpha"]["latency_ms_avg"] == 60.0
    # Sorted by accuracy desc
    assert rows[0]["subject"] == "alpha"


def test_build_class_report_to_dataframe():
    pd = pytest.importorskip("pandas")
    records = [_make_record("a", 1, 50)]
    df = to_dataframe(
        build_class_report(EvalReport(config_name="X", rounds=1, case_records=records))
    )
    assert "accuracy_mean" in df.columns
    assert df.index.name == "subject"
    assert df.loc["a", "accuracy_mean"] == 50.0
