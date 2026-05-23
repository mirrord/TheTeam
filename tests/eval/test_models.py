"""Tests for pithos.eval core dataclasses."""

from datetime import datetime, timedelta

from pithos.eval import (
    CaseRecord,
    EvalReport,
    EvalTrace,
    GradeResult,
    RunContext,
    SubjectRun,
    TaskCase,
    TraceNode,
    TrajectoryIssue,
    TrajectoryIssueSeverity,
)


def test_task_case_defaults():
    c = TaskCase(case_id="q1", task_type="multiple_choice", prompt="What is 2+2?")
    assert c.case_id == "q1"
    assert c.expected is None
    assert c.metadata == {}


def test_run_context_defaults():
    ctx = RunContext()
    assert ctx.round_num == 1
    assert ctx.case_index == 0
    assert ctx.tool_registry is None
    assert ctx.extras == {}


def test_subject_run_duration_ms():
    start = datetime(2026, 1, 1, 12, 0, 0)
    end = start + timedelta(milliseconds=250)
    run = SubjectRun(
        subject_name="a",
        case_id="q1",
        output="hi",
        started_at=start,
        ended_at=end,
    )
    assert run.duration_ms == 250.0


def test_subject_run_duration_zero_when_missing_timestamps():
    run = SubjectRun(subject_name="a", case_id="q1", output="hi")
    assert run.duration_ms == 0.0


def test_eval_trace_totals():
    nodes = [
        TraceNode(step=0, node_id="n1", node_type="PromptNode"),
        TraceNode(step=1, node_id="n2", node_type="PromptNode"),
    ]
    trace = EvalTrace(
        subject_name="a",
        case_id="q1",
        nodes=nodes,
        total_prompt_tokens=10,
        total_completion_tokens=5,
        total_response_time_ms=100.0,
    )
    assert trace.total_steps == 2
    assert trace.end_to_end_ms == 100.0


def test_eval_trace_end_to_end_prefers_timestamps():
    start = datetime(2026, 1, 1, 12, 0, 0)
    end = start + timedelta(milliseconds=500)
    trace = EvalTrace(
        subject_name="a",
        case_id="q1",
        started_at=start,
        ended_at=end,
        total_response_time_ms=999.0,
    )
    assert trace.end_to_end_ms == 500.0


def test_trajectory_issue_defaults():
    issue = TrajectoryIssue(
        analyzer="loop_detector",
        code="cycle",
        message="3-step cycle detected",
    )
    assert issue.severity is TrajectoryIssueSeverity.WARNING
    assert issue.step is None
    assert issue.detail == {}


def test_grade_result():
    g = GradeResult(grader="letter_match", score=100.0, passed=True)
    assert g.score == 100.0
    assert g.passed is True


def test_case_record_minimal():
    g = GradeResult(grader="letter_match", score=0.0, passed=False)
    rec = CaseRecord(
        subject_name="a",
        case_id="q1",
        round_num=1,
        task_type="multiple_choice",
        output="",
        grade=g,
    )
    assert rec.issues == []
    assert rec.trace is None


def test_eval_report_defaults():
    rep = EvalReport(config_name="X", rounds=1)
    assert rep.case_records == []
    assert rep.per_subject_stats == {}
    assert rep.class_report == {}
    assert isinstance(rep.generated_at, datetime)
