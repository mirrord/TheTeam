"""Tests for Subject adapters (agent / flowchart / team)."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from pithos.eval import RunContext, SubjectSpec, TaskCase
from pithos.eval.subjects import (
    AgentSubject,
    FlowchartSubject,
    TeamSubject,
    build_subject,
)


def _mock_collector_snapshot():
    return {
        "token_usage": {
            "m": {
                "prompt_tokens": 2,
                "completion_tokens": 3,
                "total_response_time_ms": 40.0,
            }
        },
        "tool_calls": {},
        "memory": {},
        "flowchart_paths": [],
    }


# --------------------------------------------------------------------------
# AgentSubject
# --------------------------------------------------------------------------


def test_agent_subject_runs_injected_instance(monkeypatch):
    fake_agent = MagicMock()
    fake_agent.send.return_value = "answer"

    # Patch MetricsCollector to a mock with a deterministic snapshot
    fake_collector = MagicMock()
    fake_collector.get_snapshot.return_value = _mock_collector_snapshot()
    monkeypatch.setattr("pithos.metrics.MetricsCollector", lambda: fake_collector)

    subject = AgentSubject("a1", {"instance": fake_agent})
    case = TaskCase(case_id="q1", task_type="free_form", prompt="2+2?")

    run = subject.run(case, RunContext())

    assert run.output == "answer"
    assert run.error is None
    assert run.subject_name == "a1"
    fake_agent.attach_metrics.assert_called_once_with(fake_collector)
    fake_agent.send.assert_called_once_with("2+2?")
    assert run.trace is not None
    assert run.trace.total_prompt_tokens == 2
    assert run.trace.total_completion_tokens == 3


def test_agent_subject_captures_exception_as_error(monkeypatch):
    fake_agent = MagicMock()
    fake_agent.send.side_effect = RuntimeError("boom")
    fake_collector = MagicMock()
    fake_collector.get_snapshot.return_value = _mock_collector_snapshot()
    monkeypatch.setattr("pithos.metrics.MetricsCollector", lambda: fake_collector)

    subject = AgentSubject("a1", {"instance": fake_agent})
    run = subject.run(
        TaskCase(case_id="q1", task_type="free_form", prompt="x"),
        RunContext(),
    )

    assert run.error is not None
    assert "RuntimeError" in run.error
    assert run.output == ""


# --------------------------------------------------------------------------
# FlowchartSubject
# --------------------------------------------------------------------------


def test_flowchart_subject_runs_injected_instance(monkeypatch):
    fake_flow = MagicMock()
    fake_flow.start_node = "start"
    fake_flow.run.return_value = "flow-result"
    fake_flow.get_execution_trace.return_value = SimpleNamespace(
        entries=[], completed=True
    )

    fake_collector = MagicMock()
    fake_collector.get_snapshot.return_value = _mock_collector_snapshot()
    monkeypatch.setattr("pithos.metrics.MetricsCollector", lambda: fake_collector)

    fake_agent = MagicMock()
    subject = FlowchartSubject(
        "f1",
        {"instance": fake_flow, "agents_instance": {"alpha": fake_agent}},
    )
    run = subject.run(
        TaskCase(case_id="q1", task_type="free_form", prompt="p"),
        RunContext(),
    )

    assert run.output == "flow-result"
    assert run.error is None
    fake_flow.attach_metrics.assert_called_once()
    fake_flow.enable_trace.assert_called_once()
    fake_flow.reset.assert_called_once()
    fake_flow.run.assert_called_once()
    fake_agent.attach_metrics.assert_called_once_with(fake_collector)


def test_flowchart_subject_requires_flowchart_name_when_no_instance():
    subject = FlowchartSubject("f1", {})
    with pytest.raises(ValueError, match="'flowchart' config key required"):
        subject.run(
            TaskCase(case_id="q1", task_type="free_form", prompt="x"),
            RunContext(),
        )


# --------------------------------------------------------------------------
# TeamSubject
# --------------------------------------------------------------------------


def test_team_subject_runs_injected_team(monkeypatch):
    fake_team = MagicMock()
    fake_team.agents = {"a": MagicMock(), "b": MagicMock()}
    fake_team.run.return_value = "team-output"

    fake_collector = MagicMock()
    fake_collector.get_snapshot.return_value = _mock_collector_snapshot()
    monkeypatch.setattr("pithos.metrics.MetricsCollector", lambda: fake_collector)

    subject = TeamSubject("t1", {"instance": fake_team})
    run = subject.run(
        TaskCase(case_id="q1", task_type="free_form", prompt="p"),
        RunContext(),
    )

    assert run.output == "team-output"
    assert run.error is None
    for member in fake_team.agents.values():
        member.attach_metrics.assert_called_once_with(fake_collector)


def test_team_subject_requires_instance():
    subject = TeamSubject("t1", {})
    with pytest.raises(ValueError, match="preconstructed"):
        subject.run(
            TaskCase(case_id="q1", task_type="free_form", prompt="x"),
            RunContext(),
        )


# --------------------------------------------------------------------------
# build_subject dispatch
# --------------------------------------------------------------------------


def test_build_subject_dispatches_by_type():
    spec_agent = SubjectSpec(name="a", type="agent", config={"model": "m"})
    spec_flow = SubjectSpec(name="f", type="flowchart", config={"flowchart": "x"})
    spec_team = SubjectSpec(name="t", type="team", config={})

    assert isinstance(build_subject(spec_agent), AgentSubject)
    assert isinstance(build_subject(spec_flow), FlowchartSubject)
    assert isinstance(build_subject(spec_team), TeamSubject)


def test_build_subject_rejects_unknown_type():
    with pytest.raises(ValueError, match="Unknown subject type"):
        build_subject(SubjectSpec(name="x", type="bogus", config={}))
