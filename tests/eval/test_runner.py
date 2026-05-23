"""Tests for EvalRunner."""

from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any

import pytest

from pithos.eval import (
    EvalConfig,
    EvalExecutionConfig,
    EvalOutputConfig,
    EvalRunner,
    EvalTrace,
    GradeResult,
    RunContext,
    SubjectRun,
    TaskCase,
    TraceNode,
)
from pithos.eval.subjects.base import Subject

# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


class _StubSubject(Subject):
    def __init__(self, name: str, *, outputs: list[str], fail_until: int = 0):
        super().__init__(name)
        self._outputs = outputs
        self._idx = 0
        self.calls = 0
        self.fail_until = fail_until

    def run(self, case: TaskCase, ctx: RunContext) -> SubjectRun:
        self.calls += 1
        if self.calls <= self.fail_until:
            return SubjectRun(
                subject_name=self.name,
                case_id=case.case_id,
                output="",
                error="boom",
                started_at=datetime.now(),
                ended_at=datetime.now(),
            )
        out = self._outputs[self._idx % len(self._outputs)]
        self._idx += 1
        trace = EvalTrace(
            subject_name=self.name,
            case_id=case.case_id,
            nodes=[TraceNode(step=0, node_id="n1", node_type="agent", duration_ms=5)],
            total_response_time_ms=5.0,
            metrics_snapshot={
                "token_usage": {"m": {"prompt_tokens": 10, "completion_tokens": 5}}
            },
        )
        return SubjectRun(
            subject_name=self.name,
            case_id=case.case_id,
            output=out,
            metrics=None,
            trace=trace,
            started_at=datetime.now(),
            ended_at=datetime.now(),
        )


class _StubTask:
    def __init__(self, cases: list[TaskCase], expected: dict[str, str]):
        self._cases = cases
        self._expected = expected
        self.name = "stub_task"

    def cases(self):
        return list(self._cases)

    def grade(self, case, output, ctx=None):
        passed = output == self._expected.get(case.case_id)
        return GradeResult(grader="stub", score=100.0 if passed else 0.0, passed=passed)


class _ExplodingAnalyzer:
    analyzer_name = "exploder"

    def analyze(self, trace, ctx=None):
        raise RuntimeError("nope")


class _CountingAnalyzer:
    analyzer_name = "counter"

    def __init__(self):
        self.seen = 0

    def analyze(self, trace, ctx=None):
        self.seen += 1
        return []


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_config(tmp_path, rounds=1, retries=0, parallelism=1) -> EvalConfig:
    return EvalConfig(
        name="ut",
        subjects={},
        tasks={},
        execution=EvalExecutionConfig(
            rounds=rounds, num_retries=retries, parallelism=parallelism
        ),
        output=EvalOutputConfig(base_dir=str(tmp_path)),
    )


def _cases() -> list[TaskCase]:
    return [
        TaskCase(case_id="q1", task_type="free_form", prompt="hi", expected="a"),
        TaskCase(case_id="q2", task_type="free_form", prompt="hi", expected="b"),
    ]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_runner_executes_cases_and_writes_jsonl(tmp_path):
    config = _make_config(tmp_path)
    subj = _StubSubject("alpha", outputs=["a", "b"])
    task = _StubTask(_cases(), expected={"q1": "a", "q2": "b"})
    counter = _CountingAnalyzer()
    runner = EvalRunner(
        config,
        subjects={"alpha": subj},
        tasks={"stub": task},
        analyzers=[counter],
    )
    records = runner.run()
    assert len(records) == 2
    assert all(r.grade.passed for r in records)
    assert counter.seen == 2

    jsonl = os.path.join(config.run_dir, "cases", "round_1", "alpha__stub.jsonl")
    assert os.path.exists(jsonl)
    lines = open(jsonl, encoding="utf-8").read().strip().splitlines()
    assert len(lines) == 2
    payload = json.loads(lines[0])
    assert payload["case_id"] == "q1"
    assert payload["grade"]["score"] == 100.0


def test_runner_resume_skips_completed_cases(tmp_path):
    config = _make_config(tmp_path)
    subj1 = _StubSubject("alpha", outputs=["a", "b"])
    task = _StubTask(_cases(), expected={"q1": "a", "q2": "b"})

    EvalRunner(config, subjects={"alpha": subj1}, tasks={"stub": task}).run()
    assert subj1.calls == 2

    subj2 = _StubSubject("alpha", outputs=["a", "b"])
    runner2 = EvalRunner(config, subjects={"alpha": subj2}, tasks={"stub": task})
    records = runner2.run()
    # Subject not re-invoked thanks to resume.
    assert subj2.calls == 0
    assert len(records) == 2


def test_runner_no_resume_reruns(tmp_path):
    config = _make_config(tmp_path)
    subj1 = _StubSubject("alpha", outputs=["a", "b"])
    task = _StubTask(_cases(), expected={"q1": "a", "q2": "b"})
    EvalRunner(config, subjects={"alpha": subj1}, tasks={"stub": task}).run()

    subj2 = _StubSubject("alpha", outputs=["a", "b"])
    runner = EvalRunner(
        config, subjects={"alpha": subj2}, tasks={"stub": task}, resume=False
    )
    runner.run()
    assert subj2.calls == 2


def test_runner_retries_on_error(tmp_path):
    config = _make_config(tmp_path, retries=2)
    subj = _StubSubject("alpha", outputs=["a", "b"], fail_until=1)
    task = _StubTask(_cases(), expected={"q1": "a", "q2": "b"})
    records = EvalRunner(config, subjects={"alpha": subj}, tasks={"stub": task}).run()
    # First case retried once (fail_until=1 means first call fails),
    # then succeeds; second case succeeds first try.
    assert subj.calls == 3
    assert all(r.error is None for r in records)


def test_runner_records_failure_when_all_retries_exhausted(tmp_path):
    config = _make_config(tmp_path, retries=1)
    subj = _StubSubject("alpha", outputs=["a", "b"], fail_until=10)
    task = _StubTask(_cases(), expected={"q1": "a", "q2": "b"})
    records = EvalRunner(config, subjects={"alpha": subj}, tasks={"stub": task}).run()
    assert all(r.error is not None for r in records)
    assert all(r.grade.score == 0.0 for r in records)


def test_runner_swallows_analyzer_exceptions(tmp_path):
    config = _make_config(tmp_path)
    subj = _StubSubject("alpha", outputs=["a", "b"])
    task = _StubTask(_cases(), expected={"q1": "a", "q2": "b"})
    runner = EvalRunner(
        config,
        subjects={"alpha": subj},
        tasks={"stub": task},
        analyzers=[_ExplodingAnalyzer()],
    )
    records = runner.run()
    assert len(records) == 2
    # Issues list stays empty because the analyzer raised.
    assert all(r.issues == [] for r in records)


def test_runner_multi_round(tmp_path):
    config = _make_config(tmp_path, rounds=2)
    subj = _StubSubject("alpha", outputs=["a", "b"])
    task = _StubTask(_cases(), expected={"q1": "a", "q2": "b"})
    records = EvalRunner(config, subjects={"alpha": subj}, tasks={"stub": task}).run()
    assert len(records) == 4
    assert sorted(r.round_num for r in records) == [1, 1, 2, 2]


def test_runner_max_cases(tmp_path):
    config = _make_config(tmp_path)
    subj = _StubSubject("alpha", outputs=["a", "b"])
    task = _StubTask(_cases(), expected={"q1": "a", "q2": "b"})
    records = EvalRunner(
        config,
        subjects={"alpha": subj},
        tasks={"stub": task},
        max_cases_per_task=1,
    ).run()
    assert len(records) == 1
    assert records[0].case_id == "q1"
