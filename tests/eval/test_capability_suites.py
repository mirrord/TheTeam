"""Phase 6 — capability suites (tool use, memory recall, self reflection)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from pithos.eval.config import GraderSpec, TaskSpec
from pithos.eval.datasets import (
    MemoryRecallDataset,
    SelfReflectionDataset,
    ToolUseDataset,
    build_dataset,
)
from pithos.eval.graders import (
    MemoryRecallGrader,
    ToolTraceGrader,
    build_grader,
)
from pithos.eval.models import EvalTrace, RunContext, SubjectRun, TaskCase, TraceNode
from pithos.eval.tasks import (
    MemoryRecallTask,
    SelfReflectionTask,
    ToolUseTask,
    build_task,
)

# ---------------------------------------------------------------------------
# ToolTraceGrader
# ---------------------------------------------------------------------------


def _trace_with_tools(*tool_names: str) -> EvalTrace:
    node = TraceNode(
        step=0,
        node_id="root",
        node_type="agent",
        tool_calls=[{"tool": t, "total_calls": 1} for t in tool_names],
    )
    return EvalTrace(subject_name="s", case_id="c", nodes=[node])


def test_tool_trace_grader_set_full_match():
    g = ToolTraceGrader()
    trace = _trace_with_tools("read_file", "list_dir")
    ctx = RunContext(extras={"trace": trace})
    res = g.grade(
        "",
        {"tools": ["read_file", "list_dir"]},
        case=TaskCase(case_id="x", task_type="tool_use", prompt=""),
        ctx=ctx,
    )
    assert res.score == 100.0
    assert res.passed
    assert res.detail["missing"] == []


def test_tool_trace_grader_set_partial():
    g = ToolTraceGrader()
    trace = _trace_with_tools("read_file")
    ctx = RunContext(extras={"trace": trace})
    res = g.grade(
        "",
        {"tools": ["read_file", "list_dir"]},
        ctx=ctx,
    )
    assert res.score == 50.0
    assert not res.passed
    assert "list_dir" in res.detail["missing"]


def test_tool_trace_grader_ordered_sequence_match():
    g = ToolTraceGrader({"ordered": True})
    trace = _trace_with_tools("a", "b", "c")
    ctx = RunContext(extras={"trace": trace})
    res = g.grade("", {"tools": ["a", "b"]}, ctx=ctx)
    assert res.score == 100.0


def test_tool_trace_grader_ordered_sequence_mismatch():
    g = ToolTraceGrader({"ordered": True})
    trace = _trace_with_tools("b", "a")
    ctx = RunContext(extras={"trace": trace})
    res = g.grade("", {"tools": ["a", "b"]}, ctx=ctx)
    assert res.score == 0.0


def test_tool_trace_grader_empty_trace_with_expectation():
    g = ToolTraceGrader()
    ctx = RunContext(extras={"trace": None})
    res = g.grade("", {"tools": ["read_file"]}, ctx=ctx)
    assert res.score == 0.0
    assert res.detail["recorded"] == []


def test_tool_trace_grader_no_expectation_no_calls_passes():
    g = ToolTraceGrader({"penalize_extras": True})
    ctx = RunContext(extras={"trace": _trace_with_tools()})
    res = g.grade("", {"tools": []}, ctx=ctx)
    assert res.score == 100.0


def test_tool_trace_grader_penalize_extras():
    g = ToolTraceGrader({"penalize_extras": True})
    ctx = RunContext(extras={"trace": _trace_with_tools("read_file", "junk", "junk2")})
    res = g.grade("", {"tools": ["read_file"]}, ctx=ctx)
    # full match (100) minus 25 per extra (2 extras) = 50
    assert res.score == 50.0


# ---------------------------------------------------------------------------
# MemoryRecallGrader
# ---------------------------------------------------------------------------


def test_memory_recall_grader_substring_match():
    g = MemoryRecallGrader()
    res = g.grade("Your favourite colour is Teal, of course.", "teal")
    assert res.score == 100.0
    assert res.passed
    assert res.detail["matched_in_output"] is True


def test_memory_recall_grader_no_match():
    g = MemoryRecallGrader()
    res = g.grade("I don't recall.", "teal")
    assert res.score == 0.0
    assert not res.passed


def test_memory_recall_grader_dict_expected():
    g = MemoryRecallGrader()
    res = g.grade("The dog is a Border Collie.", {"recall": "border collie"})
    assert res.passed


def test_memory_recall_grader_reads_memory_hits():
    g = MemoryRecallGrader()
    trace = EvalTrace(
        subject_name="s",
        case_id="c",
        nodes=[],
        metrics_snapshot={"memory": {"hits": 3}},
    )
    run = SubjectRun(subject_name="s", case_id="c", output="ok", trace=trace)
    ctx = RunContext(extras={"trace": trace, "subject_run": run})
    res = g.grade("Your name is Alice.", "alice", ctx=ctx)
    assert res.detail["memory_hits"] == 3


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------


def test_tool_use_dataset_yields_setup_free_cases(tmp_path):
    p = tmp_path / "tu.json"
    p.write_text(
        '[{"prompt": "do X", "expected_tools": ["read_file"]}]',
        encoding="utf-8",
    )
    ds = ToolUseDataset(path=str(p))
    cases = list(ds.cases())
    assert len(cases) == 1
    c = cases[0]
    assert c.task_type == "tool_use"
    assert c.expected == {"tools": ["read_file"]}
    assert c.setup_prompts == []
    assert c.metadata["expected_tools"] == ["read_file"]


def test_memory_recall_dataset_pulls_setup_prompts(tmp_path):
    p = tmp_path / "mr.json"
    p.write_text(
        '[{"seed_prompts": ["My name is Bo."], '
        '"recall_prompt": "Who am I?", "expected_recall": "Bo"}]',
        encoding="utf-8",
    )
    ds = MemoryRecallDataset(path=str(p))
    case = next(iter(ds.cases()))
    assert case.setup_prompts == ["My name is Bo."]
    assert case.prompt == "Who am I?"
    assert case.expected == "Bo"


def test_memory_recall_dataset_accepts_singular_seed(tmp_path):
    p = tmp_path / "mr.json"
    p.write_text(
        '[{"seed_prompt": "Hi.", "recall_prompt": "Q?", ' '"expected_recall": "A"}]',
        encoding="utf-8",
    )
    case = next(iter(MemoryRecallDataset(path=str(p)).cases()))
    assert case.setup_prompts == ["Hi."]


def test_self_reflection_dataset(tmp_path):
    p = tmp_path / "sr.json"
    p.write_text(
        '[{"prompt": "Capital of AU is Sydney; correct?", '
        '"expected_correction": "Canberra"}]',
        encoding="utf-8",
    )
    case = next(iter(SelfReflectionDataset(path=str(p)).cases()))
    assert case.expected == "Canberra"
    assert case.task_type == "self_reflection"


def test_builtin_resolution_via_build_dataset():
    ds = build_dataset({"type": "tool_use", "builtin": "tool_use_basic"})
    assert isinstance(ds, ToolUseDataset)
    cases = list(ds.cases())
    assert len(cases) > 0
    assert all(c.task_type == "tool_use" for c in cases)


def test_builtin_memory_recall_loads():
    ds = build_dataset({"type": "memory_recall", "builtin": "memory_recall_basic"})
    cases = list(ds.cases())
    assert cases
    assert all(c.setup_prompts for c in cases)


def test_builtin_self_reflection_loads():
    ds = build_dataset({"type": "self_reflection", "builtin": "self_reflection_basic"})
    cases = list(ds.cases())
    assert cases


def test_builtin_missing_raises():
    with pytest.raises(FileNotFoundError):
        build_dataset({"type": "tool_use", "builtin": "does_not_exist"})


def test_build_dataset_requires_path_or_builtin():
    with pytest.raises(ValueError):
        build_dataset({"type": "tool_use"})


# ---------------------------------------------------------------------------
# Tasks (registry + delegation)
# ---------------------------------------------------------------------------


def test_build_task_dispatches_all_capability_types():
    for ttype, builtin, grader in [
        ("tool_use", "tool_use_basic", GraderSpec(type="tool_trace")),
        ("memory_recall", "memory_recall_basic", GraderSpec(type="memory_recall")),
        (
            "self_reflection",
            "self_reflection_basic",
            GraderSpec(type="regex_match", config={"pattern": ".*"}),
        ),
    ]:
        spec = TaskSpec(
            name=f"t_{ttype}",
            type=ttype,
            dataset={"type": ttype, "builtin": builtin},
            grader=grader,
        )
        task = build_task(spec)
        assert task.name == f"t_{ttype}"
        assert list(task.cases())


def test_tool_use_task_delegates_to_grader():
    grader = ToolTraceGrader()
    ds = ToolUseDataset.__new__(ToolUseDataset)
    task = ToolUseTask(name="tu", dataset=ds, grader=grader)
    case = TaskCase(
        case_id="c0",
        task_type="tool_use",
        prompt="",
        expected={"tools": ["read_file"]},
    )
    ctx = RunContext(extras={"trace": _trace_with_tools("read_file")})
    res = task.grade(case, "", ctx)
    assert res.passed


def test_memory_recall_task_delegates():
    grader = MemoryRecallGrader()
    ds = MemoryRecallDataset.__new__(MemoryRecallDataset)
    task = MemoryRecallTask(name="mr", dataset=ds, grader=grader)
    case = TaskCase(case_id="c0", task_type="memory_recall", prompt="", expected="bo")
    res = task.grade(case, "Your name is Bo.", RunContext())
    assert res.passed


def test_self_reflection_task_delegates():
    from pithos.eval.graders.regex_match import RegexMatchGrader

    grader = RegexMatchGrader({"pattern": r"Canberra"})
    ds = SelfReflectionDataset.__new__(SelfReflectionDataset)
    task = SelfReflectionTask(name="sr", dataset=ds, grader=grader)
    case = TaskCase(
        case_id="c0", task_type="self_reflection", prompt="", expected="Canberra"
    )
    res = task.grade(case, "Actually, the capital is Canberra.", RunContext())
    assert res.passed


# ---------------------------------------------------------------------------
# AgentSubject setup_prompts loop
# ---------------------------------------------------------------------------


def test_agent_subject_replays_setup_prompts():
    from pithos.eval.subjects.agent import AgentSubject

    sent: list[str] = []

    class _StubAgent:
        name = "stub"

        def attach_metrics(self, collector):
            self.collector = collector

        def enable_trace(self):
            pass

        def reset(self):
            pass

        def send(self, prompt):
            sent.append(prompt)
            return f"echo:{prompt}"

    subject = AgentSubject("stub", {"instance": _StubAgent()})

    case = TaskCase(
        case_id="c0",
        task_type="memory_recall",
        prompt="What did I tell you?",
        setup_prompts=["My name is Alice.", "I live in Reykjavik."],
    )
    run = subject.run(case, RunContext())
    assert run.error is None
    assert sent == ["My name is Alice.", "I live in Reykjavik.", "What did I tell you?"]
    assert run.output == "echo:What did I tell you?"


# ---------------------------------------------------------------------------
# Runner pre-populates ctx.extras["trace"] before grading
# ---------------------------------------------------------------------------


def test_runner_populates_ctx_trace_before_grade():
    from pithos.eval.config import EvalConfig, EvalExecutionConfig
    from pithos.eval.runner import EvalRunner

    trace = _trace_with_tools("read_file")

    class _Sub:
        name = "sub"

        def run(self, case, ctx):
            from datetime import datetime

            return SubjectRun(
                subject_name=self.name,
                case_id=case.case_id,
                output="ok",
                trace=trace,
                started_at=datetime.utcnow(),
                ended_at=datetime.utcnow(),
            )

    captured: dict = {}

    class _Task:
        name = "t"

        def cases(self):
            return [
                TaskCase(
                    case_id="c0",
                    task_type="tool_use",
                    prompt="x",
                    expected={"tools": ["read_file"]},
                )
            ]

        def grade(self, case, output, ctx):
            captured["trace"] = ctx.extras.get("trace")
            captured["run"] = ctx.extras.get("subject_run")
            from pithos.eval.models import GradeResult

            return GradeResult(grader="t", score=100.0, passed=True)

    cfg = EvalConfig(
        name="x",
        subjects={},
        tasks={},
        execution=EvalExecutionConfig(rounds=1, num_retries=0, parallelism=1),
    )
    runner = EvalRunner(
        cfg,
        subjects={"sub": _Sub()},
        tasks={"t": _Task()},
        write_outputs=False,
        resume=False,
    )
    runner.run()
    assert captured["trace"] is trace
    assert captured["run"] is not None
