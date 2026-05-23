"""Showcase the major features of ``pithos.eval``.

Run with::

    .\\.venv\\Scripts\\python.exe demos/eval_demo.py

The demo is fully self-contained — it does **not** require Ollama or any
LLM backend. Every subject is a deterministic stub so the output is
reproducible and the demo doubles as a smoke test for the public API.

Sections
--------
1. Built-in dataset discovery.
2. Grader showcase (one example per built-in grader).
3. Trajectory analyzer showcase against a hand-crafted trace.
4. End-to-end mini run: two stub subjects → CaseRecords → C.L.A.S.S. report.
"""

from __future__ import annotations

import random
import textwrap
from datetime import datetime, timedelta
from typing import Optional

# Seed for reproducible output (multiple_choice datasets shuffle choices via
# the global random state, and letter_match scoring depends on the order).
random.seed(0)

from pithos.eval import (
    AnalyzerSpec,
    CaseRecord,
    EvalConfig,
    EvalExecutionConfig,
    EvalOutputConfig,
    EvalTrace,
    GradeResult,
    GraderSpec,
    Reporter,
    RunContext,
    SubjectRun,
    SubjectSpec,
    TaskCase,
    TaskSpec,
    TraceNode,
    build_analyzer,
)
from pithos.eval.datasets import build_dataset
from pithos.eval.graders import build_grader
from pithos.eval.runner import EvalRunner
from pithos.eval.subjects.base import Subject
from pithos.eval.tasks import build_task

# ---------------------------------------------------------------------------
# Small console helpers
# ---------------------------------------------------------------------------


def header(title: str) -> None:
    bar = "=" * 72
    print(f"\n{bar}\n  {title}\n{bar}")


def subheader(title: str) -> None:
    print(f"\n--- {title} ---")


# ---------------------------------------------------------------------------
# Section 1 — built-in datasets
# ---------------------------------------------------------------------------


def demo_builtin_datasets() -> None:
    header("1. Built-in datasets (src/pithos/eval/datasets/builtins/)")

    builtins = [
        ("multiple_choice", "linguistic_basic"),
        ("tool_use", "tool_use_basic"),
        ("memory_recall", "memory_recall_basic"),
        ("self_reflection", "self_reflection_basic"),
    ]
    for ttype, name in builtins:
        ds = build_dataset({"type": ttype, "builtin": name})
        cases = list(ds.cases())
        print(f"\n  {name} ({ttype}) — {len(cases)} cases")
        sample = cases[0]
        prompt_preview = textwrap.shorten(sample.prompt, width=90, placeholder=" …")
        print(f"    case_id : {sample.case_id}")
        print(f"    prompt  : {prompt_preview}")
        print(
            f"    expected: {textwrap.shorten(repr(sample.expected), 70, placeholder=' …')}"
        )
        if sample.setup_prompts:
            print(f"    setup   : {len(sample.setup_prompts)} prior turn(s)")


# ---------------------------------------------------------------------------
# Section 2 — graders
# ---------------------------------------------------------------------------


def _show_grade(label: str, result: GradeResult) -> None:
    print(f"  {label:18s} → score={result.score:5.1f}  passed={result.passed}")
    if result.detail:
        compact = {k: v for k, v in result.detail.items() if k != "raw"}
        print(f"  {'':18s}   detail={compact}")


def demo_graders() -> None:
    header("2. Graders")

    subheader('letter_match — pull {"ANSWER": "<letter>"} from output')
    g = build_grader(GraderSpec(type="letter_match"))
    _show_grade(
        "correct",
        g.grade('The answer is {"ANSWER": "B"}', expected="B"),
    )
    _show_grade(
        "wrong",
        g.grade('I think {"ANSWER": "A"}', expected="B"),
    )

    subheader("exact_match — case-insensitive string equality")
    g = build_grader(GraderSpec(type="exact_match", config={"case_sensitive": False}))
    _show_grade("equal", g.grade("Paris", expected="paris"))
    _show_grade("differs", g.grade("London", expected="paris"))

    subheader("regex_match — pattern hit anywhere in output")
    g = build_grader(GraderSpec(type="regex_match", config={"pattern": r"\b\d{4}\b"}))
    _show_grade("year found", g.grade("Founded in 1969 in NY.", expected=None))
    _show_grade("no year", g.grade("Founded long ago.", expected=None))

    subheader("tool_trace — compare recorded vs expected tool sequence")
    g = build_grader(GraderSpec(type="tool_trace", config={"ordered": True}))
    trace_nodes = [
        TraceNode(
            step=0,
            node_id="agent",
            node_type="agent",
            tool_calls=[
                {"tool": "search_web", "successes": 1, "failures": 0, "total_calls": 1},
                {"tool": "summarize", "successes": 1, "failures": 0, "total_calls": 1},
            ],
        )
    ]
    trace = EvalTrace(subject_name="demo", case_id="t1", nodes=trace_nodes)
    ctx = RunContext(extras={"trace": trace})
    case = TaskCase(
        case_id="t1",
        task_type="tool_use",
        prompt="Find and summarize…",
        expected={"tools": ["search_web", "summarize"]},
    )
    _show_grade(
        "ordered match",
        g.grade("done", expected=case.expected, case=case, ctx=ctx),
    )

    subheader("memory_recall — case-insensitive substring of expected fact")
    g = build_grader(GraderSpec(type="memory_recall"))
    case = TaskCase(
        case_id="m1",
        task_type="memory_recall",
        prompt="What is my pet's name?",
        expected="Mittens",
    )
    ctx = RunContext(extras={"subject_run": None})
    _show_grade(
        "recalls fact",
        g.grade(
            "Your cat is named Mittens.", expected=case.expected, case=case, ctx=ctx
        ),
    )
    _show_grade(
        "forgets fact",
        g.grade("I'm not sure.", expected=case.expected, case=case, ctx=ctx),
    )

    subheader("composite — weighted blend of multiple graders")
    g = build_grader(
        GraderSpec(
            type="composite",
            config={
                "components": [
                    {"type": "exact_match", "weight": 0.4},
                    {"type": "regex_match", "weight": 0.6, "pattern": r"final"},
                ]
            },
        )
    )
    _show_grade(
        "blended",
        g.grade("This is the final answer.", expected="this is the final answer."),
    )


# ---------------------------------------------------------------------------
# Section 3 — analyzers
# ---------------------------------------------------------------------------


def _node(step: int, node_id: str, **kw) -> TraceNode:
    started = datetime(2026, 5, 23, 12, 0, 0) + timedelta(milliseconds=step * 100)
    return TraceNode(
        step=step,
        node_id=node_id,
        node_type=kw.pop("node_type", "agent"),
        inputs=kw.pop("inputs", {}),
        outputs=kw.pop("outputs", []),
        tool_calls=kw.pop("tool_calls", []),
        started_at=started,
        ended_at=started + timedelta(milliseconds=kw.pop("duration_ms", 50)),
        duration_ms=kw.pop("duration_ms_value", 50.0),
        **kw,
    )


def _build_bad_trace() -> EvalTrace:
    """Hand-craft a trace exhibiting loops, redundancy, hallucinated tool,
    cost spike, and latency outliers."""
    nodes = [
        _node(0, "planner", inputs={"q": "x"}),
        _node(
            1,
            "tool",
            tool_calls=[
                {"tool": "search_web", "successes": 1, "failures": 0, "total_calls": 1}
            ],
        ),
        _node(
            2,
            "tool",
            tool_calls=[
                {"tool": "search_web", "successes": 1, "failures": 0, "total_calls": 1}
            ],
        ),
        _node(
            3,
            "tool",
            tool_calls=[
                {"tool": "search_web", "successes": 1, "failures": 0, "total_calls": 1}
            ],
            outputs=["no new results"],
        ),
        _node(
            4,
            "tool",
            tool_calls=[
                {"tool": "make_coffee", "successes": 0, "failures": 1, "total_calls": 1}
            ],
            outputs=["no new results"],
        ),
        # Two more identical-output steps so the redundancy run length exceeds
        # the default min_run_length=2.
        _node(
            5,
            "tool",
            tool_calls=[
                {"tool": "search_web", "successes": 1, "failures": 0, "total_calls": 1}
            ],
            outputs=["no new results"],
        ),
        _node(
            6,
            "tool",
            tool_calls=[
                {"tool": "search_web", "successes": 1, "failures": 0, "total_calls": 1}
            ],
            outputs=["no new results"],
        ),
        _node(7, "planner", inputs={"q": "x"}),  # revisit with same inputs → cycle
        _node(8, "planner", inputs={"q": "x"}),
        _node(9, "planner", inputs={"q": "x"}),
        _node(10, "summarize", duration_ms_value=5000.0),  # latency spike
    ]
    trace = EvalTrace(
        subject_name="demo",
        case_id="bad_case",
        nodes=nodes,
        total_prompt_tokens=12000,
        total_completion_tokens=3000,
        total_response_time_ms=6500.0,
        metrics_snapshot={
            "token_usage": {
                "demo-model": {"prompt_tokens": 12000, "completion_tokens": 3000}
            }
        },
    )
    return trace


class _ToolRegistryStub:
    """Minimal registry exposing only ``search_web`` — ``make_coffee`` is unknown."""

    def list_tool_names(self):
        return ["search_web", "summarize"]


def demo_analyzers() -> None:
    header("3. Trajectory analyzers")
    trace = _build_bad_trace()
    print(
        f"  Synthetic trace: {trace.total_steps} steps, "
        f"{trace.total_prompt_tokens} prompt + {trace.total_completion_tokens} completion tokens."
    )

    specs = [
        AnalyzerSpec(type="loop_detector", config={"max_repeats": 1}),
        # Default min_run_length=2 → runs of 3+ identical outputs are flagged.
        AnalyzerSpec(type="redundancy"),
        AnalyzerSpec(type="tool_hallucination"),
        AnalyzerSpec(
            type="cost",
            config={
                "prices": {
                    "demo-model": {"prompt_per_1k": 0.5, "completion_per_1k": 1.5}
                },
                "warn_threshold_usd": 1.0,
            },
        ),
        AnalyzerSpec(
            type="latency",
            config={"slow_step_ms": 1000, "slow_total_ms": 3000},
        ),
    ]
    from pithos.eval.trace.analyzers import AnalyzerContext

    ctx = AnalyzerContext(
        tool_registry=_ToolRegistryStub(),
        price_map={"demo-model": {"prompt_per_1k": 0.5, "completion_per_1k": 1.5}},
    )

    for spec in specs:
        analyzer = build_analyzer(spec)
        issues = analyzer.analyze(trace, ctx)
        print(f"\n  [{spec.type}] {len(issues)} issue(s)")
        for issue in issues:
            print(f"    • {issue.severity.value:7s} {issue.code:20s} {issue.message}")


# ---------------------------------------------------------------------------
# Section 4 — end-to-end runner + reporter
# ---------------------------------------------------------------------------


class StubSubject(Subject):
    """Deterministic subject backed by a ``{case_id: output}`` mapping.

    Populates a minimal :class:`EvalTrace` so the runner can feed
    analyzers and the Reporter can compute the C.L.A.S.S. table.
    """

    def __init__(
        self,
        name: str,
        responses: dict[str, str],
        *,
        latency_ms: float = 20.0,
        prompt_tokens: int = 100,
        completion_tokens: int = 30,
    ) -> None:
        super().__init__(name)
        self._responses = responses
        self._latency_ms = latency_ms
        self._prompt_tokens = prompt_tokens
        self._completion_tokens = completion_tokens

    def run(self, case: TaskCase, ctx: RunContext) -> SubjectRun:
        started = datetime.now()
        output = self._responses.get(case.case_id, "")
        ended = started + timedelta(milliseconds=self._latency_ms)
        trace = EvalTrace(
            subject_name=self.name,
            case_id=case.case_id,
            nodes=[_node(0, "agent", duration_ms_value=self._latency_ms)],
            started_at=started,
            ended_at=ended,
            total_prompt_tokens=self._prompt_tokens,
            total_completion_tokens=self._completion_tokens,
            total_response_time_ms=self._latency_ms,
            metrics_snapshot={
                "token_usage": {
                    self.name: {
                        "prompt_tokens": self._prompt_tokens,
                        "completion_tokens": self._completion_tokens,
                    }
                }
            },
        )
        return SubjectRun(
            subject_name=self.name,
            case_id=case.case_id,
            output=output,
            trace=trace,
            started_at=started,
            ended_at=ended,
        )


def demo_end_to_end() -> None:
    header("4. End-to-end run → C.L.A.S.S. report")

    # Tiny inline multiple-choice dataset (no file I/O).
    cases = [
        TaskCase(
            case_id=f"q{i}",
            task_type="multiple_choice",
            prompt=q,
            expected=ans,
            metadata={"choices": ["A", "B", "C", "D"], "correct_answer": ans},
        )
        for i, (q, ans) in enumerate(
            [
                ("Capital of France?", "A"),
                ("2 + 2 equals?", "B"),
                ("Sky color?", "C"),
            ]
        )
    ]

    class _InlineDataset:
        def __init__(self, cases):
            self._cases = cases

        def cases(self):
            return iter(self._cases)

        def __iter__(self):
            return iter(self._cases)

        def __len__(self):
            return len(self._cases)

    from pithos.eval.tasks.multiple_choice import MultipleChoiceTask

    task = MultipleChoiceTask(
        name="trivia",
        dataset=_InlineDataset(cases),
        grader=build_grader(GraderSpec(type="letter_match")),
    )

    # Two stub subjects: a "smart" one (correct on most) and a "noisy" one.
    smart = StubSubject(
        "smart-agent",
        {"q0": '{"ANSWER": "A"}', "q1": '{"ANSWER": "B"}', "q2": '{"ANSWER": "C"}'},
        latency_ms=20,
        prompt_tokens=80,
        completion_tokens=20,
    )
    noisy = StubSubject(
        "noisy-agent",
        {"q0": '{"ANSWER": "A"}', "q1": '{"ANSWER": "D"}', "q2": "Hmm, no idea."},
        latency_ms=60,
        prompt_tokens=200,
        completion_tokens=80,
    )

    cfg = EvalConfig(
        name="eval_demo",
        subjects={
            "smart-agent": SubjectSpec(name="smart-agent", type="agent"),
            "noisy-agent": SubjectSpec(name="noisy-agent", type="agent"),
        },
        tasks={
            "trivia": TaskSpec(
                name="trivia",
                type="multiple_choice",
                dataset={"type": "multiple_choice"},
                grader=GraderSpec(type="letter_match"),
            ),
        },
        analyzers=[AnalyzerSpec(type="latency", config={"slow_step_ms": 50})],
        execution=EvalExecutionConfig(rounds=2, num_retries=0, parallelism=1),
        output=EvalOutputConfig(base_dir="./_demo_results"),
    )

    runner = EvalRunner(
        cfg,
        subjects={"smart-agent": smart, "noisy-agent": noisy},
        tasks={"trivia": task},
        write_outputs=False,
        resume=False,
    )
    records: list[CaseRecord] = runner.run()
    print(
        f"\n  Produced {len(records)} CaseRecord(s) "
        f"({cfg.execution.rounds} rounds × 2 subjects × 3 cases)."
    )

    reporter = Reporter(
        config_name=cfg.name,
        rounds=cfg.execution.rounds,
        price_map={
            "smart-agent": {"prompt_per_1k": 0.1, "completion_per_1k": 0.3},
            "noisy-agent": {"prompt_per_1k": 0.1, "completion_per_1k": 0.3},
        },
    )
    report = reporter.build_report(records)

    subheader("Per-subject summary")
    for subject, stats in report.per_subject_stats.items():
        print(
            f"  {subject:14s}  mean_score={stats.get('mean_score', 0):5.1f}  "
            f"n={stats.get('case_count', 0)}  pass_rate={stats.get('pass_rate', 0):.0%}  "
            f"ci=[{stats.get('ci_lower', 0):.1f}, {stats.get('ci_upper', 0):.1f}]"
        )

    subheader("C.L.A.S.S. report")
    cols = (
        "subject",
        "accuracy_mean",
        "latency_ms_avg",
        "cost_usd",
        "stability_std_dev",
        "stability_rounds",
        "security",
        "case_count",
    )
    print("  " + "  ".join(f"{c:>17}" for c in cols))
    for subject, row in report.class_report.items():
        cells = []
        for c in cols:
            v = row.get(c, "")
            if isinstance(v, float):
                cells.append(f"{v:>17.3f}")
            else:
                cells.append(f"{str(v):>17s}")
        print("  " + "  ".join(cells))

    issues_total = sum(len(v) for v in report.issues_by_subject.values())
    print(f"\n  Trajectory issues raised: {issues_total}")
    for subject, issues in report.issues_by_subject.items():
        if issues:
            print(
                f"    {subject}: " + ", ".join(f"{i.code}@{i.step}" for i in issues[:5])
            )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    print("pithos.eval feature showcase — fully offline, no LLM required.")
    demo_builtin_datasets()
    demo_graders()
    demo_analyzers()
    demo_end_to_end()
    print("\nAll sections completed. See docs/EVALUATION.md for the full guide.\n")


if __name__ == "__main__":
    main()
