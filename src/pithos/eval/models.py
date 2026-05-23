"""Core dataclasses for the pithos.eval package.

These types are intentionally framework-agnostic: they describe *what*
was evaluated and *what was observed*, decoupled from the runtime
internals of agents, flowcharts, and teams. The trace ingestion layer
(:mod:`pithos.eval.trace.ingest`) is responsible for mapping the various
runtime trace types into the unified :class:`EvalTrace` shape defined
here, and analyzers / graders / reporters consume only these dataclasses.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional

# ---------------------------------------------------------------------------
# Tasks
# ---------------------------------------------------------------------------


@dataclass
class TaskCase:
    """A single evaluation case fed to a subject under test.

    Tasks may construct one or more :class:`TaskCase` instances from a
    dataset. A case carries everything a subject needs to attempt the
    task plus everything a grader needs to score the result.
    """

    case_id: str
    """Stable identifier for this case within its task/dataset."""

    task_type: str
    """Task class name (e.g. ``"multiple_choice"``, ``"tool_use"``)."""

    prompt: str
    """The user-facing prompt sent to the subject."""

    expected: Any = None
    """Expected answer / outcome used by graders (free-form)."""

    metadata: dict[str, Any] = field(default_factory=dict)
    """Arbitrary per-case metadata (category, difficulty, etc.)."""

    setup_prompts: list[str] = field(default_factory=list)
    """Optional pre-turns sent to the subject before :attr:`prompt`.

    Used by multi-turn capability suites (e.g. memory recall) where the
    subject must first ingest one or more setup messages and then
    respond to the final prompt that is actually graded."""


# ---------------------------------------------------------------------------
# Runtime
# ---------------------------------------------------------------------------


@dataclass
class RunContext:
    """Per-run execution context passed to subjects and analyzers.

    Holds shared resources (registries, config) the subject or analyzer
    may consult without having to thread them through every call.
    """

    round_num: int = 1
    case_index: int = 0
    tool_registry: Optional[Any] = None
    """Optional :class:`pithos.tools.ToolRegistry` for hallucination checks."""

    extras: dict[str, Any] = field(default_factory=dict)
    """Arbitrary additional context (e.g. memory store handle)."""


@dataclass
class SubjectRun:
    """Result of running a single :class:`TaskCase` against a subject.

    The :attr:`output` is the raw response text, :attr:`metrics` is the
    :class:`~pithos.metrics.MetricsCollector` used for this run (fresh
    per run so values can be attributed cleanly), and :attr:`trace`
    is the unified evaluation trace produced by the trace ingest layer.
    """

    subject_name: str
    case_id: str
    output: str
    metrics: Optional[Any] = None
    """The :class:`~pithos.metrics.MetricsCollector` for this run."""

    trace: Optional["EvalTrace"] = None
    started_at: Optional[datetime] = None
    ended_at: Optional[datetime] = None
    error: Optional[str] = None
    """Populated when the subject raised; otherwise ``None``."""

    @property
    def duration_ms(self) -> float:
        if self.started_at is None or self.ended_at is None:
            return 0.0
        return (self.ended_at - self.started_at).total_seconds() * 1000.0


# ---------------------------------------------------------------------------
# Trace
# ---------------------------------------------------------------------------


@dataclass
class TraceNode:
    """One step in a unified evaluation trace.

    A trace node abstracts both flowchart node executions and single-shot
    agent turns into a uniform shape so analyzers can operate on either.
    """

    step: int
    node_id: str
    node_type: str
    inputs: dict[str, Any] = field(default_factory=dict)
    outputs: list[Any] = field(default_factory=list)
    started_at: Optional[datetime] = None
    ended_at: Optional[datetime] = None
    duration_ms: float = 0.0
    from_node: Optional[str] = None
    """Previous node ID (edge source), or ``None`` for the first step."""

    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    """Per-step tool invocations: ``{"tool", "success", "execution_time_ms"}``."""


@dataclass
class EvalTrace:
    """Unified execution trace consumed by analyzers and the reporter.

    Constructed by :func:`pithos.eval.trace.ingest.build_eval_trace`
    from a runtime trace (:class:`pithos.flowchart.ExecutionTrace`) plus
    a :class:`pithos.metrics.MetricsCollector` snapshot. All times are
    in milliseconds; token counts come from the metrics snapshot.
    """

    subject_name: str
    case_id: str
    nodes: list[TraceNode] = field(default_factory=list)
    completed: bool = True
    started_at: Optional[datetime] = None
    ended_at: Optional[datetime] = None
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_response_time_ms: float = 0.0
    metrics_snapshot: dict[str, Any] = field(default_factory=dict)
    """Raw :meth:`MetricsCollector.get_snapshot` payload, for analyzers
    that need the full per-model / per-tool breakdown."""

    @property
    def total_steps(self) -> int:
        return len(self.nodes)

    @property
    def end_to_end_ms(self) -> float:
        if self.started_at is None or self.ended_at is None:
            return self.total_response_time_ms
        return (self.ended_at - self.started_at).total_seconds() * 1000.0


# ---------------------------------------------------------------------------
# Trajectory analysis
# ---------------------------------------------------------------------------


class TrajectoryIssueSeverity(str, Enum):
    """Severity levels for trajectory analyzer findings."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


@dataclass
class TrajectoryIssue:
    """One finding emitted by a trajectory analyzer.

    Issues are *observations*, not pass/fail decisions; the reporter
    aggregates them into the C.L.A.S.S. stability/security columns and
    presents them per-case for inspection.
    """

    analyzer: str
    """Analyzer that produced the issue (e.g. ``"loop_detector"``)."""

    code: str
    """Short machine-readable category (e.g. ``"cycle"``,
    ``"unknown_tool"``)."""

    message: str
    severity: TrajectoryIssueSeverity = TrajectoryIssueSeverity.WARNING
    step: Optional[int] = None
    """Trace step at which the issue was observed, when applicable."""

    detail: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Grading
# ---------------------------------------------------------------------------


@dataclass
class GradeResult:
    """Result of grading a single :class:`SubjectRun`.

    Scores are normalised to the ``[0, 100]`` range so the reporter can
    aggregate consistently across graders.  ``passed`` is an optional
    boolean projection useful for pass/fail summaries.
    """

    grader: str
    score: float
    passed: Optional[bool] = None
    detail: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


@dataclass
class CaseRecord:
    """Persisted result of one ``(subject, case, round)`` evaluation.

    The runner writes one ``CaseRecord`` per case per round as JSONL so
    runs are resumable and reports can be regenerated from disk.
    """

    subject_name: str
    case_id: str
    round_num: int
    task_type: str
    output: str
    grade: GradeResult
    issues: list[TrajectoryIssue] = field(default_factory=list)
    trace: Optional[EvalTrace] = None
    metrics_snapshot: dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class EvalReport:
    """Aggregated report produced by :class:`pithos.eval.reporter.Reporter`.

    Holds per-subject summary rows plus the underlying case records.
    The :attr:`class_report` is a structured dict mirroring the
    Cost/Latency/Accuracy/Stability/Security columns surfaced in the
    CLI and (eventually) the web UI.
    """

    config_name: str
    rounds: int
    case_records: list[CaseRecord] = field(default_factory=list)
    per_subject_stats: dict[str, dict[str, Any]] = field(default_factory=dict)
    class_report: dict[str, dict[str, Any]] = field(default_factory=dict)
    issues_by_subject: dict[str, list[TrajectoryIssue]] = field(default_factory=dict)
    generated_at: datetime = field(default_factory=datetime.now)
