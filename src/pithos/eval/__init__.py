"""pithos.eval — agent and workflow evaluation suite.

A unified harness for evaluating any *subject under test* (single agent,
flowchart, or team) across the C.L.A.S.S. dimensions (Cost, Latency,
Accuracy, Stability, Security) plus trajectory-level signals such as
loops, redundancy, and tool hallucinations.

See ``docs/EVALUATION.md`` for the user-facing guide.
"""

from .models import (
    CaseRecord,
    EvalReport,
    EvalTrace,
    GradeResult,
    RunContext,
    SubjectRun,
    TaskCase,
    TrajectoryIssue,
    TrajectoryIssueSeverity,
    TraceNode,
)
from .config import (
    AnalyzerSpec,
    EvalConfig,
    EvalExecutionConfig,
    EvalOutputConfig,
    GraderSpec,
    SubjectSpec,
    TaskSpec,
)
from .metrics_view import build_class_report, to_dataframe
from .trace.analyzers import Analyzer, AnalyzerContext, build_analyzer
from .runner import EvalRunner
from .reporter import Reporter, load_records_from_run_dir

__all__ = [
    # Core data
    "CaseRecord",
    "EvalReport",
    "EvalTrace",
    "GradeResult",
    "RunContext",
    "SubjectRun",
    "TaskCase",
    "TraceNode",
    "TrajectoryIssue",
    "TrajectoryIssueSeverity",
    # Config
    "AnalyzerSpec",
    "EvalConfig",
    "EvalExecutionConfig",
    "EvalOutputConfig",
    "GraderSpec",
    "SubjectSpec",
    "TaskSpec",
    # Analyzers & reporting
    "Analyzer",
    "AnalyzerContext",
    "build_analyzer",
    "build_class_report",
    "to_dataframe",
    # Runner + reporter
    "EvalRunner",
    "Reporter",
    "load_records_from_run_dir",
]
