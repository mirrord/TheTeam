"""Evaluation runner — orchestrates subject runs across cases and rounds.

The runner is the heart of ``pithos.eval``. It:

1. Builds (or accepts) the per-config subjects, tasks, and analyzers.
2. Iterates over the configured rounds and the cartesian product of
   subjects × tasks × cases.
3. For each iteration: executes the subject, runs the analyzer pipeline
   on the resulting trace, grades the output, and emits a
   :class:`~pithos.eval.models.CaseRecord`.
4. Persists each case as a single JSONL line under
   ``{run_dir}/cases/round_{n}/{subject}__{task}.jsonl`` so runs are
   resumable and reports can be regenerated offline.

The runner does **not** perform aggregation; that is the
:class:`~pithos.eval.reporter.Reporter`'s job.
"""

from __future__ import annotations

import logging
import os
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Iterable, Optional

from .config import EvalConfig
from .models import (
    CaseRecord,
    EvalTrace,
    GradeResult,
    RunContext,
    SubjectRun,
    TaskCase,
    TrajectoryIssue,
)
from .serde import dump_record, load_records

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _case_file(run_dir: str, round_num: int, subject: str, task: str) -> str:
    return os.path.join(
        run_dir, "cases", f"round_{round_num}", f"{subject}__{task}.jsonl"
    )


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


class EvalRunner:
    """Coordinates subjects, tasks, and analyzers across rounds."""

    def __init__(
        self,
        config: EvalConfig,
        *,
        subjects: Optional[dict[str, Any]] = None,
        tasks: Optional[dict[str, Any]] = None,
        analyzers: Optional[list[Any]] = None,
        analyzer_context: Optional[Any] = None,
        tool_registry: Optional[Any] = None,
        resume: bool = True,
        write_outputs: bool = True,
        max_cases_per_task: Optional[int] = None,
    ) -> None:
        self.config = config
        self.tool_registry = tool_registry
        self.resume = resume
        self.write_outputs = write_outputs
        self.max_cases_per_task = max_cases_per_task

        self.subjects = subjects or self._build_subjects()
        self.tasks = tasks or self._build_tasks()
        self.analyzers = analyzers if analyzers is not None else self._build_analyzers()
        self.analyzer_context = analyzer_context or self._build_analyzer_context()

    # ------------------------------------------------------------------
    # Construction helpers (lazy imports keep optional deps optional)
    # ------------------------------------------------------------------

    def _build_subjects(self) -> dict[str, Any]:
        from .subjects.base import build_subject

        return {
            name: build_subject(spec) for name, spec in self.config.subjects.items()
        }

    def _build_tasks(self) -> dict[str, Any]:
        from .tasks.base import build_task

        return {name: build_task(spec) for name, spec in self.config.tasks.items()}

    def _build_analyzers(self) -> list[Any]:
        from .trace.analyzers import build_analyzer

        return [build_analyzer(spec) for spec in self.config.analyzers]

    def _build_analyzer_context(self) -> Any:
        from .trace.analyzers import AnalyzerContext

        return AnalyzerContext(tool_registry=self.tool_registry)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> list[CaseRecord]:
        """Execute the configured matrix and return all case records."""
        all_records: list[CaseRecord] = []
        rounds = max(1, int(self.config.execution.rounds))

        if self.write_outputs:
            _ensure_dir(self.config.run_dir)

        for round_num in range(1, rounds + 1):
            for task_name, task in self.tasks.items():
                cases = list(task.cases())
                if self.max_cases_per_task is not None:
                    cases = cases[: self.max_cases_per_task]
                round_records = self._run_round(round_num, task_name, task, cases)
                all_records.extend(round_records)

        return all_records

    # ------------------------------------------------------------------
    # Per-round execution (parallel across subjects)
    # ------------------------------------------------------------------

    def _run_round(
        self,
        round_num: int,
        task_name: str,
        task: Any,
        cases: list[TaskCase],
    ) -> list[CaseRecord]:
        parallelism = max(1, int(self.config.execution.parallelism))
        records: list[CaseRecord] = []

        if parallelism == 1 or len(self.subjects) == 1:
            for subject_name, subject in self.subjects.items():
                records.extend(
                    self._run_subject_round(
                        round_num, subject_name, subject, task_name, task, cases
                    )
                )
            return records

        with ThreadPoolExecutor(max_workers=parallelism) as pool:
            futures = {
                pool.submit(
                    self._run_subject_round,
                    round_num,
                    subject_name,
                    subject,
                    task_name,
                    task,
                    cases,
                ): subject_name
                for subject_name, subject in self.subjects.items()
            }
            for fut in as_completed(futures):
                try:
                    records.extend(fut.result())
                except Exception:  # pragma: no cover - propagated via logger
                    logger.exception("Subject %s failed catastrophically", futures[fut])
        return records

    # ------------------------------------------------------------------
    # Per-(subject, round) execution
    # ------------------------------------------------------------------

    def _run_subject_round(
        self,
        round_num: int,
        subject_name: str,
        subject: Any,
        task_name: str,
        task: Any,
        cases: Iterable[TaskCase],
    ) -> list[CaseRecord]:
        records: list[CaseRecord] = []
        out_file = _case_file(self.config.run_dir, round_num, subject_name, task_name)
        done_case_ids: set[str] = set()
        if self.resume:
            for prior in load_records(out_file):
                cid = prior.get("case_id")
                if cid:
                    done_case_ids.add(cid)
                    records.append(self._record_from_dict(prior))

        if self.write_outputs:
            _ensure_dir(os.path.dirname(out_file))

        for index, case in enumerate(cases):
            if case.case_id in done_case_ids:
                continue
            record = self._evaluate_case(
                round_num, subject_name, subject, task_name, task, case, index
            )
            records.append(record)
            if self.write_outputs:
                with open(out_file, "a", encoding="utf-8") as fh:
                    fh.write(dump_record(record) + "\n")
        return records

    # ------------------------------------------------------------------
    # Single case evaluation (with retry on subject error)
    # ------------------------------------------------------------------

    def _evaluate_case(
        self,
        round_num: int,
        subject_name: str,
        subject: Any,
        task_name: str,
        task: Any,
        case: TaskCase,
        case_index: int,
    ) -> CaseRecord:
        ctx = RunContext(
            round_num=round_num,
            case_index=case_index,
            tool_registry=self.tool_registry,
        )

        retries = max(0, int(self.config.execution.num_retries))
        last_error: Optional[str] = None
        run: Optional[SubjectRun] = None
        for attempt in range(retries + 1):
            try:
                run = subject.run(case, ctx)
                if run.error is None:
                    last_error = None
                    break
                last_error = run.error
            except Exception as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                logger.warning(
                    "Subject %s case %s attempt %d failed: %s",
                    subject_name,
                    case.case_id,
                    attempt + 1,
                    last_error,
                )
                run = None

        if run is None:
            # All attempts crashed — fabricate a failure record.
            return CaseRecord(
                subject_name=subject_name,
                case_id=case.case_id,
                round_num=round_num,
                task_type=case.task_type,
                output="",
                grade=GradeResult(grader="error", score=0.0, passed=False),
                error=last_error or "subject raised",
                metadata=dict(case.metadata),
            )

        # Trajectory analyzers.
        trace: Optional[EvalTrace] = run.trace
        issues: list[TrajectoryIssue] = []
        if trace is not None:
            for analyzer in self.analyzers:
                try:
                    issues.extend(analyzer.analyze(trace, self.analyzer_context))
                except Exception as exc:
                    logger.warning(
                        "Analyzer %s failed on %s/%s: %s",
                        getattr(analyzer, "analyzer_name", type(analyzer).__name__),
                        subject_name,
                        case.case_id,
                        exc,
                    )

        # Grade.
        ctx.extras["trace"] = trace
        ctx.extras["subject_run"] = run
        try:
            grade = task.grade(case, run.output, ctx)
        except Exception as exc:
            logger.warning(
                "Grader for task %s raised on %s/%s: %s",
                task_name,
                subject_name,
                case.case_id,
                exc,
            )
            grade = GradeResult(
                grader="error",
                score=0.0,
                passed=False,
                detail={
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(),
                },
            )

        snapshot: dict[str, Any] = {}
        if run.metrics is not None and hasattr(run.metrics, "get_snapshot"):
            try:
                snapshot = run.metrics.get_snapshot()
            except Exception:  # pragma: no cover - defensive
                snapshot = {}

        return CaseRecord(
            subject_name=subject_name,
            case_id=case.case_id,
            round_num=round_num,
            task_type=case.task_type,
            output=run.output,
            grade=grade,
            issues=issues,
            trace=trace,
            metrics_snapshot=snapshot,
            error=last_error,
            metadata=dict(case.metadata),
        )

    # ------------------------------------------------------------------
    # JSONL → CaseRecord rehydration (shallow, for resume)
    # ------------------------------------------------------------------

    @staticmethod
    def _record_from_dict(data: dict) -> CaseRecord:
        grade_data = data.get("grade") or {}
        grade = GradeResult(
            grader=grade_data.get("grader", "unknown"),
            score=float(grade_data.get("score", 0.0)),
            passed=grade_data.get("passed"),
            detail=dict(grade_data.get("detail") or {}),
        )
        issues = []
        from .models import TrajectoryIssue, TrajectoryIssueSeverity

        for raw in data.get("issues") or []:
            sev_raw = raw.get("severity", "warning")
            try:
                severity = TrajectoryIssueSeverity(sev_raw)
            except ValueError:
                severity = TrajectoryIssueSeverity.WARNING
            issues.append(
                TrajectoryIssue(
                    analyzer=raw.get("analyzer", "unknown"),
                    code=raw.get("code", "issue"),
                    message=raw.get("message", ""),
                    severity=severity,
                    step=raw.get("step"),
                    detail=dict(raw.get("detail") or {}),
                )
            )
        return CaseRecord(
            subject_name=data.get("subject_name", ""),
            case_id=data.get("case_id", ""),
            round_num=int(data.get("round_num", 1)),
            task_type=data.get("task_type", ""),
            output=data.get("output", ""),
            grade=grade,
            issues=issues,
            trace=None,  # traces aren't re-hydrated for resume
            metrics_snapshot=dict(data.get("metrics_snapshot") or {}),
            error=data.get("error"),
            metadata=dict(data.get("metadata") or {}),
        )
