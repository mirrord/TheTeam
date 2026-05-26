"""EvalService — wraps ``pithos.eval`` for the TheTeam web server.

Provides listing of evaluation configs and past run results, plus
launching/stopping live benchmark runs with SocketIO progress events.
"""

from __future__ import annotations

import json
import logging
import threading
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import yaml

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Progress-reporting runner (module-level so it can be imported cleanly)
# ---------------------------------------------------------------------------

# Import EvalRunner at module level inside a try/except so the service
# can be imported even when pithos.eval is not fully available.
try:
    from pithos.eval.runner import EvalRunner as _EvalRunner

    class _ProgressReportingEvalRunner(_EvalRunner):
        """EvalRunner subclass that emits SocketIO progress events per case."""

        def __init__(
            self,
            config: Any,
            *,
            run_id: str,
            stop_event: threading.Event,
            runs_dict: dict,
            runs_lock: threading.Lock,
            socketio: Any,
            **kwargs: Any,
        ) -> None:
            super().__init__(config, **kwargs)
            self._run_id = run_id
            self._stop_event = stop_event
            self._runs_dict = runs_dict
            self._runs_lock = runs_lock
            self._socketio = socketio

        def _evaluate_case(
            self,
            round_num: int,
            subject_name: str,
            subject: Any,
            task_name: str,
            task: Any,
            case: Any,
            case_index: int,
        ) -> Any:
            if self._stop_event.is_set():
                raise InterruptedError("Benchmark stopped by user")

            record = super()._evaluate_case(
                round_num, subject_name, subject, task_name, task, case, case_index
            )

            completed: int
            total: int
            with self._runs_lock:
                if self._run_id in self._runs_dict:
                    self._runs_dict[self._run_id]["completed"] = (
                        self._runs_dict[self._run_id].get("completed", 0) + 1
                    )
                    completed = self._runs_dict[self._run_id]["completed"]
                    total = self._runs_dict[self._run_id].get("total", 0)
                else:
                    completed = 0
                    total = 0

            if self._socketio:
                from theteam.api.socketio_handlers import emit_to_room  # noqa: PLC0415

                emit_to_room(
                    self._socketio,
                    f"benchmark_{self._run_id}",
                    "benchmark_progress",
                    {
                        "run_id": self._run_id,
                        "round_num": round_num,
                        "subject": subject_name,
                        "task": task_name,
                        "case_id": record.case_id,
                        "status": "error" if record.error else "ok",
                        "score": record.grade.score,
                        "passed": record.grade.passed,
                        "completed": completed,
                        "total": total,
                    },
                )

            return record

except ImportError:
    # pithos.eval not available; placeholder keeps the name defined so
    # _run_benchmark can raise a clear error at runtime.
    class _ProgressReportingEvalRunner:  # type: ignore[no-redef]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise ImportError("pithos.eval is not available")


# ---------------------------------------------------------------------------
# EvalService
# ---------------------------------------------------------------------------


class EvalService:
    """Service for managing evaluation configs, past runs, and live benchmarks."""

    def __init__(
        self,
        config_dir: Optional[str] = None,
        results_dir: Optional[str] = None,
    ) -> None:
        cwd = Path.cwd()
        self.config_dir: Path = (
            Path(config_dir) if config_dir else cwd / "configs" / "eval"
        )
        self.results_dir: Path = Path(results_dir) if results_dir else cwd / "results"

        # Active benchmark runs keyed by run_id
        self.runs: dict[str, dict[str, Any]] = {}
        self.lock = threading.Lock()

    # ------------------------------------------------------------------
    # Config listing
    # ------------------------------------------------------------------

    def list_configs(self, config_dir: Optional[str] = None) -> list[dict]:
        """Return metadata for every ``*.yaml``/``*.yml`` eval config file.

        Silently skips files that cannot be parsed.
        """
        search_dir = Path(config_dir) if config_dir else self.config_dir
        results: list[dict] = []

        if not search_dir.exists():
            return results

        for yaml_file in sorted(search_dir.glob("*.yaml")) + sorted(search_dir.glob("*.yml")):  # type: ignore[operator]
            try:
                with open(yaml_file, "r", encoding="utf-8") as fh:
                    data = yaml.safe_load(fh) or {}
            except Exception as exc:
                logger.error("Failed to parse eval config %s: %s", yaml_file, exc)
                continue

            subjects: dict = data.get("subjects") or {}
            tasks: dict = data.get("tasks") or {}
            results.append(
                {
                    "name": data.get("name", yaml_file.stem),
                    "path": str(yaml_file),
                    "subject_count": len(subjects),
                    "task_count": len(tasks),
                    "subject_names": list(subjects.keys()),
                    "task_names": list(tasks.keys()),
                }
            )

        return results

    # ------------------------------------------------------------------
    # Config loading
    # ------------------------------------------------------------------

    def get_config(self, name_or_path: str) -> Optional[dict]:
        """Return the full parsed YAML dict for a named config, or ``None``."""
        # Try direct path first
        candidate = Path(name_or_path)
        if candidate.is_file():
            try:
                with open(candidate, "r", encoding="utf-8") as fh:
                    return yaml.safe_load(fh)
            except Exception as exc:
                logger.error("Failed to load config %s: %s", candidate, exc)
                return None

        # Search by stem in config_dir
        for ext in (".yaml", ".yml"):
            path = self.config_dir / f"{name_or_path}{ext}"
            if path.is_file():
                try:
                    with open(path, "r", encoding="utf-8") as fh:
                        return yaml.safe_load(fh)
                except Exception as exc:
                    logger.error("Failed to load config %s: %s", path, exc)
                    return None

        return None

    # ------------------------------------------------------------------
    # Run listing
    # ------------------------------------------------------------------

    def list_runs(self, base_dir: Optional[str] = None) -> list[dict]:
        """Return metadata for all result directories, newest first."""
        search_dir = Path(base_dir) if base_dir else self.results_dir
        runs: list[dict] = []

        if not search_dir.exists():
            return runs

        for entry in search_dir.iterdir():
            if not entry.is_dir():
                continue
            report_path = entry / "stats" / "report.json"
            meta: dict[str, Any] = {
                "name": entry.name,
                "path": str(entry),
                "config_name": "",
                "generated_at": "",
                "case_count": 0,
                "subject_names": [],
            }
            if report_path.exists():
                try:
                    with open(report_path, "r", encoding="utf-8") as fh:
                        report = json.load(fh)
                    meta["config_name"] = report.get("config_name", "")
                    meta["generated_at"] = report.get("generated_at", "")
                    per_subject: dict = report.get("per_subject_stats") or {}
                    meta["subject_names"] = list(per_subject.keys())
                    # Sum case counts across subjects/rounds
                    total_cases = 0
                    for stats in per_subject.values():
                        total_cases += stats.get("total_cases", 0)
                    meta["case_count"] = total_cases
                except Exception as exc:
                    logger.error("Failed to read report.json in %s: %s", entry, exc)
            runs.append(meta)

        # Sort newest first; directories without a timestamp fall to end
        runs.sort(key=lambda r: r.get("generated_at") or "", reverse=True)
        return runs

    # ------------------------------------------------------------------
    # Run detail
    # ------------------------------------------------------------------

    def get_run(self, run_dir: str) -> dict:
        """Load report and case details from a completed run directory.

        Raises:
            FileNotFoundError: if *run_dir* does not exist.
        """
        run_path = Path(run_dir)
        if not run_path.exists():
            raise FileNotFoundError(f"Run directory not found: {run_dir}")

        report: dict = {}
        report_path = run_path / "stats" / "report.json"
        if report_path.exists():
            try:
                with open(report_path, "r", encoding="utf-8") as fh:
                    report = json.load(fh)
            except Exception as exc:
                logger.error("Failed to read report.json: %s", exc)

        cases_by_subject: dict[str, list[dict]] = {}
        cases_root = run_path / "cases"
        if cases_root.is_dir():
            for round_dir in sorted(cases_root.iterdir()):
                if not round_dir.is_dir() or not round_dir.name.startswith("round_"):
                    continue
                try:
                    round_num = int(round_dir.name.split("_", 1)[1])
                except (IndexError, ValueError):
                    round_num = 0

                for jsonl_file in sorted(round_dir.glob("*.jsonl")):
                    try:
                        with open(jsonl_file, "r", encoding="utf-8") as fh:
                            for line in fh:
                                line = line.strip()
                                if not line:
                                    continue
                                try:
                                    obj = json.loads(line)
                                except json.JSONDecodeError:
                                    continue
                                grade = obj.get("grade") or {}
                                issues = obj.get("issues") or []
                                case_detail = {
                                    "subject_name": obj.get("subject_name", ""),
                                    "case_id": obj.get("case_id", ""),
                                    "round_num": obj.get("round_num", round_num),
                                    "task_type": obj.get("task_type", ""),
                                    "output": obj.get("output", ""),
                                    "score": grade.get("score"),
                                    "passed": grade.get("passed"),
                                    "error": obj.get("error"),
                                    "issue_count": len(issues),
                                }
                                subject = case_detail["subject_name"] or "unknown"
                                cases_by_subject.setdefault(subject, []).append(
                                    case_detail
                                )
                    except Exception as exc:
                        logger.error("Failed to read %s: %s", jsonl_file, exc)

        return {"report": report, "cases_by_subject": cases_by_subject}

    # ------------------------------------------------------------------
    # Run lifecycle
    # ------------------------------------------------------------------

    def start_run(
        self,
        config_data: dict,
        options: dict,
        client_id: Optional[str],
        socketio: Any,
    ) -> str:
        """Launch a benchmark run in a background thread.

        Returns:
            A UUID string identifying this run.
        """
        run_id = str(uuid.uuid4())
        now = datetime.now().isoformat()
        with self.lock:
            self.runs[run_id] = {
                "id": run_id,
                "config_name": config_data.get("name", ""),
                "status": "starting",
                "started_at": now,
                "client_id": client_id,
                "stop_event": threading.Event(),
                "completed": 0,
                "total": 0,
            }

        thread = threading.Thread(
            target=self._run_benchmark,
            args=(run_id, config_data, options, socketio, client_id),
            daemon=True,
        )
        thread.start()
        return run_id

    def stop_run(self, run_id: str) -> bool:
        """Signal a running benchmark to stop.

        Returns:
            ``True`` if the run was found, ``False`` otherwise.
        """
        with self.lock:
            run = self.runs.get(run_id)
        if run is None:
            return False
        run["stop_event"].set()
        return True

    def get_run_status(self, run_id: str) -> Optional[dict]:
        """Return current run state, excluding the ``stop_event`` object."""
        with self.lock:
            run = self.runs.get(run_id)
        if run is None:
            return None
        return {k: v for k, v in run.items() if k != "stop_event"}

    # ------------------------------------------------------------------
    # Background execution
    # ------------------------------------------------------------------

    def _count_total_cases(self, cfg: Any, max_cases: Optional[int] = None) -> int:
        """Estimate total case × subject × round count."""
        from pithos.eval.tasks.base import build_task  # noqa: PLC0415

        total = 0
        for task_spec in cfg.tasks.values():
            try:
                task = build_task(task_spec)
                cases = list(task.cases())
                n = len(cases) if max_cases is None else min(len(cases), max_cases)
                total += n * len(cfg.subjects) * cfg.execution.rounds
            except Exception:
                pass
        return total

    def _run_benchmark(
        self,
        run_id: str,
        config: dict,
        options: dict,
        socketio: Any,
        client_id: Optional[str],
    ) -> None:
        from pithos.eval import EvalConfig, Reporter  # noqa: PLC0415
        from theteam.api.socketio_handlers import emit_to_room  # noqa: PLC0415

        try:
            with self.lock:
                self.runs[run_id]["status"] = "running"

            # Apply option overrides before building EvalConfig
            if options.get("rounds") is not None:
                config.setdefault("execution", {})["rounds"] = int(options["rounds"])
            if options.get("output_dir"):
                config.setdefault("output", {})["base_dir"] = options["output_dir"]

            cfg = EvalConfig.from_dict(config)

            dry_run: bool = bool(options.get("dry_run", False))
            max_cases: Optional[int] = options.get("max_cases", None)
            if max_cases is not None:
                max_cases = int(max_cases)

            total = self._count_total_cases(cfg, max_cases)
            with self.lock:
                self.runs[run_id]["total"] = total

            if socketio:
                emit_to_room(
                    socketio,
                    f"benchmark_{run_id}",
                    "benchmark_started",
                    {
                        "run_id": run_id,
                        "config_name": cfg.name,
                        "started_at": datetime.now().isoformat(),
                        "total_subjects": len(cfg.subjects),
                        "total_tasks": len(cfg.tasks),
                        "rounds": cfg.execution.rounds,
                        "total_cases": total,
                    },
                )

            stop_event: threading.Event = self.runs[run_id]["stop_event"]
            runner = _ProgressReportingEvalRunner(
                cfg,
                run_id=run_id,
                stop_event=stop_event,
                runs_dict=self.runs,
                runs_lock=self.lock,
                socketio=socketio,
                resume=True,
                write_outputs=not dry_run,
                max_cases_per_task=max_cases,
            )
            records = runner.run()

            reporter = Reporter(
                config_name=cfg.name,
                rounds=cfg.execution.rounds,
            )
            report = reporter.build_report(records)

            if not dry_run:
                reporter.write(report, cfg.run_dir)

            serialized_report = {
                "class_report": report.class_report,
                "per_subject_stats": report.per_subject_stats,
                "issues_by_subject": {
                    k: [
                        {
                            "analyzer": i.analyzer,
                            "code": i.code,
                            "message": i.message,
                            "severity": (
                                i.severity.value
                                if hasattr(i.severity, "value")
                                else str(i.severity)
                            ),
                        }
                        for i in v
                    ]
                    for k, v in report.issues_by_subject.items()
                },
            }

            with self.lock:
                if run_id in self.runs:
                    self.runs[run_id]["status"] = "completed"
                    self.runs[run_id]["report"] = serialized_report

            if socketio:
                emit_to_room(
                    socketio,
                    f"benchmark_{run_id}",
                    "benchmark_complete",
                    {
                        "run_id": run_id,
                        "report": serialized_report,
                        "case_count": len(records),
                    },
                )

        except InterruptedError:
            with self.lock:
                if run_id in self.runs:
                    self.runs[run_id]["status"] = "stopped"
            if socketio:
                emit_to_room(
                    socketio,
                    f"benchmark_{run_id}",
                    "benchmark_error",
                    {"run_id": run_id, "error": "Run stopped by user"},
                )

        except Exception as exc:
            logger.error("Benchmark run %s failed: %s", run_id, exc, exc_info=True)
            with self.lock:
                if run_id in self.runs:
                    self.runs[run_id]["status"] = "failed"
                    self.runs[run_id]["error"] = str(exc)
            if socketio:
                emit_to_room(
                    socketio,
                    f"benchmark_{run_id}",
                    "benchmark_error",
                    {"run_id": run_id, "error": str(exc)},
                )
