"""Tests for :class:`theteam.services.eval_service.EvalService`."""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

from theteam.services.eval_service import EvalService

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def config_dir(tmp_path: Path) -> Path:
    d = tmp_path / "configs"
    d.mkdir()
    return d


@pytest.fixture
def results_dir(tmp_path: Path) -> Path:
    d = tmp_path / "results"
    d.mkdir()
    return d


@pytest.fixture
def service(config_dir: Path, results_dir: Path) -> EvalService:
    return EvalService(config_dir=str(config_dir), results_dir=str(results_dir))


def _write_config(directory: Path, name: str, subjects: dict, tasks: dict) -> Path:
    path = directory / f"{name}.yaml"
    path.write_text(
        yaml.safe_dump({"name": name, "subjects": subjects, "tasks": tasks}),
        encoding="utf-8",
    )
    return path


def _make_run_dir(
    results_dir: Path,
    run_name: str,
    config_name: str = "test-cfg",
    subjects: dict | None = None,
) -> Path:
    run_path = results_dir / run_name
    stats_dir = run_path / "stats"
    stats_dir.mkdir(parents=True)
    per_subject = subjects or {"sub_a": {"total_cases": 4}}
    report = {
        "config_name": config_name,
        "generated_at": "2025-01-01T00:00:00",
        "per_subject_stats": per_subject,
    }
    (stats_dir / "report.json").write_text(json.dumps(report), encoding="utf-8")
    return run_path


# ---------------------------------------------------------------------------
# list_configs
# ---------------------------------------------------------------------------


def test_list_configs_empty_dir(service: EvalService):
    assert service.list_configs() == []


def test_list_configs_missing_dir(tmp_path: Path):
    svc = EvalService(config_dir=str(tmp_path / "nonexistent"))
    assert svc.list_configs() == []


def test_list_configs_parses_yaml(service: EvalService, config_dir: Path):
    _write_config(config_dir, "cfg1", subjects={"s1": {}, "s2": {}}, tasks={"t1": {}})
    configs = service.list_configs()
    assert len(configs) == 1
    c = configs[0]
    assert c["name"] == "cfg1"
    assert c["subject_count"] == 2
    assert c["task_count"] == 1
    assert set(c["subject_names"]) == {"s1", "s2"}


def test_list_configs_skips_invalid_yaml(service: EvalService, config_dir: Path):
    (config_dir / "bad.yaml").write_text(": : : invalid", encoding="utf-8")
    _write_config(config_dir, "good", subjects={"s": {}}, tasks={"t": {}})
    configs = service.list_configs()
    # bad.yaml is skipped, good.yaml is returned
    assert len(configs) == 1
    assert configs[0]["name"] == "good"


def test_list_configs_multiple(service: EvalService, config_dir: Path):
    _write_config(config_dir, "alpha", subjects={"a": {}}, tasks={"t": {}})
    _write_config(
        config_dir, "beta", subjects={"b": {}, "c": {}}, tasks={"t": {}, "u": {}}
    )
    configs = service.list_configs()
    assert len(configs) == 2
    names = [c["name"] for c in configs]
    assert "alpha" in names and "beta" in names


# ---------------------------------------------------------------------------
# get_config
# ---------------------------------------------------------------------------


def test_get_config_by_name(service: EvalService, config_dir: Path):
    _write_config(config_dir, "mycfg", subjects={"s": {}}, tasks={})
    result = service.get_config("mycfg")
    assert result is not None
    assert result["name"] == "mycfg"


def test_get_config_by_path(service: EvalService, config_dir: Path):
    path = _write_config(config_dir, "direct", subjects={}, tasks={})
    result = service.get_config(str(path))
    assert result is not None
    assert result["name"] == "direct"


def test_get_config_missing_returns_none(service: EvalService):
    assert service.get_config("does_not_exist") is None


# ---------------------------------------------------------------------------
# list_runs
# ---------------------------------------------------------------------------


def test_list_runs_empty_dir(service: EvalService):
    assert service.list_runs() == []


def test_list_runs_reads_report_json(service: EvalService, results_dir: Path):
    _make_run_dir(
        results_dir,
        "run_001",
        config_name="eval-cfg",
        subjects={"a": {"total_cases": 5}},
    )
    runs = service.list_runs()
    assert len(runs) == 1
    r = runs[0]
    assert r["config_name"] == "eval-cfg"
    assert r["case_count"] == 5
    assert "a" in r["subject_names"]


def test_list_runs_sorted_newest_first(service: EvalService, results_dir: Path):
    _make_run_dir(results_dir, "old_run")
    # Override generated_at for ordering
    (results_dir / "old_run" / "stats" / "report.json").write_text(
        json.dumps(
            {
                "config_name": "x",
                "generated_at": "2024-01-01T00:00:00",
                "per_subject_stats": {},
            }
        ),
        encoding="utf-8",
    )
    _make_run_dir(results_dir, "new_run")
    (results_dir / "new_run" / "stats" / "report.json").write_text(
        json.dumps(
            {
                "config_name": "y",
                "generated_at": "2025-06-01T00:00:00",
                "per_subject_stats": {},
            }
        ),
        encoding="utf-8",
    )
    runs = service.list_runs()
    assert runs[0]["config_name"] == "y"
    assert runs[1]["config_name"] == "x"


def test_list_runs_dir_without_report(service: EvalService, results_dir: Path):
    (results_dir / "orphan_run").mkdir()
    runs = service.list_runs()
    assert len(runs) == 1
    assert runs[0]["name"] == "orphan_run"
    assert runs[0]["config_name"] == ""


# ---------------------------------------------------------------------------
# get_run
# ---------------------------------------------------------------------------


def test_get_run_loads_report(service: EvalService, results_dir: Path):
    run_path = _make_run_dir(results_dir, "myrun")
    result = service.get_run(str(run_path))
    assert result["report"]["config_name"] == "test-cfg"
    assert isinstance(result["cases_by_subject"], dict)


def test_get_run_loads_cases(service: EvalService, results_dir: Path):
    run_path = _make_run_dir(results_dir, "run_with_cases")
    cases_dir = run_path / "cases" / "round_1"
    cases_dir.mkdir(parents=True)
    record = {
        "subject_name": "sub_a",
        "case_id": "case-1",
        "round_num": 1,
        "task_type": "coding",
        "output": "print('hello')",
        "grade": {"score": 1.0, "passed": True},
        "issues": [],
        "error": None,
    }
    (cases_dir / "sub_a.jsonl").write_text(json.dumps(record) + "\n", encoding="utf-8")
    result = service.get_run(str(run_path))
    assert "sub_a" in result["cases_by_subject"]
    cases = result["cases_by_subject"]["sub_a"]
    assert len(cases) == 1
    assert cases[0]["case_id"] == "case-1"
    assert cases[0]["passed"] is True


def test_get_run_raises_for_missing_dir(service: EvalService):
    with pytest.raises(FileNotFoundError):
        service.get_run("/nonexistent/path/to/run")


# ---------------------------------------------------------------------------
# start_run / stop_run / get_run_status
# ---------------------------------------------------------------------------


def test_start_run_returns_run_id(service: EvalService):
    mock_socketio = None
    called_event = threading.Event()

    def fake_run_benchmark(run_id, config, options, socketio, client_id):
        called_event.set()

    service._run_benchmark = fake_run_benchmark  # type: ignore[method-assign]

    config = {"name": "test", "subjects": {}, "tasks": {}}
    run_id = service.start_run(config, {}, None, mock_socketio)
    assert isinstance(run_id, str)
    assert len(run_id) == 36  # UUID4
    called_event.wait(timeout=1.0)
    assert called_event.is_set()


def test_start_run_registers_in_runs(service: EvalService):
    service._run_benchmark = lambda *a, **kw: None  # type: ignore[method-assign]
    config = {"name": "test"}
    run_id = service.start_run(config, {}, "client-123", None)
    status = service.get_run_status(run_id)
    assert status is not None
    assert status["id"] == run_id
    assert status["client_id"] == "client-123"
    assert "stop_event" not in status  # must be excluded from public status


def test_stop_run_sets_flag(service: EvalService):
    # Pause the background thread so we can test the flag before it finishes
    gate = threading.Event()

    def slow_run(run_id, config, options, socketio, client_id):
        gate.wait(timeout=2.0)

    service._run_benchmark = slow_run  # type: ignore[method-assign]
    run_id = service.start_run({"name": "t"}, {}, None, None)
    # Give thread a moment to start
    time.sleep(0.05)
    result = service.stop_run(run_id)
    assert result is True
    stop_event = service.runs[run_id]["stop_event"]
    assert stop_event.is_set()
    gate.set()  # unblock thread


def test_stop_run_returns_false_for_missing(service: EvalService):
    assert service.stop_run("nonexistent-run-id") is False


def test_get_run_status_missing(service: EvalService):
    assert service.get_run_status("nope") is None


# ---------------------------------------------------------------------------
# Progress-reporting runner (smoke test — no live LLM)
# ---------------------------------------------------------------------------


def test_progress_events_emitted(service: EvalService, results_dir: Path):
    """Verify _ProgressReportingEvalRunner emits benchmark_progress events."""
    try:
        from pithos.eval.runner import EvalRunner
    except ImportError:
        pytest.skip("pithos.eval not available")

    from theteam.services.eval_service import _ProgressReportingEvalRunner

    emitted: list[dict] = []

    class FakeSocketIO:
        def emit(self, event, data, room=None, **kwargs):
            emitted.append({"event": event, "data": data, "room": room})

    mock_record = MagicMock()
    mock_record.error = None
    mock_record.grade.score = 1.0
    mock_record.grade.passed = True
    mock_record.case_id = "c1"

    stop_event = threading.Event()
    runs_dict: dict = {"run-1": {"completed": 0, "total": 1}}
    lock = threading.Lock()

    runner = _ProgressReportingEvalRunner.__new__(_ProgressReportingEvalRunner)
    runner._run_id = "run-1"
    runner._stop_event = stop_event
    runner._runs_dict = runs_dict
    runner._runs_lock = lock
    runner._socketio = FakeSocketIO()

    # Patch the *parent* class method so super()._evaluate_case returns mock_record
    # while the subclass's emit logic still executes.
    with patch.object(EvalRunner, "_evaluate_case", return_value=mock_record):
        _ProgressReportingEvalRunner._evaluate_case(
            runner, 1, "sub", MagicMock(), "task", MagicMock(), MagicMock(), 0
        )

    progress_events = [e for e in emitted if e["event"] == "benchmark_progress"]
    assert len(progress_events) >= 1
    assert progress_events[0]["data"]["run_id"] == "run-1"
