"""Smoke tests for the ``pithos-eval`` CLI."""

from __future__ import annotations

import json
import os

import pytest
import yaml

from pithos.eval.cli import build_parser, main


def test_cli_no_command_prints_help(capsys):
    rc = main([])
    out = capsys.readouterr().out
    assert rc == 1
    assert "pithos-eval" in out


def test_cli_list_configs(tmp_path, capsys):
    cfg_dir = tmp_path / "configs"
    cfg_dir.mkdir()
    (cfg_dir / "a.yaml").write_text("name: a\nsubjects: {}\ntasks: {}\n")
    (cfg_dir / "b.yml").write_text("name: b\nsubjects: {}\ntasks: {}\n")
    rc = main(["list-configs", "--dir", str(cfg_dir)])
    out = capsys.readouterr().out
    assert rc == 0
    assert "a.yaml" in out
    assert "b.yml" in out


def test_cli_list_configs_empty(tmp_path, capsys):
    rc = main(["list-configs", "--dir", str(tmp_path / "nope")])
    out = capsys.readouterr().out
    assert rc == 0
    assert "No eval configs" in out


def test_cli_list_suites(capsys):
    rc = main(["list-suites"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "multiple_choice" in out
    assert "free_form" in out


def test_cli_report_missing_dir_errors(tmp_path, capsys):
    rc = main(["report", "--run-dir", str(tmp_path / "ghost")])
    err = capsys.readouterr().err
    assert rc == 1
    assert "No case records" in err


def test_cli_report_aggregates_existing_jsonl(tmp_path, capsys):
    from pithos.eval import CaseRecord, EvalTrace, GradeResult, TraceNode
    from pithos.eval.serde import dump_record

    cases_dir = tmp_path / "cases" / "round_1"
    cases_dir.mkdir(parents=True)
    rec = CaseRecord(
        subject_name="alpha",
        case_id="q1",
        round_num=1,
        task_type="free_form",
        output="x",
        grade=GradeResult(grader="g", score=90.0, passed=True),
        trace=EvalTrace(
            subject_name="alpha",
            case_id="q1",
            nodes=[TraceNode(step=0, node_id="n", node_type="A")],
        ),
        metrics_snapshot={},
    )
    (cases_dir / "alpha__stub.jsonl").write_text(
        dump_record(rec) + "\n", encoding="utf-8"
    )

    rc = main(["report", "--run-dir", str(tmp_path)])
    out = capsys.readouterr().out
    assert rc == 0
    assert os.path.exists(tmp_path / "stats" / "report.json")
    assert "alpha" in out


def test_cli_run_end_to_end(tmp_path, monkeypatch, capsys):
    """Run a tiny config through the CLI using a stubbed subject + task."""
    # Write a YAML config with an "agent" subject + a "free_form" task; the
    # actual subject/task instances are injected via monkeypatching the
    # build dispatchers below so the test doesn't need Ollama.
    from datetime import datetime

    from pithos.eval import (
        EvalTrace,
        GradeResult,
        RunContext,
        SubjectRun,
        TaskCase,
        TraceNode,
    )
    from pithos.eval.subjects.base import Subject

    class _Subj(Subject):
        def run(self, case, ctx):
            now = datetime.now()
            return SubjectRun(
                subject_name=self.name,
                case_id=case.case_id,
                output="ans",
                metrics=None,
                trace=EvalTrace(
                    subject_name=self.name,
                    case_id=case.case_id,
                    nodes=[TraceNode(step=0, node_id="n", node_type="agent")],
                    total_response_time_ms=1.0,
                ),
                started_at=now,
                ended_at=now,
            )

    class _Task:
        name = "stub"

        def cases(self):
            return [
                TaskCase(
                    case_id="q1", task_type="free_form", prompt="p", expected="ans"
                )
            ]

        def grade(self, case, output, ctx=None):
            return GradeResult(grader="g", score=100.0, passed=True)

    monkeypatch.setattr(
        "pithos.eval.runner.EvalRunner._build_subjects",
        lambda self: {"alpha": _Subj("alpha")},
    )
    monkeypatch.setattr(
        "pithos.eval.runner.EvalRunner._build_tasks",
        lambda self: {"stub": _Task()},
    )

    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(
        yaml.safe_dump(
            {
                "name": "cli_test",
                "subjects": {"alpha": {"type": "agent"}},
                "tasks": {
                    "stub": {
                        "type": "free_form",
                        "dataset": {"type": "free_form", "path": "ignored"},
                        "grader": {"type": "exact_match"},
                    }
                },
                "execution": {"rounds": 1, "parallelism": 1},
                "output": {"base_dir": str(tmp_path / "out")},
            }
        )
    )

    rc = main(["run", "--config", str(cfg_path)])
    out = capsys.readouterr().out
    assert rc == 0
    assert "alpha" in out
    # Stats file should exist under the timestamped run dir.
    run_root = tmp_path / "out"
    found_stats = [
        os.path.join(dp, fn)
        for dp, _, fns in os.walk(run_root)
        for fn in fns
        if fn == "report.json"
    ]
    assert len(found_stats) == 1


def test_cli_parser_accepts_all_subcommands():
    parser = build_parser()
    for cmd in ["run", "report", "list-configs", "list-suites", "analyze"]:
        # Just verify each subparser exists.
        action = next(
            a for a in parser._subparsers._group_actions if hasattr(a, "choices")
        )
        assert cmd in action.choices
