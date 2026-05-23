"""Tests for Reporter aggregation and persistence."""

from __future__ import annotations

import json
import os

import pytest

from pithos.eval import (
    CaseRecord,
    EvalTrace,
    GradeResult,
    Reporter,
    TraceNode,
    load_records_from_run_dir,
)


def _record(subject, case_id, score, round_num=1, tokens=0):
    snap = {"token_usage": {"m": {"prompt_tokens": tokens, "completion_tokens": 0}}}
    trace = EvalTrace(
        subject_name=subject,
        case_id=case_id,
        nodes=[TraceNode(step=0, node_id="n", node_type="X", duration_ms=10.0)],
        total_response_time_ms=10.0,
        metrics_snapshot=snap,
    )
    return CaseRecord(
        subject_name=subject,
        case_id=case_id,
        round_num=round_num,
        task_type="free_form",
        output="out",
        grade=GradeResult(grader="g", score=score, passed=score >= 50),
        trace=trace,
        metrics_snapshot=snap,
    )


def test_reporter_builds_per_subject_stats():
    records = [
        _record("alpha", "q1", 100),
        _record("alpha", "q2", 60),
        _record("beta", "q1", 0),
        _record("beta", "q2", 20),
    ]
    report = Reporter(config_name="t", rounds=1, bootstrap_n=50).build_report(records)
    assert set(report.per_subject_stats) == {"alpha", "beta"}
    assert report.per_subject_stats["alpha"]["mean_score"] == 80.0
    assert report.per_subject_stats["alpha"]["pass_rate"] == 1.0
    assert report.per_subject_stats["beta"]["pass_rate"] == 0.0
    assert "alpha" in report.class_report
    assert report.class_report["alpha"]["accuracy_mean"] == 80.0


def test_reporter_uses_price_map_for_cost():
    records = [_record("alpha", "q1", 100, tokens=2000)]
    reporter = Reporter(
        config_name="t",
        rounds=1,
        bootstrap_n=10,
        price_map={"m": {"prompt_per_1k": 0.5}},
    )
    report = reporter.build_report(records)
    assert report.class_report["alpha"]["cost_usd"] == pytest.approx(1.0)


def test_reporter_write_emits_json_and_csv(tmp_path):
    records = [_record("alpha", "q1", 100), _record("alpha", "q2", 0)]
    report = Reporter(config_name="t", rounds=1, bootstrap_n=10).build_report(records)
    paths = Reporter(config_name="t").write(report, str(tmp_path))
    assert os.path.exists(paths["report_json"])
    assert os.path.exists(paths["class_report_csv"])
    data = json.loads(open(paths["report_json"], encoding="utf-8").read())
    assert "per_subject_stats" in data
    assert "class_report" in data
    csv_text = open(paths["class_report_csv"], encoding="utf-8").read()
    assert "subject" in csv_text.splitlines()[0]
    assert "alpha" in csv_text


def test_load_records_from_run_dir_roundtrip(tmp_path):
    # Manually lay out a fake JSONL run dir and ensure rehydration works.
    from pithos.eval.serde import dump_record

    cases_dir = tmp_path / "cases" / "round_1"
    cases_dir.mkdir(parents=True)
    rec = _record("alpha", "q1", 100)
    (cases_dir / "alpha__stub.jsonl").write_text(
        dump_record(rec) + "\n", encoding="utf-8"
    )
    rehydrated = load_records_from_run_dir(str(tmp_path))
    assert len(rehydrated) == 1
    assert rehydrated[0].subject_name == "alpha"
    assert rehydrated[0].grade.score == 100.0


def test_load_records_from_missing_dir_returns_empty(tmp_path):
    assert load_records_from_run_dir(str(tmp_path / "nope")) == []
