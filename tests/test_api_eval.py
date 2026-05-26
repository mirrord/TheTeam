"""Tests for the /api/v1/eval Flask blueprint."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from theteam.api import eval as eval_api

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_service():
    """Return a MagicMock that stands in for the module-level eval_service."""
    return MagicMock()


@pytest.fixture
def client(make_app, mock_service, monkeypatch):
    monkeypatch.setattr(eval_api, "eval_service", mock_service)
    app = make_app(eval_api.bp)
    yield app.test_client(), mock_service


# ---------------------------------------------------------------------------
# GET /configs
# ---------------------------------------------------------------------------


def test_list_configs_ok(client):
    c, svc = client
    svc.list_configs.return_value = [
        {"name": "cfg1", "subject_count": 2, "task_count": 3}
    ]
    resp = c.get("/api/v1/eval/configs")
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["configs"][0]["name"] == "cfg1"


def test_list_configs_service_error(client):
    c, svc = client
    svc.list_configs.side_effect = RuntimeError("disk error")
    resp = c.get("/api/v1/eval/configs")
    assert resp.status_code == 500
    assert "error" in resp.get_json()


# ---------------------------------------------------------------------------
# GET /configs/<name>
# ---------------------------------------------------------------------------


def test_get_config_ok(client):
    c, svc = client
    svc.get_config.return_value = {"name": "cfg1", "subjects": {"s1": {}}}
    resp = c.get("/api/v1/eval/configs/cfg1")
    assert resp.status_code == 200
    assert resp.get_json()["config"]["name"] == "cfg1"
    svc.get_config.assert_called_once_with("cfg1")


def test_get_config_404(client):
    c, svc = client
    svc.get_config.return_value = None
    resp = c.get("/api/v1/eval/configs/missing")
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# GET /runs
# ---------------------------------------------------------------------------


def test_list_runs_ok(client):
    c, svc = client
    svc.list_runs.return_value = [{"name": "run_001", "config_name": "cfg1"}]
    resp = c.get("/api/v1/eval/runs")
    assert resp.status_code == 200
    assert resp.get_json()["runs"][0]["name"] == "run_001"


def test_list_runs_empty(client):
    c, svc = client
    svc.list_runs.return_value = []
    resp = c.get("/api/v1/eval/runs")
    assert resp.status_code == 200
    assert resp.get_json()["runs"] == []


# ---------------------------------------------------------------------------
# GET /runs/detail
# ---------------------------------------------------------------------------


def test_get_run_detail_ok(client):
    c, svc = client
    svc.get_run.return_value = {"report": {"config_name": "c"}, "cases_by_subject": {}}
    resp = c.get("/api/v1/eval/runs/detail?run_dir=/some/run/path")
    assert resp.status_code == 200
    svc.get_run.assert_called_once_with("/some/run/path")


def test_get_run_detail_missing_param(client):
    c, svc = client
    resp = c.get("/api/v1/eval/runs/detail")
    assert resp.status_code == 400


def test_get_run_detail_not_found(client):
    c, svc = client
    svc.get_run.side_effect = FileNotFoundError("run not found")
    resp = c.get("/api/v1/eval/runs/detail?run_dir=/missing")
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# POST /runs  (start benchmark)
# ---------------------------------------------------------------------------


def test_start_run_ok(client):
    c, svc = client
    svc.start_run.return_value = "run-uuid-1234"
    payload = {"config": {"name": "cfg1"}, "options": {"rounds": 2}}
    resp = c.post(
        "/api/v1/eval/runs",
        data=json.dumps(payload),
        content_type="application/json",
    )
    assert resp.status_code == 202
    body = resp.get_json()
    assert body["run_id"] == "run-uuid-1234"
    assert body["status"] == "starting"


def test_start_run_missing_config(client):
    c, svc = client
    resp = c.post(
        "/api/v1/eval/runs",
        data=json.dumps({"options": {}}),
        content_type="application/json",
    )
    assert resp.status_code == 400


def test_start_run_no_body(client):
    c, svc = client
    resp = c.post("/api/v1/eval/runs", content_type="application/json")
    assert resp.status_code == 400


def test_start_run_service_error(client):
    c, svc = client
    svc.start_run.side_effect = ValueError("bad config")
    resp = c.post(
        "/api/v1/eval/runs",
        data=json.dumps({"config": {"name": "x"}}),
        content_type="application/json",
    )
    assert resp.status_code == 500


# ---------------------------------------------------------------------------
# GET /runs/active/<run_id>
# ---------------------------------------------------------------------------


def test_get_active_run_ok(client):
    c, svc = client
    svc.get_run_status.return_value = {
        "id": "abc",
        "status": "running",
        "completed": 3,
        "total": 10,
    }
    resp = c.get("/api/v1/eval/runs/active/abc")
    assert resp.status_code == 200
    assert resp.get_json()["run"]["status"] == "running"
    svc.get_run_status.assert_called_once_with("abc")


def test_get_active_run_404(client):
    c, svc = client
    svc.get_run_status.return_value = None
    resp = c.get("/api/v1/eval/runs/active/no-such-run")
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# DELETE /runs/active/<run_id>
# ---------------------------------------------------------------------------


def test_stop_run_ok(client):
    c, svc = client
    svc.stop_run.return_value = True
    resp = c.delete("/api/v1/eval/runs/active/abc")
    assert resp.status_code == 200
    assert "stop" in resp.get_json()["message"].lower()
    svc.stop_run.assert_called_once_with("abc")


def test_stop_run_not_found(client):
    c, svc = client
    svc.stop_run.return_value = False
    resp = c.delete("/api/v1/eval/runs/active/gone")
    assert resp.status_code == 404
