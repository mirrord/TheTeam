"""Tests for the /api/v1/flowcharts Flask blueprint."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from theteam.api import flowcharts as flowcharts_api


@pytest.fixture
def client(make_app, monkeypatch):
    fake = MagicMock()
    monkeypatch.setattr(flowcharts_api, "flowchart_service", fake)
    app = make_app(flowcharts_api.bp)
    yield app.test_client(), fake


def test_list(client):
    c, svc = client
    svc.list_flowcharts.return_value = [{"id": "f"}]
    resp = c.get("/api/v1/flowcharts/")
    assert resp.status_code == 200


def test_get_ok(client):
    c, svc = client
    svc.get_flowchart.return_value = {"id": "f"}
    resp = c.get("/api/v1/flowcharts/f")
    assert resp.status_code == 200


def test_get_404(client):
    c, svc = client
    svc.get_flowchart.return_value = None
    resp = c.get("/api/v1/flowcharts/f")
    assert resp.status_code == 404


def test_create_ok(client):
    c, svc = client
    svc.create_flowchart.return_value = "id1"
    resp = c.post("/api/v1/flowcharts/", json={"nodes": {}})
    assert resp.status_code == 201


def test_create_no_body(client):
    c, _ = client
    resp = c.post("/api/v1/flowcharts/", json={})
    assert resp.status_code == 400


def test_create_value_error(client):
    c, svc = client
    svc.create_flowchart.side_effect = ValueError("bad")
    resp = c.post("/api/v1/flowcharts/", json={"x": 1})
    assert resp.status_code == 400


def test_update_ok(client):
    c, svc = client
    svc.update_flowchart.return_value = True
    resp = c.put("/api/v1/flowcharts/f", json={"nodes": {}})
    assert resp.status_code == 200


def test_update_404(client):
    c, svc = client
    svc.update_flowchart.return_value = False
    resp = c.put("/api/v1/flowcharts/f", json={"nodes": {}})
    assert resp.status_code == 404


def test_delete_ok(client):
    c, svc = client
    svc.delete_flowchart.return_value = True
    resp = c.delete("/api/v1/flowcharts/f")
    assert resp.status_code == 200


def test_delete_404(client):
    c, svc = client
    svc.delete_flowchart.return_value = False
    resp = c.delete("/api/v1/flowcharts/f")
    assert resp.status_code == 404


def test_import_ok(client):
    c, svc = client
    svc.import_from_yaml.return_value = "id1"
    resp = c.post("/api/v1/flowcharts/import", json={"yaml": "name: x"})
    assert resp.status_code == 201


def test_import_no_yaml(client):
    c, _ = client
    resp = c.post("/api/v1/flowcharts/import", json={})
    assert resp.status_code == 400


def test_import_value_error(client):
    c, svc = client
    svc.import_from_yaml.side_effect = ValueError("bad")
    resp = c.post("/api/v1/flowcharts/import", json={"yaml": "x"})
    assert resp.status_code == 400


def test_export_ok(client):
    c, svc = client
    svc.export_to_yaml.return_value = "name: x\n"
    resp = c.get("/api/v1/flowcharts/f/export")
    assert resp.status_code == 200
    assert resp.get_json()["yaml"].startswith("name:")


def test_export_404(client):
    c, svc = client
    svc.export_to_yaml.return_value = None
    resp = c.get("/api/v1/flowcharts/f/export")
    assert resp.status_code == 404


def test_validate(client):
    c, svc = client
    svc.validate_flowchart.return_value = {"valid": True, "errors": []}
    resp = c.post("/api/v1/flowcharts/f/validate")
    assert resp.status_code == 200
    assert resp.get_json()["valid"] is True


def test_execute_ok(client):
    c, svc = client
    svc.start_execution.return_value = "exec-1"
    resp = c.post("/api/v1/flowcharts/f/execute", json={"context": {}})
    assert resp.status_code == 202
    assert resp.get_json()["execution_id"] == "exec-1"


def test_execute_value_error(client):
    c, svc = client
    svc.start_execution.side_effect = ValueError("missing")
    resp = c.post("/api/v1/flowcharts/f/execute", json={})
    assert resp.status_code == 400
