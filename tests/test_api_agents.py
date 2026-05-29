"""Tests for the /api/v1/agents Flask blueprint."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from theteam.api import agents as agents_api


@pytest.fixture
def client(make_app, monkeypatch):
    fake = MagicMock()
    monkeypatch.setattr(agents_api, "agent_service", fake)
    app = make_app(agents_api.bp)
    yield app.test_client(), fake


def test_list_ok(client):
    c, svc = client
    svc.list_agents.return_value = [{"id": "a"}]
    resp = c.get("/api/v1/agents/")
    assert resp.status_code == 200
    assert resp.get_json() == {"agents": [{"id": "a"}]}


def test_list_error(client):
    c, svc = client
    svc.list_agents.side_effect = RuntimeError("boom")
    resp = c.get("/api/v1/agents/")
    assert resp.status_code == 500
    assert "error" in resp.get_json()


def test_get_agent_ok(client):
    c, svc = client
    svc.get_agent.return_value = {"id": "a"}
    resp = c.get("/api/v1/agents/a")
    assert resp.status_code == 200
    assert resp.get_json()["agent"]["id"] == "a"


def test_get_agent_404(client):
    c, svc = client
    svc.get_agent.return_value = None
    resp = c.get("/api/v1/agents/x")
    assert resp.status_code == 404


def test_create_ok(client):
    c, svc = client
    svc.create_agent.return_value = "new-id"
    resp = c.post("/api/v1/agents/", json={"model": "m"})
    assert resp.status_code == 201
    assert resp.get_json()["agent_id"] == "new-id"


def test_create_no_body(client):
    c, _ = client
    resp = c.post("/api/v1/agents/", json={})
    assert resp.status_code == 400


def test_create_value_error(client):
    c, svc = client
    svc.create_agent.side_effect = ValueError("missing model")
    resp = c.post("/api/v1/agents/", json={"name": "x"})
    assert resp.status_code == 400


def test_update_ok(client):
    c, svc = client
    svc.update_agent.return_value = True
    resp = c.put("/api/v1/agents/a", json={"model": "m"})
    assert resp.status_code == 200


def test_update_404(client):
    c, svc = client
    svc.update_agent.return_value = False
    resp = c.put("/api/v1/agents/a", json={"model": "m"})
    assert resp.status_code == 404


def test_update_no_body(client):
    c, _ = client
    resp = c.put("/api/v1/agents/a", json={})
    assert resp.status_code == 400


def test_delete_ok(client):
    c, svc = client
    svc.delete_agent.return_value = True
    resp = c.delete("/api/v1/agents/a")
    assert resp.status_code == 200


def test_delete_404(client):
    c, svc = client
    svc.delete_agent.return_value = False
    resp = c.delete("/api/v1/agents/a")
    assert resp.status_code == 404


def test_test_agent_ok(client):
    c, svc = client
    svc.test_agent.return_value = {"prompt": "p", "response": "r", "agent_id": "a"}
    resp = c.post("/api/v1/agents/a/test", json={"prompt": "p"})
    assert resp.status_code == 200
    assert resp.get_json()["result"]["response"] == "r"


def test_test_agent_no_prompt(client):
    c, _ = client
    resp = c.post("/api/v1/agents/a/test", json={})
    assert resp.status_code == 400


def test_test_agent_value_error(client):
    c, svc = client
    svc.test_agent.side_effect = ValueError("not found")
    resp = c.post("/api/v1/agents/a/test", json={"prompt": "p"})
    assert resp.status_code == 400
