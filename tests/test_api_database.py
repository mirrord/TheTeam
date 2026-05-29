"""Tests for the /api/v1/database Flask blueprint."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from theteam.api import database as database_api


@pytest.fixture
def client(make_app, monkeypatch):
    fake = MagicMock()
    monkeypatch.setattr(database_api, "_db_manager", fake)
    app = make_app(database_api.bp)
    yield app.test_client(), fake


def test_info_ok(client):
    c, mgr = client
    mgr.get_database_info.return_value = [
        SimpleNamespace(
            name="memory",
            type="vector",
            path="/tmp/mem",
            size_bytes=0,
            available=True,
            error=None,
        )
    ]
    resp = c.get("/api/v1/database/info")
    assert resp.status_code == 200
    assert resp.get_json()["databases"][0]["name"] == "memory"


def test_clear_requires_confirm(client):
    c, _ = client
    resp = c.post("/api/v1/database/clear/memory", json={})
    assert resp.status_code == 400


def test_clear_memory(client):
    c, mgr = client
    resp = c.post("/api/v1/database/clear/memory", json={"confirm": True})
    assert resp.status_code == 200
    mgr.clear_memory.assert_called_once()


def test_clear_history(client):
    c, mgr = client
    resp = c.post("/api/v1/database/clear/history", json={"confirm": True})
    assert resp.status_code == 200
    mgr.clear_history.assert_called_once()


def test_clear_flowcharts(client):
    c, mgr = client
    resp = c.post("/api/v1/database/clear/flowcharts", json={"confirm": True})
    assert resp.status_code == 200
    mgr.clear_flowcharts.assert_called_once()


def test_clear_all(client):
    c, mgr = client
    mgr.clear_all.return_value = {"memory": True}
    resp = c.post("/api/v1/database/clear/all", json={"confirm": True})
    assert resp.status_code == 200
    mgr.clear_all.assert_called_once()


def test_clear_invalid_database(client):
    c, _ = client
    resp = c.post("/api/v1/database/clear/bogus", json={"confirm": True})
    assert resp.status_code == 400


def test_search_no_query(client):
    c, _ = client
    resp = c.post("/api/v1/database/search", json={})
    assert resp.status_code == 400


def test_search_semantic(client):
    c, mgr = client
    mgr.search_all.return_value = {
        "memory": [
            SimpleNamespace(
                database="memory",
                result_type="memory",
                content="hi",
                metadata={},
                relevance_score=0.9,
                match_type="semantic",
            )
        ]
    }
    resp = c.post("/api/v1/database/search", json={"query": "hi"})
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["query"] == "hi"
    assert body["results"]["memory"][0]["content"] == "hi"


def test_search_exact(client):
    c, mgr = client
    mgr.search_exact.return_value = {"history": []}
    resp = c.post("/api/v1/database/search", json={"query": "foo", "exact": True})
    assert resp.status_code == 200
    mgr.search_exact.assert_called_once()


def test_memory_categories(client):
    c, mgr = client
    mgr.memory.list_categories.return_value = ["a", "b"]
    resp = c.get("/api/v1/database/memory/categories")
    assert resp.status_code == 200
    assert resp.get_json()["categories"] == ["a", "b"]


def test_memory_search_no_query(client):
    c, _ = client
    resp = c.post("/api/v1/database/memory/search", json={})
    assert resp.status_code == 400


def test_memory_search_ok(client):
    c, mgr = client
    mgr.memory.search_all_categories.return_value = {
        "cat": [
            SimpleNamespace(
                id="1",
                category="cat",
                content="x",
                metadata={},
                distance=0.1,
                relevance_score=0.9,
            )
        ]
    }
    resp = c.post("/api/v1/database/memory/search", json={"query": "q"})
    assert resp.status_code == 200
    assert resp.get_json()["results"]["cat"][0]["id"] == "1"
