"""Tests for the /api/v1/tools Flask blueprint."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from theteam.api import tools as tools_api


@pytest.fixture
def client(make_app, monkeypatch):
    fake_registry = MagicMock()
    monkeypatch.setattr(tools_api, "_tool_registry", fake_registry)
    app = make_app(tools_api.bp)
    yield app.test_client(), fake_registry


def test_list_tools(client):
    c, reg = client
    reg.tools = {"a": SimpleNamespace(name="a"), "b": SimpleNamespace(name="b")}
    resp = c.get("/api/v1/tools/")
    assert resp.status_code == 200
    assert sorted(resp.get_json()["tools"]) == ["a", "b"]


def test_get_tool_ok(client):
    c, reg = client
    tool = SimpleNamespace(
        name="a",
        path="/usr/bin/a",
        description="d",
        platform="linux",
        source="builtin",
    )
    reg.tools = {"a": tool}
    resp = c.get("/api/v1/tools/a")
    assert resp.status_code == 200
    body = resp.get_json()["tool"]
    assert body["name"] == "a"
    assert body["path"] == "/usr/bin/a"


def test_get_tool_404(client):
    c, reg = client
    reg.tools = {}
    resp = c.get("/api/v1/tools/missing")
    assert resp.status_code == 404
