"""Tests for the /api/v1/system Flask blueprint."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from theteam.api import system as system_api


@pytest.fixture
def client(make_app):
    app = make_app(system_api.bp)
    return app.test_client()


def test_health(client):
    resp = client.get("/api/v1/system/health")
    assert resp.status_code == 200
    assert resp.get_json()["status"] == "healthy"


def test_info(client):
    resp = client.get("/api/v1/system/info")
    assert resp.status_code == 200
    body = resp.get_json()
    assert "python_version" in body
    assert "platform" in body


def test_models_ok(client):
    with patch("pithos.utils.get_available_models", return_value=["m1", "m2"]):
        resp = client.get("/api/v1/system/models")
    assert resp.status_code == 200
    assert resp.get_json()["models"] == ["m1", "m2"]


def test_models_error(client):
    with patch(
        "pithos.utils.get_available_models", side_effect=RuntimeError("ollama down")
    ):
        resp = client.get("/api/v1/system/models")
    assert resp.status_code == 500
    assert "error" in resp.get_json()
