"""Shared pytest fixtures for TheTeam tests.

These fixtures help build isolated Flask apps and service instances for the
API and service test modules. Each fixture uses a temporary directory so
tests do not touch any real configs/data on disk.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import pytest
from flask import Blueprint, Flask


@pytest.fixture
def make_app() -> Callable[[Blueprint], Flask]:
    """Return a factory that builds a Flask app with a single blueprint.

    Usage:
        app = make_app(bp)
        client = app.test_client()
    """

    def _factory(*blueprints: Blueprint) -> Flask:
        app = Flask(__name__)
        app.config.update(TESTING=True, SECRET_KEY="test-secret")
        for bp in blueprints:
            app.register_blueprint(bp)
        return app

    return _factory


@pytest.fixture
def tmp_agents_dir(tmp_path: Path) -> Path:
    """Return an empty temporary agents config directory."""
    d = tmp_path / "agents"
    d.mkdir()
    return d


@pytest.fixture
def tmp_flowcharts_dir(tmp_path: Path) -> Path:
    """Return an empty temporary flowcharts config directory."""
    d = tmp_path / "flowcharts"
    d.mkdir()
    return d


@pytest.fixture
def tmp_conversations_dir(tmp_path: Path) -> Path:
    """Return an empty temporary conversations storage directory."""
    d = tmp_path / "conversations"
    d.mkdir()
    return d
