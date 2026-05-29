"""Tests for :class:`theteam.services.agent_service.AgentService`."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

from theteam.services.agent_service import AgentService


@pytest.fixture
def service(tmp_agents_dir: Path) -> AgentService:
    return AgentService(config_dir=tmp_agents_dir)


# ---------------------------------------------------------------------------
# list / get
# ---------------------------------------------------------------------------


def test_list_empty(service):
    assert service.list_agents() == []


def test_create_runtime_and_list(service):
    agent_id = service.create_agent({"name": "Alice", "model": "test-model"})
    assert agent_id == "alice"
    agents = service.list_agents()
    assert len(agents) == 1
    assert agents[0]["id"] == "alice"
    assert agents[0]["source"] == "runtime"


def test_create_file_persists_yaml(service, tmp_agents_dir):
    service.create_agent({"name": "Bob", "model": "test-model", "save_to_file": True})
    yaml_file = tmp_agents_dir / "bob.yaml"
    assert yaml_file.exists()
    data = yaml.safe_load(yaml_file.read_text())
    # save_to_file flag should not be persisted
    assert "save_to_file" not in data
    assert data["model"] == "test-model"


def test_create_missing_model_rejected(service):
    with pytest.raises(ValueError, match="model"):
        service.create_agent({"name": "no-model"})


def test_create_uses_explicit_id(service):
    agent_id = service.create_agent({"id": "custom", "model": "m"})
    assert agent_id == "custom"


def test_get_runtime_agent(service):
    service.create_agent({"id": "a", "model": "m"})
    agent = service.get_agent("a")
    assert agent is not None
    assert agent["source"] == "runtime"
    assert agent["config"]["model"] == "m"


def test_get_file_agent(service, tmp_agents_dir):
    (tmp_agents_dir / "x.yaml").write_text(yaml.dump({"name": "X", "model": "m"}))
    agent = service.get_agent("x")
    assert agent is not None
    assert agent["source"] == "file"
    assert agent["config"]["model"] == "m"


def test_get_missing_returns_none(service):
    assert service.get_agent("nope") is None


def test_list_includes_file_and_runtime(service, tmp_agents_dir):
    (tmp_agents_dir / "f.yaml").write_text(yaml.dump({"model": "m"}))
    service.create_agent({"id": "r", "model": "m"})
    ids = {a["id"] for a in service.list_agents()}
    assert ids == {"f", "r"}


def test_list_skips_corrupt_yaml(service, tmp_agents_dir, caplog):
    (tmp_agents_dir / "bad.yaml").write_text("not: valid: yaml: : :")
    # malformed but loadable as None or string in some cases — write a clear fail
    (tmp_agents_dir / "bad.yaml").write_text(":\n - invalid")
    agents = service.list_agents()
    # bad file logs an error but doesn't crash
    assert all(a["id"] != "bad" or "model" in a for a in agents)


# ---------------------------------------------------------------------------
# update / delete
# ---------------------------------------------------------------------------


def test_update_runtime(service):
    service.create_agent({"id": "a", "model": "m"})
    assert service.update_agent("a", {"model": "m2"}) is True
    assert service.get_agent("a")["config"]["model"] == "m2"


def test_update_file(service, tmp_agents_dir):
    (tmp_agents_dir / "x.yaml").write_text(yaml.dump({"model": "m"}))
    assert service.update_agent("x", {"model": "m2"}) is True
    data = yaml.safe_load((tmp_agents_dir / "x.yaml").read_text())
    assert data["model"] == "m2"


def test_update_missing_returns_false(service):
    assert service.update_agent("ghost", {"model": "m"}) is False


def test_delete_runtime(service):
    service.create_agent({"id": "a", "model": "m"})
    assert service.delete_agent("a") is True
    assert service.get_agent("a") is None


def test_delete_file(service, tmp_agents_dir):
    (tmp_agents_dir / "x.yaml").write_text(yaml.dump({"model": "m"}))
    assert service.delete_agent("x") is True
    assert not (tmp_agents_dir / "x.yaml").exists()


def test_delete_missing_returns_false(service):
    assert service.delete_agent("ghost") is False


# ---------------------------------------------------------------------------
# test_agent
# ---------------------------------------------------------------------------


def test_test_agent_missing_raises(service):
    with pytest.raises(ValueError, match="not found"):
        service.test_agent("ghost", "hello")


def test_test_agent_invokes_ollama(service):
    service.create_agent({"id": "a", "model": "test-model"})
    fake_agent = MagicMock()
    fake_agent.send.return_value = "hi back"
    with patch("pithos.agent.OllamaAgent", return_value=fake_agent) as cls:
        result = service.test_agent("a", "hello")
    cls.assert_called_once()
    fake_agent.send.assert_called_once_with("hello")
    assert result == {"prompt": "hello", "response": "hi back", "agent_id": "a"}
