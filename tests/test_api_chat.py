"""Tests for the /api/v1/chat Flask blueprint."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from theteam.api import chat as chat_api


@pytest.fixture
def client(make_app, monkeypatch):
    fake = MagicMock()
    monkeypatch.setattr(chat_api, "chat_service", fake)
    app = make_app(chat_api.bp)
    yield app.test_client(), fake


def test_list_conversations(client):
    c, svc = client
    svc.list_conversations.return_value = [{"id": "1"}]
    resp = c.get("/api/v1/chat/conversations")
    assert resp.status_code == 200
    assert resp.get_json()["conversations"] == [{"id": "1"}]


def test_list_error(client):
    c, svc = client
    svc.list_conversations.side_effect = RuntimeError
    resp = c.get("/api/v1/chat/conversations")
    assert resp.status_code == 500


def test_get_conversation_ok(client):
    c, svc = client
    svc.get_conversation.return_value = {"id": "1"}
    resp = c.get("/api/v1/chat/conversations/1")
    assert resp.status_code == 200


def test_get_conversation_404(client):
    c, svc = client
    svc.get_conversation.return_value = None
    resp = c.get("/api/v1/chat/conversations/1")
    assert resp.status_code == 404


def test_create_ok(client):
    c, svc = client
    svc.create_conversation.return_value = "id1"
    resp = c.post("/api/v1/chat/conversations", json={"agent_id": "a", "title": "t"})
    assert resp.status_code == 201
    assert resp.get_json()["conversation_id"] == "id1"


def test_create_no_body(client):
    c, svc = client
    svc.create_conversation.return_value = "id1"
    resp = c.post("/api/v1/chat/conversations", json={})
    assert resp.status_code == 201


def test_delete_ok(client):
    c, svc = client
    svc.delete_conversation.return_value = True
    resp = c.delete("/api/v1/chat/conversations/1")
    assert resp.status_code == 200


def test_delete_404(client):
    c, svc = client
    svc.delete_conversation.return_value = False
    resp = c.delete("/api/v1/chat/conversations/1")
    assert resp.status_code == 404


def test_send_message_ok(client):
    c, svc = client
    svc.send_message.return_value = "msg-1"
    resp = c.post("/api/v1/chat/conversations/1/messages", json={"message": "hi"})
    assert resp.status_code == 202
    assert resp.get_json()["message_id"] == "msg-1"


def test_send_message_no_body(client):
    c, _ = client
    resp = c.post("/api/v1/chat/conversations/1/messages", json={})
    assert resp.status_code == 400


def test_send_message_value_error(client):
    c, svc = client
    svc.send_message.side_effect = ValueError("conversation not found")
    resp = c.post("/api/v1/chat/conversations/1/messages", json={"message": "hi"})
    assert resp.status_code == 400


def test_update_agent_ok(client):
    c, svc = client
    svc.update_conversation_agent.return_value = True
    resp = c.put("/api/v1/chat/conversations/1/agent", json={"agent_id": "a"})
    assert resp.status_code == 200


def test_update_agent_no_id(client):
    c, _ = client
    resp = c.put("/api/v1/chat/conversations/1/agent", json={})
    assert resp.status_code == 400


def test_update_agent_404(client):
    c, svc = client
    svc.update_conversation_agent.return_value = False
    resp = c.put("/api/v1/chat/conversations/1/agent", json={"agent_id": "a"})
    assert resp.status_code == 404


def test_update_base_model_ok(client):
    c, svc = client
    svc.update_conversation_base_model.return_value = True
    resp = c.put("/api/v1/chat/conversations/1/agent", json={"base_model": "llama3:8b"})
    assert resp.status_code == 200
    svc.update_conversation_base_model.assert_called_once_with("1", "llama3:8b")


def test_update_base_model_404(client):
    c, svc = client
    svc.update_conversation_base_model.return_value = False
    resp = c.put("/api/v1/chat/conversations/1/agent", json={"base_model": "m"})
    assert resp.status_code == 404


def test_update_tools_ok(client):
    c, svc = client
    svc.update_conversation_tools.return_value = True
    resp = c.put("/api/v1/chat/conversations/1/tools", json={"enabled_tools": ["git"]})
    assert resp.status_code == 200
    svc.update_conversation_tools.assert_called_once_with("1", ["git"])


def test_update_tools_clear(client):
    c, svc = client
    svc.update_conversation_tools.return_value = True
    resp = c.put("/api/v1/chat/conversations/1/tools", json={"enabled_tools": None})
    assert resp.status_code == 200


def test_update_tools_missing(client):
    c, _ = client
    resp = c.put("/api/v1/chat/conversations/1/tools", json={})
    assert resp.status_code == 400


def test_update_tools_bad_type(client):
    c, _ = client
    resp = c.put("/api/v1/chat/conversations/1/tools", json={"enabled_tools": "git"})
    assert resp.status_code == 400


def test_update_tools_404(client):
    c, svc = client
    svc.update_conversation_tools.return_value = False
    resp = c.put("/api/v1/chat/conversations/1/tools", json={"enabled_tools": []})
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Rename
# ---------------------------------------------------------------------------


def test_rename_ok(client):
    c, svc = client
    svc.rename_conversation.return_value = True
    resp = c.put("/api/v1/chat/conversations/1/title", json={"title": "New Name"})
    assert resp.status_code == 200
    svc.rename_conversation.assert_called_once_with("1", "New Name")


def test_rename_404(client):
    c, svc = client
    svc.rename_conversation.return_value = False
    resp = c.put("/api/v1/chat/conversations/1/title", json={"title": "New Name"})
    assert resp.status_code == 404


def test_rename_missing_title(client):
    c, _ = client
    resp = c.put("/api/v1/chat/conversations/1/title", json={})
    assert resp.status_code == 400


def test_rename_empty_title(client):
    c, _ = client
    resp = c.put("/api/v1/chat/conversations/1/title", json={"title": "   "})
    assert resp.status_code == 400
