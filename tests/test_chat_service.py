"""Tests for :class:`theteam.services.chat_service.ChatService`."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from theteam.services.chat_service import ChatService, Conversation, Message


@pytest.fixture
def service(tmp_conversations_dir: Path) -> ChatService:
    return ChatService(storage_dir=tmp_conversations_dir)


# ---------------------------------------------------------------------------
# CRUD
# ---------------------------------------------------------------------------


def test_list_empty(service):
    assert service.list_conversations() == []


def test_create_and_get(service):
    cid = service.create_conversation(agent_id="agent-1", title="hi")
    convo = service.get_conversation(cid)
    assert convo is not None
    assert convo["agent_id"] == "agent-1"
    assert convo["title"] == "hi"
    assert convo["messages"] == []


def test_create_default_title(service):
    cid = service.create_conversation()
    convo = service.get_conversation(cid)
    assert convo["title"].startswith("Conversation ")


def test_create_persists_to_disk(service, tmp_conversations_dir):
    cid = service.create_conversation()
    files = list(tmp_conversations_dir.glob("*.json"))
    assert any(f.stem == cid for f in files)


def test_get_missing_returns_none(service):
    assert service.get_conversation("ghost") is None


def test_list_sorted_by_updated_desc(service):
    # create in order, but bump updated_at for older one
    a = service.create_conversation(title="A")
    b = service.create_conversation(title="B")
    # Ensure deterministic ordering by updating B's timestamp later
    service.add_system_message(b, "ping")
    convos = service.list_conversations()
    assert convos[0]["id"] == b
    assert convos[1]["id"] == a


def test_delete(service, tmp_conversations_dir):
    cid = service.create_conversation()
    assert service.delete_conversation(cid) is True
    assert service.get_conversation(cid) is None
    assert not (tmp_conversations_dir / f"{cid}.json").exists()


def test_delete_missing(service):
    assert service.delete_conversation("ghost") is False


def test_update_agent(service):
    cid = service.create_conversation(agent_id="old")
    assert service.update_conversation_agent(cid, "new") is True
    assert service.get_conversation(cid)["agent_id"] == "new"


def test_update_agent_missing(service):
    assert service.update_conversation_agent("ghost", "x") is False


def test_add_system_message(service):
    cid = service.create_conversation()
    assert service.add_system_message(cid, "hello") is True
    convo = service.get_conversation(cid)
    assert len(convo["messages"]) == 1
    assert convo["messages"][0]["role"] == "system"


def test_add_system_message_missing(service):
    assert service.add_system_message("ghost", "x") is False


def test_load_existing_conversations(tmp_conversations_dir):
    # Pre-seed a conversation file and ensure ChatService loads it.
    convo = Conversation(
        id="c1",
        title="t",
        agent_id=None,
        created_at="2024-01-01T00:00:00",
        updated_at="2024-01-01T00:00:00",
        messages=[Message(id="m1", role="user", content="hi", timestamp="t")],
    )
    (tmp_conversations_dir / "c1.json").write_text(json.dumps(convo.to_dict()))
    svc = ChatService(storage_dir=tmp_conversations_dir)
    loaded = svc.get_conversation("c1")
    assert loaded is not None
    assert loaded["messages"][0]["content"] == "hi"


# ---------------------------------------------------------------------------
# send_message + streaming
# ---------------------------------------------------------------------------


def test_send_message_missing_conversation(service):
    with pytest.raises(ValueError, match="not found"):
        service.send_message("ghost", "hello")


def test_send_message_appends_user_message_and_streams(service):
    cid = service.create_conversation(agent_id="a")

    fake_agent = MagicMock()
    fake_agent.contexts = {"default": MagicMock(add_message=MagicMock())}
    fake_agent.create_context = MagicMock()
    fake_agent.stream = MagicMock(return_value=iter(["hel", "lo"]))

    fake_agent_service = MagicMock()
    fake_agent_service.get_agent.return_value = {"config": {"model": "m", "name": "A"}}

    socketio = MagicMock()
    # Run synchronously: invoke target directly instead of background_task.
    socketio.start_background_task.side_effect = lambda fn, *a, **kw: fn(*a, **kw)

    with (
        patch("pithos.agent.OllamaAgent", return_value=fake_agent),
        patch(
            "theteam.services.agent_service.AgentService",
            return_value=fake_agent_service,
        ),
        patch("theteam.api.socketio_handlers.emit_to_client") as emit_mock,
    ):
        msg_id = service.send_message(cid, "hello", client_id="c1", socketio=socketio)

    assert msg_id  # returned user message id
    convo = service.get_conversation(cid)
    roles = [m["role"] for m in convo["messages"]]
    assert roles == ["user", "assistant"]
    assert convo["messages"][1]["content"] == "hello"  # streamed assistant text

    # Streaming events emitted in expected order.
    events = [call.args[2] for call in emit_mock.call_args_list]
    assert events[0] == "message_processing"
    assert "stream_start" in events
    assert events.count("stream_chunk") == 2
    assert events[-1] == "stream_end"


def test_send_message_emits_error_on_exception(service):
    cid = service.create_conversation(agent_id="a")

    fake_agent_service = MagicMock()
    fake_agent_service.get_agent.return_value = {"config": {"model": "m"}}

    socketio = MagicMock()
    socketio.start_background_task.side_effect = lambda fn, *a, **kw: fn(*a, **kw)

    with (
        patch("pithos.agent.OllamaAgent", side_effect=RuntimeError("boom")),
        patch(
            "theteam.services.agent_service.AgentService",
            return_value=fake_agent_service,
        ),
        patch("theteam.api.socketio_handlers.emit_to_client") as emit_mock,
    ):
        service.send_message(cid, "hi", client_id="c1", socketio=socketio)

    events = [call.args[2] for call in emit_mock.call_args_list]
    assert "message_error" in events


# ---------------------------------------------------------------------------
# Base-model and per-conversation tools (new in this revision)
# ---------------------------------------------------------------------------


def test_update_base_model_clears_agent(service):
    cid = service.create_conversation(agent_id="agent-1")
    assert service.update_conversation_base_model(cid, "llama3:8b") is True
    convo = service.get_conversation(cid)
    assert convo["base_model"] == "llama3:8b"
    assert convo["agent_id"] is None


def test_update_agent_clears_base_model(service):
    cid = service.create_conversation()
    service.update_conversation_base_model(cid, "llama3:8b")
    assert service.update_conversation_agent(cid, "agent-2") is True
    convo = service.get_conversation(cid)
    assert convo["agent_id"] == "agent-2"
    assert convo["base_model"] is None


def test_update_tools_allow_list(service):
    cid = service.create_conversation()
    assert service.update_conversation_tools(cid, ["git", "python"]) is True
    convo = service.get_conversation(cid)
    assert convo["enabled_tools"] == ["git", "python"]


def test_update_tools_clear_override(service):
    cid = service.create_conversation()
    service.update_conversation_tools(cid, ["git"])
    assert service.update_conversation_tools(cid, None) is True
    convo = service.get_conversation(cid)
    assert convo["enabled_tools"] is None


def test_update_tools_disable_all(service):
    cid = service.create_conversation()
    assert service.update_conversation_tools(cid, []) is True
    convo = service.get_conversation(cid)
    assert convo["enabled_tools"] == []


def test_base_model_persists_across_reload(service, tmp_conversations_dir):
    cid = service.create_conversation()
    service.update_conversation_base_model(cid, "llama3:8b")
    service.update_conversation_tools(cid, ["git"])
    # Reload from disk
    fresh = ChatService(storage_dir=tmp_conversations_dir)
    convo = fresh.get_conversation(cid)
    assert convo["base_model"] == "llama3:8b"
    assert convo["enabled_tools"] == ["git"]


# ---------------------------------------------------------------------------
# Rename
# ---------------------------------------------------------------------------


def test_rename_conversation_ok(service):
    cid = service.create_conversation(title="Original")
    assert service.rename_conversation(cid, "New Title") is True
    convo = service.get_conversation(cid)
    assert convo["title"] == "New Title"


def test_rename_updates_updated_at(service):
    import time

    cid = service.create_conversation(title="Original")
    original_updated = service.get_conversation(cid)["updated_at"]
    time.sleep(0.01)
    service.rename_conversation(cid, "Changed")
    new_updated = service.get_conversation(cid)["updated_at"]
    assert new_updated > original_updated


def test_rename_persists_to_disk(service, tmp_conversations_dir):
    cid = service.create_conversation(title="Old")
    service.rename_conversation(cid, "Persisted")
    fresh = ChatService(storage_dir=tmp_conversations_dir)
    assert fresh.get_conversation(cid)["title"] == "Persisted"


def test_rename_missing_conversation(service):
    assert service.rename_conversation("nonexistent-id", "Whatever") is False


def test_update_missing_returns_false(service):
    assert service.update_conversation_base_model("ghost", "m") is False
    assert service.update_conversation_tools("ghost", []) is False
