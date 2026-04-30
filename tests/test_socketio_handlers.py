"""Tests for :mod:`theteam.api.socketio_handlers`.

Uses Flask-SocketIO's in-process test client, which avoids spinning up a
real eventlet server.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from flask import Flask
from flask_socketio import SocketIO

from theteam.api import socketio_handlers
from theteam.api.socketio_handlers import (
    active_connections,
    emit_to_client,
    emit_to_room,
    register_handlers,
)


@pytest.fixture
def sio_app():
    app = Flask(__name__)
    app.config["TESTING"] = True
    app.config["SECRET_KEY"] = "test"
    sio = SocketIO(app, async_mode="threading")
    register_handlers(sio)
    yield app, sio
    active_connections.clear()


def _client(sio_app):
    app, sio = sio_app
    return sio.test_client(app)


# ---------------------------------------------------------------------------
# Connection lifecycle
# ---------------------------------------------------------------------------


def test_connect_emits_established(sio_app):
    client = _client(sio_app)
    received = client.get_received()
    events = [r["name"] for r in received]
    assert "connection_established" in events
    assert active_connections  # at least one entry


def test_disconnect_clears_connection(sio_app):
    client = _client(sio_app)
    assert active_connections
    client.disconnect()
    assert active_connections == {}


def test_ping_replies_pong(sio_app):
    client = _client(sio_app)
    client.get_received()  # drain
    client.emit("ping", {"x": 1})
    received = client.get_received()
    assert any(r["name"] == "pong" for r in received)


# ---------------------------------------------------------------------------
# Rooms
# ---------------------------------------------------------------------------


def test_join_and_leave_room(sio_app):
    client = _client(sio_app)
    client.get_received()
    client.emit("join_room", {"room": "r1"})
    joined = [r for r in client.get_received() if r["name"] == "room_joined"]
    assert joined and joined[0]["args"][0]["room"] == "r1"

    client.emit("leave_room", {"room": "r1"})
    left = [r for r in client.get_received() if r["name"] == "room_left"]
    assert left


def test_join_room_missing_name_emits_error(sio_app):
    client = _client(sio_app)
    client.get_received()
    client.emit("join_room", {})
    received = client.get_received()
    assert any(r["name"] == "error" for r in received)


# ---------------------------------------------------------------------------
# Chat / execute message dispatch
# ---------------------------------------------------------------------------


def test_chat_message_invokes_chat_service(sio_app, monkeypatch):
    fake_chat_service = MagicMock()
    fake_chat_service.send_message.return_value = "msg-1"

    # The handler imports lazily: from theteam.api.chat import chat_service
    import theteam.api.chat as chat_api

    monkeypatch.setattr(chat_api, "chat_service", fake_chat_service)

    client = _client(sio_app)
    client.get_received()
    client.emit("chat_message", {"conversation_id": "c1", "message": "hi"})
    received = client.get_received()
    events = {r["name"] for r in received}
    assert "message_sent" in events
    fake_chat_service.send_message.assert_called_once()


def test_chat_message_missing_args_emits_error(sio_app):
    client = _client(sio_app)
    client.get_received()
    client.emit("chat_message", {"conversation_id": "c1"})
    received = client.get_received()
    assert any(r["name"] == "error" for r in received)


def test_execute_flowchart_handler(sio_app, monkeypatch):
    fake_service = MagicMock()
    fake_service.start_execution.return_value = "exec-1"

    # Handler does `from theteam.services.flowchart_service import FlowchartService`
    # then calls `FlowchartService()`. Patch the class.
    import theteam.services.flowchart_service as fc_module

    monkeypatch.setattr(fc_module, "FlowchartService", lambda *a, **kw: fake_service)

    client = _client(sio_app)
    client.get_received()
    client.emit("execute_flowchart", {"flowchart_id": "f1", "context": {}})
    received = client.get_received()
    events = {r["name"] for r in received}
    assert "execution_started" in events


def test_execute_flowchart_missing_id(sio_app):
    client = _client(sio_app)
    client.get_received()
    client.emit("execute_flowchart", {})
    received = client.get_received()
    assert any(r["name"] == "error" for r in received)


# ---------------------------------------------------------------------------
# emit helpers
# ---------------------------------------------------------------------------


def test_emit_to_client_calls_socketio_emit():
    sio = MagicMock()
    emit_to_client(sio, "client-1", "evt", {"x": 1})
    sio.emit.assert_called_once_with("evt", {"x": 1}, room="client-1")


def test_emit_to_client_swallows_exception():
    sio = MagicMock()
    sio.emit.side_effect = RuntimeError("dead socket")
    # Should not raise.
    emit_to_client(sio, "client-1", "evt", {})


def test_emit_to_room_calls_socketio_emit():
    sio = MagicMock()
    emit_to_room(sio, "room-1", "evt", {})
    sio.emit.assert_called_once_with("evt", {}, room="room-1")
