"""
SocketIO event handlers for real-time communication.
Handles connection management, reconnection, and real-time updates.
"""

import logging
from flask_socketio import emit, join_room, leave_room
from datetime import datetime

logger = logging.getLogger(__name__)

# Store active connections
active_connections = {}


def register_handlers(socketio):
    """Register all SocketIO event handlers."""

    @socketio.on("connect")
    def handle_connect():
        """Handle client connection."""
        from flask import request

        client_id = request.sid
        active_connections[client_id] = {
            "connected_at": datetime.now().isoformat(),
            "rooms": set(),
        }
        logger.info(f"Client connected: {client_id}")
        emit(
            "connection_established",
            {
                "client_id": client_id,
                "timestamp": datetime.now().isoformat(),
                "message": "Connected to TheTeam server",
            },
        )

    @socketio.on("disconnect")
    def handle_disconnect():
        """Handle client disconnection."""
        from flask import request

        client_id = request.sid
        if client_id in active_connections:
            del active_connections[client_id]
        logger.info(f"Client disconnected: {client_id}")

    @socketio.on("ping")
    def handle_ping(data):
        """Handle ping for connection monitoring."""
        # Don't log ping/pong to avoid cluttering logs
        emit("pong", {"timestamp": datetime.now().isoformat(), "received": data})

    @socketio.on("join_room")
    def handle_join_room(data):
        """Join a specific room for targeted updates."""
        from flask import request

        client_id = request.sid
        room = data.get("room")

        if not room:
            emit("error", {"message": "No room specified"})
            return

        join_room(room)
        if client_id in active_connections:
            active_connections[client_id]["rooms"].add(room)

        logger.info(f"Client {client_id} joined room: {room}")
        emit("room_joined", {"room": room, "timestamp": datetime.now().isoformat()})

    @socketio.on("leave_room")
    def handle_leave_room(data):
        """Leave a specific room."""
        from flask import request

        client_id = request.sid
        room = data.get("room")

        if not room:
            emit("error", {"message": "No room specified"})
            return

        leave_room(room)
        if (
            client_id in active_connections
            and room in active_connections[client_id]["rooms"]
        ):
            active_connections[client_id]["rooms"].remove(room)

        logger.info(f"Client {client_id} left room: {room}")
        emit("room_left", {"room": room, "timestamp": datetime.now().isoformat()})

    @socketio.on("chat_message")
    def handle_chat_message(data):
        """Handle incoming chat message (triggers async processing)."""
        from flask import request
        from theteam.api.chat import chat_service

        client_id = request.sid
        conversation_id = data.get("conversation_id")
        message = data.get("message")

        if not conversation_id or not message:
            emit("error", {"message": "Missing conversation_id or message"})
            return

        try:
            message_id = chat_service.send_message(
                conversation_id=conversation_id,
                message=message,
                client_id=client_id,
                socketio=socketio,
            )

            # Immediate acknowledgment
            emit(
                "message_sent",
                {
                    "message_id": message_id,
                    "conversation_id": conversation_id,
                    "timestamp": datetime.now().isoformat(),
                },
            )

        except Exception as e:
            logger.error(f"Error handling chat message: {e}", exc_info=True)
            emit("error", {"message": str(e)})

    @socketio.on("execute_flowchart")
    def handle_execute_flowchart(data):
        """Handle flowchart execution request."""
        from flask import request
        from theteam.services.flowchart_service import FlowchartService

        client_id = request.sid
        flowchart_id = data.get("flowchart_id")
        context = data.get("context", {})

        if not flowchart_id:
            emit("error", {"message": "Missing flowchart_id"})
            return

        try:
            flowchart_service = FlowchartService()
            execution_id = flowchart_service.start_execution(
                flowchart_id=flowchart_id,
                initial_context=context,
                client_id=client_id,
                socketio=socketio,
            )

            # Immediate acknowledgment
            emit(
                "execution_started",
                {
                    "execution_id": execution_id,
                    "flowchart_id": flowchart_id,
                    "timestamp": datetime.now().isoformat(),
                },
            )

        except Exception as e:
            logger.error(f"Error executing flowchart: {e}", exc_info=True)
            emit("error", {"message": str(e)})

    @socketio.on("stop_execution")
    def handle_stop_execution(data):
        """Handle request to stop a running execution."""
        from theteam.services.flowchart_service import FlowchartService

        execution_id = data.get("execution_id")
        if not execution_id:
            emit("error", {"message": "Missing execution_id"})
            return

        try:
            flowchart_service = FlowchartService()
            success = flowchart_service.stop_execution(execution_id)

            if success:
                emit(
                    "execution_stopped",
                    {
                        "execution_id": execution_id,
                        "timestamp": datetime.now().isoformat(),
                    },
                )
            else:
                emit("error", {"message": "Execution not found or already completed"})

        except Exception as e:
            logger.error(f"Error stopping execution: {e}", exc_info=True)
            emit("error", {"message": str(e)})

    @socketio.on("subscribe_execution")
    def handle_subscribe_execution(data):
        """Subscribe to execution updates."""
        from flask import request

        client_id = request.sid
        execution_id = data.get("execution_id")

        if not execution_id:
            emit("error", {"message": "Missing execution_id"})
            return

        room = f"execution_{execution_id}"
        join_room(room)
        if client_id in active_connections:
            active_connections[client_id]["rooms"].add(room)

        emit(
            "subscribed",
            {"execution_id": execution_id, "timestamp": datetime.now().isoformat()},
        )

    @socketio.on("error")
    def handle_error(error):
        """Handle errors from client."""
        logger.error(f"Client error: {error}")


def emit_to_client(socketio, client_id, event, data):
    """Emit event to a specific client."""
    try:
        socketio.emit(event, data, room=client_id)
    except Exception as e:
        logger.error(f"Error emitting to client {client_id}: {e}")


def emit_to_room(socketio, room, event, data):
    """Emit event to all clients in a room."""
    try:
        socketio.emit(event, data, room=room)
    except Exception as e:
        logger.error(f"Error emitting to room {room}: {e}")
