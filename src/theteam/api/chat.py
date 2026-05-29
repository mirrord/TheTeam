"""
Chat API endpoints.
Handles conversation management and message exchange.
"""

import logging
from flask import Blueprint, request, jsonify
from theteam.api import API_PREFIX
from theteam.services.chat_service import ChatService

logger = logging.getLogger(__name__)
bp = Blueprint("chat", __name__, url_prefix=f"{API_PREFIX}/chat")
chat_service = ChatService()


@bp.route("/conversations", methods=["GET"])
def list_conversations():
    """List all conversations."""
    try:
        conversations = chat_service.list_conversations()
        return jsonify({"conversations": conversations}), 200
    except Exception as e:
        logger.error(f"Error listing conversations: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@bp.route("/conversations/<conversation_id>", methods=["GET"])
def get_conversation(conversation_id):
    """Get a specific conversation with its messages."""
    try:
        conversation = chat_service.get_conversation(conversation_id)
        if not conversation:
            return jsonify({"error": "Conversation not found"}), 404
        return jsonify({"conversation": conversation}), 200
    except Exception as e:
        logger.error(
            f"Error getting conversation {conversation_id}: {e}", exc_info=True
        )
        return jsonify({"error": str(e)}), 500


@bp.route("/conversations", methods=["POST"])
def create_conversation():
    """Create a new conversation."""
    try:
        data = request.get_json() or {}
        conversation_id = chat_service.create_conversation(
            agent_id=data.get("agent_id"), title=data.get("title")
        )
        return (
            jsonify(
                {"conversation_id": conversation_id, "message": "Conversation created"}
            ),
            201,
        )
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        logger.error(f"Error creating conversation: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@bp.route("/conversations/<conversation_id>", methods=["DELETE"])
def delete_conversation(conversation_id):
    """Delete a conversation."""
    try:
        success = chat_service.delete_conversation(conversation_id)
        if not success:
            return jsonify({"error": "Conversation not found"}), 404
        return jsonify({"message": "Conversation deleted"}), 200
    except Exception as e:
        logger.error(
            f"Error deleting conversation {conversation_id}: {e}", exc_info=True
        )
        return jsonify({"error": str(e)}), 500


@bp.route("/conversations/<conversation_id>/messages", methods=["POST"])
def send_message(conversation_id):
    """Send a message in a conversation (async via SocketIO)."""
    try:
        data = request.get_json()
        if not data or "message" not in data:
            return jsonify({"error": "No message provided"}), 400

        message_id = chat_service.send_message(
            conversation_id=conversation_id,
            message=data["message"],
            client_id=data.get("client_id"),
        )

        return jsonify({"message_id": message_id, "message": "Message sent"}), 202
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        logger.error(f"Error sending message to {conversation_id}: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@bp.route("/conversations/<conversation_id>/agent", methods=["PUT"])
def update_conversation_agent(conversation_id):
    """Update the agent (or base model) for a conversation.

    Accepts one of:
    - ``{"agent_id": "<id>"}``  — switch to a configured agent.
    - ``{"base_model": "<model>"}`` — switch to base-model mode using
      the given Ollama model with no system prompt or tools.
    """
    try:
        data = request.get_json() or {}
        agent_id = data.get("agent_id")
        base_model = data.get("base_model")

        if base_model and not agent_id:
            success = chat_service.update_conversation_base_model(
                conversation_id, base_model
            )
            if not success:
                return jsonify({"error": "Conversation not found"}), 404
            return jsonify({"message": "Base model updated"}), 200

        if not agent_id:
            return jsonify({"error": "No agent_id or base_model provided"}), 400

        success = chat_service.update_conversation_agent(conversation_id, agent_id)
        if not success:
            return jsonify({"error": "Conversation not found"}), 404

        return jsonify({"message": "Agent updated"}), 200
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        logger.error(f"Error updating agent for {conversation_id}: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@bp.route("/conversations/<conversation_id>/tools", methods=["PUT"])
def update_conversation_tools(conversation_id):
    """Set the per-conversation enabled-tools allow-list.

    Body: ``{"enabled_tools": ["git", "python"]}`` to allow-list,
    ``{"enabled_tools": []}`` to disable all tools, or
    ``{"enabled_tools": null}`` to clear the override.
    """
    try:
        data = request.get_json() or {}
        if "enabled_tools" not in data:
            return jsonify({"error": "No enabled_tools provided"}), 400
        tools = data["enabled_tools"]
        if tools is not None and not isinstance(tools, list):
            return jsonify({"error": "enabled_tools must be a list or null"}), 400

        success = chat_service.update_conversation_tools(conversation_id, tools)
        if not success:
            return jsonify({"error": "Conversation not found"}), 404
        return jsonify({"message": "Tools updated"}), 200
    except Exception as e:
        logger.error(f"Error updating tools for {conversation_id}: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500
