"""
Agent management API endpoints.
Handles agent listing, configuration, creation, and deletion.
"""

import logging
from flask import Blueprint, request, jsonify
from theteam.api import API_PREFIX
from theteam.services.agent_service import AgentService

logger = logging.getLogger(__name__)
bp = Blueprint("agents", __name__, url_prefix=f"{API_PREFIX}/agents")
agent_service = AgentService()


@bp.route("/", methods=["GET"])
def list_agents():
    """List all available agents."""
    try:
        agents = agent_service.list_agents()
        return jsonify({"agents": agents}), 200
    except Exception as e:
        logger.error(f"Error listing agents: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@bp.route("/<agent_id>", methods=["GET"])
def get_agent(agent_id):
    """Get a specific agent's configuration."""
    try:
        agent = agent_service.get_agent(agent_id)
        if not agent:
            return jsonify({"error": "Agent not found"}), 404
        return jsonify({"agent": agent}), 200
    except Exception as e:
        logger.error(f"Error getting agent {agent_id}: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@bp.route("/", methods=["POST"])
def create_agent():
    """Create a new agent."""
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided"}), 400

        agent_id = agent_service.create_agent(data)
        return (
            jsonify({"agent_id": agent_id, "message": "Agent created successfully"}),
            201,
        )
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        logger.error(f"Error creating agent: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@bp.route("/<agent_id>", methods=["PUT"])
def update_agent(agent_id):
    """Update an existing agent."""
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided"}), 400

        success = agent_service.update_agent(agent_id, data)
        if not success:
            return jsonify({"error": "Agent not found"}), 404

        return jsonify({"message": "Agent updated successfully"}), 200
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        logger.error(f"Error updating agent {agent_id}: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@bp.route("/<agent_id>", methods=["DELETE"])
def delete_agent(agent_id):
    """Delete an agent."""
    try:
        success = agent_service.delete_agent(agent_id)
        if not success:
            return jsonify({"error": "Agent not found"}), 404

        return jsonify({"message": "Agent deleted successfully"}), 200
    except Exception as e:
        logger.error(f"Error deleting agent {agent_id}: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@bp.route("/<agent_id>/test", methods=["POST"])
def test_agent(agent_id):
    """Test an agent with a sample prompt."""
    try:
        data = request.get_json()
        prompt = data.get("prompt") if data else None
        if not prompt:
            return jsonify({"error": "No prompt provided"}), 400

        result = agent_service.test_agent(agent_id, prompt)
        return jsonify({"result": result}), 200
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        logger.error(f"Error testing agent {agent_id}: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500
