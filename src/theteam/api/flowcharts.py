"""
Flowchart management API endpoints.
Handles flowchart CRUD operations, import/export, and validation.
"""

import logging
from flask import Blueprint, request, jsonify
from theteam.api import API_PREFIX
from theteam.services.flowchart_service import FlowchartService

logger = logging.getLogger(__name__)
bp = Blueprint("flowcharts", __name__, url_prefix=f"{API_PREFIX}/flowcharts")
flowchart_service = FlowchartService()


@bp.route("/", methods=["GET"])
def list_flowcharts():
    """List all available flowcharts."""
    try:
        flowcharts = flowchart_service.list_flowcharts()
        return jsonify({"flowcharts": flowcharts}), 200
    except Exception as e:
        logger.error(f"Error listing flowcharts: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@bp.route("/<flowchart_id>", methods=["GET"])
def get_flowchart(flowchart_id):
    """Get a specific flowchart."""
    try:
        flowchart = flowchart_service.get_flowchart(flowchart_id)
        if not flowchart:
            return jsonify({"error": "Flowchart not found"}), 404
        return jsonify({"flowchart": flowchart}), 200
    except Exception as e:
        logger.error(f"Error getting flowchart {flowchart_id}: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@bp.route("/", methods=["POST"])
def create_flowchart():
    """Create a new flowchart."""
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided"}), 400

        flowchart_id = flowchart_service.create_flowchart(data)
        return (
            jsonify(
                {
                    "flowchart_id": flowchart_id,
                    "message": "Flowchart created successfully",
                }
            ),
            201,
        )
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        logger.error(f"Error creating flowchart: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@bp.route("/<flowchart_id>", methods=["PUT"])
def update_flowchart(flowchart_id):
    """Update an existing flowchart."""
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided"}), 400

        success = flowchart_service.update_flowchart(flowchart_id, data)
        if not success:
            return jsonify({"error": "Flowchart not found"}), 404

        return jsonify({"message": "Flowchart updated successfully"}), 200
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        logger.error(f"Error updating flowchart {flowchart_id}: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@bp.route("/<flowchart_id>", methods=["DELETE"])
def delete_flowchart(flowchart_id):
    """Delete a flowchart."""
    try:
        success = flowchart_service.delete_flowchart(flowchart_id)
        if not success:
            return jsonify({"error": "Flowchart not found"}), 404

        return jsonify({"message": "Flowchart deleted successfully"}), 200
    except Exception as e:
        logger.error(f"Error deleting flowchart {flowchart_id}: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@bp.route("/import", methods=["POST"])
def import_flowchart():
    """Import a flowchart from YAML."""
    try:
        data = request.get_json()
        if not data or "yaml" not in data:
            return jsonify({"error": "No YAML data provided"}), 400

        flowchart_id = flowchart_service.import_from_yaml(
            data["yaml"], data.get("name")
        )
        return (
            jsonify(
                {
                    "flowchart_id": flowchart_id,
                    "message": "Flowchart imported successfully",
                }
            ),
            201,
        )
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        logger.error(f"Error importing flowchart: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@bp.route("/<flowchart_id>/export", methods=["GET"])
def export_flowchart(flowchart_id):
    """Export a flowchart to YAML."""
    try:
        yaml_data = flowchart_service.export_to_yaml(flowchart_id)
        if not yaml_data:
            return jsonify({"error": "Flowchart not found"}), 404

        return jsonify({"yaml": yaml_data}), 200
    except Exception as e:
        logger.error(f"Error exporting flowchart {flowchart_id}: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@bp.route("/<flowchart_id>/validate", methods=["POST"])
def validate_flowchart(flowchart_id):
    """Validate a flowchart structure."""
    try:
        validation_result = flowchart_service.validate_flowchart(flowchart_id)
        return jsonify(validation_result), 200
    except Exception as e:
        logger.error(f"Error validating flowchart {flowchart_id}: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@bp.route("/<flowchart_id>/execute", methods=["POST"])
def execute_flowchart(flowchart_id):
    """Execute a flowchart (async via SocketIO)."""
    try:
        data = request.get_json() or {}
        execution_id = flowchart_service.start_execution(
            flowchart_id,
            initial_context=data.get("context", {}),
            client_id=data.get("client_id"),
        )
        return (
            jsonify({"execution_id": execution_id, "message": "Execution started"}),
            202,
        )
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        logger.error(f"Error executing flowchart {flowchart_id}: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500
