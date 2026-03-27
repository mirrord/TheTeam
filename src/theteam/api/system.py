"""
System API endpoints.
Handles system-level operations like health checks and configuration.
"""

import logging
from flask import Blueprint, jsonify
from theteam.api import API_PREFIX

logger = logging.getLogger(__name__)
bp = Blueprint("system", __name__, url_prefix=f"{API_PREFIX}/system")


@bp.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint."""
    return jsonify({"status": "healthy", "service": "theteam", "version": "0.1.1"}), 200


@bp.route("/info", methods=["GET"])
def system_info():
    """Get system information."""
    try:
        import sys
        import platform

        return (
            jsonify(
                {
                    "python_version": sys.version,
                    "platform": platform.platform(),
                    "system": platform.system(),
                    "machine": platform.machine(),
                }
            ),
            200,
        )
    except Exception as e:
        logger.error(f"Error getting system info: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@bp.route("/models", methods=["GET"])
def list_models():
    """Get available Ollama models."""
    try:
        from pithos.utils import get_available_models

        models = get_available_models()
        return jsonify({"models": models}), 200
    except Exception as e:
        logger.error(f"Error getting models: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500
