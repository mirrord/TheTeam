"""Eval/Benchmark API endpoints.

Exposes listing of eval configs, past run results, and benchmark lifecycle
(start/stop/status) under the ``/api/v1/eval`` prefix.
"""

import logging
from flask import Blueprint, request, jsonify
from theteam.api import API_PREFIX
from theteam.services.eval_service import EvalService

logger = logging.getLogger(__name__)
bp = Blueprint("eval", __name__, url_prefix=f"{API_PREFIX}/eval")

# Module-level service instance (mirrors chat.py / tools.py pattern).
eval_service = EvalService()


# ---------------------------------------------------------------------------
# Configs
# ---------------------------------------------------------------------------


@bp.route("/configs", methods=["GET"])
def list_configs():
    """List all available eval config files."""
    try:
        configs = eval_service.list_configs()
        return jsonify({"configs": configs}), 200
    except Exception as exc:
        logger.error("Error listing eval configs: %s", exc, exc_info=True)
        return jsonify({"error": str(exc)}), 500


@bp.route("/configs/<name>", methods=["GET"])
def get_config(name):
    """Return the full parsed YAML dict for a named config."""
    try:
        config = eval_service.get_config(name)
        if config is None:
            return jsonify({"error": "Config not found"}), 404
        return jsonify({"config": config}), 200
    except Exception as exc:
        logger.error("Error getting eval config %s: %s", name, exc, exc_info=True)
        return jsonify({"error": str(exc)}), 500


# ---------------------------------------------------------------------------
# Past runs
# ---------------------------------------------------------------------------


@bp.route("/runs", methods=["GET"])
def list_runs():
    """List metadata for all completed/past benchmark runs."""
    try:
        runs = eval_service.list_runs()
        return jsonify({"runs": runs}), 200
    except Exception as exc:
        logger.error("Error listing eval runs: %s", exc, exc_info=True)
        return jsonify({"error": str(exc)}), 500


@bp.route("/runs/detail", methods=["GET"])
def get_run_detail():
    """Load full report + case details for a past run.

    Query params:
        run_dir: absolute or relative path to the run directory.
    """
    run_dir = request.args.get("run_dir", "")
    if not run_dir:
        return jsonify({"error": "run_dir query parameter required"}), 400
    try:
        result = eval_service.get_run(run_dir)
        return jsonify(result), 200
    except FileNotFoundError as exc:
        return jsonify({"error": str(exc)}), 404
    except Exception as exc:
        logger.error("Error loading run detail %s: %s", run_dir, exc, exc_info=True)
        return jsonify({"error": str(exc)}), 500


# ---------------------------------------------------------------------------
# Live run lifecycle
# ---------------------------------------------------------------------------


@bp.route("/runs", methods=["POST"])
def start_run():
    """Start a new benchmark run.

    Expected JSON body::

        {
            "config": {...},          // full eval config dict (required)
            "options": {              // optional overrides
                "rounds": 3,
                "max_cases": 10,
                "dry_run": false,
                "output_dir": "results/my_run"
            },
            "client_id": "..."        // optional SocketIO client SID
        }

    Returns 202 with ``{"run_id": "..."}`` on success.
    """
    data = request.get_json(silent=True)
    if not data or "config" not in data:
        return jsonify({"error": "JSON body with 'config' field is required"}), 400

    config_data = data["config"]
    options = data.get("options") or {}
    client_id = data.get("client_id")

    try:
        # Retrieve the SocketIO extension from the current Flask app so that
        # the background thread can emit live progress events.
        from flask import current_app  # noqa: PLC0415

        socketio = current_app.extensions.get("socketio")

        run_id = eval_service.start_run(config_data, options, client_id, socketio)
        return jsonify({"run_id": run_id, "status": "starting"}), 202
    except Exception as exc:
        logger.error("Error starting benchmark run: %s", exc, exc_info=True)
        return jsonify({"error": str(exc)}), 500


@bp.route("/runs/active/<run_id>", methods=["GET"])
def get_active_run(run_id):
    """Return current status of an in-progress or recently finished run."""
    try:
        status = eval_service.get_run_status(run_id)
        if status is None:
            return jsonify({"error": "Run not found"}), 404
        return jsonify({"run": status}), 200
    except Exception as exc:
        logger.error("Error getting run status %s: %s", run_id, exc, exc_info=True)
        return jsonify({"error": str(exc)}), 500


@bp.route("/runs/active/<run_id>", methods=["DELETE"])
def stop_run(run_id):
    """Send a stop signal to a running benchmark."""
    try:
        success = eval_service.stop_run(run_id)
        if not success:
            return jsonify({"error": "Run not found"}), 404
        return jsonify({"message": "Stop signal sent", "run_id": run_id}), 200
    except Exception as exc:
        logger.error("Error stopping run %s: %s", run_id, exc, exc_info=True)
        return jsonify({"error": str(exc)}), 500
