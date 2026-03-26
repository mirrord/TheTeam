"""
Tools API endpoints.
Handles tool discovery and listing for agent configuration.
"""

import logging
from flask import Blueprint, jsonify
from theteam.api import API_PREFIX
from pithos.tools import ToolRegistry
from pithos.config_manager import ConfigManager

logger = logging.getLogger(__name__)
bp = Blueprint("tools", __name__, url_prefix=f"{API_PREFIX}/tools")

# Global tool registry instance
_tool_registry = None


def get_tool_registry():
    """Get or create the global tool registry."""
    global _tool_registry
    if _tool_registry is None:
        config_manager = ConfigManager()
        _tool_registry = ToolRegistry(config_manager)
    return _tool_registry


@bp.route("/", methods=["GET"])
def list_tools():
    """List all available CLI tools."""
    try:
        registry = get_tool_registry()
        tools = [tool.name for tool in registry.tools.values()]
        return jsonify({"tools": tools}), 200
    except Exception as e:
        logger.error(f"Error listing tools: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@bp.route("/<tool_name>", methods=["GET"])
def get_tool(tool_name):
    """Get information about a specific tool."""
    try:
        registry = get_tool_registry()
        tool = registry.tools.get(tool_name)
        if not tool:
            return jsonify({"error": "Tool not found"}), 404

        return (
            jsonify(
                {
                    "tool": {
                        "name": tool.name,
                        "path": tool.path,
                        "description": tool.description,
                        "platform": tool.platform,
                        "source": tool.source,
                    }
                }
            ),
            200,
        )
    except Exception as e:
        logger.error(f"Error getting tool {tool_name}: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500
