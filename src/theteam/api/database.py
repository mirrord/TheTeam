"""
Database API endpoints.
Handles database management operations (clearing, searching, info).
"""

import logging
from flask import Blueprint, jsonify, request
from theteam.api import API_PREFIX
from pithos.database_manager import DatabaseManager

logger = logging.getLogger(__name__)
bp = Blueprint("database", __name__, url_prefix=f"{API_PREFIX}/database")

# Global database manager instance
_db_manager: DatabaseManager | None = None


def get_db_manager() -> DatabaseManager:
    """Get or create the global database manager."""
    global _db_manager
    if _db_manager is None:
        _db_manager = DatabaseManager()
    return _db_manager


@bp.route("/info", methods=["GET"])
def database_info():
    """Get information about all databases."""
    try:
        manager = get_db_manager()
        info_list = manager.get_database_info()
        return (
            jsonify(
                {
                    "databases": [
                        {
                            "name": info.name,
                            "type": info.type,
                            "path": info.path,
                            "size_bytes": info.size_bytes,
                            "available": info.available,
                            "error": info.error,
                        }
                        for info in info_list
                    ]
                }
            ),
            200,
        )
    except Exception as e:
        logger.error(f"Error getting database info: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@bp.route("/clear/<database>", methods=["POST"])
def clear_database(database: str):
    """Clear a specific database or all databases.

    Args:
        database: One of 'memory', 'history', 'flowcharts', or 'all'
    """
    try:
        data = request.get_json() or {}
        confirm = data.get("confirm", False)

        if not confirm:
            return (
                jsonify(
                    {"error": "Must set confirm=true in request body to clear database"}
                ),
                400,
            )

        manager = get_db_manager()

        if database == "all":
            results = manager.clear_all(confirm=True)
            return jsonify({"results": results}), 200
        elif database == "memory":
            manager.clear_memory()
            return jsonify({"message": "Memory store cleared"}), 200
        elif database == "history":
            manager.clear_history()
            return jsonify({"message": "Conversation history cleared"}), 200
        elif database == "flowcharts":
            manager.clear_flowcharts()
            return jsonify({"message": "Flowchart store cleared"}), 200
        else:
            return jsonify({"error": f"Invalid database: {database}"}), 400

    except Exception as e:
        logger.error(f"Error clearing database {database}: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@bp.route("/search", methods=["POST"])
def search_databases():
    """Search across all databases."""
    try:
        data = request.get_json()
        if not data or "query" not in data:
            return jsonify({"error": "Missing 'query' in request body"}), 400

        query = data["query"]
        exact = data.get("exact", False)
        semantic = data.get("semantic", True)
        limit = data.get("limit", 10)

        manager = get_db_manager()

        if exact:
            databases = data.get("databases")
            results = manager.search_exact(text=query, databases=databases)
        else:
            results = manager.search_all(query=query, semantic=semantic, limit=limit)
            # Convert UnifiedSearchResult objects to dicts
            serialized_results = {}
            for db, db_results in results.items():
                serialized_results[db] = [
                    {
                        "database": r.database,
                        "result_type": r.result_type,
                        "content": r.content,
                        "metadata": r.metadata,
                        "relevance_score": r.relevance_score,
                        "match_type": r.match_type,
                    }
                    for r in db_results
                ]
            results = serialized_results

        return jsonify({"results": results, "query": query, "exact": exact}), 200

    except Exception as e:
        logger.error(f"Error searching databases: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@bp.route("/memory/categories", methods=["GET"])
def list_memory_categories():
    """List all memory categories."""
    try:
        manager = get_db_manager()
        categories = manager.memory.list_categories()
        return jsonify({"categories": categories}), 200
    except Exception as e:
        logger.error(f"Error listing memory categories: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@bp.route("/memory/search", methods=["POST"])
def search_memory():
    """Search memory store across all categories."""
    try:
        data = request.get_json()
        if not data or "query" not in data:
            return jsonify({"error": "Missing 'query' in request body"}), 400

        query = data["query"]
        n_results = data.get("n_results", 10)
        min_relevance = data.get("min_relevance")
        categories = data.get("categories")

        manager = get_db_manager()
        results = manager.memory.search_all_categories(
            query=query,
            n_results=n_results,
            min_relevance=min_relevance,
            categories=categories,
        )

        # Serialize results
        serialized = {}
        for category, category_results in results.items():
            serialized[category] = [
                {
                    "id": r.id,
                    "category": r.category,
                    "content": r.content,
                    "metadata": r.metadata,
                    "distance": r.distance,
                    "relevance_score": r.relevance_score,
                }
                for r in category_results
            ]

        return jsonify({"results": serialized, "query": query}), 200

    except Exception as e:
        logger.error(f"Error searching memory: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@bp.route("/flowcharts", methods=["GET"])
def list_database_flowcharts():
    """List flowcharts from the database."""
    try:
        tags = request.args.get("tags")
        limit = request.args.get("limit", type=int)

        tag_list = [t.strip() for t in tags.split(",")] if tags else None

        manager = get_db_manager()
        flowcharts = manager.flowcharts.list_flowcharts(tags=tag_list, limit=limit)

        return (
            jsonify(
                {
                    "flowcharts": [
                        {
                            "id": fc.id,
                            "name": fc.name,
                            "description": fc.description,
                            "tags": fc.tags,
                            "notes": fc.notes,
                            "created_at": fc.created_at,
                            "updated_at": fc.updated_at,
                            "source": fc.source,
                        }
                        for fc in flowcharts
                    ]
                }
            ),
            200,
        )
    except Exception as e:
        logger.error(f"Error listing flowcharts: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@bp.route("/flowcharts/<flowchart_id>", methods=["GET"])
def get_database_flowchart(flowchart_id: str):
    """Get a flowchart from the database."""
    try:
        manager = get_db_manager()
        flowchart = manager.flowcharts.get_flowchart(flowchart_id)

        if not flowchart:
            return jsonify({"error": "Flowchart not found"}), 404

        return jsonify({"flowchart": flowchart.to_dict()}), 200

    except Exception as e:
        logger.error(f"Error getting flowchart {flowchart_id}: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@bp.route("/flowcharts", methods=["POST"])
def store_database_flowchart():
    """Store a flowchart in the database."""
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "Missing request body"}), 400

        name = data.get("name")
        config = data.get("config")

        if not name or not config:
            return jsonify({"error": "Missing 'name' or 'config' in request body"}), 400

        description = data.get("description", "")
        notes = data.get("notes", "")
        tags = data.get("tags", [])
        flowchart_id = data.get("id")

        manager = get_db_manager()
        flowchart_id = manager.flowcharts.store_flowchart(
            name=name,
            config=config,
            description=description,
            notes=notes,
            tags=tags,
            flowchart_id=flowchart_id,
        )

        return (
            jsonify({"flowchart_id": flowchart_id, "message": "Flowchart stored"}),
            201,
        )

    except Exception as e:
        logger.error(f"Error storing flowchart: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@bp.route("/flowcharts/<flowchart_id>", methods=["PUT"])
def update_database_flowchart_notes(flowchart_id: str):
    """Update flowchart notes."""
    try:
        data = request.get_json()
        if not data or "notes" not in data:
            return jsonify({"error": "Missing 'notes' in request body"}), 400

        manager = get_db_manager()
        success = manager.flowcharts.update_notes(flowchart_id, data["notes"])

        if not success:
            return jsonify({"error": "Flowchart not found"}), 404

        return jsonify({"message": "Notes updated"}), 200

    except Exception as e:
        logger.error(f"Error updating flowchart {flowchart_id}: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@bp.route("/flowcharts/<flowchart_id>/tags", methods=["POST"])
def add_database_flowchart_tags(flowchart_id: str):
    """Add tags to a flowchart."""
    try:
        data = request.get_json()
        if not data or "tags" not in data:
            return jsonify({"error": "Missing 'tags' in request body"}), 400

        manager = get_db_manager()
        manager.flowcharts.add_tags(flowchart_id, data["tags"])

        return jsonify({"message": "Tags added"}), 200

    except Exception as e:
        logger.error(
            f"Error adding tags to flowchart {flowchart_id}: {e}", exc_info=True
        )
        return jsonify({"error": str(e)}), 500


@bp.route("/flowcharts/<flowchart_id>", methods=["DELETE"])
def delete_database_flowchart(flowchart_id: str):
    """Delete a flowchart from the database."""
    try:
        manager = get_db_manager()
        success = manager.flowcharts.delete_flowchart(flowchart_id)

        if not success:
            return jsonify({"error": "Flowchart not found"}), 404

        return jsonify({"message": "Flowchart deleted"}), 200

    except Exception as e:
        logger.error(f"Error deleting flowchart {flowchart_id}: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@bp.route("/flowcharts/search", methods=["POST"])
def search_database_flowcharts():
    """Search flowcharts."""
    try:
        data = request.get_json()
        if not data or "query" not in data:
            return jsonify({"error": "Missing 'query' in request body"}), 400

        query = data["query"]
        tags = data.get("tags")
        limit = data.get("limit", 10)
        exact = data.get("exact", False)
        semantic = data.get("semantic", True)

        manager = get_db_manager()

        if exact:
            results = manager.flowcharts.search_exact(text=query, tags=tags)
            serialized = [fc.to_dict() for fc in results[:limit]]
        else:
            results = manager.flowcharts.search(
                query=query, tags=tags, limit=limit, semantic=semantic
            )
            serialized = [
                {
                    "flowchart": r.flowchart.to_dict(),
                    "relevance_score": r.relevance_score,
                    "match_type": r.match_type,
                }
                for r in results
            ]

        return jsonify({"results": serialized, "query": query}), 200

    except Exception as e:
        logger.error(f"Error searching flowcharts: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@bp.route("/flowcharts/tags", methods=["GET"])
def list_flowchart_tags():
    """List all flowchart tags."""
    try:
        manager = get_db_manager()
        tags = manager.flowcharts.list_tags()

        return (
            jsonify({"tags": [{"tag": tag, "count": count} for tag, count in tags]}),
            200,
        )
    except Exception as e:
        logger.error(f"Error listing flowchart tags: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500
