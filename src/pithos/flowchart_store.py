"""Persistent flowchart storage with metadata and semantic search.

Stores flowchart configurations in SQLite with support for tags, notes, and
semantic search via ChromaDB vector embeddings.
"""

import json
import os
import sqlite3
import hashlib
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any, Optional
import yaml

try:
    import chromadb
    from chromadb.config import Settings as ChromaSettings

    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False
    chromadb = None
    ChromaSettings = None

_CHROMA_COLLECTION = "flowcharts"

_SCHEMA_SQL = """
PRAGMA foreign_keys = ON;
PRAGMA journal_mode = WAL;

CREATE TABLE IF NOT EXISTS flowcharts (
    id           TEXT PRIMARY KEY,
    name         TEXT NOT NULL,
    description  TEXT,
    config       TEXT NOT NULL,  -- JSON-serialized flowchart config
    notes        TEXT,           -- Freetext notes
    created_at   TEXT NOT NULL,
    updated_at   TEXT NOT NULL,
    source       TEXT DEFAULT 'database'
);

CREATE INDEX IF NOT EXISTS idx_flowcharts_name ON flowcharts(name);
CREATE INDEX IF NOT EXISTS idx_flowcharts_updated ON flowcharts(updated_at);

CREATE TABLE IF NOT EXISTS flowchart_tags (
    flowchart_id TEXT NOT NULL,
    tag          TEXT NOT NULL,
    UNIQUE(flowchart_id, tag),
    FOREIGN KEY(flowchart_id) REFERENCES flowcharts(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_flowchart_tags_tag ON flowchart_tags(tag);
CREATE INDEX IF NOT EXISTS idx_flowchart_tags_flowchart ON flowchart_tags(flowchart_id);

CREATE VIRTUAL TABLE IF NOT EXISTS flowcharts_fts
    USING fts5(id UNINDEXED, name, description, notes);
"""


@dataclass
class FlowchartRecord:
    """A stored flowchart with metadata."""

    id: str
    name: str
    description: str
    config: dict[str, Any]
    notes: str
    created_at: str
    updated_at: str
    source: str = "database"
    tags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class FlowchartSearchResult:
    """A search result from the flowchart store."""

    flowchart: FlowchartRecord
    relevance_score: float  # 0.0–1.0, higher is better
    match_type: str  # 'semantic', 'text', or 'exact'


class FlowchartStore:
    """Persistent store for flowchart configurations with metadata."""

    def __init__(self, persist_directory: str = "./data/flowcharts") -> None:
        """Initialize flowchart store.

        Args:
            persist_directory: Directory for the SQLite database and, when
                ChromaDB is available, the vector index.  Created automatically
                if it does not exist.
        """
        os.makedirs(persist_directory, exist_ok=True)
        self._db_path = os.path.join(persist_directory, "flowcharts.db")
        self._conn = self._init_db()

        # Optional ChromaDB for semantic search
        self._chroma_client: Any = None
        self._chroma_collection: Any = None
        if CHROMADB_AVAILABLE and chromadb is not None and ChromaSettings is not None:
            try:
                client = chromadb.PersistentClient(
                    path=persist_directory,
                    settings=ChromaSettings(
                        anonymized_telemetry=False, allow_reset=True
                    ),
                )
                self._chroma_client = client
                self._chroma_collection = client.get_or_create_collection(
                    name=_CHROMA_COLLECTION
                )
            except Exception:
                # Non-fatal: fall back to text search
                self._chroma_client = None
                self._chroma_collection = None

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _init_db(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.executescript(_SCHEMA_SQL)
        conn.commit()
        return conn

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _generate_id(name: str) -> str:
        """Generate a unique ID for a flowchart."""
        ts = datetime.now(timezone.utc).isoformat()
        raw = f"{name}{ts}"
        digest = hashlib.sha256(raw.encode()).hexdigest()[:16]
        return f"fc_{digest}"

    def _fetch_tags(self, flowchart_ids: list[str]) -> dict[str, list[str]]:
        """Fetch tags for multiple flowcharts."""
        if not flowchart_ids:
            return {}
        placeholders = ",".join("?" * len(flowchart_ids))
        rows = self._conn.execute(
            f"SELECT flowchart_id, tag FROM flowchart_tags WHERE flowchart_id IN ({placeholders})",
            flowchart_ids,
        ).fetchall()
        result: dict[str, list[str]] = {fid: [] for fid in flowchart_ids}
        for row in rows:
            result[row["flowchart_id"]].append(row["tag"])
        return result

    def _rows_to_records(self, rows: list) -> list[FlowchartRecord]:
        """Convert SQLite rows to FlowchartRecord objects."""
        if not rows:
            return []
        ids = [r["id"] for r in rows]
        tags_by_id = self._fetch_tags(ids)
        return [
            FlowchartRecord(
                id=r["id"],
                name=r["name"],
                description=r["description"] or "",
                config=json.loads(r["config"]),
                notes=r["notes"] or "",
                created_at=r["created_at"],
                updated_at=r["updated_at"],
                source=r["source"],
                tags=tags_by_id.get(r["id"], []),
            )
            for r in rows
        ]

    @staticmethod
    def _fts5_escape(query: str) -> str:
        """Escape a plain-text string for use as an FTS5 phrase query."""
        escaped = query.replace('"', '""')
        return f'"{escaped}"'

    def _filter_by_tags(
        self, records: list[FlowchartRecord], tags: list[str]
    ) -> list[FlowchartRecord]:
        """Filter records to those having at least one of the specified tags."""
        tag_set = set(tags)
        return [r for r in records if tag_set.intersection(set(r.tags))]

    # ------------------------------------------------------------------
    # Public write API
    # ------------------------------------------------------------------

    def store_flowchart(
        self,
        name: str,
        config: dict[str, Any],
        description: str = "",
        notes: str = "",
        tags: Optional[list[str]] = None,
        flowchart_id: Optional[str] = None,
    ) -> str:
        """Store a flowchart configuration.

        Args:
            name: Human-readable flowchart name.
            config: Flowchart configuration dictionary.
            description: Optional description.
            notes: Optional freetext notes.
            tags: Optional list of tags.
            flowchart_id: Optional ID to use (if None, generates new ID).

        Returns:
            The flowchart ID.
        """
        if flowchart_id is None:
            flowchart_id = self._generate_id(name)

        timestamp = datetime.now(timezone.utc).isoformat()
        config_json = json.dumps(config, indent=2)

        # Check if flowchart exists
        existing = self._conn.execute(
            "SELECT id FROM flowcharts WHERE id = ?", (flowchart_id,)
        ).fetchone()

        if existing:
            # Update existing
            self._conn.execute(
                "UPDATE flowcharts SET name=?, description=?, config=?, notes=?, updated_at=? WHERE id=?",
                (name, description, config_json, notes, timestamp, flowchart_id),
            )
        else:
            # Insert new
            self._conn.execute(
                "INSERT INTO flowcharts (id, name, description, config, notes, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    flowchart_id,
                    name,
                    description,
                    config_json,
                    notes,
                    timestamp,
                    timestamp,
                ),
            )

        # Update FTS index
        self._conn.execute("DELETE FROM flowcharts_fts WHERE id = ?", (flowchart_id,))
        self._conn.execute(
            "INSERT INTO flowcharts_fts (id, name, description, notes) VALUES (?, ?, ?, ?)",
            (flowchart_id, name, description or "", notes or ""),
        )

        self._conn.commit()

        # Add tags
        if tags:
            self.add_tags(flowchart_id, tags)

        # Index for semantic search
        if self._chroma_collection is not None:
            # Create searchable text from flowchart metadata
            searchable_text = f"{name}\n{description}\n{notes}"
            try:
                # Check if already exists in vector DB
                try:
                    self._chroma_collection.get(ids=[flowchart_id])
                    # Update existing
                    self._chroma_collection.update(
                        ids=[flowchart_id],
                        documents=[searchable_text],
                        metadatas=[
                            {
                                "name": name,
                                "description": description,
                                "updated_at": timestamp,
                            }
                        ],
                    )
                except Exception:
                    # Add new
                    self._chroma_collection.add(
                        ids=[flowchart_id],
                        documents=[searchable_text],
                        metadatas=[
                            {
                                "name": name,
                                "description": description,
                                "updated_at": timestamp,
                            }
                        ],
                    )
            except Exception:
                pass  # Vector index failure is non-fatal

        return flowchart_id

    def add_tags(self, flowchart_id: str, tags: list[str]) -> None:
        """Add tags to a flowchart.

        Args:
            flowchart_id: Flowchart ID.
            tags: List of tag strings to add.
        """
        rows = [(flowchart_id, tag.strip()) for tag in tags if tag.strip()]
        if rows:
            self._conn.executemany(
                "INSERT OR IGNORE INTO flowchart_tags (flowchart_id, tag) VALUES (?, ?)",
                rows,
            )
            self._conn.commit()

    def remove_tags(self, flowchart_id: str, tags: list[str]) -> None:
        """Remove tags from a flowchart.

        Args:
            flowchart_id: Flowchart ID.
            tags: List of tag strings to remove.
        """
        if not tags:
            return
        placeholders = ",".join("?" * len(tags))
        self._conn.execute(
            f"DELETE FROM flowchart_tags WHERE flowchart_id = ? AND tag IN ({placeholders})",
            [flowchart_id] + tags,
        )
        self._conn.commit()

    def update_notes(self, flowchart_id: str, notes: str) -> bool:
        """Update the notes field for a flowchart.

        Args:
            flowchart_id: Flowchart ID.
            notes: New notes text.

        Returns:
            True if updated, False if flowchart not found.
        """
        timestamp = datetime.now(timezone.utc).isoformat()
        cursor = self._conn.execute(
            "UPDATE flowcharts SET notes=?, updated_at=? WHERE id=?",
            (notes, timestamp, flowchart_id),
        )
        self._conn.commit()

        if cursor.rowcount > 0:
            # Update FTS index
            row = self._conn.execute(
                "SELECT name, description FROM flowcharts WHERE id=?", (flowchart_id,)
            ).fetchone()
            if row:
                self._conn.execute(
                    "UPDATE flowcharts_fts SET notes=? WHERE id=?",
                    (notes, flowchart_id),
                )
                self._conn.commit()
            return True
        return False

    def delete_flowchart(self, flowchart_id: str) -> bool:
        """Delete a flowchart.

        Args:
            flowchart_id: Flowchart ID.

        Returns:
            True if deleted, False if not found.
        """
        cursor = self._conn.execute(
            "DELETE FROM flowcharts WHERE id=?", (flowchart_id,)
        )
        self._conn.execute("DELETE FROM flowcharts_fts WHERE id=?", (flowchart_id,))
        self._conn.commit()

        # Remove from vector index
        if self._chroma_collection is not None:
            try:
                self._chroma_collection.delete(ids=[flowchart_id])
            except Exception:
                pass

        return cursor.rowcount > 0

    # ------------------------------------------------------------------
    # Public read / search API
    # ------------------------------------------------------------------

    def get_flowchart(self, flowchart_id: str) -> Optional[FlowchartRecord]:
        """Retrieve a flowchart by ID.

        Args:
            flowchart_id: Flowchart ID.

        Returns:
            FlowchartRecord if found, None otherwise.
        """
        row = self._conn.execute(
            "SELECT * FROM flowcharts WHERE id=?", (flowchart_id,)
        ).fetchone()
        if row:
            records = self._rows_to_records([row])
            return records[0] if records else None
        return None

    def list_flowcharts(
        self, tags: Optional[list[str]] = None, limit: Optional[int] = None
    ) -> list[FlowchartRecord]:
        """List all flowcharts.

        Args:
            tags: Optional list of tags to filter by.
            limit: Maximum number of results.

        Returns:
            List of FlowchartRecord objects ordered by updated_at descending.
        """
        query = "SELECT * FROM flowcharts ORDER BY updated_at DESC"
        if limit:
            query += f" LIMIT {limit}"

        rows = self._conn.execute(query).fetchall()
        records = self._rows_to_records(list(rows))

        if tags:
            records = self._filter_by_tags(records, tags)

        return records

    def search_text(
        self,
        query: str,
        tags: Optional[list[str]] = None,
        limit: int = 10,
    ) -> list[FlowchartSearchResult]:
        """Full-text search using SQLite FTS5.

        Args:
            query: Search query.
            tags: Optional list of tags to filter by.
            limit: Maximum number of results.

        Returns:
            List of FlowchartSearchResult objects.
        """
        fts_query = self._fts5_escape(query)

        try:
            fts_rows = self._conn.execute(
                "SELECT id FROM flowcharts_fts WHERE flowcharts_fts MATCH ? "
                "ORDER BY rank LIMIT ?",
                (fts_query, limit * 2),
            ).fetchall()
        except sqlite3.OperationalError:
            # Malformed FTS query — fall back to LIKE
            like_pattern = f"%{query}%"
            fts_rows = self._conn.execute(
                "SELECT id FROM flowcharts WHERE name LIKE ? OR description LIKE ? OR notes LIKE ? LIMIT ?",
                (like_pattern, like_pattern, like_pattern, limit * 2),
            ).fetchall()

        fts_ids = [r["id"] for r in fts_rows]
        if not fts_ids:
            return []

        placeholders = ",".join("?" * len(fts_ids))
        rows = self._conn.execute(
            f"SELECT * FROM flowcharts WHERE id IN ({placeholders})", fts_ids
        ).fetchall()

        records = self._rows_to_records(list(rows))

        if tags:
            records = self._filter_by_tags(records, tags)

        # Preserve FTS rank order
        id_order = {fid: idx for idx, fid in enumerate(fts_ids)}
        records.sort(key=lambda r: id_order.get(r.id, len(fts_ids)))

        return [
            FlowchartSearchResult(flowchart=r, relevance_score=1.0, match_type="text")
            for r in records[:limit]
        ]

    def search_semantic(
        self,
        query: str,
        tags: Optional[list[str]] = None,
        limit: int = 10,
    ) -> list[FlowchartSearchResult]:
        """Semantic search using ChromaDB vector embeddings.

        Falls back to text search when ChromaDB is not available.

        Args:
            query: Natural-language search query.
            tags: Optional list of tags to filter by.
            limit: Maximum number of results.

        Returns:
            List of FlowchartSearchResult objects ordered by relevance.
        """
        if self._chroma_collection is None:
            return self.search_text(query, tags, limit)

        try:
            result = self._chroma_collection.query(
                query_texts=[query],
                n_results=max(limit * 2, 20),
            )
        except Exception:
            return self.search_text(query, tags, limit)

        if not result or not result.get("ids") or not result["ids"][0]:
            return []

        fc_ids: list[str] = result["ids"][0]
        distances: list[float] = (
            result["distances"][0] if result.get("distances") else [0.0] * len(fc_ids)
        )
        id_to_distance = dict(zip(fc_ids, distances))

        # Fetch full records
        placeholders = ",".join("?" * len(fc_ids))
        rows = self._conn.execute(
            f"SELECT * FROM flowcharts WHERE id IN ({placeholders})", fc_ids
        ).fetchall()

        records = self._rows_to_records(list(rows))

        if tags:
            records = self._filter_by_tags(records, tags)

        results: list[FlowchartSearchResult] = []
        import math

        for record in records[:limit]:
            distance = id_to_distance.get(record.id, 0.0)
            relevance = math.exp(-distance)
            results.append(
                FlowchartSearchResult(
                    flowchart=record,
                    relevance_score=relevance,
                    match_type="semantic",
                )
            )

        # Sort by descending relevance
        results.sort(key=lambda r: r.relevance_score, reverse=True)
        return results

    def search(
        self,
        query: str,
        tags: Optional[list[str]] = None,
        limit: int = 10,
        semantic: bool = True,
    ) -> list[FlowchartSearchResult]:
        """Search flowcharts with automatic mode selection.

        Args:
            query: Search query.
            tags: Optional list of tags to filter by.
            limit: Maximum number of results.
            semantic: Use semantic search if available.

        Returns:
            List of FlowchartSearchResult objects.
        """
        if semantic and self._chroma_collection is not None:
            return self.search_semantic(query, tags, limit)
        return self.search_text(query, tags, limit)

    def search_exact(
        self,
        text: str,
        tags: Optional[list[str]] = None,
    ) -> list[FlowchartRecord]:
        """Search for exact text matches in flowchart metadata or config.

        Args:
            text: Exact text to search for (case-insensitive).
            tags: Optional list of tags to filter by.

        Returns:
            List of matching FlowchartRecord objects.
        """
        like_pattern = f"%{text}%"
        rows = self._conn.execute(
            "SELECT * FROM flowcharts WHERE "
            "name LIKE ? OR description LIKE ? OR notes LIKE ? OR config LIKE ? "
            "ORDER BY updated_at DESC",
            (like_pattern, like_pattern, like_pattern, like_pattern),
        ).fetchall()

        records = self._rows_to_records(list(rows))

        if tags:
            records = self._filter_by_tags(records, tags)

        return records

    def list_tags(self) -> list[tuple[str, int]]:
        """List all tags with usage counts.

        Returns:
            List of (tag, count) tuples ordered by count descending.
        """
        rows = self._conn.execute(
            "SELECT tag, COUNT(*) as count FROM flowchart_tags "
            "GROUP BY tag ORDER BY count DESC"
        ).fetchall()
        return [(r["tag"], r["count"]) for r in rows]

    def export_flowchart(self, flowchart_id: str, output_path: str) -> bool:
        """Export a flowchart to YAML file.

        Args:
            flowchart_id: Flowchart ID.
            output_path: Path to save the YAML file.

        Returns:
            True if exported, False if flowchart not found.
        """
        flowchart = self.get_flowchart(flowchart_id)
        if not flowchart:
            return False

        # Prepare export data
        export_data = flowchart.config.copy()
        export_data["name"] = flowchart.name
        export_data["description"] = flowchart.description
        if flowchart.notes:
            export_data["notes"] = flowchart.notes
        if flowchart.tags:
            export_data["tags"] = flowchart.tags

        with open(output_path, "w", encoding="utf-8") as f:
            yaml.dump(export_data, f, default_flow_style=False, sort_keys=False)

        return True

    def import_flowchart(
        self, input_path: str, flowchart_id: Optional[str] = None
    ) -> str:
        """Import a flowchart from YAML file.

        Args:
            input_path: Path to the YAML file.
            flowchart_id: Optional ID to use (if None, generates new ID).

        Returns:
            The flowchart ID.
        """
        with open(input_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        # Extract metadata
        name = data.pop("name", os.path.basename(input_path).replace(".yaml", ""))
        description = data.pop("description", "")
        notes = data.pop("notes", "")
        tags = data.pop("tags", [])

        # Remaining data is the config
        config = data

        return self.store_flowchart(
            name=name,
            config=config,
            description=description,
            notes=notes,
            tags=tags,
            flowchart_id=flowchart_id,
        )

    @property
    def semantic_search_available(self) -> bool:
        """True when ChromaDB is available and the vector index is ready."""
        return self._chroma_collection is not None

    def clear_all(self) -> None:
        """Clear all flowchart data. Use with caution!"""
        self._conn.execute("DELETE FROM flowchart_tags")
        self._conn.execute("DELETE FROM flowcharts_fts")
        self._conn.execute("DELETE FROM flowcharts")
        self._conn.commit()

        # Clear vector index if available
        if self._chroma_client is not None:
            try:
                self._chroma_client.delete_collection(name=_CHROMA_COLLECTION)
                # Recreate empty collection
                self._chroma_collection = self._chroma_client.get_or_create_collection(
                    name=_CHROMA_COLLECTION
                )
            except Exception:
                pass

    def close(self) -> None:
        """Close all database connections."""
        import gc

        self._conn.close()
        if self._chroma_client is not None:
            try:
                system = getattr(self._chroma_client, "_system", None)
                if system is not None and hasattr(system, "stop"):
                    system.stop()
            except Exception:
                pass
            try:
                self._chroma_client.clear_system_cache()
            except Exception:
                pass
            self._chroma_collection = None
            self._chroma_client = None
            gc.collect()


def main():
    """CLI for flowchart database management."""
    import argparse

    parser = argparse.ArgumentParser(
        description="pithos Flowchart Store - Flowchart database management"
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # Store command
    store_parser = subparsers.add_parser("store", help="Store a flowchart")
    store_parser.add_argument("name", help="Flowchart name")
    store_parser.add_argument("yaml_file", help="Path to YAML config file")
    store_parser.add_argument("--description", default="", help="Description")
    store_parser.add_argument("--notes", default="", help="Freetext notes")
    store_parser.add_argument("--tags", help="Comma-separated tags")

    # Import command
    import_parser = subparsers.add_parser("import", help="Import flowchart from YAML")
    import_parser.add_argument("yaml_file", help="Path to YAML file")
    import_parser.add_argument(
        "--id", dest="flowchart_id", help="Optional flowchart ID"
    )

    # Export command
    export_parser = subparsers.add_parser("export", help="Export flowchart to YAML")
    export_parser.add_argument("flowchart_id", help="Flowchart ID")
    export_parser.add_argument("output_file", help="Output YAML file path")

    # Get command
    get_parser = subparsers.add_parser("get", help="Get a flowchart")
    get_parser.add_argument("flowchart_id", help="Flowchart ID")

    # List command
    list_parser = subparsers.add_parser("list", help="List all flowcharts")
    list_parser.add_argument("--tags", help="Filter by comma-separated tags")
    list_parser.add_argument("--limit", type=int, help="Maximum number of results")

    # Search command
    search_parser = subparsers.add_parser("search", help="Search flowcharts")
    search_parser.add_argument("query", help="Search query")
    search_parser.add_argument("--exact", action="store_true", help="Exact text match")
    search_parser.add_argument("--tags", help="Filter by comma-separated tags")
    search_parser.add_argument("--limit", type=int, default=10, help="Maximum results")

    # Tags command
    subparsers.add_parser("tags", help="List all tags")

    # Add tags command
    add_tags_parser = subparsers.add_parser("add-tags", help="Add tags to flowchart")
    add_tags_parser.add_argument("flowchart_id", help="Flowchart ID")
    add_tags_parser.add_argument("tags", help="Comma-separated tags to add")

    # Update notes command
    notes_parser = subparsers.add_parser("notes", help="Update flowchart notes")
    notes_parser.add_argument("flowchart_id", help="Flowchart ID")
    notes_parser.add_argument("notes", help="New notes text")

    # Delete command
    delete_parser = subparsers.add_parser("delete", help="Delete a flowchart")
    delete_parser.add_argument("flowchart_id", help="Flowchart ID")

    # Clear command
    clear_parser = subparsers.add_parser("clear", help="Clear all flowcharts")
    clear_parser.add_argument(
        "--confirm", action="store_true", help="Confirm clear operation"
    )

    args = parser.parse_args()

    # Initialize store
    store = FlowchartStore()

    try:
        if args.command == "store":
            with open(args.yaml_file, "r") as f:
                config = yaml.safe_load(f)
            tags = [t.strip() for t in args.tags.split(",")] if args.tags else []
            flowchart_id = store.store_flowchart(
                name=args.name,
                config=config,
                description=args.description,
                notes=args.notes,
                tags=tags,
            )
            print(f"Stored flowchart with ID: {flowchart_id}")

        elif args.command == "import":
            flowchart_id = store.import_flowchart(
                args.yaml_file, flowchart_id=args.flowchart_id
            )
            print(f"Imported flowchart with ID: {flowchart_id}")

        elif args.command == "export":
            success = store.export_flowchart(args.flowchart_id, args.output_file)
            if success:
                print(f"Exported flowchart to: {args.output_file}")
            else:
                print(f"Flowchart not found: {args.flowchart_id}")

        elif args.command == "get":
            flowchart = store.get_flowchart(args.flowchart_id)
            if flowchart:
                print(f"\nFlowchart: {flowchart.name}")
                print(f"Description: {flowchart.description}")
                print(f"Tags: {', '.join(flowchart.tags)}")
                print(f"Created: {flowchart.created_at}")
                print(f"Updated: {flowchart.updated_at}")
                if flowchart.notes:
                    print(f"\nNotes:\n{flowchart.notes}")
                print(f"\nConfig:\n{json.dumps(flowchart.config, indent=2)}")
            else:
                print(f"Flowchart not found: {args.flowchart_id}")

        elif args.command == "list":
            tags = [t.strip() for t in args.tags.split(",")] if args.tags else None
            flowcharts = store.list_flowcharts(tags=tags, limit=args.limit)
            print(f"\nFound {len(flowcharts)} flowchart(s):\n")
            for fc in flowcharts:
                print(f"  [{fc.id}] {fc.name}")
                if fc.description:
                    print(f"    {fc.description}")
                if fc.tags:
                    print(f"    Tags: {', '.join(fc.tags)}")
                print()

        elif args.command == "search":
            tags = [t.strip() for t in args.tags.split(",")] if args.tags else None
            if args.exact:
                results = store.search_exact(args.query, tags=tags)
                print(f"\nFound {len(results)} exact match(es):\n")
                for fc in results[: args.limit]:
                    print(f"  [{fc.id}] {fc.name}")
                    if fc.description:
                        print(f"    {fc.description}")
                    print()
            else:
                results = store.search(args.query, tags=tags, limit=args.limit)
                print(f"\nFound {len(results)} result(s):\n")
                for result in results:
                    fc = result.flowchart
                    print(
                        f"  [{fc.id}] {fc.name} (score: {result.relevance_score:.3f})"
                    )
                    if fc.description:
                        print(f"    {fc.description}")
                    print()

        elif args.command == "tags":
            tags = store.list_tags()
            print(f"\nAll tags ({len(tags)}):\n")
            for tag, count in tags:
                print(f"  {tag} ({count})")

        elif args.command == "add-tags":
            tags = [t.strip() for t in args.tags.split(",")]
            store.add_tags(args.flowchart_id, tags)
            print(f"Added tags to flowchart {args.flowchart_id}")

        elif args.command == "notes":
            success = store.update_notes(args.flowchart_id, args.notes)
            if success:
                print(f"Updated notes for flowchart {args.flowchart_id}")
            else:
                print(f"Flowchart not found: {args.flowchart_id}")

        elif args.command == "delete":
            success = store.delete_flowchart(args.flowchart_id)
            if success:
                print(f"Deleted flowchart {args.flowchart_id}")
            else:
                print(f"Flowchart not found: {args.flowchart_id}")

        elif args.command == "clear":
            if not args.confirm:
                print(
                    "WARNING: This will delete all flowcharts. Use --confirm to proceed."
                )
            else:
                store.clear_all()
                print("Cleared all flowcharts")

        else:
            parser.print_help()

    finally:
        store.close()


if __name__ == "__main__":
    main()
