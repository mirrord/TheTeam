"""Persistent conversation history storage with text and semantic (RAG) search.

Provides ``ConversationStore``, a lightweight store that persists every message
exchanged between an agent and users so that conversations can later be
retrieved by:

* **Full-text search** — backed by SQLite FTS5 (always available).
* **Semantic / RAG search** — backed by ChromaDB vector embeddings (requires
  ``pip install chromadb``; gracefully degrades to text search when absent).

Messages can be annotated with arbitrary string tags after the fact, enabling
filtered retrieval such as ``tags=["important", "bug-fix"]``.

Typical usage via the Agent API::

    agent.enable_history("./data/conversations")
    response = agent.send("How do I fix the authentication error?")
    agent.tag_current_message(["important", "bug-fix"])
    results = agent.search_history("authentication error")
"""

import hashlib
import os
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

try:
    import chromadb
    from chromadb.config import Settings as ChromaSettings

    CHROMADB_AVAILABLE = True
except ImportError:  # pragma: no cover
    CHROMADB_AVAILABLE = False
    chromadb = None  # type: ignore[assignment]
    ChromaSettings = None  # type: ignore[assignment,misc]

_CHROMA_COLLECTION = "conversation_history"

_SCHEMA_SQL = """
PRAGMA foreign_keys = ON;
PRAGMA journal_mode = WAL;

CREATE TABLE IF NOT EXISTS messages (
    id           TEXT PRIMARY KEY,
    session_id   TEXT NOT NULL,
    agent_name   TEXT NOT NULL,
    context_name TEXT NOT NULL,
    role         TEXT NOT NULL,
    content      TEXT NOT NULL,
    timestamp    TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_messages_session   ON messages(session_id);
CREATE INDEX IF NOT EXISTS idx_messages_agent     ON messages(agent_name);
CREATE INDEX IF NOT EXISTS idx_messages_timestamp ON messages(timestamp);

CREATE TABLE IF NOT EXISTS tags (
    message_id TEXT NOT NULL,
    tag        TEXT NOT NULL,
    UNIQUE(message_id, tag),
    FOREIGN KEY(message_id) REFERENCES messages(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_tags_tag       ON tags(tag);
CREATE INDEX IF NOT EXISTS idx_tags_message   ON tags(message_id);

CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts
    USING fts5(id UNINDEXED, content);
"""


@dataclass
class MessageRecord:
    """A single stored conversation message."""

    id: str
    session_id: str
    agent_name: str
    context_name: str
    role: str  # 'user', 'assistant', or 'system'
    content: str
    timestamp: str
    tags: list[str] = field(default_factory=list)


@dataclass
class HistorySearchResult:
    """A search result from the conversation history."""

    message: MessageRecord
    relevance_score: float  # 0.0–1.0, higher is better
    match_type: str  # 'semantic' or 'text'


class ConversationStore:
    """Persistent store for agent conversation histories.

    Args:
        persist_directory: Directory for the SQLite database and, when
            ChromaDB is available, the vector index.  Created automatically
            if it does not exist.
    """

    def __init__(self, persist_directory: str = "./data/conversations") -> None:
        os.makedirs(persist_directory, exist_ok=True)
        self._db_path = os.path.join(persist_directory, "history.db")
        self._conn = self._init_db()

        # Optional ChromaDB for semantic search
        self._chroma_client: Any = None
        self._chroma_collection: Any = None
        if CHROMADB_AVAILABLE and chromadb is not None:
            try:
                client = chromadb.PersistentClient(
                    path=persist_directory,
                    settings=ChromaSettings(anonymized_telemetry=False),
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
    def _generate_id(session_id: str, role: str, content: str) -> str:
        ts = datetime.now(timezone.utc).isoformat()
        raw = f"{session_id}{role}{content}{ts}"
        digest = hashlib.sha256(raw.encode()).hexdigest()[:16]
        return f"msg_{digest}"

    def _fetch_tags(self, message_ids: list[str]) -> dict[str, list[str]]:
        if not message_ids:
            return {}
        placeholders = ",".join("?" * len(message_ids))
        rows = self._conn.execute(
            f"SELECT message_id, tag FROM tags WHERE message_id IN ({placeholders})",
            message_ids,
        ).fetchall()
        result: dict[str, list[str]] = {mid: [] for mid in message_ids}
        for row in rows:
            result[row["message_id"]].append(row["tag"])
        return result

    def _rows_to_records(self, rows: list) -> list[MessageRecord]:
        if not rows:
            return []
        ids = [r["id"] for r in rows]
        tags_by_id = self._fetch_tags(ids)
        return [
            MessageRecord(
                id=r["id"],
                session_id=r["session_id"],
                agent_name=r["agent_name"],
                context_name=r["context_name"],
                role=r["role"],
                content=r["content"],
                timestamp=r["timestamp"],
                tags=tags_by_id.get(r["id"], []),
            )
            for r in rows
        ]

    @staticmethod
    def _fts5_escape(query: str) -> str:
        """Escape a plain-text string for use as an FTS5 phrase query."""
        # Wrap in double-quotes to treat as a phrase; escape internal quotes.
        escaped = query.replace('"', '""')
        return f'"{escaped}"'

    def _filter_by_tags(
        self, records: list[MessageRecord], tags: list[str]
    ) -> list[MessageRecord]:
        tag_set = set(tags)
        return [r for r in records if tag_set.intersection(set(r.tags))]

    # ------------------------------------------------------------------
    # Public write API
    # ------------------------------------------------------------------

    def store_message(
        self,
        session_id: str,
        agent_name: str,
        context_name: str,
        role: str,
        content: str,
    ) -> str:
        """Persist a single message and return its unique ID.

        The message is stored in both the SQLite database (for structured
        queries and FTS) and, when ChromaDB is available, the vector index.

        Args:
            session_id: Identifier for the conversation session.
            agent_name: Name of the agent.
            context_name: Name of the active context.
            role: Message role (``'user'``, ``'assistant'``, or ``'system'``).
            content: Message content.

        Returns:
            Unique string ID for the stored message.
        """
        msg_id = self._generate_id(session_id, role, content)
        timestamp = datetime.now(timezone.utc).isoformat()

        self._conn.execute(
            "INSERT OR IGNORE INTO messages "
            "(id, session_id, agent_name, context_name, role, content, timestamp) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (msg_id, session_id, agent_name, context_name, role, content, timestamp),
        )
        self._conn.execute(
            "INSERT INTO messages_fts (id, content) VALUES (?, ?)",
            (msg_id, content),
        )
        self._conn.commit()

        # Optionally index for semantic search
        if self._chroma_collection is not None:
            try:
                self._chroma_collection.add(
                    ids=[msg_id],
                    documents=[content],
                    metadatas=[
                        {
                            "session_id": session_id,
                            "agent_name": agent_name,
                            "context_name": context_name,
                            "role": role,
                            "timestamp": timestamp,
                        }
                    ],
                )
            except Exception:
                pass  # Vector index failure is non-fatal

        return msg_id

    def add_tags(self, message_id: str, tags: list[str]) -> None:
        """Attach string tags to a stored message.

        Tags are deduplicated automatically (the same tag can be added
        multiple times without error).

        Args:
            message_id: ID returned by :meth:`store_message`.
            tags: List of tag strings to attach.
        """
        rows = [(message_id, tag.strip()) for tag in tags if tag.strip()]
        if rows:
            self._conn.executemany(
                "INSERT OR IGNORE INTO tags (message_id, tag) VALUES (?, ?)", rows
            )
            self._conn.commit()

    # ------------------------------------------------------------------
    # Public read / search API
    # ------------------------------------------------------------------

    def search_text(
        self,
        query: str,
        n_results: int = 10,
        agent_name: Optional[str] = None,
        session_id: Optional[str] = None,
        tags: Optional[list[str]] = None,
        role: Optional[str] = None,
    ) -> list[HistorySearchResult]:
        """Full-text search using SQLite FTS5.

        The query is treated as a phrase (e.g. ``"authentication error"``).
        If FTS5 parsing fails the search falls back to an SQL ``LIKE`` scan.

        Args:
            query: Plain-text search phrase.
            n_results: Maximum number of results to return.
            agent_name: Filter by agent name.
            session_id: Filter by session ID.
            tags: If provided, only return messages that have *at least one*
                of these tags.
            role: Filter by role (``'user'`` or ``'assistant'``).

        Returns:
            List of :class:`HistorySearchResult` ordered by FTS5 rank.
        """
        fts_query = self._fts5_escape(query)

        # Fetch candidate IDs from FTS index
        try:
            fts_rows = self._conn.execute(
                "SELECT id FROM messages_fts WHERE content MATCH ? "
                "ORDER BY rank LIMIT ?",
                (fts_query, n_results * 4),
            ).fetchall()
        except sqlite3.OperationalError:
            # Malformed FTS query — fall back to LIKE
            like_pattern = f"%{query}%"
            fts_rows = self._conn.execute(
                "SELECT id FROM messages WHERE content LIKE ? LIMIT ?",
                (like_pattern, n_results * 4),
            ).fetchall()

        fts_ids = [r["id"] for r in fts_rows]
        if not fts_ids:
            return []

        # Fetch full records with optional filters
        filter_conds: list[str] = [f"id IN ({','.join('?' * len(fts_ids))})"]
        params: list[Any] = list(fts_ids)

        if agent_name:
            filter_conds.append("agent_name = ?")
            params.append(agent_name)
        if session_id:
            filter_conds.append("session_id = ?")
            params.append(session_id)
        if role:
            filter_conds.append("role = ?")
            params.append(role)

        where = " AND ".join(filter_conds)
        rows = self._conn.execute(
            f"SELECT * FROM messages WHERE {where}", params
        ).fetchall()

        records = self._rows_to_records(rows)

        if tags:
            records = self._filter_by_tags(records, tags)

        # Preserve FTS rank order
        id_order = {mid: idx for idx, mid in enumerate(fts_ids)}
        records.sort(key=lambda r: id_order.get(r.id, len(fts_ids)))

        return [
            HistorySearchResult(message=r, relevance_score=1.0, match_type="text")
            for r in records[:n_results]
        ]

    def search_semantic(
        self,
        query: str,
        n_results: int = 10,
        agent_name: Optional[str] = None,
        session_id: Optional[str] = None,
        tags: Optional[list[str]] = None,
        role: Optional[str] = None,
    ) -> list[HistorySearchResult]:
        """Semantic search using ChromaDB vector embeddings.

        Falls back to :meth:`search_text` when ChromaDB is not available or
        when the vector query fails.

        Args:
            query: Natural-language search query.
            n_results: Maximum number of results to return.
            agent_name: Filter by agent name.
            session_id: Filter by session ID.
            tags: If provided, only return messages that have *at least one*
                of these tags.
            role: Filter by role.

        Returns:
            List of :class:`HistorySearchResult` ordered by relevance score.
        """
        if self._chroma_collection is None:
            return self.search_text(
                query, n_results, agent_name, session_id, tags, role
            )

        # Build ChromaDB metadata filter
        where_parts: list[dict[str, Any]] = []
        if agent_name:
            where_parts.append({"agent_name": {"$eq": agent_name}})
        if session_id:
            where_parts.append({"session_id": {"$eq": session_id}})
        if role:
            where_parts.append({"role": {"$eq": role}})

        chroma_where: Optional[dict[str, Any]] = None
        if len(where_parts) == 1:
            chroma_where = where_parts[0]
        elif len(where_parts) > 1:
            chroma_where = {"$and": where_parts}

        try:
            result = self._chroma_collection.query(
                query_texts=[query],
                n_results=max(n_results * 2, 20),
                where=chroma_where,
            )
        except Exception:
            return self.search_text(
                query, n_results, agent_name, session_id, tags, role
            )

        if not result or not result.get("ids") or not result["ids"][0]:
            return []

        msg_ids: list[str] = result["ids"][0]
        distances: list[float] = (
            result["distances"][0] if result.get("distances") else [0.0] * len(msg_ids)
        )
        id_to_distance = dict(zip(msg_ids, distances))

        # Fetch full records from SQLite to get tags and all fields
        placeholders = ",".join("?" * len(msg_ids))
        rows = self._conn.execute(
            f"SELECT * FROM messages WHERE id IN ({placeholders})", msg_ids
        ).fetchall()

        records = self._rows_to_records(list(rows))

        if tags:
            records = self._filter_by_tags(records, tags)

        results: list[HistorySearchResult] = []
        for record in records[:n_results]:
            distance = id_to_distance.get(record.id, 0.0)
            relevance = 1.0 / (1.0 + distance)
            results.append(
                HistorySearchResult(
                    message=record,
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
        n_results: int = 10,
        agent_name: Optional[str] = None,
        session_id: Optional[str] = None,
        tags: Optional[list[str]] = None,
        role: Optional[str] = None,
        semantic: bool = True,
    ) -> list[HistorySearchResult]:
        """Search conversation history with automatic mode selection.

        Uses semantic (vector) search when ``semantic=True`` and ChromaDB is
        available; otherwise falls back to full-text search.

        Args:
            query: Search query string.
            n_results: Maximum number of results to return.
            agent_name: Filter by agent name.
            session_id: Filter by session ID.
            tags: If provided, only return messages with at least one of
                these tags.
            role: Filter by role (``'user'`` or ``'assistant'``).
            semantic: Prefer semantic search when ChromaDB is available.

        Returns:
            List of :class:`HistorySearchResult` objects.
        """
        if semantic and self._chroma_collection is not None:
            return self.search_semantic(
                query, n_results, agent_name, session_id, tags, role
            )
        return self.search_text(query, n_results, agent_name, session_id, tags, role)

    def get_session_messages(
        self,
        session_id: str,
        role: Optional[str] = None,
        context_name: Optional[str] = None,
    ) -> list[MessageRecord]:
        """Retrieve all messages for a session in chronological order.

        Args:
            session_id: Session identifier.
            role: If given, restrict to this role.
            context_name: If given, restrict to this context.

        Returns:
            Ordered list of :class:`MessageRecord`.
        """
        conds = ["session_id = ?"]
        params: list[Any] = [session_id]
        if role:
            conds.append("role = ?")
            params.append(role)
        if context_name:
            conds.append("context_name = ?")
            params.append(context_name)

        where = " AND ".join(conds)
        rows = self._conn.execute(
            f"SELECT * FROM messages WHERE {where} ORDER BY timestamp ASC",
            params,
        ).fetchall()
        return self._rows_to_records(list(rows))

    def list_sessions(
        self,
        agent_name: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        """List all known sessions, newest first.

        Args:
            agent_name: If given, restrict to sessions for this agent.

        Returns:
            List of dicts with keys ``session_id``, ``agent_name``,
            ``started``, and ``message_count``.
        """
        if agent_name:
            rows = self._conn.execute(
                "SELECT session_id, agent_name, "
                "MIN(timestamp) AS started, COUNT(*) AS message_count "
                "FROM messages WHERE agent_name = ? "
                "GROUP BY session_id ORDER BY started DESC",
                (agent_name,),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT session_id, agent_name, "
                "MIN(timestamp) AS started, COUNT(*) AS message_count "
                "FROM messages GROUP BY session_id ORDER BY started DESC"
            ).fetchall()
        return [dict(r) for r in rows]

    @property
    def semantic_search_available(self) -> bool:
        """``True`` when ChromaDB is available and the vector index is ready."""
        return self._chroma_collection is not None

    def clear_all(self) -> None:
        """Clear all conversation history data.

        Removes all messages, tags, and vector embeddings. Use with caution!
        """
        # Clear SQLite tables
        self._conn.execute("DELETE FROM tags")
        self._conn.execute("DELETE FROM messages_fts")
        self._conn.execute("DELETE FROM messages")
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

    def search_exact(
        self,
        text: str,
        agent_name: Optional[str] = None,
        session_id: Optional[str] = None,
        role: Optional[str] = None,
    ) -> list[MessageRecord]:
        """Search for exact text matches in conversation history.

        Args:
            text: Exact text to search for (case-insensitive substring match).
            agent_name: Filter by agent name.
            session_id: Filter by session ID.
            role: Filter by role.

        Returns:
            List of matching MessageRecord objects.
        """
        conds = ["content LIKE ?"]
        params: list[Any] = [f"%{text}%"]

        if agent_name:
            conds.append("agent_name = ?")
            params.append(agent_name)
        if session_id:
            conds.append("session_id = ?")
            params.append(session_id)
        if role:
            conds.append("role = ?")
            params.append(role)

        where = " AND ".join(conds)
        rows = self._conn.execute(
            f"SELECT * FROM messages WHERE {where} ORDER BY timestamp DESC",
            params,
        ).fetchall()

        return self._rows_to_records(list(rows))

    def close(self) -> None:
        """Close all underlying database connections.

        Releases both the SQLite connection and, when ChromaDB was
        initialised, the ChromaDB PersistentClient.  On Windows this is
        required before the containing directory can be deleted, because
        both ``chroma.sqlite3`` and HNSW index files are held open by
        the underlying chromadb System.
        """
        import gc

        self._conn.close()
        if self._chroma_client is not None:
            try:
                # Stop all chromadb components (SegmentManager, HNSW indices,
                # SQLite connections) before clearing the global cache.
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
            gc.collect()  # ensure C-extension destructors run promptly
