"""Database manager for unified database operations.

Provides centralized management for all pithos databases:
- Memory (vector knowledge store)
- Conversation history
- Flowcharts

Supports clearing, searching, and managing all databases from a single interface.
"""

import logging
from pathlib import Path
from typing import Any, Optional
from dataclasses import dataclass

from pithos.tools.memory_tool import MemoryStore, CHROMADB_AVAILABLE
from pithos.agent.history import ConversationStore
from pithos.flowchart_store import FlowchartStore
from pithos.config_manager import ConfigManager

logger = logging.getLogger(__name__)


@dataclass
class DatabaseInfo:
    """Information about a database."""

    name: str
    type: str
    path: str
    size_bytes: int
    available: bool
    error: Optional[str] = None


@dataclass
class UnifiedSearchResult:
    """Unified search result from any database."""

    database: str  # 'memory', 'history', or 'flowcharts'
    result_type: str  # 'knowledge', 'conversation', or 'flowchart'
    content: str
    metadata: dict[str, Any]
    relevance_score: float
    match_type: str


class DatabaseManager:
    """Unified manager for all pithos databases."""

    def __init__(
        self,
        memory_dir: str = "./data/memory",
        history_dir: str = "./data/conversations",
        flowchart_dir: str = "./data/flowcharts",
        config_manager: Optional[ConfigManager] = None,
    ):
        """Initialize database manager.

        Args:
            memory_dir: Directory for memory (vector) database.
            history_dir: Directory for conversation history database.
            flowchart_dir: Directory for flowchart database.
            config_manager: Optional configuration manager.
        """
        self.memory_dir = Path(memory_dir)
        self.history_dir = Path(history_dir)
        self.flowchart_dir = Path(flowchart_dir)
        self.config_manager = config_manager or ConfigManager()

        # Initialize stores (lazy loading)
        self._memory_store: Optional[MemoryStore] = None
        self._history_store: Optional[ConversationStore] = None
        self._flowchart_store: Optional[FlowchartStore] = None

    @property
    def memory(self) -> MemoryStore:
        """Get or create memory store."""
        if self._memory_store is None:
            if not CHROMADB_AVAILABLE:
                raise RuntimeError(
                    "ChromaDB is not installed. Install with: pip install chromadb"
                )
            self._memory_store = MemoryStore(
                config_manager=self.config_manager,
                persist_directory=str(self.memory_dir),
            )
        return self._memory_store

    @property
    def history(self) -> ConversationStore:
        """Get or create conversation history store."""
        if self._history_store is None:
            self._history_store = ConversationStore(
                persist_directory=str(self.history_dir)
            )
        return self._history_store

    @property
    def flowcharts(self) -> FlowchartStore:
        """Get or create flowchart store."""
        if self._flowchart_store is None:
            self._flowchart_store = FlowchartStore(
                persist_directory=str(self.flowchart_dir)
            )
        return self._flowchart_store

    # ------------------------------------------------------------------
    # Database info & status
    # ------------------------------------------------------------------

    def get_database_info(self) -> list[DatabaseInfo]:
        """Get information about all databases.

        Returns:
            List of DatabaseInfo objects.
        """
        info_list = []

        # Memory database
        try:
            if CHROMADB_AVAILABLE:
                memory_path = self.memory_dir / "chroma.sqlite3"
                size = memory_path.stat().st_size if memory_path.exists() else 0
                info_list.append(
                    DatabaseInfo(
                        name="Memory (Vector Store)",
                        type="chromadb",
                        path=str(self.memory_dir),
                        size_bytes=size,
                        available=True,
                    )
                )
            else:
                info_list.append(
                    DatabaseInfo(
                        name="Memory (Vector Store)",
                        type="chromadb",
                        path=str(self.memory_dir),
                        size_bytes=0,
                        available=False,
                        error="ChromaDB not installed",
                    )
                )
        except Exception as e:
            info_list.append(
                DatabaseInfo(
                    name="Memory (Vector Store)",
                    type="chromadb",
                    path=str(self.memory_dir),
                    size_bytes=0,
                    available=False,
                    error=str(e),
                )
            )

        # History database
        try:
            history_path = self.history_dir / "history.db"
            size = history_path.stat().st_size if history_path.exists() else 0
            info_list.append(
                DatabaseInfo(
                    name="Conversation History",
                    type="sqlite",
                    path=str(history_path),
                    size_bytes=size,
                    available=True,
                )
            )
        except Exception as e:
            info_list.append(
                DatabaseInfo(
                    name="Conversation History",
                    type="sqlite",
                    path=str(self.history_dir),
                    size_bytes=0,
                    available=False,
                    error=str(e),
                )
            )

        # Flowchart database
        try:
            flowchart_path = self.flowchart_dir / "flowcharts.db"
            size = flowchart_path.stat().st_size if flowchart_path.exists() else 0
            info_list.append(
                DatabaseInfo(
                    name="Flowcharts",
                    type="sqlite",
                    path=str(flowchart_path),
                    size_bytes=size,
                    available=True,
                )
            )
        except Exception as e:
            info_list.append(
                DatabaseInfo(
                    name="Flowcharts",
                    type="sqlite",
                    path=str(self.flowchart_dir),
                    size_bytes=0,
                    available=False,
                    error=str(e),
                )
            )

        return info_list

    # ------------------------------------------------------------------
    # Clear operations
    # ------------------------------------------------------------------

    def clear_memory(self) -> None:
        """Clear all memory (vector) store data."""
        try:
            self.memory.clear_all()
            logger.info("Memory store cleared")
        except Exception as e:
            logger.error(f"Error clearing memory store: {e}")
            raise

    def clear_history(self) -> None:
        """Clear all conversation history data."""
        try:
            self.history.clear_all()
            logger.info("Conversation history cleared")
        except Exception as e:
            logger.error(f"Error clearing conversation history: {e}")
            raise

    def clear_flowcharts(self) -> None:
        """Clear all flowchart data."""
        try:
            self.flowcharts.clear_all()
            logger.info("Flowchart store cleared")
        except Exception as e:
            logger.error(f"Error clearing flowchart store: {e}")
            raise

    def clear_all(self, confirm: bool = False) -> dict[str, str]:
        """Clear all databases.

        Args:
            confirm: Must be True to actually clear (safety check).

        Returns:
            Dictionary with results for each database.

        Raises:
            ValueError: If confirm is not True.
        """
        if not confirm:
            raise ValueError(
                "Must set confirm=True to clear all databases. This operation cannot be undone!"
            )

        results = {}

        # Clear memory
        try:
            self.clear_memory()
            results["memory"] = "cleared"
        except Exception as e:
            results["memory"] = f"error: {e}"

        # Clear history
        try:
            self.clear_history()
            results["history"] = "cleared"
        except Exception as e:
            results["history"] = f"error: {e}"

        # Clear flowcharts
        try:
            self.clear_flowcharts()
            results["flowcharts"] = "cleared"
        except Exception as e:
            results["flowcharts"] = f"error: {e}"

        return results

    # ------------------------------------------------------------------
    # Universal search
    # ------------------------------------------------------------------

    def search_all(
        self,
        query: str,
        semantic: bool = True,
        limit: int = 10,
    ) -> dict[str, list[UnifiedSearchResult]]:
        """Search across all databases.

        Args:
            query: Search query text.
            semantic: Use semantic search if available.
            limit: Maximum results per database.

        Returns:
            Dictionary mapping database names to lists of results.
        """
        all_results = {}

        # Search memory store
        try:
            if CHROMADB_AVAILABLE:
                memory_results = self.memory.search_all_categories(
                    query=query,
                    n_results=limit,
                )
                unified_results = []
                for category, results in memory_results.items():
                    for result in results[:limit]:
                        unified_results.append(
                            UnifiedSearchResult(
                                database="memory",
                                result_type="knowledge",
                                content=result.content,
                                metadata={
                                    "category": category,
                                    **result.metadata,
                                },
                                relevance_score=result.relevance_score,
                                match_type="semantic",
                            )
                        )
                if unified_results:
                    # Sort by relevance
                    unified_results.sort(key=lambda r: r.relevance_score, reverse=True)
                    all_results["memory"] = unified_results[:limit]
        except Exception as e:
            logger.warning(f"Memory search failed: {e}")

        # Search conversation history
        try:
            history_results = self.history.search(
                query=query,
                n_results=limit,
                semantic=semantic,
            )
            unified_results = [
                UnifiedSearchResult(
                    database="history",
                    result_type="conversation",
                    content=result.message.content,
                    metadata={
                        "role": result.message.role,
                        "agent_name": result.message.agent_name,
                        "session_id": result.message.session_id,
                        "timestamp": result.message.timestamp,
                        "tags": result.message.tags,
                    },
                    relevance_score=result.relevance_score,
                    match_type=result.match_type,
                )
                for result in history_results
            ]
            if unified_results:
                all_results["history"] = unified_results
        except Exception as e:
            logger.warning(f"History search failed: {e}")

        # Search flowcharts
        try:
            flowchart_results = self.flowcharts.search(
                query=query,
                semantic=semantic,
                limit=limit,
            )
            unified_results = [
                UnifiedSearchResult(
                    database="flowcharts",
                    result_type="flowchart",
                    content=f"{result.flowchart.name}: {result.flowchart.description}",
                    metadata={
                        "id": result.flowchart.id,
                        "name": result.flowchart.name,
                        "description": result.flowchart.description,
                        "tags": result.flowchart.tags,
                        "notes": result.flowchart.notes,
                        "updated_at": result.flowchart.updated_at,
                    },
                    relevance_score=result.relevance_score,
                    match_type=result.match_type,
                )
                for result in flowchart_results
            ]
            if unified_results:
                all_results["flowcharts"] = unified_results
        except Exception as e:
            logger.warning(f"Flowchart search failed: {e}")

        return all_results

    def search_exact(
        self,
        text: str,
        databases: Optional[list[str]] = None,
    ) -> dict[str, Any]:
        """Search for exact text matches across databases.

        Args:
            text: Exact text to search for.
            databases: List of databases to search ('memory', 'history', 'flowcharts').
                      If None, searches all.

        Returns:
            Dictionary with results from each database.
        """
        if databases is None:
            databases = ["memory", "history", "flowcharts"]

        all_results = {}

        # Search memory
        if "memory" in databases:
            try:
                if CHROMADB_AVAILABLE:
                    results = self.memory.search_exact(text)
                    all_results["memory"] = results
            except Exception as e:
                logger.warning(f"Memory exact search failed: {e}")
                all_results["memory"] = {"error": str(e)}

        # Search history
        if "history" in databases:
            try:
                results = self.history.search_exact(text)
                all_results["history"] = [
                    {
                        "id": r.id,
                        "session_id": r.session_id,
                        "agent_name": r.agent_name,
                        "context_name": r.context_name,
                        "role": r.role,
                        "content": r.content,
                        "timestamp": r.timestamp,
                        "tags": r.tags,
                    }
                    for r in results
                ]
            except Exception as e:
                logger.warning(f"History exact search failed: {e}")
                all_results["history"] = {"error": str(e)}

        # Search flowcharts
        if "flowcharts" in databases:
            try:
                results = self.flowcharts.search_exact(text)
                all_results["flowcharts"] = [r.to_dict() for r in results]
            except Exception as e:
                logger.warning(f"Flowchart exact search failed: {e}")
                all_results["flowcharts"] = {"error": str(e)}

        return all_results

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Close all database connections."""
        if self._memory_store is not None:
            try:
                self._memory_store.close()
            except Exception as e:
                logger.warning(f"Error closing memory store: {e}")
        if self._history_store is not None:
            try:
                self._history_store.close()
            except Exception as e:
                logger.warning(f"Error closing history store: {e}")
        if self._flowchart_store is not None:
            try:
                self._flowchart_store.close()
            except Exception as e:
                logger.warning(f"Error closing flowchart store: {e}")


def main():
    """CLI for database management."""
    import argparse

    parser = argparse.ArgumentParser(
        description="pithos Database Manager - Unified database operations"
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # Info command
    subparsers.add_parser("info", help="Show database information")

    # Clear commands
    clear_parser = subparsers.add_parser("clear", help="Clear database(s)")
    clear_parser.add_argument(
        "database",
        choices=["memory", "history", "flowcharts", "all"],
        help="Database to clear",
    )
    clear_parser.add_argument(
        "--confirm",
        action="store_true",
        help="Confirm the clear operation",
    )

    # Search commands
    search_parser = subparsers.add_parser("search", help="Search across all databases")
    search_parser.add_argument("query", help="Search query")
    search_parser.add_argument("--exact", action="store_true", help="Exact text match")
    search_parser.add_argument(
        "--semantic",
        action="store_true",
        default=True,
        help="Use semantic search (default)",
    )
    search_parser.add_argument(
        "--limit", type=int, default=10, help="Maximum results per database"
    )

    args = parser.parse_args()

    # Initialize manager
    manager = DatabaseManager()

    try:
        if args.command == "info":
            info_list = manager.get_database_info()
            print("\n=== Database Information ===\n")
            for info in info_list:
                print(f"{info.name}:")
                print(f"  Type: {info.type}")
                print(f"  Path: {info.path}")
                if info.available:
                    print(f"  Size: {info.size_bytes:,} bytes")
                    print("  Status: Available")
                else:
                    print("  Status: Unavailable")
                    if info.error:
                        print(f"  Error: {info.error}")
                print()

        elif args.command == "clear":
            if not args.confirm:
                print(
                    "WARNING: This will permanently delete data. Use --confirm to proceed."
                )
                return

            if args.database == "all":
                results = manager.clear_all(confirm=True)
                print("\n=== Clear All Databases ===\n")
                for db, status in results.items():
                    print(f"{db}: {status}")
            elif args.database == "memory":
                manager.clear_memory()
                print("Memory store cleared")
            elif args.database == "history":
                manager.clear_history()
                print("Conversation history cleared")
            elif args.database == "flowcharts":
                manager.clear_flowcharts()
                print("Flowchart store cleared")

        elif args.command == "search":
            if args.exact:
                results = manager.search_exact(args.query)
                print(f"\n=== Exact Search Results for '{args.query}' ===\n")
                for db, db_results in results.items():
                    if isinstance(db_results, dict) and "error" in db_results:
                        print(f"{db}: Error - {db_results['error']}")
                    elif db_results:
                        print(f"\n{db.upper()} ({len(db_results)} results):")
                        for i, result in enumerate(db_results[: args.limit], 1):
                            if isinstance(result, dict):
                                content = result.get("content", "")[:200]
                                print(f"  {i}. {content}...")
                            else:
                                print(f"  {i}. {result}")
            else:
                results = manager.search_all(
                    query=args.query,
                    semantic=args.semantic,
                    limit=args.limit,
                )
                print(f"\n=== Search Results for '{args.query}' ===\n")
                for db, db_results in results.items():
                    print(f"\n{db.upper()} ({len(db_results)} results):")
                    for i, result in enumerate(db_results, 1):
                        print(
                            f"  {i}. [Score: {result.relevance_score:.3f}] {result.content[:200]}..."
                        )
                        print(f"     Type: {result.result_type}")

        else:
            parser.print_help()

    finally:
        manager.close()


if __name__ == "__main__":
    main()
