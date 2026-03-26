"""Vector database memory system for pithos agents - knowledge storage and retrieval."""

import os
import json
import time
from collections import OrderedDict
from dataclasses import dataclass, asdict
from typing import Optional, Any
from datetime import datetime
import hashlib

try:
    import chromadb
    from chromadb.config import Settings

    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False
    chromadb = None

from ..config_manager import ConfigManager
from .tag_suggester import CategoryTagSuggester, TagSuggestion


@dataclass
class MemoryEntry:
    """A single memory entry in the knowledge base."""

    id: str
    category: str
    content: str
    metadata: dict[str, Any]
    timestamp: str
    embedding: Optional[list[float]] = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class SearchResult:
    """Result from a memory search."""

    id: str
    category: str
    content: str
    metadata: dict[str, Any]
    distance: float  # Similarity distance (lower is better)
    relevance_score: float  # 0.0 to 1.0, higher is better


@dataclass
class _CacheEntry:
    """Internal cache entry holding a ChromaDB collection with access metadata."""

    collection: Any
    last_accessed: float  # time.monotonic() value
    created_at: float  # time.monotonic() value


class MemoryStore:
    """Vector database store for organizing knowledge by categories."""

    def __init__(
        self,
        config_manager: Optional[ConfigManager] = None,
        persist_directory: Optional[str] = None,
    ):
        """Initialize memory store.

        Args:
            config_manager: Configuration manager for loading memory configs.
            persist_directory: Directory for persistent storage. If None, uses default.

        Raises:
            RuntimeError: If ChromaDB is not available.
        """
        if not CHROMADB_AVAILABLE:
            raise RuntimeError(
                "ChromaDB is not installed. Install with: pip install chromadb"
            )

        self.config_manager = config_manager
        self.config = self._load_config()

        # Set up persistence directory
        if persist_directory is None:
            persist_directory = self.config.get("persist_directory", "./data/memory")

        # Ensure directory exists
        os.makedirs(persist_directory, exist_ok=True)

        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True,
            ),
        )

        # Collection cache with TTL and LRU eviction.
        # OrderedDict insertion/access order tracks LRU (front = oldest).
        self._collections: OrderedDict[str, _CacheEntry] = OrderedDict()
        cache_config = self.config.get("collection_cache", {})
        self._cache_ttl: float = float(cache_config.get("ttl_seconds", 300))
        self._cache_max_size: int = int(cache_config.get("max_size", 50))

        # Optional LLM-backed tag suggestions (disabled until enable_tag_suggestions() is called).
        self._tag_suggester: Optional[CategoryTagSuggester] = None

    def _load_config(self) -> dict:
        """Load memory tool configuration."""
        if self.config_manager:
            config = self.config_manager.get_config("memory_config", "tools")
            if config:
                return config

        # Default configuration
        return {
            "enabled": True,
            "persist_directory": "./data/memory",
            "embedding_function": "default",
            "max_results": 10,
            "similarity_threshold": 0.5,
            "default_metadata": {},
        }

    def _generate_id(self, content: str, category: str) -> str:
        """Generate a unique ID for a memory entry.

        Args:
            content: The content to store.
            category: The category/collection name.

        Returns:
            Unique ID string.
        """
        # Use content hash + timestamp for uniqueness
        content_hash = hashlib.sha256(content.encode()).hexdigest()[:16]
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S%f")
        return f"{category}_{content_hash}_{timestamp}"

    def _get_collection(self, category: str):
        """Get or create a collection for a category, with TTL and LRU caching.

        TTL eviction: a cached entry accessed more than ``_cache_ttl`` seconds ago
        is considered stale and re-fetched from ChromaDB (0 = disabled).
        Size eviction: when the cache reaches ``_cache_max_size`` entries the
        least-recently-used entry is evicted before the new one is inserted
        (0 = unlimited).

        Args:
            category: The category/collection name.

        Returns:
            ChromaDB collection object.
        """
        now = time.monotonic()

        if category in self._collections:
            entry = self._collections[category]
            if self._cache_ttl > 0 and (now - entry.last_accessed) > self._cache_ttl:
                # Entry has expired — remove it and fall through to re-fetch.
                del self._collections[category]
            else:
                # Cache hit — refresh access time and promote to MRU position.
                entry.last_accessed = now
                self._collections.move_to_end(category)
                return entry.collection

        # Cache miss (or just expired) — fetch from ChromaDB.
        collection = self.client.get_or_create_collection(
            name=category,
            metadata={"category": category, "created": datetime.now().isoformat()},
        )

        # Evict LRU entries to stay within max_size (0 = unlimited).
        if self._cache_max_size > 0:
            while len(self._collections) >= self._cache_max_size:
                self._collections.popitem(last=False)  # Remove LRU (front)

        self._collections[category] = _CacheEntry(
            collection=collection,
            last_accessed=now,
            created_at=now,
        )
        return collection

    def store(
        self,
        category: str,
        content: str,
        metadata: Optional[dict[str, Any]] = None,
    ) -> str:
        """Store a piece of knowledge in a category.

        Args:
            category: The category to store the knowledge in.
            content: The text content to store.
            metadata: Optional metadata to attach.

        Returns:
            The ID of the stored entry.
        """
        if not content or not content.strip():
            raise ValueError("Content cannot be empty")

        # Generate ID
        entry_id = self._generate_id(content, category)

        # Prepare metadata
        entry_metadata = self.config.get("default_metadata", {}).copy()
        if metadata:
            entry_metadata.update(metadata)
        entry_metadata["timestamp"] = datetime.now().isoformat()
        entry_metadata["category"] = category

        # Attach LLM-suggested tags when auto-suggestions are enabled.
        if self._tag_suggester is not None:
            try:
                existing = self.list_categories()
                suggestions = self._tag_suggester.suggest(content, existing)
                if suggestions:
                    entry_metadata["suggested_tags"] = [
                        s.category for s in suggestions
                    ]
                    entry_metadata["suggested_tags_confidence"] = [
                        round(s.confidence, 4) for s in suggestions
                    ]
            except Exception:
                pass  # Suggestions are advisory; never block a store.

        # Get collection
        collection = self._get_collection(category)

        # Store in ChromaDB
        collection.add(
            ids=[entry_id],
            documents=[content],
            metadatas=[entry_metadata],
        )

        return entry_id

    def store_batch(
        self,
        category: str,
        contents: list[str],
        metadatas: Optional[list[dict[str, Any]]] = None,
    ) -> list[str]:
        """Store multiple pieces of knowledge at once.

        Args:
            category: The category to store the knowledge in.
            contents: List of text contents to store.
            metadatas: Optional list of metadata dicts (same length as contents).

        Returns:
            List of IDs for the stored entries.
        """
        if not contents:
            raise ValueError("Contents list cannot be empty")

        if metadatas and len(metadatas) != len(contents):
            raise ValueError("Metadatas list must match contents list length")

        # Generate IDs
        entry_ids = [self._generate_id(content, category) for content in contents]

        # Prepare metadatas
        entry_metadatas = []
        default_metadata = self.config.get("default_metadata", {})
        timestamp = datetime.now().isoformat()

        for i, content in enumerate(contents):
            entry_metadata = default_metadata.copy()
            if metadatas and i < len(metadatas):
                entry_metadata.update(metadatas[i])
            entry_metadata["timestamp"] = timestamp
            entry_metadata["category"] = category
            entry_metadatas.append(entry_metadata)

        # Get collection
        collection = self._get_collection(category)

        # Store in ChromaDB
        collection.add(
            ids=entry_ids,
            documents=contents,
            metadatas=entry_metadatas,
        )

        return entry_ids

    def retrieve(
        self,
        category: str,
        query: str,
        n_results: Optional[int] = None,
        where: Optional[dict[str, Any]] = None,
        min_relevance: Optional[float] = None,
    ) -> list[SearchResult]:
        """Retrieve relevant knowledge from a category.

        Args:
            category: The category to search in.
            query: The search query text.
            n_results: Maximum number of results to return. Uses config default if None.
            where: Optional metadata filter (e.g., {"source": "manual"}).
            min_relevance: Minimum relevance score (0–1) for a result to be
                included in the response.  When provided, overrides the
                ``similarity_threshold`` value from config so that callers
                (e.g. :class:`~pithos.agent.recall.AutoRecall`) can specify
                their own threshold without modifying the global config.

        Returns:
            List of SearchResult objects, ordered by relevance.
        """
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")

        # Get collection
        collection = self._get_collection(category)

        # Set n_results
        if n_results is None:
            n_results = self.config.get("max_results", 10)

        # Query the collection
        results = collection.query(
            query_texts=[query],
            n_results=n_results,
            where=where,
        )

        # Parse results
        search_results = []
        if results and results["ids"] and results["ids"][0]:
            for i, entry_id in enumerate(results["ids"][0]):
                distance = results["distances"][0][i] if results["distances"] else 0.0
                # Convert distance to relevance score (0.0 to 1.0, higher is better)
                # Using exponential decay: score = e^(-distance)
                import math

                relevance_score = math.exp(-distance)

                search_results.append(
                    SearchResult(
                        id=entry_id,
                        category=category,
                        content=results["documents"][0][i],
                        metadata=(
                            results["metadatas"][0][i] if results["metadatas"] else {}
                        ),
                        distance=distance,
                        relevance_score=relevance_score,
                    )
                )

        # Filter by similarity threshold.  If the caller supplied an explicit
        # min_relevance, use that; otherwise fall back to the config value.
        if min_relevance is not None:
            threshold = min_relevance
        else:
            threshold = self.config.get("similarity_threshold", 0.7)
        search_results = [r for r in search_results if r.relevance_score >= threshold]

        return search_results

    def delete(self, category: str, entry_id: str) -> bool:
        """Delete a specific memory entry.

        Args:
            category: The category containing the entry.
            entry_id: The ID of the entry to delete.

        Returns:
            True if deleted, False if not found.
        """
        try:
            collection = self._get_collection(category)
            collection.delete(ids=[entry_id])
            return True
        except Exception:
            return False

    def delete_category(self, category: str) -> bool:
        """Delete an entire category and all its contents.

        Args:
            category: The category to delete.

        Returns:
            True if deleted, False if not found.
        """
        try:
            self.client.delete_collection(name=category)
            if category in self._collections:
                del self._collections[category]
            return True
        except Exception:
            return False

    def invalidate_cache(self, category: Optional[str] = None) -> None:
        """Invalidate cached collection(s) without affecting persisted data.

        Args:
            category: If provided, invalidates only that category's cache entry.
                      If None, invalidates all cached entries.
        """
        if category is None:
            self._collections.clear()
        elif category in self._collections:
            del self._collections[category]

    def cache_info(self) -> dict[str, Any]:
        """Return current cache statistics.

        Returns:
            Dictionary with ``size``, ``max_size``, ``ttl_seconds``, and
            per-entry age information.
        """
        now = time.monotonic()
        return {
            "size": len(self._collections),
            "max_size": self._cache_max_size,
            "ttl_seconds": self._cache_ttl,
            "entries": {
                cat: {
                    "idle_seconds": now - entry.last_accessed,
                    "age_seconds": now - entry.created_at,
                }
                for cat, entry in self._collections.items()
            },
        }

    # ------------------------------------------------------------------
    # LLM-backed tag suggestions
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Close the ChromaDB client and release all database resources.

        On Windows, ChromaDB holds file locks on ``chroma.sqlite3`` and
        HNSW index files until the underlying system is stopped.  Call
        this method before deleting the persistence directory to avoid a
        ``PermissionError: [WinError 32]``.
        """
        import gc

        self._collections.clear()
        if self.client is not None:
            try:
                system = getattr(self.client, "_system", None)
                if system is not None and hasattr(system, "stop"):
                    system.stop()
            except Exception:
                pass
            try:
                self.client.clear_system_cache()
            except Exception:
                pass
            self.client = None
            gc.collect()  # ensure C-extension destructors run promptly

    def enable_tag_suggestions(
        self,
        model: str,
        max_suggestions: int = 3,
        temperature: float = 0.2,
        timeout: int = 30,
    ) -> None:
        """Enable automatic LLM-generated category tag suggestions.

        When enabled, :meth:`store` will call the LLM to suggest up to
        *max_suggestions* category tags for each piece of content stored.
        The suggestions are attached to the entry's metadata under the keys
        ``suggested_tags`` (list of tag strings) and
        ``suggested_tags_confidence`` (parallel list of float scores).

        Tag suggestions are purely advisory — they do not change the
        *category* argument passed to :meth:`store` and a failure in the LLM
        call will never prevent an entry from being stored.

        Args:
            model: Ollama model name to use for suggestions.
            max_suggestions: Maximum number of suggestions per entry (1-10).
            temperature: LLM sampling temperature (lower = more deterministic).
            timeout: HTTP timeout in seconds for the LLM request.
        """
        self._tag_suggester = CategoryTagSuggester(
            model=model,
            max_suggestions=max_suggestions,
            temperature=temperature,
            timeout=timeout,
        )

    def disable_tag_suggestions(self) -> None:
        """Disable automatic LLM tag suggestions."""
        self._tag_suggester = None

    @property
    def tag_suggestions_enabled(self) -> bool:
        """``True`` if LLM tag suggestions are currently active."""
        return self._tag_suggester is not None

    def suggest_categories(
        self,
        content: str,
        max_suggestions: int = 3,
        model: Optional[str] = None,
    ) -> list[TagSuggestion]:
        """Ask the LLM to suggest category tags for *content*.

        This is a one-shot utility method that does **not** require
        :meth:`enable_tag_suggestions` to be called first.  If a
        :class:`~.tag_suggester.CategoryTagSuggester` is already configured
        (via :meth:`enable_tag_suggestions`) and *model* is ``None``, the
        existing suggester is reused; otherwise a temporary suggester is
        created with the given *model*.

        Args:
            content: Text to generate category suggestions for.
            max_suggestions: Maximum number of suggestions to return.
            model: Ollama model name.  Required when
                :meth:`enable_tag_suggestions` has not been called.

        Returns:
            List of :class:`~.tag_suggester.TagSuggestion` objects sorted by
            descending confidence.  Empty list on error or if no LLM is
            configured.

        Raises:
            ValueError: If no model is available (neither configured via
                :meth:`enable_tag_suggestions` nor passed as *model*).
        """
        if self._tag_suggester is not None and model is None:
            suggester = self._tag_suggester
        elif model:
            suggester = CategoryTagSuggester(
                model=model,
                max_suggestions=max_suggestions,
            )
        else:
            raise ValueError(
                "No LLM model configured. Pass model= or call enable_tag_suggestions() first."
            )

        existing = []
        try:
            existing = self.list_categories()
        except Exception:
            pass

        return suggester.suggest(content, existing_categories=existing)

    # ------------------------------------------------------------------

    def list_categories(self) -> list[str]:
        """List all available categories.

        Returns:
            List of category names.
        """
        collections = self.client.list_collections()
        return [col.name for col in collections]

    def get_category_info(self, category: str) -> dict[str, Any]:
        """Get information about a category.

        Args:
            category: The category name.

        Returns:
            Dictionary with category information.
        """
        try:
            collection = self._get_collection(category)
            count = collection.count()
            metadata = collection.metadata or {}

            return {
                "name": category,
                "count": count,
                "metadata": metadata,
            }
        except Exception as e:
            return {
                "name": category,
                "error": str(e),
            }

    def get_all_entries(self, category: str) -> list[dict[str, Any]]:
        """Get all entries in a category (without embedding vectors).

        Args:
            category: The category name.

        Returns:
            List of entry dictionaries.
        """
        try:
            collection = self._get_collection(category)
            results = collection.get()

            entries = []
            if results and results["ids"]:
                for i, entry_id in enumerate(results["ids"]):
                    entries.append(
                        {
                            "id": entry_id,
                            "content": (
                                results["documents"][i] if results["documents"] else ""
                            ),
                            "metadata": (
                                results["metadatas"][i] if results["metadatas"] else {}
                            ),
                        }
                    )

            return entries
        except Exception:
            return []

    def search_all_categories(
        self,
        query: str,
        n_results: Optional[int] = None,
        min_relevance: Optional[float] = None,
        categories: Optional[list[str]] = None,
    ) -> dict[str, list[SearchResult]]:
        """Search across all or specified categories.

        Args:
            query: The search query text.
            n_results: Maximum number of results per category.
            min_relevance: Minimum relevance score (0–1).
            categories: List of categories to search. If None, searches all.

        Returns:
            Dictionary mapping category names to lists of SearchResult objects.
        """
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")

        if categories is None:
            categories = self.list_categories()

        if n_results is None:
            n_results = self.config.get("max_results", 10)

        results_by_category = {}
        for category in categories:
            try:
                results = self.retrieve(
                    category=category,
                    query=query,
                    n_results=n_results,
                    min_relevance=min_relevance,
                )
                if results:
                    results_by_category[category] = results
            except Exception:
                # Skip categories that error out
                continue

        return results_by_category

    def search_exact(
        self,
        text: str,
        categories: Optional[list[str]] = None,
    ) -> dict[str, list[dict[str, Any]]]:
        """Search for exact text matches across categories.

        Args:
            text: Exact text to search for.
            categories: List of categories to search. If None, searches all.

        Returns:
            Dictionary mapping category names to lists of matching entries.
        """
        if not text or not text.strip():
            raise ValueError("Search text cannot be empty")

        if categories is None:
            categories = self.list_categories()

        results_by_category = {}
        for category in categories:
            try:
                entries = self.get_all_entries(category)
                matches = [
                    entry for entry in entries if text.lower() in entry["content"].lower()
                ]
                if matches:
                    results_by_category[category] = matches
            except Exception:
                # Skip categories that error out
                continue

        return results_by_category

    def clear_all(self) -> None:
        """Clear all data (use with caution!)."""
        self.client.reset()
        self._collections.clear()

    def export_category(self, category: str, output_path: str) -> None:
        """Export a category to a JSON file.

        Args:
            category: The category to export.
            output_path: Path to save the JSON file.
        """
        entries = self.get_all_entries(category)
        info = self.get_category_info(category)

        export_data = {
            "category": category,
            "info": info,
            "entries": entries,
            "exported_at": datetime.now().isoformat(),
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)

    def import_category(self, input_path: str, category: Optional[str] = None) -> str:
        """Import a category from a JSON file.

        Args:
            input_path: Path to the JSON file.
            category: Optional category name to use (overrides file's category name).

        Returns:
            The category name that was imported.
        """
        with open(input_path, "r", encoding="utf-8") as f:
            import_data = json.load(f)

        # Determine category name
        target_category = category or import_data.get("category", "imported")

        # Import entries
        entries = import_data.get("entries", [])
        if entries:
            contents = [entry["content"] for entry in entries]
            metadatas = [entry.get("metadata", {}) for entry in entries]
            self.store_batch(target_category, contents, metadatas)

        return target_category


def main():
    """CLI for memory tool management."""
    import argparse

    parser = argparse.ArgumentParser(
        description="pithos Memory Tool - Vector database knowledge management"
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # Store command
    store_parser = subparsers.add_parser("store", help="Store knowledge in a category")
    store_parser.add_argument("category", help="Category name")
    store_parser.add_argument("content", help="Content to store")
    store_parser.add_argument(
        "--metadata", help="JSON string of metadata", default=None
    )

    # Retrieve command
    retrieve_parser = subparsers.add_parser(
        "retrieve", help="Retrieve knowledge from a category"
    )
    retrieve_parser.add_argument("category", help="Category name")
    retrieve_parser.add_argument("query", help="Search query")
    retrieve_parser.add_argument("--limit", type=int, default=5, help="Maximum results")

    # List categories command
    subparsers.add_parser("list", help="List all categories")

    # Info command
    info_parser = subparsers.add_parser("info", help="Get category information")
    info_parser.add_argument("category", help="Category name")

    # Delete command
    delete_parser = subparsers.add_parser("delete", help="Delete an entry")
    delete_parser.add_argument("category", help="Category name")
    delete_parser.add_argument("entry_id", help="Entry ID to delete")

    # Export command
    export_parser = subparsers.add_parser("export", help="Export category to JSON")
    export_parser.add_argument("category", help="Category name")
    export_parser.add_argument("output", help="Output JSON file path")

    # Import command
    import_parser = subparsers.add_parser("import", help="Import category from JSON")
    import_parser.add_argument("input", help="Input JSON file path")
    import_parser.add_argument(
        "--category", help="Target category name (optional)", default=None
    )

    args = parser.parse_args()

    # Initialize memory store
    config_manager = ConfigManager()
    memory = MemoryStore(config_manager)

    # Execute command
    if args.command == "store":
        metadata = json.loads(args.metadata) if args.metadata else None
        entry_id = memory.store(args.category, args.content, metadata)
        print(f"Stored with ID: {entry_id}")

    elif args.command == "retrieve":
        results = memory.retrieve(args.category, args.query, n_results=args.limit)
        print(f"\nFound {len(results)} results:\n")
        for i, result in enumerate(results, 1):
            print(f"{i}. [Score: {result.relevance_score:.3f}]")
            print(f"   {result.content}")
            print(f"   Metadata: {result.metadata}")
            print()

    elif args.command == "list":
        categories = memory.list_categories()
        print(f"\nAvailable categories ({len(categories)}):")
        for cat in categories:
            info = memory.get_category_info(cat)
            print(f"  - {cat}: {info.get('count', 0)} entries")

    elif args.command == "info":
        info = memory.get_category_info(args.category)
        print(f"\nCategory: {args.category}")
        print(f"Entries: {info.get('count', 0)}")
        print(f"Metadata: {info.get('metadata', {})}")

    elif args.command == "delete":
        success = memory.delete(args.category, args.entry_id)
        if success:
            print(f"Deleted entry {args.entry_id}")
        else:
            print(f"Entry {args.entry_id} not found")

    elif args.command == "export":
        memory.export_category(args.category, args.output)
        print(f"Exported {args.category} to {args.output}")

    elif args.command == "import":
        category = memory.import_category(args.input, args.category)
        print(f"Imported to category: {category}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
