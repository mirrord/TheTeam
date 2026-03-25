"""Tests for memory tool - vector database knowledge storage and retrieval."""

import pytest
import tempfile
import shutil
import os
import time

# Skip all tests if ChromaDB is not available
try:
    from pithos.tools import (
        MemoryStore,
        MemoryEntry,
        SearchResult,
        CHROMADB_AVAILABLE,
    )
except ImportError:
    CHROMADB_AVAILABLE = False

pytestmark = pytest.mark.skipif(not CHROMADB_AVAILABLE, reason="ChromaDB not installed")


class TestMemoryStore:
    """Tests for MemoryStore class."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for testing."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        # Cleanup
        shutil.rmtree(temp_dir, ignore_errors=True)

    @pytest.fixture
    def memory_store(self, temp_dir):
        """Create a MemoryStore instance for testing."""
        return MemoryStore(config_manager=None, persist_directory=temp_dir)

    @pytest.fixture
    def populated_store(self, memory_store):
        """Create a memory store with some test data."""
        # Add test data
        memory_store.store(
            "python",
            "Python is a high-level programming language with dynamic typing.",
            {"source": "test", "topic": "intro"},
        )
        memory_store.store(
            "python",
            "Python uses indentation to define code blocks instead of braces.",
            {"source": "test", "topic": "syntax"},
        )
        memory_store.store(
            "python",
            "Python has a rich ecosystem of libraries for data science, web development, and more.",
            {"source": "test", "topic": "ecosystem"},
        )
        memory_store.store(
            "javascript",
            "JavaScript is a dynamic programming language commonly used for web development.",
            {"source": "test", "topic": "intro"},
        )
        return memory_store

    def test_memory_store_init(self, temp_dir):
        """Test MemoryStore initialization."""
        store = MemoryStore(persist_directory=temp_dir)
        assert store is not None
        assert store.client is not None
        assert os.path.exists(temp_dir)

    def test_store_single_entry(self, memory_store):
        """Test storing a single entry."""
        entry_id = memory_store.store(
            "test_category", "This is a test entry.", {"key": "value"}
        )
        assert entry_id is not None
        assert isinstance(entry_id, str)
        assert "test_category" in entry_id

    def test_store_empty_content(self, memory_store):
        """Test that storing empty content raises ValueError."""
        with pytest.raises(ValueError, match="Content cannot be empty"):
            memory_store.store("test", "")

        with pytest.raises(ValueError, match="Content cannot be empty"):
            memory_store.store("test", "   ")

    def test_store_batch(self, memory_store):
        """Test storing multiple entries at once."""
        contents = [
            "First entry",
            "Second entry",
            "Third entry",
        ]
        metadatas = [
            {"index": 1},
            {"index": 2},
            {"index": 3},
        ]

        entry_ids = memory_store.store_batch("batch_test", contents, metadatas)
        assert len(entry_ids) == 3
        assert all(isinstance(id, str) for id in entry_ids)

    def test_store_batch_mismatched_metadata(self, memory_store):
        """Test that mismatched metadata length raises error."""
        contents = ["First", "Second"]
        metadatas = [{"index": 1}]  # Wrong length

        with pytest.raises(ValueError, match="must match"):
            memory_store.store_batch("test", contents, metadatas)

    def test_retrieve_relevant_results(self, populated_store):
        """Test retrieving relevant results."""
        results = populated_store.retrieve("python", "What is Python?")
        assert len(results) > 0
        assert all(isinstance(r, SearchResult) for r in results)
        assert all(r.category == "python" for r in results)
        assert all(r.relevance_score > 0 for r in results)

    def test_retrieve_with_limit(self, populated_store):
        """Test retrieving with result limit."""
        results = populated_store.retrieve("python", "Python", n_results=2)
        assert len(results) <= 2

    def test_retrieve_empty_query(self, memory_store):
        """Test that empty query raises ValueError."""
        with pytest.raises(ValueError, match="Query cannot be empty"):
            memory_store.retrieve("test", "")

    def test_retrieve_from_empty_category(self, memory_store):
        """Test retrieving from an empty category."""
        results = memory_store.retrieve("nonexistent", "test query")
        assert len(results) == 0

    def test_retrieve_with_metadata_filter(self, populated_store):
        """Test retrieving with metadata filtering."""
        results = populated_store.retrieve(
            "python", "Python", where={"topic": "syntax"}
        )
        # Should find results with topic=syntax
        assert all("topic" in r.metadata for r in results)

    def test_delete_entry(self, populated_store):
        """Test deleting a specific entry."""
        # Store a new entry
        entry_id = populated_store.store("test_delete", "Delete me")

        # Verify it exists
        entries = populated_store.get_all_entries("test_delete")
        assert len(entries) == 1
        assert entries[0]["id"] == entry_id

        # Delete it
        success = populated_store.delete("test_delete", entry_id)
        assert success is True

        # Verify it's gone
        entries_after = populated_store.get_all_entries("test_delete")
        assert len(entries_after) == 0

    def test_delete_category(self, populated_store):
        """Test deleting an entire category."""
        # Verify category exists
        categories = populated_store.list_categories()
        assert "python" in categories

        # Delete it
        success = populated_store.delete_category("python")
        assert success is True

        # Verify it's gone
        categories = populated_store.list_categories()
        assert "python" not in categories

    def test_list_categories(self, populated_store):
        """Test listing all categories."""
        categories = populated_store.list_categories()
        assert "python" in categories
        assert "javascript" in categories
        assert len(categories) >= 2

    def test_get_category_info(self, populated_store):
        """Test getting category information."""
        info = populated_store.get_category_info("python")
        assert info["name"] == "python"
        assert info["count"] == 3
        assert "metadata" in info

    def test_get_all_entries(self, populated_store):
        """Test getting all entries from a category."""
        entries = populated_store.get_all_entries("python")
        assert len(entries) == 3
        assert all("id" in e for e in entries)
        assert all("content" in e for e in entries)
        assert all("metadata" in e for e in entries)

    def test_clear_all(self, populated_store):
        """Test clearing all data."""
        # Verify data exists
        categories = populated_store.list_categories()
        assert len(categories) > 0

        # Clear all
        populated_store.clear_all()

        # Verify everything is gone
        categories = populated_store.list_categories()
        assert len(categories) == 0

    def test_export_category(self, populated_store, temp_dir):
        """Test exporting a category to JSON."""
        output_path = os.path.join(temp_dir, "export.json")
        populated_store.export_category("python", output_path)

        assert os.path.exists(output_path)

        # Verify JSON content
        import json

        with open(output_path, "r") as f:
            data = json.load(f)

        assert data["category"] == "python"
        assert "entries" in data
        assert len(data["entries"]) == 3
        assert "exported_at" in data

    def test_import_category(self, memory_store, temp_dir):
        """Test importing a category from JSON."""
        # Create test data
        test_data = {
            "category": "imported",
            "entries": [
                {"content": "Entry 1", "metadata": {"index": 1}},
                {"content": "Entry 2", "metadata": {"index": 2}},
            ],
        }

        import_path = os.path.join(temp_dir, "import.json")
        import json

        with open(import_path, "w") as f:
            json.dump(test_data, f)

        # Import
        category = memory_store.import_category(import_path)
        assert category == "imported"

        # Verify data was imported
        entries = memory_store.get_all_entries("imported")
        assert len(entries) == 2

    def test_import_with_custom_category(self, memory_store, temp_dir):
        """Test importing with a custom category name."""
        # Create test data
        test_data = {
            "category": "original",
            "entries": [{"content": "Test", "metadata": {}}],
        }

        import_path = os.path.join(temp_dir, "import.json")
        import json

        with open(import_path, "w") as f:
            json.dump(test_data, f)

        # Import with custom category
        category = memory_store.import_category(import_path, category="custom")
        assert category == "custom"

        # Verify in custom category
        entries = memory_store.get_all_entries("custom")
        assert len(entries) == 1

    def test_persistence(self, temp_dir):
        """Test that data persists across store instances."""
        # Create first store and add data
        store1 = MemoryStore(persist_directory=temp_dir)
        store1.store("persist_test", "This should persist")
        del store1

        # Create new store with same directory
        store2 = MemoryStore(persist_directory=temp_dir)
        categories = store2.list_categories()
        assert "persist_test" in categories

        entries = store2.get_all_entries("persist_test")
        assert len(entries) == 1
        assert "This should persist" in entries[0]["content"]

    def test_relevance_scores(self, populated_store):
        """Test that relevance scores are calculated correctly."""
        results = populated_store.retrieve("python", "Python programming")

        # Scores should be between 0 and 1
        assert all(0 <= r.relevance_score <= 1 for r in results)

        # Results should be ordered by relevance (descending)
        scores = [r.relevance_score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_similarity_threshold(self, temp_dir):
        """Test that similarity threshold filters results."""
        # Create store with high threshold
        config_data = {"similarity_threshold": 0.95}

        # Mock config manager
        class MockConfigManager:
            def get_config(self, name, category):
                return config_data

        store = MemoryStore(
            config_manager=MockConfigManager(), persist_directory=temp_dir
        )

        # Add diverse content
        store.store("test", "The quick brown fox jumps over the lazy dog")
        store.store("test", "A completely different topic about space exploration")

        # Query with something closer to first entry
        results = store.retrieve("test", "quick fox")

        # With high threshold, should get fewer results
        # (only the very relevant one)
        assert len(results) <= 1


class TestMemoryStoreConfiguration:
    """Tests for MemoryStore configuration loading."""

    def test_default_config(self, tmp_path):
        """Test default configuration when no config manager provided."""
        store = MemoryStore(config_manager=None, persist_directory=str(tmp_path))
        assert store.config["enabled"] is True
        assert store.config["max_results"] == 10
        assert store.config["similarity_threshold"] == 0.5

    def test_custom_config(self, tmp_path):
        """Test loading custom configuration."""

        class MockConfigManager:
            def get_config(self, name, category):
                return {
                    "enabled": True,
                    "max_results": 5,
                    "similarity_threshold": 0.8,
                    "persist_directory": str(tmp_path),
                }

        store = MemoryStore(
            config_manager=MockConfigManager(), persist_directory=str(tmp_path)
        )
        assert store.config["max_results"] == 5
        assert store.config["similarity_threshold"] == 0.8


class TestCollectionCache:
    """Tests for collection cache TTL and LRU size-based eviction."""

    @pytest.fixture
    def temp_dir(self):
        temp = tempfile.mkdtemp()
        yield temp
        shutil.rmtree(temp, ignore_errors=True)

    def test_cache_hit_returns_same_object(self, temp_dir):
        """Second access to the same category returns the cached collection."""
        store = MemoryStore(persist_directory=temp_dir)
        col1 = store._get_collection("test")
        col2 = store._get_collection("test")
        assert col1 is col2

    def test_ttl_eviction_on_access(self, temp_dir):
        """A stale cache entry is evicted and re-fetched on the next access."""
        store = MemoryStore(persist_directory=temp_dir)
        store._cache_ttl = 0.05  # 50 ms TTL

        store._get_collection("test")
        assert "test" in store._collections

        time.sleep(0.1)  # Exceed TTL

        # Accessing again should detect expiry and remove the entry first.
        store._get_collection("test")
        # Entry should now be back in cache but be a fresh fetch.
        assert "test" in store._collections
        # The idle time of the refreshed entry should be very small.
        assert store._collections["test"].last_accessed > 0

    def test_ttl_zero_disables_expiry(self, temp_dir):
        """Setting TTL=0 never evicts entries based on time."""
        store = MemoryStore(persist_directory=temp_dir)
        store._cache_ttl = 0

        col1 = store._get_collection("test")
        # Backdate the entry to simulate a very old access.
        store._collections["test"].last_accessed -= 10_000

        col2 = store._get_collection("test")
        assert col2 is col1  # Same object — no re-fetch.

    def test_size_eviction_removes_lru(self, temp_dir):
        """When the cache is full the least-recently-used entry is evicted."""
        store = MemoryStore(persist_directory=temp_dir)
        store._cache_max_size = 3
        store._cache_ttl = 0

        store._get_collection("cat_a")
        store._get_collection("cat_b")
        store._get_collection("cat_c")
        assert len(store._collections) == 3

        # cat_a is LRU — adding cat_d must evict it.
        store._get_collection("cat_d")
        assert len(store._collections) == 3
        assert "cat_a" not in store._collections
        assert "cat_b" in store._collections
        assert "cat_c" in store._collections
        assert "cat_d" in store._collections

    def test_access_promotes_to_mru(self, temp_dir):
        """Re-accessing a cached entry moves it to MRU, protecting it from eviction."""
        store = MemoryStore(persist_directory=temp_dir)
        store._cache_max_size = 3
        store._cache_ttl = 0

        store._get_collection("cat_a")
        store._get_collection("cat_b")
        store._get_collection("cat_c")

        # Re-access cat_a — it moves to MRU; cat_b becomes LRU.
        store._get_collection("cat_a")

        store._get_collection("cat_d")
        assert "cat_a" in store._collections
        assert "cat_b" not in store._collections

    def test_max_size_zero_unlimited(self, temp_dir):
        """Setting max_size=0 allows unlimited cache growth."""
        store = MemoryStore(persist_directory=temp_dir)
        store._cache_max_size = 0
        store._cache_ttl = 0

        for i in range(20):
            store._get_collection(f"cat_{i}")

        assert len(store._collections) == 20

    def test_delete_category_invalidates_cache(self, temp_dir):
        """Deleting a category also removes it from the collection cache."""
        store = MemoryStore(persist_directory=temp_dir)
        store.store("cached_cat", "some content")
        assert "cached_cat" in store._collections

        store.delete_category("cached_cat")
        assert "cached_cat" not in store._collections

    def test_invalidate_cache_single_category(self, temp_dir):
        """invalidate_cache(category) removes only that entry."""
        store = MemoryStore(persist_directory=temp_dir)
        store._get_collection("cat_x")
        store._get_collection("cat_y")
        assert "cat_x" in store._collections

        store.invalidate_cache("cat_x")
        assert "cat_x" not in store._collections
        assert "cat_y" in store._collections

    def test_invalidate_cache_all(self, temp_dir):
        """invalidate_cache() with no argument clears the entire cache."""
        store = MemoryStore(persist_directory=temp_dir)
        store._get_collection("cat_x")
        store._get_collection("cat_y")

        store.invalidate_cache()
        assert len(store._collections) == 0

    def test_cache_info_reports_correct_stats(self, temp_dir):
        """cache_info() returns accurate size and configuration values."""
        store = MemoryStore(persist_directory=temp_dir)
        store._cache_max_size = 10
        store._cache_ttl = 60

        store._get_collection("col_a")
        store._get_collection("col_b")

        info = store.cache_info()
        assert info["size"] == 2
        assert info["max_size"] == 10
        assert info["ttl_seconds"] == 60
        assert "col_a" in info["entries"]
        assert "col_b" in info["entries"]
        assert info["entries"]["col_a"]["idle_seconds"] >= 0
        assert info["entries"]["col_a"]["age_seconds"] >= 0

    def test_cache_config_loaded_from_config_dict(self, temp_dir):
        """Cache TTL and max_size are read from the 'collection_cache' config key."""

        class MockConfigManager:
            def get_config(self, name, category):
                return {
                    "enabled": True,
                    "persist_directory": temp_dir,
                    "collection_cache": {
                        "ttl_seconds": 120,
                        "max_size": 25,
                    },
                }

        store = MemoryStore(
            config_manager=MockConfigManager(), persist_directory=temp_dir
        )
        assert store._cache_ttl == 120.0
        assert store._cache_max_size == 25


class TestMemoryEntry:
    """Tests for MemoryEntry dataclass."""

    def test_memory_entry_creation(self):
        """Test creating a MemoryEntry."""
        entry = MemoryEntry(
            id="test_id",
            category="test",
            content="test content",
            metadata={"key": "value"},
            timestamp="2024-01-01T00:00:00",
        )
        assert entry.id == "test_id"
        assert entry.category == "test"
        assert entry.content == "test content"

    def test_memory_entry_to_dict(self):
        """Test converting MemoryEntry to dictionary."""
        entry = MemoryEntry(
            id="test_id",
            category="test",
            content="test content",
            metadata={"key": "value"},
            timestamp="2024-01-01T00:00:00",
        )
        d = entry.to_dict()
        assert d["id"] == "test_id"
        assert d["category"] == "test"
        assert d["content"] == "test content"
        assert d["metadata"]["key"] == "value"


class TestSearchResult:
    """Tests for SearchResult dataclass."""

    def test_search_result_creation(self):
        """Test creating a SearchResult."""
        result = SearchResult(
            id="test_id",
            category="test",
            content="test content",
            metadata={"key": "value"},
            distance=0.5,
            relevance_score=0.8,
        )
        assert result.id == "test_id"
        assert result.distance == 0.5
        assert result.relevance_score == 0.8


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
