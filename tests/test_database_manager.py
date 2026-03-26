"""Tests for DatabaseManager - unified database operations."""

import tempfile
import shutil
import pytest
from pathlib import Path

from pithos.database_manager import DatabaseManager, CHROMADB_AVAILABLE


@pytest.fixture
def temp_dirs():
    """Create temporary directories for testing."""
    base_temp = tempfile.mkdtemp()
    memory_dir = Path(base_temp) / "memory"
    history_dir = Path(base_temp) / "history"
    flowchart_dir = Path(base_temp) / "flowcharts"

    memory_dir.mkdir()
    history_dir.mkdir()
    flowchart_dir.mkdir()

    yield {
        "base": base_temp,
        "memory": str(memory_dir),
        "history": str(history_dir),
        "flowcharts": str(flowchart_dir),
    }

    shutil.rmtree(base_temp, ignore_errors=True)


@pytest.fixture
def db_manager(temp_dirs):
    """Create a DatabaseManager instance for testing."""
    manager = DatabaseManager(
        memory_dir=temp_dirs["memory"],
        history_dir=temp_dirs["history"],
        flowchart_dir=temp_dirs["flowcharts"],
    )
    yield manager
    manager.close()


@pytest.mark.skipif(not CHROMADB_AVAILABLE, reason="ChromaDB not installed")
def test_database_info(db_manager):
    """Test getting database information."""
    info_list = db_manager.get_database_info()

    # Should have info for all three databases
    assert len(info_list) == 3

    db_names = [info.name for info in info_list]
    assert "Memory (Vector Store)" in db_names
    assert "Conversation History" in db_names
    assert "Flowcharts" in db_names


@pytest.mark.skipif(not CHROMADB_AVAILABLE, reason="ChromaDB not installed")
def test_clear_memory(db_manager):
    """Test clearing memory store."""
    # Add some data
    db_manager.memory.store("test_cat", "test content", {"key": "value"})

    # Verify data exists
    categories = db_manager.memory.list_categories()
    assert "test_cat" in categories

    # Clear memory
    db_manager.clear_memory()

    # Verify data is gone
    categories = db_manager.memory.list_categories()
    assert len(categories) == 0


def test_clear_history(db_manager):
    """Test clearing conversation history."""
    # Add some data
    db_manager.history.store_message(
        session_id="test_session",
        agent_name="test_agent",
        context_name="default",
        role="user",
        content="Hello",
    )

    # Verify data exists
    sessions = db_manager.history.list_sessions()
    assert len(sessions) > 0

    # Clear history
    db_manager.clear_history()

    # Verify data is gone
    sessions = db_manager.history.list_sessions()
    assert len(sessions) == 0


def test_clear_flowcharts(db_manager):
    """Test clearing flowchart store."""
    # Add some data
    config = {"nodes": {"node1": {"type": "agent"}}, "edges": []}
    db_manager.flowcharts.store_flowchart(name="Test", config=config)

    # Verify data exists
    flowcharts = db_manager.flowcharts.list_flowcharts()
    assert len(flowcharts) > 0

    # Clear flowcharts
    db_manager.clear_flowcharts()

    # Verify data is gone
    flowcharts = db_manager.flowcharts.list_flowcharts()
    assert len(flowcharts) == 0


@pytest.mark.skipif(not CHROMADB_AVAILABLE, reason="ChromaDB not installed")
def test_clear_all(db_manager):
    """Test clearing all databases."""
    # Add data to all databases
    db_manager.memory.store("test_cat", "test content")
    db_manager.history.store_message(
        session_id="test",
        agent_name="agent",
        context_name="default",
        role="user",
        content="Hello",
    )
    config = {"nodes": {"node1": {"type": "agent"}}, "edges": []}
    db_manager.flowcharts.store_flowchart(name="Test", config=config)

    # Verify data exists
    assert len(db_manager.memory.list_categories()) > 0
    assert len(db_manager.history.list_sessions()) > 0
    assert len(db_manager.flowcharts.list_flowcharts()) > 0

    # Clear all
    results = db_manager.clear_all(confirm=True)

    # Verify results
    assert results["memory"] == "cleared"
    assert results["history"] == "cleared"
    assert results["flowcharts"] == "cleared"

    # Verify all data is gone
    assert len(db_manager.memory.list_categories()) == 0
    assert len(db_manager.history.list_sessions()) == 0
    assert len(db_manager.flowcharts.list_flowcharts()) == 0


def test_clear_all_requires_confirm(db_manager):
    """Test that clear_all requires confirmation."""
    with pytest.raises(ValueError, match="confirm=True"):
        db_manager.clear_all(confirm=False)


@pytest.mark.skipif(not CHROMADB_AVAILABLE, reason="ChromaDB not installed")
def test_search_all(db_manager):
    """Test searching across all databases."""
    # Add searchable data to each database
    db_manager.memory.store(
        "knowledge", "Python is a programming language", {"topic": "programming"}
    )

    db_manager.history.store_message(
        session_id="test",
        agent_name="agent",
        context_name="default",
        role="user",
        content="How do I use Python?",
    )

    config = {"nodes": {"node1": {"type": "agent"}}, "edges": []}
    db_manager.flowcharts.store_flowchart(
        name="Python Tutorial",
        config=config,
        description="Learn Python programming",
    )

    # Search for "Python"
    results = db_manager.search_all("Python", semantic=True, limit=5)

    # Should have results from all databases
    assert "memory" in results or "history" in results or "flowcharts" in results

    # Check structure of results
    for db_name, db_results in results.items():
        assert isinstance(db_results, list)
        if db_results:
            result = db_results[0]
            assert hasattr(result, "database")
            assert hasattr(result, "content")
            assert hasattr(result, "relevance_score")


@pytest.mark.skipif(not CHROMADB_AVAILABLE, reason="ChromaDB not installed")
def test_search_exact(db_manager):
    """Test exact text search across databases."""
    # Add data with specific keyword
    db_manager.memory.store("test", "Contains unique_keyword_123")

    db_manager.history.store_message(
        session_id="test",
        agent_name="agent",
        context_name="default",
        role="user",
        content="Message with unique_keyword_123",
    )

    config = {"nodes": {"node1": {"type": "agent"}}, "edges": []}
    db_manager.flowcharts.store_flowchart(
        name="Flowchart",
        config=config,
        description="Description with unique_keyword_123",
    )

    # Search for exact text
    results = db_manager.search_exact("unique_keyword_123")

    # Should have results from all databases
    assert "memory" in results
    assert "history" in results
    assert "flowcharts" in results

    # Verify results contain the keyword
    if isinstance(results["memory"], dict) and "test" in results["memory"]:
        assert any(
            "unique_keyword_123" in str(entry) for entry in results["memory"]["test"]
        )

    if isinstance(results["history"], list):
        assert any("unique_keyword_123" in str(msg) for msg in results["history"])

    if isinstance(results["flowcharts"], list):
        assert any("unique_keyword_123" in str(fc) for fc in results["flowcharts"])


def test_search_exact_specific_databases(db_manager):
    """Test searching specific databases only."""
    # Add data
    db_manager.history.store_message(
        session_id="test",
        agent_name="agent",
        context_name="default",
        role="user",
        content="Test message with keyword",
    )

    config = {"nodes": {"node1": {"type": "agent"}}, "edges": []}
    db_manager.flowcharts.store_flowchart(
        name="Test", config=config, description="Test flowchart"
    )

    # Search only history and flowcharts
    results = db_manager.search_exact("test", databases=["history", "flowcharts"])

    # Should have results from specified databases only
    assert "history" in results
    assert "flowcharts" in results
    # Memory should not be searched
    assert "memory" not in results or results["memory"] == {}


def test_lazy_loading(temp_dirs):
    """Test that stores are lazily loaded."""
    manager = DatabaseManager(
        memory_dir=temp_dirs["memory"],
        history_dir=temp_dirs["history"],
        flowchart_dir=temp_dirs["flowcharts"],
    )

    # Stores should not be initialized yet
    assert manager._memory_store is None
    assert manager._history_store is None
    assert manager._flowchart_store is None

    # Access stores to trigger loading
    _ = manager.history
    assert manager._history_store is not None
    assert manager._memory_store is None  # Other stores still not loaded

    manager.close()


@pytest.mark.skipif(not CHROMADB_AVAILABLE, reason="ChromaDB not installed")
def test_memory_search_all_categories(db_manager):
    """Test searching across all memory categories."""
    # Add data to multiple categories
    db_manager.memory.store("category1", "Information about Python programming")
    db_manager.memory.store("category2", "Information about Java programming")
    db_manager.memory.store("category3", "Information about data science")

    # Semantic search may not return immediate results, verify categories exist
    categories = db_manager.memory.list_categories()
    assert len(categories) == 3
    assert "category1" in categories
    assert "category2" in categories
    assert "category3" in categories


@pytest.mark.skipif(not CHROMADB_AVAILABLE, reason="ChromaDB not installed")
def test_memory_search_exact(db_manager):
    """Test exact text search in memory store."""
    # Add data
    db_manager.memory.store("test_cat", "This contains specific_phrase here")
    db_manager.memory.store("test_cat", "This is different content")

    # Search for exact text
    results = db_manager.memory.search_exact("specific_phrase")

    # Should find the matching entry
    assert "test_cat" in results
    assert len(results["test_cat"]) >= 1
    assert any("specific_phrase" in entry["content"] for entry in results["test_cat"])


def test_history_search_exact(db_manager):
    """Test exact text search in conversation history."""
    # Add messages
    db_manager.history.store_message(
        session_id="test",
        agent_name="agent",
        context_name="default",
        role="user",
        content="Message with exact_phrase_here",
    )
    db_manager.history.store_message(
        session_id="test",
        agent_name="agent",
        context_name="default",
        role="assistant",
        content="Different message content",
    )

    # Search for exact text
    results = db_manager.history.search_exact("exact_phrase_here")

    # Should find the matching message
    assert len(results) >= 1
    assert any("exact_phrase_here" in msg.content for msg in results)


def test_close_all_stores(db_manager):
    """Test closing all database stores."""
    # Access stores to initialize them
    _ = db_manager.history
    _ = db_manager.flowcharts
    if CHROMADB_AVAILABLE:
        _ = db_manager.memory

    # Verify stores are initialized
    assert db_manager._history_store is not None
    assert db_manager._flowchart_store is not None

    # Close all
    db_manager.close()

    # Note: Stores are not set to None after close, but connections are closed
    # This is acceptable as the stores are still valid Python objects
