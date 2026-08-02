"""Integration tests for agent memory tool functionality."""

import pytest
import tempfile
import shutil
from unittest.mock import Mock, MagicMock, patch
from pithos.agent import OllamaAgent
from pithos.config_manager import ConfigManager
from pithos.tools.memory_ops import MemoryOpExtractor

# Skip all tests if ChromaDB is not available
try:
    from pithos.tools.memory_tool import CHROMADB_AVAILABLE
except ImportError:
    CHROMADB_AVAILABLE = False

pytestmark = pytest.mark.skipif(not CHROMADB_AVAILABLE, reason="ChromaDB not installed")


class TestAgentMemoryIntegration:
    """Tests for agent memory tool integration."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for testing."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)

    @pytest.fixture
    def agent(self):
        """Create a test agent."""
        return OllamaAgent(
            default_model="test-model", system_prompt="You are a helpful assistant."
        )

    @pytest.fixture
    def config_manager(self, tmp_path):
        """Create a ConfigManager with test configuration."""
        config_dir = tmp_path / "configs" / "tools"
        config_dir.mkdir(parents=True)

        config_file = config_dir / "memory_config.yaml"
        config_content = """
enabled: true
persist_directory: "./data/memory"
max_results: 5
similarity_threshold: 0.7
default_metadata:
  source: "test"
"""
        config_file.write_text(config_content)

        return ConfigManager(str(tmp_path / "configs"))

    def test_enable_memory(self, agent, config_manager, temp_dir):
        """Test enabling memory for an agent."""
        agent.enable_memory(config_manager, persist_directory=temp_dir)

        assert agent.memory_enabled is True
        assert agent.memory_store is not None

    def test_memory_prompt_enhancement(self, agent, config_manager, temp_dir):
        """Test that memory prompt is added to system prompt."""
        initial_prompt = agent.contexts["default"].get_system_prompt()
        agent.enable_memory(config_manager, persist_directory=temp_dir)

        enhanced_prompt = agent.contexts["default"].get_system_prompt()
        assert len(enhanced_prompt) > len(initial_prompt)
        assert "STORE[" in enhanced_prompt
        assert "RETRIEVE[" in enhanced_prompt
        assert "knowledge memory system" in enhanced_prompt

    def test_extract_memory_ops_store(self, agent):
        """Test extracting store memory operations."""
        extractor = MemoryOpExtractor("legacy")
        content1 = (
            'Let me save this: storemem(facts, "Python is a programming language")'
        )
        ops1 = extractor.extract(content1)
        assert len(ops1) == 1
        assert ops1[0].operation == "store"
        assert ops1[0].category == "facts"
        assert "Python" in ops1[0].content

    def test_extract_memory_ops_retrieve(self, agent):
        """Test extracting retrieve memory operations."""
        extractor = MemoryOpExtractor("legacy")
        content = 'Let me check: retrievemem(facts, "programming language")'
        ops = extractor.extract(content)
        assert len(ops) == 1
        assert ops[0].operation == "retrieve"
        assert ops[0].category == "facts"
        assert ops[0].query == "programming language"

    def test_extract_memory_ops_multiple(self, agent):
        """Test extracting multiple memory operations."""
        extractor = MemoryOpExtractor("legacy")
        content = """First storemem(notes, "Important fact") then retrievemem(notes, "fact")"""
        ops = extractor.extract(content)
        assert len(ops) == 2
        assert ops[0].operation == "store"
        assert ops[1].operation == "retrieve"

    def test_extract_memory_ops_none(self, agent):
        """Test extracting when no memory operations present."""
        extractor = MemoryOpExtractor("legacy")
        content = "Just a regular response with no memory operations"
        ops = extractor.extract(content)
        assert len(ops) == 0

    def test_execute_memory_ops_store(self, agent, config_manager, temp_dir):
        """Test executing a store operation."""
        from pithos.tools import MemoryOpRequest

        agent.enable_memory(config_manager, persist_directory=temp_dir)

        operations = [
            MemoryOpRequest(operation="store", category="test", content="Test content")
        ]

        result = agent._memory_provider._execute_ops(
            operations, agent.memory_store, agent.metrics
        )
        assert "Stored in test" in result
        assert "Test content" in result

    def test_execute_memory_ops_retrieve(self, agent, config_manager, temp_dir):
        """Test executing a retrieve operation."""
        from pithos.tools import MemoryOpRequest

        agent.enable_memory(config_manager, persist_directory=temp_dir)

        # First store something
        agent.memory_store.store("test", "Python is a programming language")

        # Then retrieve
        operations = [
            MemoryOpRequest(operation="retrieve", category="test", query="Python")
        ]

        result = agent._memory_provider._execute_ops(
            operations, agent.memory_store, agent.metrics
        )
        assert "Retrieved" in result or "No relevant results" in result

    def test_execute_memory_ops_retrieve_with_results(
        self, agent, config_manager, temp_dir
    ):
        """Test retrieving with actual results."""
        from pithos.tools import MemoryOpRequest

        agent.enable_memory(config_manager, persist_directory=temp_dir)

        # Store some very similar data
        agent.memory_store.store("languages", "Python programming language")
        agent.memory_store.store("languages", "Python is a high-level language")
        agent.memory_store.store("languages", "Python for beginners")

        # Retrieve with very similar query
        operations = [
            MemoryOpRequest(
                operation="retrieve", category="languages", query="Python programming"
            )
        ]

        result = agent._memory_provider._execute_ops(
            operations, agent.memory_store, agent.metrics
        )
        # Should find results or indicate no results
        assert "Retrieved" in result or "No relevant results" in result

    def test_execute_memory_ops_error_handling(self, agent):
        """Test error handling in memory operations when memory is not enabled."""
        from pithos.tools import MemoryOpRequest
        from pithos.tools.memory_provider import MemoryToolProvider

        # Use provider directly with no memory_store
        provider = MemoryToolProvider()
        operations = [
            MemoryOpRequest(operation="store", category="test", content="content")
        ]

        result = provider._execute_ops(operations, None)
        assert "not available" in result

    def test_memory_with_tools_enabled(self, agent, config_manager, temp_dir):
        """Test that memory and tools can be enabled together."""
        # Enable tools first
        agent.enable_tools(config_manager)
        assert agent.tools_enabled is True

        # Then enable memory
        agent.enable_memory(config_manager, persist_directory=temp_dir)
        assert agent.memory_enabled is True

        # Both should be in the prompt
        prompt = agent.contexts["default"].get_system_prompt()
        assert "[RUN]" in prompt
        assert "STORE[" in prompt

    def test_memory_prompt_includes_categories(self, agent, config_manager, temp_dir):
        """Test that memory prompt is injected into the context system prompt."""
        agent.enable_memory(config_manager, persist_directory=temp_dir)

        # Verify the prompt was injected at enable_memory() time
        prompt = agent.contexts["default"].get_system_prompt()
        assert "knowledge memory system" in prompt
        assert "storemem" in prompt or "STORE" in prompt
        assert "retrievemem" in prompt or "RETRIEVE" in prompt

    def test_format_memory_results(self, agent, config_manager, temp_dir):
        """Test formatting of memory operation results."""
        from pithos.tools import MemoryOpRequest

        agent.enable_memory(config_manager, persist_directory=temp_dir)

        # Store multiple items
        agent.memory_store.store("docs", "First document about Python")
        agent.memory_store.store("docs", "Second document about JavaScript")
        agent.memory_store.store("docs", "Third document about TypeScript")
        agent.memory_store.store("docs", "Fourth document about Rust")

        # Retrieve and check formatting
        operations = [
            MemoryOpRequest(operation="retrieve", category="docs", query="document")
        ]

        result = agent._memory_provider._execute_ops(
            operations, agent.memory_store, agent.metrics
        )

        # Should show top results with scores
        assert "Score:" in result or "No relevant results" in result

    @patch("pithos.agent.ollama_agent.chat")
    def test_memory_in_conversation_flow(
        self, mock_chat, agent, config_manager, temp_dir
    ):
        """Test memory operations in actual conversation flow."""
        agent.enable_memory(config_manager, persist_directory=temp_dir)

        # Mock LLM response with memory operation
        mock_chunk = Mock()
        mock_chunk.message.content = "I'll save that: STORE[facts]: The sky is blue"
        mock_chat.return_value = iter([mock_chunk])

        # Send a message
        response = agent.send("Remember that the sky is blue")

        # Verify memory operation was extracted and executed
        # The system message with memory result should be added
        context = agent.contexts["default"]
        messages = context.message_history

        # Look for the memory operation result in messages
        memory_messages = [
            m for m in messages if "Stored in facts" in m.get("content", "")
        ]
        assert len(memory_messages) > 0


class TestMemoryConfiguration:
    """Tests for memory configuration handling."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for testing."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)

    def test_default_memory_config(self, temp_dir):
        """Test default memory configuration."""
        from pithos.tools import MemoryStore

        store = MemoryStore(config_manager=None, persist_directory=temp_dir)
        assert store.config["enabled"] is True
        assert store.config["max_results"] == 10
        assert store.config["similarity_threshold"] == 0.5

    def test_custom_memory_config(self, temp_dir):
        """Test custom memory configuration."""
        from pithos.tools import MemoryStore

        class MockConfigManager:
            def get_config(self, name, category):
                return {
                    "enabled": True,
                    "max_results": 20,
                    "similarity_threshold": 0.85,
                }

        store = MemoryStore(
            config_manager=MockConfigManager(), persist_directory=temp_dir
        )
        assert store.config["max_results"] == 20
        assert store.config["similarity_threshold"] == 0.85


class TestMemoryToolProviderUnit:
    """Unit tests for MemoryToolProvider._execute_ops and extract_and_execute."""

    def _make_provider(self):
        from pithos.tools.memory_provider import MemoryToolProvider

        return MemoryToolProvider()

    def test_execute_ops_no_memory_store(self):
        """Returns error message when memory_store is None."""
        provider = self._make_provider()
        from pithos.tools import MemoryOpRequest

        ops = [MemoryOpRequest(operation="store", category="x", content="y")]
        result = provider._execute_ops(ops, None)
        assert "not available" in result

    def test_execute_ops_store_success(self):
        """Formats store result correctly."""
        from pithos.tools import MemoryOpRequest

        mock_store = MagicMock()
        mock_store.store.return_value = "abc123"
        mock_store.tag_suggestions_enabled = False

        provider = self._make_provider()
        ops = [
            MemoryOpRequest(operation="store", category="facts", content="Sky is blue")
        ]
        result = provider._execute_ops(ops, mock_store)

        assert "Stored in facts" in result
        assert "Sky is blue" in result
        assert "abc123" in result

    def test_execute_ops_store_records_metrics(self):
        """Records metric when store succeeds."""
        from pithos.tools import MemoryOpRequest

        mock_store = MagicMock()
        mock_store.store.return_value = "id1"
        mock_store.tag_suggestions_enabled = False

        mock_metrics = MagicMock()

        provider = self._make_provider()
        ops = [MemoryOpRequest(operation="store", category="x", content="data")]
        provider._execute_ops(ops, mock_store, mock_metrics)

        mock_metrics.record_memory_store.assert_called_once()

    def test_execute_ops_retrieve_no_results(self):
        """Formats no-results retrieve message correctly."""
        from pithos.tools import MemoryOpRequest

        mock_store = MagicMock()
        mock_store.retrieve.return_value = []

        provider = self._make_provider()
        ops = [MemoryOpRequest(operation="retrieve", category="notes", query="python")]
        result = provider._execute_ops(ops, mock_store)

        assert "No relevant results" in result

    def test_execute_ops_retrieve_with_results(self):
        """Formats retrieve result with scores correctly."""
        from pithos.tools import MemoryOpRequest
        from pithos.tools.memory_tool import SearchResult

        sr = SearchResult(
            id="1",
            category="notes",
            content="Python is a language",
            metadata={},
            distance=0.1,
            relevance_score=0.9,
        )
        mock_store = MagicMock()
        mock_store.retrieve.return_value = [sr]

        provider = self._make_provider()
        ops = [MemoryOpRequest(operation="retrieve", category="notes", query="Python")]
        result = provider._execute_ops(ops, mock_store)

        assert "Retrieved 1 results" in result
        assert "0.90" in result
        assert "Python is a language" in result

    def test_execute_ops_empty_content_store(self):
        """Store with no content returns hint message."""
        from pithos.tools import MemoryOpRequest

        mock_store = MagicMock()
        provider = self._make_provider()
        ops = [MemoryOpRequest(operation="store", category="x", content="")]
        result = provider._execute_ops(ops, mock_store)

        assert "No content provided" in result
        mock_store.store.assert_not_called()

    def test_execute_ops_empty_query_retrieve(self):
        """Retrieve with no query returns hint message."""
        from pithos.tools import MemoryOpRequest

        mock_store = MagicMock()
        provider = self._make_provider()
        ops = [MemoryOpRequest(operation="retrieve", category="x", query="")]
        result = provider._execute_ops(ops, mock_store)

        assert "No query provided" in result
        mock_store.retrieve.assert_not_called()

    def test_extract_and_execute_returns_none_for_no_ops(self):
        """Returns None when response has no memory operations."""
        provider = self._make_provider()

        mock_agent = MagicMock()
        mock_agent.memory_store = MagicMock()
        mock_agent.metrics = None

        result = provider.extract_and_execute("No ops here.", mock_agent)
        assert result is None

    def test_extract_and_execute_processes_ops(self):
        """Extracts and executes ops found in response text."""
        provider = self._make_provider()

        mock_store = MagicMock()
        mock_store.store.return_value = "id99"
        mock_store.tag_suggestions_enabled = False

        mock_agent = MagicMock()
        mock_agent.memory_store = mock_store
        mock_agent.metrics = None

        result = provider.extract_and_execute(
            "Saving this: STORE[facts]: Important thing", mock_agent
        )
        assert result is not None
        assert "Stored in facts" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
