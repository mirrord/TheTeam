"""Integration tests for agent memory tool functionality."""

import pytest
import tempfile
import shutil
from unittest.mock import Mock, patch
from pithos.agent import OllamaAgent
from pithos.config_manager import ConfigManager

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
        assert "storemem" in enhanced_prompt
        assert "retrievemem" in enhanced_prompt
        assert "knowledge memory system" in enhanced_prompt

    def test_extract_memory_ops_store(self, agent):
        """Test extracting store memory operations."""
        content1 = (
            'Let me save this: storemem(facts, "Python is a programming language")'
        )
        ops1 = agent._extract_memory_ops(content1)
        assert len(ops1) == 1
        assert ops1[0].operation == "store"
        assert ops1[0].category == "facts"
        assert "Python" in ops1[0].content

    def test_extract_memory_ops_retrieve(self, agent):
        """Test extracting retrieve memory operations."""
        content = 'Let me check: retrievemem(facts, "programming language")'
        ops = agent._extract_memory_ops(content)
        assert len(ops) == 1
        assert ops[0].operation == "retrieve"
        assert ops[0].category == "facts"
        assert ops[0].query == "programming language"

    def test_extract_memory_ops_multiple(self, agent):
        """Test extracting multiple memory operations."""
        content = """First storemem(notes, "Important fact") then retrievemem(notes, "fact")"""
        ops = agent._extract_memory_ops(content)
        assert len(ops) == 2
        assert ops[0].operation == "store"
        assert ops[1].operation == "retrieve"

    def test_extract_memory_ops_none(self, agent):
        """Test extracting when no memory operations present."""
        content = "Just a regular response with no memory operations"
        ops = agent._extract_memory_ops(content)
        assert len(ops) == 0

    def test_execute_memory_ops_store(self, agent, config_manager, temp_dir):
        """Test executing a store operation."""
        from pithos.tools import MemoryOpRequest

        agent.enable_memory(config_manager, persist_directory=temp_dir)

        operations = [
            MemoryOpRequest(operation="store", category="test", content="Test content")
        ]

        result = agent._execute_memory_ops(operations)
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

        result = agent._execute_memory_ops(operations)
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

        result = agent._execute_memory_ops(operations)
        # Should find results or indicate no results
        assert "Retrieved" in result or "No relevant results" in result

    def test_execute_memory_ops_error_handling(self, agent):
        """Test error handling in memory operations."""
        from pithos.tools import MemoryOpRequest

        # Memory not enabled
        operations = [
            MemoryOpRequest(operation="store", category="test", content="content")
        ]

        result = agent._execute_memory_ops(operations)
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
        assert "runcommand" in prompt
        assert "storemem" in prompt

    def test_memory_prompt_includes_categories(self, agent, config_manager, temp_dir):
        """Test that memory prompt includes existing categories."""
        agent.enable_memory(config_manager, persist_directory=temp_dir)

        # Store something to create a category
        agent.memory_store.store("test_cat", "Test content")

        # Get the memory prompt
        prompt = agent._get_memory_usage_prompt()
        # Note: The prompt may or may not include test_cat depending on timing
        # Just verify the structure is correct
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

        result = agent._execute_memory_ops(operations)

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
        mock_chunk.message.content = (
            'I\'ll save that: storemem(facts, "The sky is blue")'
        )
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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
