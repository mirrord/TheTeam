"""Tests for tool and memory operation extractors."""

import pytest
from pithos.tools import ToolCallExtractor, MemoryOpExtractor


class TestToolCallExtractor:
    """Tests for ToolCallExtractor."""

    @pytest.fixture
    def extractor(self):
        """Create a ToolCallExtractor."""
        return ToolCallExtractor()

    def test_cli_format_extraction(self, extractor):
        """Test extraction of CLI-style tool calls."""
        content = """
Let me check the Python version:
RUN: python --version
And also execute:
EXEC: git status
"""
        requests = extractor.extract(content)
        assert len(requests) == 2
        assert requests[0].command == "python --version"
        assert requests[0].format == "cli"
        assert requests[1].command == "git status"
        assert requests[1].format == "cli"

    def test_function_format_extraction(self, extractor):
        """Test extraction of function-style tool calls."""
        content = """
Let me run(python --version) and then
tool(git status) to check the repo.
"""
        requests = extractor.extract(content)
        assert len(requests) == 2
        assert requests[0].command == "python --version"
        assert requests[0].format == "function"
        assert requests[1].command == "git status"
        assert requests[1].format == "function"

    def test_bracket_format_extraction(self, extractor):
        """Test extraction of bracket-style tool calls."""
        content = """
First [RUN]python --version[/RUN] and then
<RUN>git status</RUN> to check everything.
"""
        requests = extractor.extract(content)
        assert len(requests) == 2
        assert requests[0].command == "python --version"
        assert requests[0].format == "bracket"
        assert requests[1].command == "git status"
        assert requests[1].format == "bracket"

    def test_legacy_format_extraction(self, extractor):
        """Test extraction of legacy runcommand() format."""
        content = 'Let me check: runcommand("python --version")'
        requests = extractor.extract(content)
        assert len(requests) == 1
        assert requests[0].command == "python --version"
        assert requests[0].format == "legacy"

    def test_mixed_formats(self, extractor):
        """Test extraction with multiple format types."""
        content = """
First runcommand("python --version")
Then RUN: git status
And finally run(echo done)
"""
        requests = extractor.extract(content)
        assert len(requests) == 3

        # Check all three formats are present (order doesn't matter)
        formats = [req.format for req in requests]
        assert "legacy" in formats
        assert "cli" in formats
        assert "function" in formats

        # Check all commands are present
        commands = [req.command for req in requests]
        assert "python --version" in commands
        assert "git status" in commands
        assert "echo done" in commands

    def test_no_extractions(self, extractor):
        """Test content with no tool calls."""
        content = "This is just regular text without any tool calls."
        requests = extractor.extract(content)
        assert len(requests) == 0

    def test_usage_examples(self, extractor):
        """Test that usage examples are properly formatted."""
        examples = extractor.get_usage_examples()
        assert "RUN:" in examples
        assert "run(" in examples
        assert "[RUN]" in examples
        assert "runcommand" in examples


class TestMemoryOpExtractor:
    """Tests for MemoryOpExtractor."""

    @pytest.fixture
    def extractor(self):
        """Create a MemoryOpExtractor."""
        return MemoryOpExtractor()

    def test_cli_store_extraction(self, extractor):
        """Test extraction of CLI-style store operations."""
        content = """
Let me save this:
STORE[facts]: Python is a programming language
"""
        requests = extractor.extract(content)
        assert len(requests) == 1
        assert requests[0].operation == "store"
        assert requests[0].category == "facts"
        assert requests[0].content == "Python is a programming language"
        assert requests[0].format == "cli"

    def test_cli_retrieve_extraction(self, extractor):
        """Test extraction of CLI-style retrieve operations."""
        content = """
Let me search:
RETRIEVE[facts]: programming languages
"""
        requests = extractor.extract(content)
        assert len(requests) == 1
        assert requests[0].operation == "retrieve"
        assert requests[0].category == "facts"
        assert requests[0].query == "programming languages"
        assert requests[0].format == "cli"

    def test_function_store_extraction(self, extractor):
        """Test extraction of function-style store operations."""
        content = "Let me store(facts, Python is great) for later."
        requests = extractor.extract(content)
        assert len(requests) == 1
        assert requests[0].operation == "store"
        assert requests[0].category == "facts"
        assert requests[0].content == "Python is great"
        assert requests[0].format == "function"

    def test_function_retrieve_extraction(self, extractor):
        """Test extraction of function-style retrieve operations."""
        content = "Let me retrieve(facts, Python information) now."
        requests = extractor.extract(content)
        assert len(requests) == 1
        assert requests[0].operation == "retrieve"
        assert requests[0].category == "facts"
        assert requests[0].query == "Python information"
        assert requests[0].format == "function"

    def test_legacy_store_extraction(self, extractor):
        """Test extraction of legacy storemem() format."""
        content = 'Let me save: storemem(facts, "Python is great")'
        requests = extractor.extract(content)
        assert len(requests) == 1
        assert requests[0].operation == "store"
        assert requests[0].category == "facts"
        assert requests[0].content == "Python is great"
        assert requests[0].format == "legacy"

    def test_legacy_retrieve_extraction(self, extractor):
        """Test extraction of legacy retrievemem() format."""
        content = 'Let me search: retrievemem(facts, "Python info")'
        requests = extractor.extract(content)
        assert len(requests) == 1
        assert requests[0].operation == "retrieve"
        assert requests[0].category == "facts"
        assert requests[0].query == "Python info"
        assert requests[0].format == "legacy"

    def test_mixed_operations(self, extractor):
        """Test extraction with multiple operations."""
        content = """
First STORE[facts]: Important information
Then retrieve(facts, search term)
And storemem(notes, "More data")
"""
        requests = extractor.extract(content)
        assert len(requests) == 3
        assert requests[0].operation == "store"
        assert requests[1].operation == "retrieve"
        assert requests[2].operation == "store"

    def test_no_extractions(self, extractor):
        """Test content with no memory operations."""
        content = "This is just regular text without any memory ops."
        requests = extractor.extract(content)
        assert len(requests) == 0

    def test_usage_examples(self, extractor):
        """Test that usage examples are properly formatted."""
        examples = extractor.get_usage_examples()
        assert "STORE[" in examples
        assert "RETRIEVE[" in examples
        assert "store(" in examples
        assert "retrieve(" in examples
