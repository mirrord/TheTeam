"""Tests for tool and memory operation extractors."""

import pytest
from pithos.tools import ToolCallExtractor, MemoryOpExtractor


class TestToolCallExtractor:
    """Tests for ToolCallExtractor."""

    @pytest.fixture
    def extractor(self):
        """Create a ToolCallExtractor."""
        return ToolCallExtractor()

    def test_cli_format_not_extracted(self, extractor):
        """CLI-style syntax is no longer supported and must be ignored."""
        content = """
Let me check the Python version:
RUN: python --version
And also execute:
EXEC: git status
"""
        requests = extractor.extract(content)
        assert requests == []

    def test_function_format_not_extracted(self, extractor):
        """Function-style syntax is no longer supported and must be ignored."""
        content = """
Let me run(python --version) and then
tool(git status) to check the repo.
"""
        requests = extractor.extract(content)
        assert requests == []

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

    def test_legacy_format_not_extracted(self, extractor):
        """Legacy runcommand() syntax is no longer supported and must be ignored."""
        content = 'Let me check: runcommand("python --version")'
        requests = extractor.extract(content)
        assert requests == []

    def test_exec_bracket_extraction(self, extractor):
        """Test extraction of [EXEC]...[/EXEC] bracket calls."""
        content = "Now [EXEC]echo done[/EXEC] to finish."
        requests = extractor.extract(content)
        assert len(requests) == 1
        assert requests[0].command == "echo done"
        assert requests[0].format == "bracket"

    def test_no_extractions(self, extractor):
        """Test content with no tool calls."""
        content = "This is just regular text without any tool calls."
        requests = extractor.extract(content)
        assert len(requests) == 0

    def test_usage_examples(self, extractor):
        """Test that usage examples advertise only the bracket format."""
        examples = extractor.get_usage_examples()
        assert "[RUN]" in examples
        assert "[/RUN]" in examples
        # Removed formats must not be advertised.
        assert "RUN:" not in examples
        assert "run(" not in examples
        assert "runcommand" not in examples


class TestMemoryOpExtractor:
    """Tests for MemoryOpExtractor."""

    @pytest.fixture
    def extractor(self):
        """Create a default (cli) MemoryOpExtractor."""
        return MemoryOpExtractor()

    @pytest.fixture
    def extractor_function(self):
        """Create a function-format MemoryOpExtractor."""
        return MemoryOpExtractor("function")

    @pytest.fixture
    def extractor_legacy(self):
        """Create a legacy-format MemoryOpExtractor."""
        return MemoryOpExtractor("legacy")

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

    def test_function_store_extraction(self, extractor_function):
        """Test extraction of function-style store operations."""
        content = "Let me store(facts, Python is great) for later."
        requests = extractor_function.extract(content)
        assert len(requests) == 1
        assert requests[0].operation == "store"
        assert requests[0].category == "facts"
        assert requests[0].content == "Python is great"
        assert requests[0].format == "function"

    def test_function_retrieve_extraction(self, extractor_function):
        """Test extraction of function-style retrieve operations."""
        content = "Let me retrieve(facts, Python information) now."
        requests = extractor_function.extract(content)
        assert len(requests) == 1
        assert requests[0].operation == "retrieve"
        assert requests[0].category == "facts"
        assert requests[0].query == "Python information"
        assert requests[0].format == "function"

    def test_legacy_store_extraction(self, extractor_legacy):
        """Test extraction of legacy storemem() format."""
        content = 'Let me save: storemem(facts, "Python is great")'
        requests = extractor_legacy.extract(content)
        assert len(requests) == 1
        assert requests[0].operation == "store"
        assert requests[0].category == "facts"
        assert requests[0].content == "Python is great"
        assert requests[0].format == "legacy"

    def test_legacy_retrieve_extraction(self, extractor_legacy):
        """Test extraction of legacy retrievemem() format."""
        content = 'Let me search: retrievemem(facts, "Python info")'
        requests = extractor_legacy.extract(content)
        assert len(requests) == 1
        assert requests[0].operation == "retrieve"
        assert requests[0].category == "facts"
        assert requests[0].query == "Python info"
        assert requests[0].format == "legacy"

    def test_cli_ignores_other_formats(self, extractor):
        """CLI extractor only matches CLI syntax; other formats are ignored."""
        content = """
First STORE[facts]: Important information
Then retrieve(facts, search term)
And storemem(notes, "More data")
"""
        requests = extractor.extract(content)
        # Only the CLI STORE[...] line is matched
        assert len(requests) == 1
        assert requests[0].operation == "store"
        assert requests[0].format == "cli"

    def test_cli_multiple_operations(self, extractor):
        """CLI extractor finds multiple store and retrieve ops."""
        content = """
STORE[facts]: Important information
RETRIEVE[facts]: search term
STORE[notes]: More data
"""
        requests = extractor.extract(content)
        assert len(requests) == 3
        # stores are collected first, then retrieves
        ops = {(r.operation, r.category) for r in requests}
        assert ("store", "facts") in ops
        assert ("store", "notes") in ops
        assert ("retrieve", "facts") in ops

    def test_no_extractions(self, extractor):
        """Test content with no memory operations."""
        content = "This is just regular text without any memory ops."
        requests = extractor.extract(content)
        assert len(requests) == 0

    def test_usage_examples_cli(self, extractor):
        """CLI extractor usage examples contain only CLI-format strings."""
        examples = extractor.get_usage_examples()
        assert "STORE[" in examples
        assert "RETRIEVE[" in examples
        assert "store(" not in examples
        assert "storemem(" not in examples

    def test_usage_examples_function(self, extractor_function):
        """Function extractor usage examples contain only function-format strings."""
        examples = extractor_function.get_usage_examples()
        assert "store(" in examples
        assert "retrieve(" in examples
        assert "STORE[" not in examples
        assert "storemem(" not in examples

    def test_usage_examples_legacy(self, extractor_legacy):
        """Legacy extractor usage examples contain only legacy-format strings."""
        examples = extractor_legacy.get_usage_examples()
        assert "storemem(" in examples
        assert "retrievemem(" in examples
        assert "STORE[" not in examples
        assert "store(" not in examples

    def test_invalid_format_raises(self):
        """Unknown format raises ValueError at construction time."""
        with pytest.raises(ValueError, match="Unknown memory operation format"):
            MemoryOpExtractor("bracket")
