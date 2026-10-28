"""Tests for tool and memory operation extractors."""

import pytest
from pithos.tools import ToolCallExtractor, MemoryOpExtractor


class TestToolCallExtractor:
    """Tests for ToolCallExtractor (JSON tool-call format)."""

    @pytest.fixture
    def extractor(self):
        """Create a ToolCallExtractor."""
        return ToolCallExtractor()

    def _block(self, command):
        return (
            '```tool_call\n'
            '{"name": "run", "arguments": {"command": "' + command + '"}}\n'
            '```'
        )

    def test_bracket_format_not_extracted(self, extractor):
        """The old [RUN]...[/RUN] bracket syntax is no longer supported."""
        content = "First [RUN]python --version[/RUN] and then <RUN>git status</RUN>."
        requests = extractor.extract(content)
        assert requests == []

    def test_cli_format_not_extracted(self, extractor):
        """CLI-style syntax is not supported and must be ignored."""
        content = "RUN: python --version\nEXEC: git status\n"
        requests = extractor.extract(content)
        assert requests == []

    def test_function_format_not_extracted(self, extractor):
        """Bare function-style syntax is not supported and must be ignored."""
        content = "Let me run(python --version) and then tool(git status)."
        requests = extractor.extract(content)
        assert requests == []

    def test_json_run_extraction(self, extractor):
        """A run tool call is extracted into a command string."""
        content = "Check:\n" + self._block("python --version")
        requests = extractor.extract(content)
        assert len(requests) == 1
        assert requests[0].command == "python --version"
        assert requests[0].format == "tool_call"

    def test_multiple_calls_extracted_in_order(self, extractor):
        content = self._block("python --version") + "\nthen\n" + self._block("git status")
        requests = extractor.extract(content)
        assert [r.command for r in requests] == ["python --version", "git status"]

    def test_json_array_in_one_block(self, extractor):
        content = (
            '```tool_call\n'
            '[{"name": "run", "arguments": {"command": "echo a"}}, '
            '{"name": "run", "arguments": {"command": "echo b"}}]\n'
            '```'
        )
        requests = extractor.extract(content)
        assert [r.command for r in requests] == ["echo a", "echo b"]

    def test_directly_named_tool_is_flattened(self, extractor):
        """A call naming a tool directly becomes '<tool> <args>'."""
        content = (
            '```tool_call\n'
            '{"name": "git", "arguments": {"command": "status"}}\n'
            '```'
        )
        requests = extractor.extract(content)
        assert requests[0].command == "git status"

    def test_raw_text_is_full_block(self, extractor):
        """raw_text spans the whole fenced block so it can be stripped."""
        block = self._block("echo done")
        requests = extractor.extract("Now " + block + " to finish.")
        assert requests[0].raw_text == block

    def test_no_extractions(self, extractor):
        """Content with no tool calls yields nothing."""
        content = "This is just regular text without any tool calls."
        assert extractor.extract(content) == []

    def test_non_tool_json_block_ignored(self, extractor):
        """A json block without a recognised tool name is not a call."""
        content = '```json\n{"foo": "bar"}\n```'
        assert extractor.extract(content) == []

    def test_usage_examples(self, extractor):
        """Usage examples advertise only the JSON tool-call format."""
        examples = extractor.get_usage_examples()
        assert "tool_call" in examples
        assert '"name": "run"' in examples
        # Removed formats must not be advertised.
        assert "[RUN]" not in examples
        assert "RUN:" not in examples
        assert "runcommand" not in examples


class TestMemoryOpExtractor:
    """Tests for MemoryOpExtractor (JSON ``memory`` tool-call format)."""

    @pytest.fixture
    def extractor(self):
        """Create a MemoryOpExtractor."""
        return MemoryOpExtractor()

    def _store(self, category, content):
        return (
            '```tool_call\n{"name": "memory", "arguments": {"operation": "store", '
            '"category": "' + category + '", "content": "' + content + '"}}\n```'
        )

    def _retrieve(self, category, query):
        return (
            '```tool_call\n{"name": "memory", "arguments": {"operation": "retrieve", '
            '"category": "' + category + '", "query": "' + query + '"}}\n```'
        )

    def test_store_extraction(self, extractor):
        content = "Let me save this:\n" + self._store(
            "facts", "Python is a programming language"
        )
        requests = extractor.extract(content)
        assert len(requests) == 1
        assert requests[0].operation == "store"
        assert requests[0].category == "facts"
        assert requests[0].content == "Python is a programming language"
        assert requests[0].format == "tool_call"

    def test_retrieve_extraction(self, extractor):
        content = "Let me search:\n" + self._retrieve("facts", "programming languages")
        requests = extractor.extract(content)
        assert len(requests) == 1
        assert requests[0].operation == "retrieve"
        assert requests[0].category == "facts"
        assert requests[0].query == "programming languages"
        assert requests[0].format == "tool_call"

    def test_sub_name_operation(self, extractor):
        """``memory:retrieve`` sets the operation from the sub-name."""
        content = (
            '```tool_call\n{"name": "memory:retrieve", "arguments": '
            '{"category": "facts", "query": "x"}}\n```'
        )
        requests = extractor.extract(content)
        assert requests[0].operation == "retrieve"
        assert requests[0].query == "x"

    def test_remember_alias(self, extractor):
        """The ``remember`` alias implies a store operation."""
        content = (
            '```tool_call\n{"name": "remember", "arguments": '
            '{"category": "notes", "content": "hi"}}\n```'
        )
        requests = extractor.extract(content)
        assert requests[0].operation == "store"
        assert requests[0].content == "hi"

    def test_ignores_non_memory_tool_calls(self, extractor):
        """Non-memory tool calls are not memory operations."""
        content = '```tool_call\n{"name": "run", "arguments": {"command": "ls"}}\n```'
        assert extractor.extract(content) == []

    def test_multiple_operations(self, extractor):
        content = (
            self._store("facts", "Important information") + "\n"
            + self._retrieve("facts", "search term") + "\n"
            + self._store("notes", "More data")
        )
        requests = extractor.extract(content)
        assert len(requests) == 3
        ops = {(r.operation, r.category) for r in requests}
        assert ("store", "facts") in ops
        assert ("store", "notes") in ops
        assert ("retrieve", "facts") in ops

    def test_no_extractions(self, extractor):
        content = "This is just regular text without any memory ops."
        assert extractor.extract(content) == []

    def test_usage_examples(self, extractor):
        examples = extractor.get_usage_examples()
        assert "tool_call" in examples
        assert '"name": "memory"' in examples
        assert '"operation": "store"' in examples
        assert '"operation": "retrieve"' in examples
        # Removed legacy formats must not be advertised.
        assert "STORE[" not in examples
        assert "storemem(" not in examples
