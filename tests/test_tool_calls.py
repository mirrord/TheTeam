"""Tests for the unified JSON tool-call parser (pithos.tools.tool_calls)."""

from __future__ import annotations

import json

from pithos.tools.tool_calls import (
    ParsedToolCall,
    extract_tool_calls,
    safe_prefix_len,
)


def _block(payload, lang: str = "tool_call") -> str:
    return f"```{lang}\n{json.dumps(payload)}\n```"


class TestExtractToolCalls:
    def test_single_call(self) -> None:
        calls = extract_tool_calls(
            "THOUGHT\n" + _block({"name": "run", "arguments": {"command": "ls"}})
        )
        assert len(calls) == 1
        assert isinstance(calls[0], ParsedToolCall)
        assert calls[0].name == "run"
        assert calls[0].arguments["command"] == "ls"

    def test_raw_is_full_block(self) -> None:
        block = _block({"name": "run", "arguments": {"command": "ls"}})
        calls = extract_tool_calls("before " + block + " after")
        assert calls[0].raw == block

    def test_array_in_one_block(self) -> None:
        calls = extract_tool_calls(
            _block(
                [
                    {"name": "run", "arguments": {"command": "a"}},
                    {"name": "edit", "arguments": {"path": "x"}},
                ]
            )
        )
        assert [c.name for c in calls] == ["run", "edit"]

    def test_tool_calls_wrapper(self) -> None:
        calls = extract_tool_calls(
            _block({"tool_calls": [{"name": "run", "arguments": {"command": "a"}}]})
        )
        assert calls[0].name == "run"

    def test_flattened_arguments_merged(self) -> None:
        calls = extract_tool_calls(_block({"name": "run", "command": "ls"}))
        assert calls[0].arguments["command"] == "ls"

    def test_arguments_json_string(self) -> None:
        calls = extract_tool_calls(
            _block({"name": "run", "arguments": json.dumps({"command": "ls"})})
        )
        assert calls[0].arguments["command"] == "ls"

    def test_literal_newlines_tolerated(self) -> None:
        body = (
            '{"name": "edit", "arguments": {"path": "f.py", '
            '"content": "a\nb\n"}}'
        )
        calls = extract_tool_calls(f"```tool_call\n{body}\n```")
        assert calls[0].arguments["content"] == "a\nb\n"

    def test_json_lang_accepted(self) -> None:
        calls = extract_tool_calls(
            _block({"name": "run", "arguments": {"command": "ls"}}, lang="json")
        )
        assert calls[0].name == "run"

    def test_non_tool_lang_ignored(self) -> None:
        assert extract_tool_calls("```python\nprint('hi')\n```") == []

    def test_non_tool_object_ignored(self) -> None:
        assert extract_tool_calls(_block({"foo": "bar"})) == []

    def test_unfenced_fallback(self) -> None:
        calls = extract_tool_calls(
            'THOUGHT {"name": "run", "arguments": {"command": "ls"}}'
        )
        assert calls[0].name == "run"

    def test_no_calls(self) -> None:
        assert extract_tool_calls("just prose") == []


class TestSafePrefixLen:
    def test_plain_text_all_safe(self) -> None:
        buf = "hello world, nothing special"
        assert safe_prefix_len(buf) == len(buf)

    def test_holds_back_from_tool_fence(self) -> None:
        buf = "answer: \n```tool_call\n{"
        # Everything from the opening fence is held back.
        assert safe_prefix_len(buf) == buf.index("```")

    def test_holds_back_trailing_partial_backticks(self) -> None:
        assert safe_prefix_len("text so far ``") == len("text so far ")
        assert safe_prefix_len("text so far `") == len("text so far ")

    def test_incomplete_info_string_held(self) -> None:
        buf = "prose\n```tool_ca"
        assert safe_prefix_len(buf) == buf.index("```")

    def test_bare_fence_held(self) -> None:
        buf = "prose\n```\n{"
        assert safe_prefix_len(buf) == buf.index("```")

    def test_json_fence_held(self) -> None:
        buf = "prose\n```json\n{"
        assert safe_prefix_len(buf) == buf.index("```")

    def test_ordinary_code_fence_released(self) -> None:
        buf = "here:\n```python\nprint('hi')\n```\nmore"
        # A closed, non-tool fence does not hold anything back.
        assert safe_prefix_len(buf) == len(buf)

    def test_ordinary_fence_then_tool_fence(self) -> None:
        buf = "```python\nx=1\n```\nnow\n```tool_call\n{"
        assert safe_prefix_len(buf) == buf.rindex("```tool_call")
