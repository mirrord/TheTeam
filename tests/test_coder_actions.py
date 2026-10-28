"""Tests for the coding-harness action parser (pithos.tools.coder.actions).

The harness now uses JSON tool calls inside fenced ``tool_call`` code blocks.
"""

from __future__ import annotations

import json

from pithos.tools.coder.actions import (
    Action,
    edit_action_to_spec,
    parse_actions,
)


def _block(payload, lang: str = "tool_call") -> str:
    """Render a tool call (dict or list) as a fenced code block."""
    return f"```{lang}\n{json.dumps(payload)}\n```"


class TestParseRun:
    def test_run_call(self) -> None:
        text = "THOUGHT: check\n" + _block(
            {"name": "run", "arguments": {"command": "python -m pytest -q"}}
        )
        actions = parse_actions(text)
        assert len(actions) == 1
        assert actions[0].kind == "run"
        assert actions[0].command == "python -m pytest -q"

    def test_run_alias_name_and_cmd_key(self) -> None:
        actions = parse_actions(_block({"name": "exec", "arguments": {"cmd": "ls"}}))
        assert actions[0].kind == "run"
        assert actions[0].command == "ls"

    def test_empty_command_ignored(self) -> None:
        assert parse_actions(_block({"name": "run", "arguments": {"command": "  "}})) == []


class TestParseEdit:
    def test_edit_replace(self) -> None:
        text = _block(
            {
                "name": "edit",
                "arguments": {
                    "path": "src/foo.py",
                    "mode": "replace",
                    "content": "print('hi')",
                },
            }
        )
        actions = parse_actions(text)
        assert len(actions) == 1
        a = actions[0]
        assert a.kind == "edit"
        assert a.path == "src/foo.py"
        assert a.mode == "replace"
        assert a.body == "print('hi')"

    def test_edit_default_mode_is_replace(self) -> None:
        actions = parse_actions(
            _block({"name": "edit", "arguments": {"path": "x.txt", "content": "c"}})
        )
        assert actions[0].mode == "replace"

    def test_edit_unknown_mode_falls_back_to_replace(self) -> None:
        actions = parse_actions(
            _block(
                {
                    "name": "edit",
                    "arguments": {"path": "x.txt", "mode": "bogus", "content": "c"},
                }
            )
        )
        assert actions[0].mode == "replace"

    def test_edit_create_mode(self) -> None:
        actions = parse_actions(
            _block(
                {
                    "name": "edit",
                    "arguments": {"path": "a b.py", "mode": "create", "content": "x"},
                }
            )
        )
        assert actions[0].path == "a b.py"
        assert actions[0].mode == "create"

    def test_edit_insert_line_parsed(self) -> None:
        actions = parse_actions(
            _block(
                {
                    "name": "edit",
                    "arguments": {
                        "path": "x.py",
                        "mode": "insert_at_line",
                        "line": 3,
                        "content": "NEW",
                    },
                }
            )
        )
        assert actions[0].line == 3
        assert actions[0].mode == "insert_at_line"

    def test_edit_patch_uses_patch_key(self) -> None:
        actions = parse_actions(
            _block(
                {
                    "name": "edit",
                    "arguments": {
                        "path": "x.py",
                        "mode": "patch",
                        "patch": "@@ -1 +1 @@",
                    },
                }
            )
        )
        assert actions[0].mode == "patch"
        assert actions[0].body == "@@ -1 +1 @@"

    def test_multiline_content_preserved(self) -> None:
        actions = parse_actions(
            _block(
                {
                    "name": "edit",
                    "arguments": {
                        "path": "x.py",
                        "mode": "create",
                        "content": "line1\n\nline3",
                    },
                }
            )
        )
        assert actions[0].body == "line1\n\nline3"

    def test_file_key_alias(self) -> None:
        actions = parse_actions(
            _block({"name": "edit", "arguments": {"file": "y.py", "content": "z"}})
        )
        assert actions[0].path == "y.py"


class TestParseSubmit:
    def test_submit_call(self) -> None:
        actions = parse_actions(
            _block({"name": "submit", "arguments": {"summary": "Done it."}})
        )
        assert len(actions) == 1
        assert actions[0].kind == "submit"
        assert actions[0].summary == "Done it."

    def test_submit_aliases(self) -> None:
        for name in ("done", "finish"):
            actions = parse_actions(
                _block({"name": name, "arguments": {"summary": "wrapped up"}})
            )
            assert actions[0].kind == "submit"
            assert actions[0].summary == "wrapped up"


class TestParsingShapes:
    def test_multiple_blocks_in_order(self) -> None:
        text = (
            "first\n"
            + _block(
                {"name": "edit", "arguments": {"path": "a.py", "content": "x"}}
            )
            + "\nthen\n"
            + _block({"name": "run", "arguments": {"command": "python a.py"}})
        )
        actions = parse_actions(text)
        assert [a.kind for a in actions] == ["edit", "run"]

    def test_json_array_in_one_block(self) -> None:
        text = _block(
            [
                {"name": "edit", "arguments": {"path": "a.py", "content": "x"}},
                {"name": "run", "arguments": {"command": "python a.py"}},
            ]
        )
        actions = parse_actions(text)
        assert [a.kind for a in actions] == ["edit", "run"]

    def test_tool_calls_wrapper(self) -> None:
        text = _block(
            {
                "tool_calls": [
                    {"name": "run", "arguments": {"command": "ls"}},
                    {"name": "submit", "arguments": {"summary": "done"}},
                ]
            }
        )
        actions = parse_actions(text)
        assert [a.kind for a in actions] == ["run", "submit"]

    def test_back_to_back_objects(self) -> None:
        body = (
            '{"name": "run", "arguments": {"command": "ls"}}\n'
            '{"name": "submit", "arguments": {"summary": "ok"}}'
        )
        actions = parse_actions(f"```tool_call\n{body}\n```")
        assert [a.kind for a in actions] == ["run", "submit"]

    def test_json_fence_lang_accepted(self) -> None:
        actions = parse_actions(
            _block({"name": "run", "arguments": {"command": "ls"}}, lang="json")
        )
        assert actions[0].kind == "run"

    def test_arguments_as_json_string(self) -> None:
        # Some models emit the arguments object as a JSON-encoded string.
        actions = parse_actions(
            _block({"name": "run", "arguments": json.dumps({"command": "ls"})})
        )
        assert actions[0].command == "ls"

    def test_flattened_arguments(self) -> None:
        # No nested "arguments": fields live directly on the call object.
        actions = parse_actions(_block({"name": "run", "command": "ls"}))
        assert actions[0].command == "ls"

    def test_tool_and_function_name_keys(self) -> None:
        a1 = parse_actions(_block({"tool": "run", "arguments": {"command": "ls"}}))
        a2 = parse_actions(_block({"function": "run", "arguments": {"command": "ls"}}))
        assert a1[0].kind == a2[0].kind == "run"

    def test_literal_newlines_in_content(self) -> None:
        # Local models often emit real newlines inside the "content" string
        # instead of escaping them as \\n; the lenient decoder must accept it.
        body = (
            '{"name": "edit", "arguments": {"path": "foo.py", "mode": "create", '
            '"content": "def main():\n    return 1\n"}}'
        )
        actions = parse_actions(f"```tool_call\n{body}\n```")
        assert len(actions) == 1
        assert actions[0].kind == "edit"
        assert actions[0].body == "def main():\n    return 1\n"

    def test_unfenced_json_fallback(self) -> None:
        text = 'THOUGHT\n{"name": "run", "arguments": {"command": "ls"}}'
        actions = parse_actions(text)
        assert actions[0].kind == "run"
        assert actions[0].command == "ls"

    def test_no_action_returns_empty(self) -> None:
        assert parse_actions("just some prose, no tool calls") == []

    def test_non_tool_fenced_json_ignored(self) -> None:
        # A plain data object without a recognised tool name is not an action.
        assert parse_actions(_block({"foo": "bar"})) == []

    def test_unknown_tool_name_ignored(self) -> None:
        assert parse_actions(_block({"name": "frobnicate", "arguments": {}})) == []

    def test_non_tool_code_fence_ignored(self) -> None:
        text = "```python\nprint('not a tool call')\n```"
        assert parse_actions(text) == []


class TestEditActionToSpec:
    def test_content_spec(self) -> None:
        action = Action(kind="edit", path="x.py", mode="replace", body="data")
        spec = edit_action_to_spec(action)
        assert spec == {"path": "x.py", "mode": "replace", "content": "data"}

    def test_patch_spec(self) -> None:
        action = Action(kind="edit", path="x.py", mode="patch", body="@@ -1 +1 @@")
        spec = edit_action_to_spec(action)
        assert spec["patch"] == "@@ -1 +1 @@"
        assert "content" not in spec

    def test_insert_line_included(self) -> None:
        action = Action(
            kind="edit", path="x.py", mode="insert_at_line", line=2, body="y"
        )
        spec = edit_action_to_spec(action)
        assert spec["line"] == 2
