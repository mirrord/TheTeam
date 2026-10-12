"""Tests for the coding-harness action parser (pithos.tools.coder.actions)."""

from __future__ import annotations

from pithos.tools.coder.actions import (
    Action,
    edit_action_to_spec,
    parse_actions,
)


class TestParseActions:
    def test_run_block(self) -> None:
        actions = parse_actions("THOUGHT: check\n[RUN]python -m pytest -q[/RUN]")
        assert len(actions) == 1
        assert actions[0].kind == "run"
        assert actions[0].command == "python -m pytest -q"

    def test_exec_alias(self) -> None:
        actions = parse_actions("[EXEC]ls[/EXEC]")
        assert actions[0].kind == "run"
        assert actions[0].command == "ls"

    def test_edit_block_replace(self) -> None:
        text = "[EDIT path=src/foo.py mode=replace]\nprint('hi')\n[/EDIT]"
        actions = parse_actions(text)
        assert len(actions) == 1
        a = actions[0]
        assert a.kind == "edit"
        assert a.path == "src/foo.py"
        assert a.mode == "replace"
        assert a.body == "print('hi')"

    def test_edit_default_mode_is_replace(self) -> None:
        actions = parse_actions("[EDIT path=x.txt]\ncontent\n[/EDIT]")
        assert actions[0].mode == "replace"

    def test_edit_quoted_path(self) -> None:
        actions = parse_actions('[EDIT path="a b.py" mode=create]\nx\n[/EDIT]')
        assert actions[0].path == "a b.py"
        assert actions[0].mode == "create"

    def test_edit_insert_line_parsed(self) -> None:
        actions = parse_actions(
            "[EDIT path=x.py mode=insert_at_line line=3]\nNEW\n[/EDIT]"
        )
        assert actions[0].line == 3

    def test_submit_block(self) -> None:
        actions = parse_actions("[SUBMIT]\nDone it.\n[/SUBMIT]")
        assert len(actions) == 1
        assert actions[0].kind == "submit"
        assert actions[0].summary == "Done it."

    def test_multiple_actions_in_order(self) -> None:
        text = (
            "first\n[EDIT path=a.py mode=replace]\nx\n[/EDIT]\n"
            "then\n[RUN]python a.py[/RUN]"
        )
        actions = parse_actions(text)
        assert [a.kind for a in actions] == ["edit", "run"]

    def test_no_action_returns_empty(self) -> None:
        assert parse_actions("just some prose, no blocks") == []

    def test_empty_run_ignored(self) -> None:
        assert parse_actions("[RUN]   [/RUN]") == []

    def test_multiline_edit_body_preserved(self) -> None:
        text = "[EDIT path=x.py mode=create]\nline1\n\nline3\n[/EDIT]"
        actions = parse_actions(text)
        assert actions[0].body == "line1\n\nline3"


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
