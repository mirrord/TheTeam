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


class TestEditAliases:
    """Mode-named edit blocks models commonly emit are accepted as edits."""

    def test_create_alias(self) -> None:
        # The exact form that previously caused format_error_limit failures.
        text = "THOUGHT: make it\n[CREATE path=asteroids.py]\nimport pygame\n[/CREATE]"
        actions = parse_actions(text)
        assert len(actions) == 1
        a = actions[0]
        assert a.kind == "edit"
        assert a.path == "asteroids.py"
        assert a.mode == "create"
        assert a.body == "import pygame"

    def test_replace_and_write_aliases(self) -> None:
        for tag in ("REPLACE", "WRITE"):
            actions = parse_actions(f"[{tag} path=x.py]\nbody\n[/{tag}]")
            assert actions[0].kind == "edit"
            assert actions[0].mode == "replace"

    def test_patch_alias(self) -> None:
        actions = parse_actions("[PATCH path=x.py]\n@@ -1 +1 @@\n[/PATCH]")
        assert actions[0].mode == "patch"

    def test_insert_aliases(self) -> None:
        a1 = parse_actions("[INSERT path=x.py line=2]\nNEW\n[/INSERT]")[0]
        assert a1.mode == "insert_at_line"
        assert a1.line == 2
        a2 = parse_actions("[INSERT_AT_LINE path=x.py line=3]\nNEW\n[/INSERT_AT_LINE]")[0]
        assert a2.mode == "insert_at_line"
        assert a2.line == 3

    def test_explicit_mode_attr_wins_over_tag(self) -> None:
        actions = parse_actions("[CREATE path=x.py mode=replace]\nb\n[/CREATE]")
        assert actions[0].mode == "replace"

    def test_lowercase_tags_accepted(self) -> None:
        actions = parse_actions("[create path=x.py]\nb\n[/create]")
        assert actions[0].kind == "edit"
        assert actions[0].mode == "create"

    def test_submit_aliases(self) -> None:
        for tag in ("DONE", "FINISH"):
            actions = parse_actions(f"[{tag}]\nwrapped up\n[/{tag}]")
            assert actions[0].kind == "submit"
            assert actions[0].summary == "wrapped up"

    def test_alias_mixed_with_run(self) -> None:
        text = "[CREATE path=a.py]\nx=1\n[/CREATE]\n[RUN]python a.py[/RUN]"
        actions = parse_actions(text)
        assert [a.kind for a in actions] == ["edit", "run"]


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
