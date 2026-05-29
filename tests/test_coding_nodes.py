"""Tests for coding-workflow node types: RouterNode, JsonParseNode,
ListFilesNode, EditFileNode, GitNode."""

import json
import os
import shutil
import subprocess
import textwrap
from pathlib import Path

import pytest

from pithos.message import Message, MessageRouter, NodeInputState
from pithos.coding_nodes import (
    RouterNode,
    JsonParseNode,
    ListFilesNode,
    EditFileNode,
    GitNode,
)
from pithos.flownode import create_node

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _run(node, data="", *, router=None, key="default", ports=None):
    """Execute a node with a single default input."""
    if router is None:
        router = MessageRouter()
    inputs = ports or node.required_inputs
    state = NodeInputState(node_id="t", required_inputs=inputs)
    state.receive_message(Message(data=data, input_key=key))
    return node.execute_with_messages(state, message_router=router), router


# ---------------------------------------------------------------------------
# RouterNode
# ---------------------------------------------------------------------------


class TestRouterNode:
    def test_routes_by_token_match(self):
        node = RouterNode(
            routes={"apply": "APPROVE", "revise": "REVISE"},
            default_route="revise",
            outputs=["apply", "revise"],
        )
        msgs, _ = _run(node, "Final: APPROVE this change")
        assert len(msgs) == 1
        assert msgs[0].input_key == "apply"
        assert msgs[0].data == "Final: APPROVE this change"

    def test_falls_back_to_default(self):
        node = RouterNode(
            routes={"apply": "APPROVE"},
            default_route="revise",
            outputs=["apply", "revise"],
        )
        msgs, _ = _run(node, "no token present")
        assert msgs[0].input_key == "revise"

    def test_routes_by_regex(self):
        node = RouterNode(
            routes={"high": r"score\s*[:=]\s*9|10"},
            default_route="low",
            match_mode="regex",
            outputs=["high", "low"],
        )
        msgs, _ = _run(node, "Final score: 9")
        assert msgs[0].input_key == "high"

    def test_routes_by_state_var(self):
        node = RouterNode(
            routes={"more": "go", "done": "stop"},
            default_route="done",
            source="decision",
            outputs=["more", "done"],
        )
        router = MessageRouter()
        router.shared_context["decision"] = "go"
        msgs, _ = _run(node, "input", router=router)
        assert msgs[0].input_key == "more"

    def test_unknown_default_route_raises(self):
        with pytest.raises(ValueError, match="default_route"):
            RouterNode(routes={"a": "x"}, default_route="missing", outputs=["a"])

    def test_factory(self):
        n = create_node(
            "router",
            {
                "routes": {"a": "X"},
                "default_route": "a",
                "outputs": ["a"],
            },
        )
        assert isinstance(n, RouterNode)


# ---------------------------------------------------------------------------
# JsonParseNode
# ---------------------------------------------------------------------------


class TestJsonParseNode:
    def test_parses_plain_json(self):
        node = JsonParseNode(save_to="parsed")
        msgs, router = _run(node, '{"x": 1, "y": [2, 3]}')
        assert msgs[0].data == {"x": 1, "y": [2, 3]}

    def test_parses_fenced_block(self):
        text = 'Here you go:\n```json\n{"ok": true}\n```\nthanks'
        node = JsonParseNode(save_to="parsed")
        msgs, _ = _run(node, text)
        assert msgs[0].data == {"ok": True}

    def test_finds_first_object_in_prose(self):
        text = 'preamble {"a": 1, "b": {"c": 2}} trailing'
        node = JsonParseNode(save_to="parsed")
        msgs, _ = _run(node, text)
        assert msgs[0].data == {"a": 1, "b": {"c": 2}}

    def test_finds_first_array(self):
        node = JsonParseNode(save_to="parsed")
        msgs, _ = _run(node, "list: [1, 2, 3] end")
        assert msgs[0].data == [1, 2, 3]

    def test_invalid_json_raises_by_default(self):
        node = JsonParseNode(save_to="parsed")
        with pytest.raises(ValueError, match="No JSON"):
            _run(node, "no json anywhere")

    def test_invalid_json_lenient_returns_none(self):
        node = JsonParseNode(save_to="parsed", on_error="none")
        msgs, _ = _run(node, "no json anywhere")
        assert msgs[0].data is None

    def test_schema_validation_required_keys(self):
        node = JsonParseNode(
            save_to="parsed",
            schema={"type": "object", "required": ["steps"]},
        )
        with pytest.raises(ValueError, match="required"):
            _run(node, '{"other": 1}')

    def test_schema_validation_type(self):
        node = JsonParseNode(
            save_to="parsed",
            schema={"type": "array"},
        )
        with pytest.raises(ValueError, match="type"):
            _run(node, '{"obj": 1}')

    def test_factory(self):
        n = create_node("jsonparse", {"save_to": "x"})
        assert isinstance(n, JsonParseNode)


# ---------------------------------------------------------------------------
# ListFilesNode
# ---------------------------------------------------------------------------


class TestListFilesNode:
    @pytest.fixture
    def repo(self, tmp_path):
        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "a.py").write_text("a")
        (tmp_path / "src" / "b.py").write_text("b")
        (tmp_path / "src" / "c.txt").write_text("c")
        (tmp_path / "node_modules").mkdir()
        (tmp_path / "node_modules" / "junk.py").write_text("j")
        return tmp_path

    def test_lists_with_include_glob(self, repo):
        node = ListFilesNode(roots=[str(repo)], include=["**/*.py"], save_to="files")
        msgs, _ = _run(node, "")
        files = msgs[0].data
        names = sorted(Path(f).name for f in files)
        assert "a.py" in names and "b.py" in names and "junk.py" in names
        assert "c.txt" not in names

    def test_exclude_glob(self, repo):
        node = ListFilesNode(
            roots=[str(repo)],
            include=["**/*.py"],
            exclude=["**/node_modules/**"],
            save_to="files",
        )
        msgs, _ = _run(node, "")
        files = msgs[0].data
        assert all("node_modules" not in f for f in files)

    def test_max_files_caps_output(self, repo):
        node = ListFilesNode(
            roots=[str(repo)], include=["**/*"], max_files=2, save_to="files"
        )
        msgs, _ = _run(node, "")
        assert len(msgs[0].data) == 2

    def test_root_template_from_context(self, repo):
        node = ListFilesNode(
            roots=["{repo_root}"], include=["**/*.py"], save_to="files"
        )
        router = MessageRouter()
        router.shared_context["repo_root"] = str(repo)
        msgs, _ = _run(node, "", router=router)
        assert len(msgs[0].data) >= 2

    def test_missing_root_raises(self, tmp_path):
        node = ListFilesNode(
            roots=[str(tmp_path / "nope")], include=["*"], save_to="files"
        )
        with pytest.raises(FileNotFoundError):
            _run(node, "")

    def test_factory(self):
        n = create_node(
            "listfiles", {"roots": ["."], "include": ["*.py"], "save_to": "f"}
        )
        assert isinstance(n, ListFilesNode)


# ---------------------------------------------------------------------------
# EditFileNode
# ---------------------------------------------------------------------------


class TestEditFileNode:
    @pytest.fixture
    def root(self, tmp_path):
        (tmp_path / "a.py").write_text("line1\nline2\nline3\n")
        return tmp_path

    def test_replace_whole_file(self, root):
        node = EditFileNode(
            allowed_root=str(root),
            edits_from="edits",
            backup=True,
        )
        router = MessageRouter()
        router.shared_context["edits"] = [
            {"path": "a.py", "mode": "replace", "content": "new content"}
        ]
        msgs, _ = _run(node, "", router=router)
        assert (root / "a.py").read_text() == "new content"
        assert (root / "a.py.bak").exists()
        assert msgs[0].data["applied"][0]["path"].endswith("a.py")

    def test_create_new_file(self, root):
        node = EditFileNode(allowed_root=str(root), edits_from="edits", backup=False)
        router = MessageRouter()
        router.shared_context["edits"] = [
            {"path": "new/sub/b.py", "mode": "create", "content": "hello"}
        ]
        _run(node, "", router=router)
        assert (root / "new" / "sub" / "b.py").read_text() == "hello"

    def test_create_fails_if_exists(self, root):
        node = EditFileNode(allowed_root=str(root), edits_from="edits", backup=False)
        router = MessageRouter()
        router.shared_context["edits"] = [
            {"path": "a.py", "mode": "create", "content": "x"}
        ]
        with pytest.raises(FileExistsError):
            _run(node, "", router=router)

    def test_insert_at_line(self, root):
        node = EditFileNode(allowed_root=str(root), edits_from="edits", backup=False)
        router = MessageRouter()
        router.shared_context["edits"] = [
            {"path": "a.py", "mode": "insert_at_line", "line": 2, "content": "INS\n"}
        ]
        _run(node, "", router=router)
        text = (root / "a.py").read_text()
        assert text.splitlines()[1] == "INS"

    def test_patch_unified_diff(self, root):
        # Create a valid unified diff that converts line2 -> LINE_TWO
        original = (root / "a.py").read_text()
        patched = original.replace("line2", "LINE_TWO")
        import difflib

        diff = "".join(
            difflib.unified_diff(
                original.splitlines(keepends=True),
                patched.splitlines(keepends=True),
                fromfile="a.py",
                tofile="a.py",
            )
        )
        node = EditFileNode(allowed_root=str(root), edits_from="edits", backup=False)
        router = MessageRouter()
        router.shared_context["edits"] = [
            {"path": "a.py", "mode": "patch", "patch": diff}
        ]
        _run(node, "", router=router)
        assert "LINE_TWO" in (root / "a.py").read_text()

    def test_rejects_path_escape(self, root):
        node = EditFileNode(allowed_root=str(root), edits_from="edits", backup=False)
        router = MessageRouter()
        router.shared_context["edits"] = [
            {"path": "../escape.py", "mode": "create", "content": "x"}
        ]
        with pytest.raises(PermissionError):
            _run(node, "", router=router)

    def test_rejects_binary_file(self, root):
        (root / "bin.dat").write_bytes(b"AB\x00CD")
        node = EditFileNode(allowed_root=str(root), edits_from="edits", backup=False)
        router = MessageRouter()
        router.shared_context["edits"] = [
            {"path": "bin.dat", "mode": "replace", "content": "x"}
        ]
        with pytest.raises(ValueError, match="binary"):
            _run(node, "", router=router)

    def test_factory(self):
        n = create_node(
            "editfile",
            {"allowed_root": ".", "edits_from": "edits"},
        )
        assert isinstance(n, EditFileNode)


# ---------------------------------------------------------------------------
# GitNode
# ---------------------------------------------------------------------------


def _git_available():
    return shutil.which("git") is not None


@pytest.mark.skipif(not _git_available(), reason="git not on PATH")
class TestGitNode:
    @pytest.fixture
    def repo(self, tmp_path):
        subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
        subprocess.run(["git", "config", "user.email", "t@t"], cwd=tmp_path, check=True)
        subprocess.run(["git", "config", "user.name", "t"], cwd=tmp_path, check=True)
        (tmp_path / "a.txt").write_text("hi\n")
        subprocess.run(["git", "add", "."], cwd=tmp_path, check=True)
        subprocess.run(["git", "commit", "-q", "-m", "init"], cwd=tmp_path, check=True)
        return tmp_path

    def test_status_clean(self, repo):
        node = GitNode(subcommand="status", repo_root=str(repo), args=["--porcelain"])
        msgs, _ = _run(node, "")
        assert msgs[0].data["success"] is True
        assert msgs[0].data["stdout"].strip() == ""

    def test_status_dirty(self, repo):
        (repo / "a.txt").write_text("changed\n")
        node = GitNode(subcommand="status", repo_root=str(repo), args=["--porcelain"])
        msgs, _ = _run(node, "")
        assert "a.txt" in msgs[0].data["stdout"]

    def test_add_and_commit(self, repo):
        (repo / "b.txt").write_text("new\n")
        add = GitNode(subcommand="add", repo_root=str(repo), args=["b.txt"])
        _run(add, "")
        commit = GitNode(subcommand="commit", repo_root=str(repo), args=["-m", "add b"])
        msgs, _ = _run(commit, "")
        assert msgs[0].data["success"] is True

    def test_destructive_blocked_without_opt_in(self, repo):
        with pytest.raises(PermissionError):
            GitNode(subcommand="reset", repo_root=str(repo), args=["--hard"])

    def test_destructive_allowed_with_opt_in(self, repo):
        node = GitNode(
            subcommand="reset",
            repo_root=str(repo),
            args=["--hard", "HEAD"],
            allow_destructive=True,
        )
        msgs, _ = _run(node, "")
        assert msgs[0].data["success"] is True

    def test_unknown_subcommand_rejected(self):
        with pytest.raises(ValueError, match="subcommand"):
            GitNode(subcommand="rm-rf-everything", repo_root=".")

    def test_args_templated_from_context(self, repo):
        node = GitNode(subcommand="log", repo_root=str(repo), args=["-n", "{n}"])
        router = MessageRouter()
        router.shared_context["n"] = "1"
        msgs, _ = _run(node, "", router=router)
        assert msgs[0].data["success"] is True
        assert "init" in msgs[0].data["stdout"]

    def test_factory(self):
        n = create_node("git", {"subcommand": "status", "repo_root": "."})
        assert isinstance(n, GitNode)


# ---------------------------------------------------------------------------
# ToolCallNode success / failure ports (back-compat additive)
# ---------------------------------------------------------------------------


class TestToolCallNodePorts:
    def _build(self, *, success, outputs):
        from unittest.mock import Mock

        from pithos.flownode import ToolCallNode
        from pithos.tools import ToolResult

        node = ToolCallNode(
            command="cmd",
            save_to="result",
            inputs=["default"],
            outputs=outputs,
        )
        router = MessageRouter()
        router.shared_context["tool_executor"] = Mock()
        router.shared_context["tool_registry"] = Mock()
        router.shared_context["tool_executor"].run.return_value = ToolResult(
            success=success,
            stdout="ok" if success else "",
            stderr="" if success else "boom",
            exit_code=0 if success else 1,
            execution_time=0.0,
            command="cmd",
        )
        state = NodeInputState(node_id="t", required_inputs=["default"])
        state.receive_message(Message(data="", input_key="default"))
        return node.execute_with_messages(state, message_router=router)

    def test_default_port_back_compat(self):
        msgs = self._build(success=True, outputs=["default"])
        assert len(msgs) == 1
        assert msgs[0].input_key == "default"
        assert msgs[0].data["success"] is True

    def test_success_port_emitted(self):
        msgs = self._build(success=True, outputs=["success", "failure"])
        ports = {m.input_key for m in msgs}
        assert "success" in ports
        assert "failure" not in ports

    def test_failure_port_emitted(self):
        node_msgs = self._build(success=False, outputs=["success", "failure"])
        # error_handling default is "continue", so failure does NOT raise
        ports = {m.input_key for m in node_msgs}
        assert "failure" in ports
        assert "success" not in ports

    def test_all_three_ports_together(self):
        msgs = self._build(success=True, outputs=["default", "success", "failure"])
        ports = {m.input_key for m in msgs}
        assert "default" in ports
        assert "success" in ports
        assert "failure" not in ports
