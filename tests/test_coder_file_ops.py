"""Tests for the root-jailed file-edit helpers in pithos.tools.coder.file_ops."""

from __future__ import annotations

from pathlib import Path

import pytest

from pithos.tools.coder.file_ops import (
    apply_edit,
    apply_unified_diff,
    reject_if_binary,
    resolve_in_root,
)


class TestResolveInRoot:
    def test_resolves_relative_path(self, tmp_path: Path) -> None:
        root = tmp_path.resolve()
        resolved = resolve_in_root("sub/file.py", root)
        assert resolved == (root / "sub" / "file.py").resolve()

    def test_escape_raises(self, tmp_path: Path) -> None:
        root = (tmp_path / "ws").resolve()
        root.mkdir()
        with pytest.raises(PermissionError):
            resolve_in_root("../secret.txt", root)

    def test_absolute_escape_raises(self, tmp_path: Path) -> None:
        root = (tmp_path / "ws").resolve()
        root.mkdir()
        outside = (tmp_path / "outside.txt").resolve()
        with pytest.raises(PermissionError):
            resolve_in_root(str(outside), root)


class TestApplyEdit:
    def test_create_new_file(self, tmp_path: Path) -> None:
        root = tmp_path.resolve()
        result = apply_edit(
            {"path": "a/b.txt", "mode": "create", "content": "hello"}, root
        )
        assert result == {"path": str((root / "a" / "b.txt").resolve()), "mode": "create"}
        assert (root / "a" / "b.txt").read_text() == "hello"

    def test_create_existing_fails(self, tmp_path: Path) -> None:
        root = tmp_path.resolve()
        (root / "x.txt").write_text("orig")
        with pytest.raises(FileExistsError):
            apply_edit({"path": "x.txt", "mode": "create", "content": "new"}, root)

    def test_replace_writes_backup(self, tmp_path: Path) -> None:
        root = tmp_path.resolve()
        (root / "x.txt").write_text("orig")
        apply_edit(
            {"path": "x.txt", "mode": "replace", "content": "new"}, root, backup=True
        )
        assert (root / "x.txt").read_text() == "new"
        assert (root / "x.txt.bak").read_text() == "orig"

    def test_replace_no_backup(self, tmp_path: Path) -> None:
        root = tmp_path.resolve()
        (root / "x.txt").write_text("orig")
        apply_edit(
            {"path": "x.txt", "mode": "replace", "content": "new"}, root, backup=False
        )
        assert not (root / "x.txt.bak").exists()

    def test_replace_missing_fails(self, tmp_path: Path) -> None:
        root = tmp_path.resolve()
        with pytest.raises(FileNotFoundError):
            apply_edit({"path": "nope.txt", "mode": "replace", "content": "x"}, root)

    def test_insert_at_line(self, tmp_path: Path) -> None:
        root = tmp_path.resolve()
        (root / "x.txt").write_text("l1\nl2\nl3\n")
        apply_edit(
            {"path": "x.txt", "mode": "insert_at_line", "line": 2, "content": "NEW"},
            root,
            backup=False,
        )
        assert (root / "x.txt").read_text() == "l1\nNEW\nl2\nl3\n"

    def test_patch_mode(self, tmp_path: Path) -> None:
        root = tmp_path.resolve()
        (root / "x.txt").write_text("a\nb\nc\n")
        patch = "@@ -1,3 +1,3 @@\n a\n-b\n+B\n c\n"
        apply_edit({"path": "x.txt", "mode": "patch", "patch": patch}, root, backup=False)
        assert (root / "x.txt").read_text() == "a\nB\nc\n"

    def test_unknown_mode_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError):
            apply_edit({"path": "x.txt", "mode": "bogus"}, tmp_path.resolve())

    def test_missing_path_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError):
            apply_edit({"mode": "create", "content": "x"}, tmp_path.resolve())

    def test_escape_raises(self, tmp_path: Path) -> None:
        root = (tmp_path / "ws").resolve()
        root.mkdir()
        with pytest.raises(PermissionError):
            apply_edit(
                {"path": "../evil.txt", "mode": "create", "content": "x"}, root
            )

    def test_reject_binary_edit(self, tmp_path: Path) -> None:
        root = tmp_path.resolve()
        (root / "bin").write_bytes(b"a\x00b")
        with pytest.raises(ValueError):
            apply_edit({"path": "bin", "mode": "replace", "content": "x"}, root)


class TestApplyUnifiedDiff:
    def test_context_mismatch_raises(self) -> None:
        with pytest.raises(ValueError):
            apply_unified_diff("a\nb\n", "@@ -1,2 +1,2 @@\n x\n-b\n+B\n")

    def test_malformed_header_raises(self) -> None:
        with pytest.raises(ValueError):
            apply_unified_diff("a\n", "@@ bogus @@\n+x\n")


class TestRejectIfBinary:
    def test_text_ok(self, tmp_path: Path) -> None:
        p = tmp_path / "t.txt"
        p.write_text("plain text")
        reject_if_binary(p)  # no raise

    def test_binary_raises(self, tmp_path: Path) -> None:
        p = tmp_path / "b.bin"
        p.write_bytes(b"x\x00y")
        with pytest.raises(ValueError):
            reject_if_binary(p)
