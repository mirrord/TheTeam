"""Shared, root-jailed file-edit helpers.

These functions implement safe, ``allowed_root``-confined file mutations used by
both :class:`pithos.coding_nodes.EditFileNode` and the coding harness
(:mod:`pithos.tools.coder`).  Keeping the logic in one place avoids divergence
between the two call sites.

Supported edit modes (each edit is a ``dict``):

``"replace"``        - overwrite the file with ``content``.
``"create"``         - create a new file with ``content`` (fails if it exists).
``"insert_at_line"`` - insert ``content`` before 1-based ``line``.
``"patch"``          - apply a unified ``patch`` produced by
                       :func:`difflib.unified_diff`.

A ``.bak`` snapshot of the original file is written before any in-place edit
when ``backup`` is true.  Any attempt to touch a path outside ``allowed_root``
raises :class:`PermissionError`.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

#: Number of leading bytes sniffed to decide whether a file is binary.
BINARY_SNIFF_BYTES = 8192

#: The edit modes understood by :func:`apply_edit`.
EDIT_MODES = frozenset({"replace", "create", "insert_at_line", "patch"})


def resolve_in_root(rel_path: str, root: Path) -> Path:
    """Resolve *rel_path* against *root*, rejecting any escape.

    Args:
        rel_path: Path relative to *root* (absolute paths are also resolved and
            must still fall inside *root*).
        root: The already-resolved jail root.

    Returns:
        The resolved, in-jail :class:`~pathlib.Path`.

    Raises:
        PermissionError: If the resolved path escapes *root*.
    """
    candidate = (root / rel_path).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise PermissionError(
            f"path {rel_path!r} escapes allowed_root {root}"
        ) from exc
    return candidate


def reject_if_binary(path: Path) -> None:
    """Raise :class:`ValueError` if *path* looks like a binary file."""
    try:
        with path.open("rb") as fh:
            chunk = fh.read(BINARY_SNIFF_BYTES)
    except OSError:
        return
    if b"\x00" in chunk:
        raise ValueError(f"refusing to edit binary file {path}")


def apply_unified_diff(original: str, patch_text: str) -> str:
    """Apply a unified diff using a small, dependency-free patcher.

    Only the body hunks are interpreted; file headers are ignored.  Each hunk
    must match the surrounding context exactly.

    Args:
        original: Current file contents.
        patch_text: Unified-diff text to apply.

    Returns:
        The patched contents.

    Raises:
        ValueError: On a malformed hunk header or a context/deletion mismatch.
    """
    src_lines = original.splitlines(keepends=True)
    out_lines: list[str] = []
    src_idx = 0

    lines = patch_text.splitlines(keepends=True)
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("@@"):
            m = re.match(r"@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@", line)
            if not m:
                raise ValueError(f"malformed hunk header: {line!r}")
            old_start = int(m.group(1))
            while src_idx < old_start - 1:
                out_lines.append(src_lines[src_idx])
                src_idx += 1
            i += 1
            while i < len(lines) and not lines[i].startswith("@@"):
                hl = lines[i]
                if hl.startswith("---") or hl.startswith("+++"):
                    i += 1
                    continue
                if hl.startswith(" "):
                    if src_idx >= len(src_lines) or src_lines[src_idx] != hl[1:]:
                        raise ValueError("patch context mismatch")
                    out_lines.append(src_lines[src_idx])
                    src_idx += 1
                elif hl.startswith("-"):
                    if src_idx >= len(src_lines) or src_lines[src_idx] != hl[1:]:
                        raise ValueError("patch deletion mismatch")
                    src_idx += 1
                elif hl.startswith("+"):
                    out_lines.append(hl[1:])
                elif hl.strip() == "":
                    pass
                else:
                    raise ValueError(f"unrecognised patch line {hl!r}")
                i += 1
            continue
        i += 1

    while src_idx < len(src_lines):
        out_lines.append(src_lines[src_idx])
        src_idx += 1

    return "".join(out_lines)


def apply_edit(edit: dict[str, Any], root: Path, backup: bool = True) -> dict[str, Any]:
    """Apply a single *edit* under *root* and return a small result dict.

    Args:
        edit: Edit spec with at least ``path`` and ``mode`` keys.
        root: The already-resolved jail root.
        backup: Write a ``.bak`` snapshot before in-place edits.

    Returns:
        ``{"path": str, "mode": str}`` describing the applied edit.

    Raises:
        ValueError: On a malformed edit (bad type, unknown mode, missing path).
        PermissionError: If the target escapes *root*.
        FileExistsError / FileNotFoundError: For create/edit pre-conditions.
    """
    if not isinstance(edit, dict):
        raise ValueError(f"each edit must be a dict, got {type(edit).__name__}")
    mode = edit.get("mode")
    if mode not in EDIT_MODES:
        raise ValueError(f"unsupported mode {mode!r}")

    rel_path = edit.get("path")
    if not rel_path:
        raise ValueError("edit missing 'path'")

    target = resolve_in_root(rel_path, root)

    if mode == "create":
        if target.exists():
            raise FileExistsError(f"{target} already exists")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(edit.get("content", ""), encoding="utf-8")
        return {"path": str(target), "mode": mode}

    # All other modes operate on an existing file.
    if not target.exists():
        raise FileNotFoundError(f"{target} does not exist")
    reject_if_binary(target)

    original = target.read_text(encoding="utf-8")
    if backup:
        backup_path = target.with_suffix(target.suffix + ".bak")
        backup_path.write_text(original, encoding="utf-8")

    if mode == "replace":
        target.write_text(edit.get("content", ""), encoding="utf-8")
    elif mode == "insert_at_line":
        line = int(edit.get("line", 1))
        content = edit.get("content", "")
        if not content.endswith("\n"):
            content += "\n"
        lines = original.splitlines(keepends=True)
        idx = max(0, min(line - 1, len(lines)))
        lines.insert(idx, content)
        target.write_text("".join(lines), encoding="utf-8")
    elif mode == "patch":
        patch_text = edit.get("patch")
        if not patch_text:
            raise ValueError("patch mode requires 'patch'")
        patched = apply_unified_diff(original, patch_text)
        target.write_text(patched, encoding="utf-8")

    return {"path": str(target), "mode": mode}
