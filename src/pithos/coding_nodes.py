"""Flow nodes for general-purpose coding workflows.

This module provides node types that complement the core nodes in
:mod:`pithos.flownode` with capabilities a coding agent needs:

* :class:`RouterNode`     -- multi-port routing based on token / regex / state.
* :class:`JsonParseNode`  -- robust JSON extraction from agent output.
* :class:`ListFilesNode`  -- glob-based filesystem enumeration.
* :class:`EditFileNode`   -- safe, root-jailed file edits (replace / create /
  insert_at_line / patch) with automatic ``.bak`` snapshots.
* :class:`GitNode`        -- whitelisted ``git`` subcommands with explicit
  opt-in for destructive operations.

All nodes integrate with the message-driven flowchart runtime in
:mod:`pithos.flownode` by subclassing :class:`~pithos.flownode.FlowNode`.
"""

from __future__ import annotations

import difflib
import fnmatch
import json
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any, Iterable, Optional

from .flownode import FlowNode
from .message import Message

# ---------------------------------------------------------------------------
# RouterNode
# ---------------------------------------------------------------------------


class RouterNode(FlowNode):
    """Route the incoming message to one of several named output ports.

    Routing is decided by inspecting either ``current_input`` (default) or
    any value in the flowchart state (``source: <state_var>``).  Each entry
    in ``routes`` maps an output port name to a *match expression*.  The
    first port whose expression matches wins; if none match the message is
    emitted on ``default_route``.

    Match modes:

    * ``"token"`` (default) - case-insensitive substring search.
    * ``"regex"``           - :func:`re.search` with the expression as the
      pattern.
    """

    def __init__(
        self,
        routes: dict[str, str],
        default_route: str,
        source: str = "current_input",
        match_mode: str = "token",
        extraction: Optional[dict[str, str]] = None,
        **kwargs: Any,
    ) -> None:
        if not routes:
            raise ValueError("RouterNode requires at least one route")
        if match_mode not in ("token", "regex"):
            raise ValueError(
                f"RouterNode match_mode must be 'token' or 'regex', got {match_mode!r}"
            )

        explicit_outputs = list(kwargs.get("outputs") or [])
        if explicit_outputs:
            # When the caller declared outputs explicitly, every route and the
            # default_route MUST already be present -- mismatches are config bugs.
            for port in routes:
                if port not in explicit_outputs:
                    raise ValueError(
                        f"RouterNode route {port!r} must appear in outputs"
                    )
            if default_route not in explicit_outputs:
                raise ValueError(
                    f"RouterNode default_route {default_route!r} must appear in outputs"
                )
            kwargs["outputs"] = explicit_outputs
        else:
            # No explicit outputs -- derive them from routes + default_route.
            outputs: list[str] = []
            for port in list(routes) + [default_route]:
                if port not in outputs:
                    outputs.append(port)
            kwargs["outputs"] = outputs

        super().__init__(extraction=extraction, **kwargs)

        self.routes = dict(routes)
        self.default_route = default_route
        self.source = source
        self.match_mode = match_mode

    def _execute(self, context: dict[str, Any]) -> Any:
        value = context.get(self.source, "")
        text = "" if value is None else str(value)

        chosen = self.default_route
        for port, expr in self.routes.items():
            if self._matches(text, expr):
                chosen = port
                break

        # Return a dict whose only populated key is the chosen port so that
        # _create_output_messages emits exactly one message on that port.
        payload = context.get("current_input", value)
        return {chosen: payload}

    def _matches(self, text: str, expr: str) -> bool:
        if self.match_mode == "regex":
            try:
                return re.search(expr, text) is not None
            except re.error:
                return False
        return expr.lower() in text.lower()


# ---------------------------------------------------------------------------
# JsonParseNode
# ---------------------------------------------------------------------------


_FENCED_JSON_RE = re.compile(
    r"```(?:json)?\s*(?P<body>\{.*?\}|\[.*?\])\s*```", re.DOTALL | re.IGNORECASE
)


class JsonParseNode(FlowNode):
    """Extract and parse JSON from arbitrary text.

    The node first looks for a fenced ```` ```json ... ``` ```` block.  If
    none is found it falls back to balanced-bracket scanning to locate the
    first standalone object or array in the text and tries to parse it.

    Optional ``schema`` validates the parsed value at a minimal level:

    .. code-block:: yaml

        schema:
          type: object        # or "array"
          required: [steps]   # only meaningful when type == "object"

    ``on_error`` controls failure handling:

    * ``"raise"`` (default) - raise :class:`ValueError`.
    * ``"none"``            - emit ``None`` on the output port.
    """

    def __init__(
        self,
        save_to: str = "parsed",
        schema: Optional[dict[str, Any]] = None,
        on_error: str = "raise",
        extraction: Optional[dict[str, str]] = None,
        **kwargs: Any,
    ) -> None:
        if on_error not in ("raise", "none"):
            raise ValueError("JsonParseNode on_error must be 'raise' or 'none'")
        super().__init__(extraction=extraction, **kwargs)
        self.save_to = save_to
        self.schema = schema
        self.on_error = on_error

    def _execute(self, context: dict[str, Any]) -> Any:
        text = context.get("current_input", "")
        if not isinstance(text, str):
            text = str(text)

        try:
            value = self._extract_json(text)
            if self.schema:
                self._validate_schema(value)
        except ValueError:
            if self.on_error == "none":
                return {self.save_to: None, "current_input": None, "_force_emit": True}
            raise

        return {self.save_to: value, "current_input": value}

    def _create_output_messages(self, result: Any, context: dict[str, Any]):
        # Special-case lenient None: the base class drops None values, but for
        # JsonParseNode the consumer needs to receive an explicit None.
        if isinstance(result, dict) and result.get("_force_emit"):
            from .message import Message

            return [
                Message(data=None, source_node=None, input_key=key)
                for key in self.output_keys
            ]
        return super()._create_output_messages(result, context)

    # -- helpers -----------------------------------------------------------

    @classmethod
    def _extract_json(cls, text: str) -> Any:
        # 1. fenced block
        match = _FENCED_JSON_RE.search(text)
        candidates: list[str] = []
        if match:
            candidates.append(match.group("body"))

        # 2. balanced-bracket scan over the raw text
        candidates.extend(cls._scan_balanced(text))

        for candidate in candidates:
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                continue

        # 3. final attempt: parse the entire string verbatim
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        raise ValueError("No JSON object or array found in input")

    @staticmethod
    def _scan_balanced(text: str) -> Iterable[str]:
        for open_ch, close_ch in (("{", "}"), ("[", "]")):
            depth = 0
            start = -1
            in_str = False
            escape = False
            for idx, ch in enumerate(text):
                if in_str:
                    if escape:
                        escape = False
                    elif ch == "\\":
                        escape = True
                    elif ch == '"':
                        in_str = False
                    continue
                if ch == '"':
                    in_str = True
                    continue
                if ch == open_ch:
                    if depth == 0:
                        start = idx
                    depth += 1
                elif ch == close_ch and depth > 0:
                    depth -= 1
                    if depth == 0 and start >= 0:
                        yield text[start : idx + 1]
                        start = -1

    def _validate_schema(self, value: Any) -> None:
        expected = self.schema.get("type") if self.schema else None
        if expected == "object" and not isinstance(value, dict):
            raise ValueError(
                f"Expected JSON object (type='object'), got {type(value).__name__}"
            )
        if expected == "array" and not isinstance(value, list):
            raise ValueError(
                f"Expected JSON array (type='array'), got {type(value).__name__}"
            )
        required = self.schema.get("required") if self.schema else None
        if required and isinstance(value, dict):
            missing = [k for k in required if k not in value]
            if missing:
                raise ValueError(f"Parsed JSON missing required keys: {missing}")


# ---------------------------------------------------------------------------
# ListFilesNode
# ---------------------------------------------------------------------------


class ListFilesNode(FlowNode):
    """Enumerate files beneath one or more root directories using globs."""

    def __init__(
        self,
        roots: list[str],
        include: Optional[list[str]] = None,
        exclude: Optional[list[str]] = None,
        max_files: int = 5000,
        save_to: str = "files",
        extraction: Optional[dict[str, str]] = None,
        **kwargs: Any,
    ) -> None:
        if not roots:
            raise ValueError("ListFilesNode requires at least one root")
        super().__init__(extraction=extraction, **kwargs)
        self.roots = list(roots)
        self.include = list(include or ["**/*"])
        self.exclude = list(exclude or [])
        self.max_files = int(max_files)
        self.save_to = save_to

    def _execute(self, context: dict[str, Any]) -> Any:
        resolved_roots = [self._stateful_format(r, context) for r in self.roots]
        results: list[str] = []

        for root_str in resolved_roots:
            root = Path(root_str).expanduser()
            if not root.exists():
                raise FileNotFoundError(f"ListFilesNode root not found: {root}")
            if not root.is_dir():
                raise NotADirectoryError(
                    f"ListFilesNode root is not a directory: {root}"
                )

            seen: set[str] = set()
            for pattern in self.include:
                for path in root.glob(pattern):
                    if not path.is_file():
                        continue
                    rel = path.as_posix()
                    if rel in seen:
                        continue
                    if self._is_excluded(path, root):
                        continue
                    seen.add(rel)
                    results.append(rel)
                    if len(results) >= self.max_files:
                        break
                if len(results) >= self.max_files:
                    break

        results.sort()
        return {self.save_to: results, "current_input": results}

    def _is_excluded(self, path: Path, root: Path) -> bool:
        try:
            rel = path.relative_to(root).as_posix()
        except ValueError:
            rel = path.as_posix()
        full = path.as_posix()
        for pattern in self.exclude:
            if fnmatch.fnmatch(rel, pattern) or fnmatch.fnmatch(full, pattern):
                return True
        return False


# ---------------------------------------------------------------------------
# EditFileNode
# ---------------------------------------------------------------------------


_BINARY_SNIFF_BYTES = 8192


class EditFileNode(FlowNode):
    """Apply a list of edits to files beneath an allowed root directory.

    Each edit is a dict with at minimum ``path`` and ``mode``.  Supported
    modes:

    ``"replace"``        - overwrite the file with ``content``.
    ``"create"``         - create a new file with ``content`` (fails if it
                            already exists).
    ``"insert_at_line"`` - insert ``content`` before 1-based ``line``.
    ``"patch"``          - apply a unified ``patch`` produced by
                            :func:`difflib.unified_diff`.

    A ``.bak`` snapshot of the original file is created before any edit when
    ``backup=True`` (the default).  All paths are resolved against
    ``allowed_root`` and any attempt to escape it raises
    :class:`PermissionError`.
    """

    _MODES = frozenset({"replace", "create", "insert_at_line", "patch"})

    def __init__(
        self,
        allowed_root: str,
        edits_from: str = "edits",
        backup: bool = True,
        save_to: str = "edit_result",
        extraction: Optional[dict[str, str]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(extraction=extraction, **kwargs)
        self.allowed_root = allowed_root
        self.edits_from = edits_from
        self.backup = bool(backup)
        self.save_to = save_to

    def _execute(self, context: dict[str, Any]) -> Any:
        root_str = self._stateful_format(self.allowed_root, context)
        root = Path(root_str).expanduser().resolve()

        edits = context.get(self.edits_from)
        if edits is None:
            raise ValueError(
                f"EditFileNode: no edits found at state key {self.edits_from!r}"
            )
        if isinstance(edits, dict):
            edits = [edits]
        if not isinstance(edits, list):
            raise ValueError(
                f"EditFileNode: edits_from must yield a list of dicts, got {type(edits).__name__}"
            )

        applied: list[dict[str, Any]] = []
        for edit in edits:
            applied.append(self._apply_edit(edit, root))

        result = {"applied": applied, "count": len(applied)}
        return {self.save_to: result, "current_input": result, "edit_result": result}

    # -- helpers -----------------------------------------------------------

    def _apply_edit(self, edit: dict[str, Any], root: Path) -> dict[str, Any]:
        if not isinstance(edit, dict):
            raise ValueError(
                f"EditFileNode: each edit must be a dict, got {type(edit).__name__}"
            )
        mode = edit.get("mode")
        if mode not in self._MODES:
            raise ValueError(f"EditFileNode: unsupported mode {mode!r}")

        rel_path = edit.get("path")
        if not rel_path:
            raise ValueError("EditFileNode: edit missing 'path'")

        target = self._resolve_path(rel_path, root)

        if mode == "create":
            if target.exists():
                raise FileExistsError(f"EditFileNode: {target} already exists")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(edit.get("content", ""), encoding="utf-8")
            return {"path": str(target), "mode": mode}

        # All other modes operate on an existing file.
        if not target.exists():
            raise FileNotFoundError(f"EditFileNode: {target} does not exist")
        self._reject_if_binary(target)

        original = target.read_text(encoding="utf-8")
        if self.backup:
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
                raise ValueError("EditFileNode: patch mode requires 'patch'")
            patched = self._apply_unified_diff(original, patch_text)
            target.write_text(patched, encoding="utf-8")

        return {"path": str(target), "mode": mode}

    def _resolve_path(self, rel_path: str, root: Path) -> Path:
        candidate = (root / rel_path).resolve()
        try:
            candidate.relative_to(root)
        except ValueError as exc:
            raise PermissionError(
                f"EditFileNode: path {rel_path!r} escapes allowed_root {root}"
            ) from exc
        return candidate

    @staticmethod
    def _reject_if_binary(path: Path) -> None:
        try:
            with path.open("rb") as fh:
                chunk = fh.read(_BINARY_SNIFF_BYTES)
        except OSError:
            return
        if b"\x00" in chunk:
            raise ValueError(f"EditFileNode: refusing to edit binary file {path}")

    @staticmethod
    def _apply_unified_diff(original: str, patch_text: str) -> str:
        """Apply a unified diff using a small, dependency-free patcher.

        Only the body hunks are interpreted; file headers are ignored.  Each
        hunk must match the surrounding context exactly.
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
                    raise ValueError(f"EditFileNode: malformed hunk header: {line!r}")
                old_start = int(m.group(1))
                # Copy untouched src up to the hunk
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
                            raise ValueError("EditFileNode: patch context mismatch")
                        out_lines.append(src_lines[src_idx])
                        src_idx += 1
                    elif hl.startswith("-"):
                        if src_idx >= len(src_lines) or src_lines[src_idx] != hl[1:]:
                            raise ValueError("EditFileNode: patch deletion mismatch")
                        src_idx += 1
                    elif hl.startswith("+"):
                        out_lines.append(hl[1:])
                    elif hl.strip() == "":
                        # blank trailing line — tolerate
                        pass
                    else:
                        raise ValueError(
                            f"EditFileNode: unrecognised patch line {hl!r}"
                        )
                    i += 1
                continue
            i += 1

        # tail
        while src_idx < len(src_lines):
            out_lines.append(src_lines[src_idx])
            src_idx += 1

        return "".join(out_lines)


# ---------------------------------------------------------------------------
# GitNode
# ---------------------------------------------------------------------------


_GIT_SAFE_SUBCOMMANDS: frozenset[str] = frozenset(
    {
        "status",
        "diff",
        "add",
        "commit",
        "log",
        "branch",
        "checkout",
        "restore",
        "show",
        "rev-parse",
    }
)

_GIT_DESTRUCTIVE_SUBCOMMANDS: frozenset[str] = frozenset(
    {"reset", "clean", "push", "rebase", "merge"}
)


class GitNode(FlowNode):
    """Run a whitelisted ``git`` subcommand inside ``repo_root``.

    Destructive subcommands (``reset``, ``clean``, ``push``, ``rebase``,
    ``merge``) require ``allow_destructive: true`` and are rejected
    otherwise.  Unknown subcommands are always rejected.
    """

    def __init__(
        self,
        subcommand: str,
        repo_root: str,
        args: Optional[list[str]] = None,
        allow_destructive: bool = False,
        timeout: float = 30.0,
        save_to: str = "git_result",
        extraction: Optional[dict[str, str]] = None,
        **kwargs: Any,
    ) -> None:
        if subcommand in _GIT_DESTRUCTIVE_SUBCOMMANDS and not allow_destructive:
            raise PermissionError(
                f"GitNode: subcommand {subcommand!r} is destructive; "
                "set allow_destructive: true to enable"
            )
        if (
            subcommand not in _GIT_SAFE_SUBCOMMANDS
            and subcommand not in _GIT_DESTRUCTIVE_SUBCOMMANDS
        ):
            raise ValueError(f"GitNode: unsupported subcommand {subcommand!r}")

        super().__init__(extraction=extraction, **kwargs)
        self.subcommand = subcommand
        self.repo_root = repo_root
        self.args = list(args or [])
        self.allow_destructive = bool(allow_destructive)
        self.timeout = float(timeout)
        self.save_to = save_to

    def _execute(self, context: dict[str, Any]) -> Any:
        repo_root = self._stateful_format(self.repo_root, context)
        formatted_args = [self._stateful_format(a, context) for a in self.args]

        git_exe = shutil.which("git")
        if git_exe is None:
            raise RuntimeError("GitNode: git executable not found on PATH")

        cmd = [git_exe, self.subcommand, *formatted_args]
        try:
            proc = subprocess.run(
                cmd,
                cwd=repo_root,
                capture_output=True,
                text=True,
                timeout=self.timeout,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise TimeoutError(f"GitNode: git {self.subcommand} timed out") from exc

        result = {
            "subcommand": self.subcommand,
            "args": formatted_args,
            "stdout": proc.stdout,
            "stderr": proc.stderr,
            "exit_code": proc.returncode,
            "success": proc.returncode == 0,
        }
        return {self.save_to: result, "current_input": result}
