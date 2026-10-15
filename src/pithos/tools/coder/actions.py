"""Parse a coding subagent's reply into executable actions.

The harness uses a small block grammar so multi-line file content can be
embedded safely (unlike a single-line command string).  Three block types are
recognised, matched in the order they appear in the reply:

* **Run a command** (routed through the workspace-jailed tool registry)::

      [RUN]python -m pytest -q[/RUN]

* **Edit a file** (routed through the root-jailed file-edit helpers)::

      [EDIT path=src/foo.py mode=replace]
      <new file contents>
      [/EDIT]

  ``mode`` is one of ``replace`` / ``create`` / ``insert_at_line`` / ``patch``
  (default ``replace``).  ``line`` is required for ``insert_at_line``.  For
  ``patch`` the body is a unified diff.

  For robustness the parser also accepts **mode-named edit blocks** that models
  frequently emit instead of ``[EDIT ... mode=X]``: ``[CREATE ...]``,
  ``[REPLACE ...]``, ``[WRITE ...]`` (= replace), ``[PATCH ...]`` and
  ``[INSERT ...]`` / ``[INSERT_AT_LINE ...]``.  The block name sets the mode
  (an explicit ``mode=`` attribute still wins).

* **Submit the final result** and end the run (``[DONE]`` / ``[FINISH]`` are
  accepted as aliases)::

      [SUBMIT]
      <summary of what was done>
      [/SUBMIT]

Parsing is case-insensitive.  Any reply that contains no recognised block is a
*format error*.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

# Edit-block tag names (longest alternatives first so e.g. INSERT_AT_LINE is
# preferred over INSERT). Each maps to a file-edit mode; ``EDIT`` defers to the
# block's ``mode=`` attribute.
_TAG_TO_MODE: dict[str, Optional[str]] = {
    "EDIT": None,
    "CREATE": "create",
    "REPLACE": "replace",
    "WRITE": "replace",
    "PATCH": "patch",
    "INSERT_AT_LINE": "insert_at_line",
    "INSERT": "insert_at_line",
}
_EDIT_TAGS = "EDIT|CREATE|REPLACE|WRITE|PATCH|INSERT_AT_LINE|INSERT"
_SUBMIT_TAGS = "SUBMIT|DONE|FINISH"
_RUN_TAGS = "RUN|EXEC"

# One master matcher. ``re.DOTALL`` so bodies span multiple lines; non-greedy so
# adjacent blocks do not merge. ``re.IGNORECASE`` tolerates lower-case tags.
# Closing tags use backreferences so an open/close pair must agree.
_BLOCK_RE = re.compile(
    rf"\[(?P<etag>{_EDIT_TAGS})\b(?P<eattrs>[^\]]*)\](?P<ebody>.*?)\[/(?P=etag)\]"
    rf"|\[(?P<stag>{_SUBMIT_TAGS})\b[^\]]*\](?P<sbody>.*?)\[/(?P=stag)\]"
    rf"|\[(?P<rtag>{_RUN_TAGS})\](?P<rbody>.*?)\[/(?P=rtag)\]",
    re.DOTALL | re.IGNORECASE,
)
_ATTR_RE = re.compile(r"(\w+)\s*=\s*(\"[^\"]*\"|'[^']*'|\S+)")


@dataclass
class Action:
    """A single parsed action."""

    kind: str  # "run" | "edit" | "submit"
    raw: str = ""
    # run
    command: str = ""
    # edit
    path: str = ""
    mode: str = "replace"
    line: Optional[int] = None
    body: str = ""
    # submit
    summary: str = ""


def _parse_attrs(attr_text: str) -> dict[str, str]:
    attrs: dict[str, str] = {}
    for key, value in _ATTR_RE.findall(attr_text or ""):
        if value and value[0] in "\"'" and value[-1] == value[0]:
            value = value[1:-1]
        attrs[key.lower()] = value
    return attrs


def _strip_block_body(body: str) -> str:
    """Trim a single leading/trailing newline that frames a block body.

    File content inside edit blocks is usually written on its own lines, which
    introduces a newline right after ``]`` and before the closing tag.  We
    strip exactly one such framing newline on each side while preserving the
    author's internal blank lines and trailing content newline intent.
    """
    if body.startswith("\r\n"):
        body = body[2:]
    elif body.startswith("\n"):
        body = body[1:]
    if body.endswith("\r\n"):
        body = body[:-2]
    elif body.endswith("\n"):
        body = body[:-1]
    return body


def _build_edit_action(tag: str, attrs_text: str, body_raw: str, raw: str) -> Action:
    attrs = _parse_attrs(attrs_text)
    body = _strip_block_body(body_raw)
    # An explicit mode= attribute wins; otherwise the block name sets the mode.
    tag_mode = _TAG_TO_MODE.get(tag.upper())
    mode = attrs.get("mode") or tag_mode or "replace"
    line_val = attrs.get("line")
    line = None
    if line_val is not None:
        try:
            line = int(line_val)
        except ValueError:
            line = None
    return Action(
        kind="edit",
        raw=raw,
        path=attrs.get("path", ""),
        mode=mode,
        line=line,
        body=body,
    )


def parse_actions(text: str) -> list[Action]:
    """Parse *text* into an ordered list of :class:`Action` objects.

    Blocks are returned in the order they appear.  An empty list means the
    reply contained no recognised action (a format error).
    """
    actions: list[Action] = []
    for match in _BLOCK_RE.finditer(text or ""):
        raw = match.group(0)
        if match.group("etag"):
            actions.append(
                _build_edit_action(
                    match.group("etag"),
                    match.group("eattrs"),
                    match.group("ebody"),
                    raw,
                )
            )
        elif match.group("stag"):
            summary = _strip_block_body(match.group("sbody") or "").strip()
            actions.append(Action(kind="submit", raw=raw, summary=summary))
        elif match.group("rtag"):
            command = (match.group("rbody") or "").strip()
            if command:
                actions.append(Action(kind="run", raw=raw, command=command))
    return actions


def edit_action_to_spec(action: Action) -> dict:
    """Convert an ``edit`` :class:`Action` into a ``file_ops.apply_edit`` spec."""
    spec: dict = {"path": action.path, "mode": action.mode}
    if action.mode == "patch":
        spec["patch"] = action.body
    else:
        spec["content"] = action.body
    if action.line is not None:
        spec["line"] = action.line
    return spec
