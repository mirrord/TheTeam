"""Parse a coding subagent's reply into executable actions.

The harness uses a small, unambiguous block grammar so multi-line file content
can be embedded safely (unlike a single-line command string).  Three block
types are recognised, matched in the order they appear in the reply:

* **Run a command** (routed through the workspace-jailed tool registry)::

      [RUN]python -m pytest -q[/RUN]

* **Edit a file** (routed through the root-jailed file-edit helpers)::

      [EDIT path=src/foo.py mode=replace]
      <new file contents>
      [/EDIT]

  ``mode`` is one of ``replace`` / ``create`` / ``insert_at_line`` / ``patch``
  (default ``replace``).  ``line`` is required for ``insert_at_line``.  For
  ``patch`` the body is a unified diff.

* **Submit the final result** and end the run::

      [SUBMIT]
      <summary of what was done>
      [/SUBMIT]

Any reply that contains no recognised block is a *format error*.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

# Block matchers. ``re.DOTALL`` so bodies may span multiple lines; non-greedy so
# adjacent blocks do not merge.
_EDIT_RE = re.compile(r"\[EDIT\b([^\]]*)\](.*?)\[/EDIT\]", re.DOTALL)
_SUBMIT_RE = re.compile(r"\[SUBMIT\](.*?)\[/SUBMIT\]", re.DOTALL)
_RUN_RE = re.compile(r"\[(?:RUN|EXEC)\](.*?)\[/(?:RUN|EXEC)\]", re.DOTALL)
_COMBINED_RE = re.compile(
    r"\[EDIT\b[^\]]*\].*?\[/EDIT\]"
    r"|\[SUBMIT\].*?\[/SUBMIT\]"
    r"|\[(?:RUN|EXEC)\].*?\[/(?:RUN|EXEC)\]",
    re.DOTALL,
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

    File content inside ``[EDIT]`` blocks is usually written on its own lines,
    which introduces a newline right after ``]`` and before ``[/EDIT]``.  We
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


def _build_edit_action(raw: str) -> Optional[Action]:
    m = _EDIT_RE.match(raw)
    if not m:
        return None
    attrs = _parse_attrs(m.group(1))
    body = _strip_block_body(m.group(2))
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
        mode=attrs.get("mode", "replace"),
        line=line,
        body=body,
    )


def parse_actions(text: str) -> list[Action]:
    """Parse *text* into an ordered list of :class:`Action` objects.

    Blocks are returned in the order they appear.  An empty list means the
    reply contained no recognised action (a format error).
    """
    actions: list[Action] = []
    for match in _COMBINED_RE.finditer(text or ""):
        raw = match.group(0)
        if raw.startswith("[EDIT"):
            action = _build_edit_action(raw)
            if action is not None:
                actions.append(action)
        elif raw.startswith("[SUBMIT]"):
            sm = _SUBMIT_RE.match(raw)
            summary = _strip_block_body(sm.group(1)) if sm else ""
            actions.append(Action(kind="submit", raw=raw, summary=summary.strip()))
        else:  # RUN / EXEC
            rm = _RUN_RE.match(raw)
            command = rm.group(1).strip() if rm else ""
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
