"""Parse a coding subagent's reply into executable actions.

The harness uses the project-wide JSON tool-call syntax (see
:mod:`pithos.tools.tool_calls`): each turn the agent emits one or more tool
calls inside fenced ``tool_call`` code blocks.  This module is a thin adapter
that maps those parsed calls onto the three tools the coding harness exposes::

    ```tool_call
    {"name": "run", "arguments": {"command": "python -m pytest -q"}}
    ```

* **run** — execute a shell / CLI command in the workspace root
  (``{"name": "run", "arguments": {"command": "..."}}``).

* **edit** — create or modify a file, jailed to the workspace root
  (``{"name": "edit", "arguments": {"path": "src/foo.py", "mode": "replace",
  "content": "..."}}``).  ``mode`` is one of ``replace`` / ``create`` /
  ``insert_at_line`` / ``patch`` (default ``replace``); ``line`` is required
  for ``insert_at_line``; for ``patch`` the unified diff is supplied as
  ``patch`` instead of ``content``.

* **submit** — finish the run and report what was done
  (``{"name": "submit", "arguments": {"summary": "..."}}``).

A reply that yields no recognised tool call is a *format error*.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from ..tool_calls import ParsedToolCall, extract_tool_calls, lookup, to_str

#: Tool-name aliases -> canonical action kind.
_NAME_TO_KIND: dict[str, str] = {
    "run": "run",
    "exec": "run",
    "shell": "run",
    "bash": "run",
    "command": "run",
    "edit": "edit",
    "submit": "submit",
    "done": "submit",
    "finish": "submit",
}

_VALID_MODES = ("replace", "create", "insert_at_line", "patch")


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


def _call_to_action(call: ParsedToolCall) -> Optional[Action]:
    """Map a parsed JSON tool call onto a harness :class:`Action`."""
    kind = _NAME_TO_KIND.get(call.name.strip().lower())
    if kind is None:
        return None
    args = call.arguments
    raw = call.raw

    if kind == "run":
        command = to_str(lookup(args, ("command", "cmd", "shell", "run"))).strip()
        if not command:
            return None
        return Action(kind="run", raw=raw, command=command)

    if kind == "submit":
        summary = to_str(
            lookup(args, ("summary", "message", "text", "result", "report"))
        ).strip()
        return Action(kind="submit", raw=raw, summary=summary)

    # edit
    path = to_str(lookup(args, ("path", "file", "filename", "filepath"))).strip()
    mode = to_str(lookup(args, ("mode",))).strip().lower() or "replace"
    if mode not in _VALID_MODES:
        mode = "replace"

    line: Optional[int] = None
    line_val = lookup(args, ("line", "line_number", "at_line"))
    if line_val is not None:
        try:
            line = int(line_val)
        except (ValueError, TypeError):
            line = None

    if mode == "patch":
        body = lookup(args, ("patch", "diff", "content", "body"))
    else:
        body = lookup(args, ("content", "body", "text"))
    return Action(
        kind="edit",
        raw=raw,
        path=path,
        mode=mode,
        line=line,
        body=to_str(body),
    )


def parse_actions(text: str) -> list[Action]:
    """Parse *text* into an ordered list of :class:`Action` objects.

    Tool calls are read (via the shared :func:`extract_tool_calls`) in the order
    they appear.  An empty list means the reply contained no recognised tool
    call (a format error).
    """
    actions: list[Action] = []
    for call in extract_tool_calls(text):
        action = _call_to_action(call)
        if action is not None:
            actions.append(action)
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
