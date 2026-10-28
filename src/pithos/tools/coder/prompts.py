"""Prompt templates for the coding harness.

These builders render the system message, the first task (instance) message,
per-step observation messages, and format-error nudges.  They are plain Python
string builders (no Jinja) so they have no dependencies and are easy to test.
"""

from __future__ import annotations

from typing import Optional

#: The action grammar shown to the agent, shared by the system + format-error
#: prompts so the two never drift.
ACTION_GRAMMAR = """\
Respond each turn with a short THOUGHT explaining your reasoning, then one or
more tool calls. Emit each tool call as a JSON object inside a fenced
```tool_call code block. Put several calls in one block as a JSON array, or use
a separate block per call. Use ONLY these tools:

1. run — execute a command (shell / CLI tool) in the workspace root:

   ```tool_call
   {"name": "run", "arguments": {"command": "python -m pytest -q"}}
   ```

2. edit — create or modify a file (path is relative to the workspace root):

   ```tool_call
   {"name": "edit", "arguments": {"path": "src/foo.py", "mode": "replace", "content": "<the complete new file contents>"}}
   ```

   mode is one of: replace | create | insert_at_line | patch
     - replace: overwrite the whole file with "content"
     - create: create a new file from "content" (fails if it already exists)
     - insert_at_line: insert "content" before 1-based line N (add "line": N)
     - patch: apply "patch" (a unified diff) to the file instead of "content"

3. submit — finish the task and report what you did:

   ```tool_call
   {"name": "submit", "arguments": {"summary": "<summary of the changes you made and how you verified them>"}}
   ```
"""


def system_prompt() -> str:
    """Return the harness system message."""
    return f"""\
You are an autonomous software engineering agent working directly in a local
code workspace. You can read and search the codebase, edit files, and run
commands (including tests) to complete the task you are given.

{ACTION_GRAMMAR}

Important rules:
- Every reply MUST contain at least one tool call. If you have nothing left
  to do, call submit.
- Work in small, verifiable steps: inspect first, make a focused change, then
  run the relevant tests or commands to confirm it.
- Each run command executes independently in a fresh subshell rooted at the
  workspace. Directory changes and environment variables do NOT persist between
  commands. Combine setup into a single command when needed.
- All file edits and commands are confined to the workspace root; paths that
  escape it are rejected.
- Dangerous commands (shell-injection operators, mass deletes, piping the
  network into a shell) are blocked. Prefer the edit tool over shell
  here-docs / sed for changing files.
- Each tool call must be valid JSON; put multi-line file content in the
  "content" string with newlines escaped as \\n.
- If a command's output is truncated, narrow it (head/tail/grep, or a more
  specific test target) rather than repeating it.
- When the task is complete and verified, issue a single submit tool call.
"""


def build_instance_prompt(
    task: str,
    *,
    root: str,
    system_info: str = "",
    file_listing: str = "",
    test_command: Optional[str] = None,
    tools_summary: str = "",
) -> str:
    """Return the first user message that states the task + workspace context."""
    parts = [
        "## Task",
        "",
        task.strip(),
        "",
        "## Workspace",
        "",
        f"Root (all paths are relative to this): {root}",
    ]
    if system_info:
        parts += ["", f"System: {system_info}"]
    if test_command:
        parts += ["", f"Suggested test command: {test_command}"]
    if tools_summary:
        parts += ["", "## Available commands", "", tools_summary]
    if file_listing:
        parts += ["", "## Files", "", file_listing]
    parts += [
        "",
        "## Recommended workflow",
        "",
        "1. Explore the relevant files to understand the code.",
        "2. Make a focused edit with the edit tool.",
        "3. Run the tests / reproduction command with the run tool.",
        "4. Iterate until it passes, then call submit with a summary.",
        "",
        "Begin now with your first THOUGHT and action.",
    ]
    return "\n".join(parts)


def truncate_output(
    text: str,
    *,
    max_chars: int,
    head_chars: int,
    tail_chars: int,
) -> tuple[str, int]:
    """Return ``(shaped_text, elided_chars)`` for an observation body.

    When *text* is within ``max_chars`` it is returned unchanged with ``0``
    elided.  Otherwise the middle is dropped, keeping the head and tail.
    """
    if text is None:
        return "", 0
    if len(text) <= max_chars:
        return text, 0
    head = text[:head_chars]
    tail = text[-tail_chars:] if tail_chars > 0 else ""
    elided = len(text) - len(head) - len(tail)
    return f"{head}\n... [{elided} characters elided] ...\n{tail}", elided


def build_observation(blocks: list[str]) -> str:
    """Join per-action observation *blocks* into a single user message."""
    body = "\n\n".join(b for b in blocks if b is not None)
    return f"## Observation\n\n{body}\n\nContinue with your next tool call, or call submit if done."


def format_error_prompt(reason: str, consecutive: int, limit: int) -> str:
    """Return the nudge sent when a reply contained no valid action."""
    suffix = ""
    if limit > 0:
        suffix = (
            f"\n\n(This was format error {consecutive} of a maximum {limit}; "
            "the run will stop if you keep replying without a valid action.)"
        )
    return (
        f"Your previous reply could not be parsed: {reason}\n\n"
        f"{ACTION_GRAMMAR}"
        f"{suffix}"
    )
