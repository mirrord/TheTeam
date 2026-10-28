"""Unified JSON tool-call parsing shared across pithos.

Current models are trained on JSON function-calling, so every agent in the
project expresses tool calls the *same* way: one or more JSON objects inside
fenced ``tool_call`` code blocks::

    ```tool_call
    {"name": "run", "arguments": {"command": "git status"}}
    ```

This module is the single source of truth for that syntax.  Both the main agent
tool loop (:class:`pithos.tools.extractor.ToolCallExtractor`) and the coding
harness (:mod:`pithos.tools.coder.actions`) parse replies through
:func:`extract_tool_calls` so the format can never drift between them.

Parsing is intentionally lenient to maximise model adherence:

* a block may hold one object, a JSON array of calls, an object wrapping a
  ``tool_calls`` list, or several objects back-to-back;
* the common key spellings models emit are accepted (``tool``/``function`` for
  the name; ``args``/``parameters`` for the arguments; ``arguments`` may itself
  be a JSON-encoded string);
* the decoder tolerates raw control characters (notably literal newlines inside
  a file's ``content``) that strictly-invalid JSON would reject;
* unfenced JSON tool-call objects are recovered as a fallback.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Iterator, Optional

#: Fence info-strings accepted as tool-call blocks (compared case-insensitively).
#: An empty info-string (a bare ``` fence) is also accepted.
TOOL_FENCE_LANGS = frozenset(
    {"tool_call", "tool_calls", "toolcall", "tool", "json"}
)

#: A fenced code block: ```<lang>\n<body>```  (``lang`` may be empty).
_FENCE_RE = re.compile(r"```[ \t]*([^\n`]*)\r?\n(.*?)```", re.DOTALL)

#: Keys models use for the tool name and for its arguments object.
_NAME_KEYS = ("name", "tool", "function", "action", "tool_name")
_ARG_KEYS = ("arguments", "args", "parameters", "params", "input")

#: A lenient JSON decoder.  ``strict=False`` allows raw control characters
#: (notably literal newlines and tabs) inside string values, which local models
#: frequently emit in a file's ``content`` instead of escaping them as ``\\n``.
_DECODER = json.JSONDecoder(strict=False)


@dataclass
class ParsedToolCall:
    """A single JSON tool call parsed from agent output.

    ``arguments`` merges any nested ``arguments`` object with keys flattened
    directly onto the call object.  ``raw`` is the exact source text of the call
    (the whole fenced block, or the bare JSON object) so callers can strip it
    from a display buffer.
    """

    name: str
    arguments: dict = field(default_factory=dict)
    raw: str = ""


def lookup(data: dict, keys: tuple[str, ...]) -> Any:
    """Return the first present, non-null value among *keys* (case-insensitive)."""
    for key in keys:
        if data.get(key) is not None:
            return data[key]
    lowered = {str(k).lower(): v for k, v in data.items()}
    for key in keys:
        if lowered.get(key) is not None:
            return lowered[key]
    return None


def coerce_args(raw: Any) -> dict:
    """Coerce an ``arguments`` payload (dict or JSON string) into a dict."""
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            parsed = _DECODER.decode(raw)
        except (ValueError, TypeError):
            return {}
        if isinstance(parsed, dict):
            return parsed
    return {}


def to_str(value: Any) -> str:
    """Render a JSON scalar/list argument as a string (lists join on spaces)."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)):
        return " ".join(to_str(v) for v in value)
    return str(value)


def _as_tool_call(value: Any, raw: str) -> Optional[ParsedToolCall]:
    """Normalise a parsed JSON value into a :class:`ParsedToolCall`."""
    if not isinstance(value, dict):
        return None
    name = lookup(value, _NAME_KEYS)
    if not isinstance(name, str) or not name.strip():
        return None
    args = coerce_args(lookup(value, _ARG_KEYS))
    reserved = set(_NAME_KEYS) | set(_ARG_KEYS)
    merged: dict = {k: v for k, v in value.items() if k not in reserved}
    merged.update(args)
    return ParsedToolCall(name=name.strip(), arguments=merged, raw=raw)


def _iter_json_values(body: str) -> Iterator[Any]:
    """Yield the JSON value(s) contained in a fenced block *body*."""
    body = body.strip()
    if not body:
        return
    try:
        parsed = _DECODER.decode(body)
    except ValueError:
        parsed = None
    if parsed is not None:
        if isinstance(parsed, list):
            yield from parsed
            return
        if isinstance(parsed, dict):
            for key in ("tool_calls", "calls", "actions"):
                inner = parsed.get(key)
                if isinstance(inner, list):
                    yield from inner
                    return
            yield parsed
            return
        return
    # Fall back to decoding consecutive JSON values (e.g. objects on separate
    # lines without an enclosing array).
    idx, length = 0, len(body)
    while idx < length:
        while idx < length and body[idx] in " \t\r\n,":
            idx += 1
        if idx >= length:
            break
        try:
            value, idx = _DECODER.raw_decode(body, idx)
        except ValueError:
            break
        yield value


def _scan_bare_json(text: str) -> Iterator[tuple[dict, str]]:
    """Best-effort recovery of unfenced JSON tool-call objects from *text*.

    Yields ``(object, source_text)`` pairs for top-level ``{...}`` objects that
    carry a recognised tool-name key.
    """
    consumed_to = -1
    for match in re.finditer(r"\{", text):
        start = match.start()
        if start < consumed_to:
            continue
        try:
            value, end = _DECODER.raw_decode(text, start)
        except ValueError:
            continue
        consumed_to = end
        if isinstance(value, dict) and lookup(value, _NAME_KEYS) is not None:
            yield value, text[start:end]


def extract_tool_calls(text: str) -> list[ParsedToolCall]:
    """Parse *text* into an ordered list of :class:`ParsedToolCall` objects.

    Tool calls are read from fenced ``tool_call`` code blocks in the order they
    appear.  If the text contains no fenced tool calls, unfenced JSON tool-call
    objects are recovered as a fallback.
    """
    text = text or ""
    calls: list[ParsedToolCall] = []
    for match in _FENCE_RE.finditer(text):
        lang = (match.group(1) or "").strip().lower()
        if lang and lang not in TOOL_FENCE_LANGS:
            continue
        raw = match.group(0)
        for value in _iter_json_values(match.group(2)):
            call = _as_tool_call(value, raw)
            if call is not None:
                calls.append(call)

    if not calls:
        for value, raw in _scan_bare_json(text):
            call = _as_tool_call(value, raw)
            if call is not None:
                calls.append(call)

    return calls


def safe_prefix_len(buf: str) -> int:
    """Return how many leading characters of *buf* are safe to display.

    Holds back any text from the point where an in-progress tool-call fence may
    be starting, so partial ``tool_call`` markup is never flashed to the user.
    Ordinary code fences (e.g. ```python```) are released once their closing
    fence is seen; only tool-call fences (and bare/``json`` fences, which the
    parser also treats as tool calls) are withheld until they complete.
    """
    n = len(buf)
    i = 0
    while True:
        idx = buf.find("```", i)
        if idx == -1:
            # Hold back a trailing partial fence opener (1-2 backticks).
            if buf.endswith("``"):
                return n - 2
            if buf.endswith("`"):
                return n - 1
            return n
        nl = buf.find("\n", idx + 3)
        if nl == -1:
            # Info string not finished yet; this could become a tool fence.
            return idx
        lang = buf[idx + 3:nl].strip().lower()
        if lang == "" or lang in TOOL_FENCE_LANGS:
            # Potential tool-call block in progress -> hold back from here.
            return idx
        # An ordinary code fence: skip past its closing fence and keep scanning.
        close = buf.find("```", nl + 1)
        if close == -1:
            return n
        i = close + 3
