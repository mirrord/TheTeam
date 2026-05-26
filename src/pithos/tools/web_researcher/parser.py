"""Parser + extractor for the WebResearcher subagent reply grammar.

Primary grammar (line-based, friendly to small local models):

    FETCH: <url>            - fetch a page and store relevant excerpts
    SEARCH: <domain> <query> - request a new per-domain search
    NOTE: <text>             - record a free-form note (kept in trace only)
    STOP                     - end the crawl loop

Fallback: a single JSON object of the shape ``{"actions": [{"op": "fetch",
"url": "..."}, {"op": "stop"}]}``. Used when the model emits structured
JSON instead of the line-based format.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Optional


@dataclass
class ResearchAction:
    """A single action requested by the subagent."""

    op: str  # 'fetch' | 'search' | 'note' | 'stop'
    url: Optional[str] = None
    domain: Optional[str] = None
    query: Optional[str] = None
    note: Optional[str] = None


_FETCH_RE = re.compile(r"^\s*FETCH\s*:\s*(\S+.*?)\s*$", re.MULTILINE | re.IGNORECASE)
_SEARCH_RE = re.compile(
    r"^\s*SEARCH\s*:\s*(\S+)\s+(.+?)\s*$", re.MULTILINE | re.IGNORECASE
)
_NOTE_RE = re.compile(r"^\s*NOTE\s*:\s*(.+?)\s*$", re.MULTILINE | re.IGNORECASE)
_STOP_RE = re.compile(r"^\s*STOP\s*\.?\s*$", re.MULTILINE | re.IGNORECASE)
_JSON_BLOCK_RE = re.compile(r"\{[\s\S]*\}")


def extract_actions(content: str) -> list[ResearchAction]:
    """Extract :class:`ResearchAction` objects from a subagent message.

    The line-based grammar is tried first. If it yields *no* actions, a
    JSON fallback is attempted. Unknown lines are silently ignored.
    """
    if not content:
        return []

    actions: list[ResearchAction] = []
    seen_fetch: set[str] = set()

    for m in _FETCH_RE.finditer(content):
        url = m.group(1).strip().rstrip(",;.")
        # Strip surrounding quotes/brackets if the model added them.
        url = url.strip("\"'<>[]() ")
        if url and url not in seen_fetch:
            seen_fetch.add(url)
            actions.append(ResearchAction(op="fetch", url=url))

    for m in _SEARCH_RE.finditer(content):
        domain = m.group(1).strip().strip("\"'<>[]() ")
        query = m.group(2).strip().strip("\"'")
        if domain and query:
            actions.append(ResearchAction(op="search", domain=domain, query=query))

    for m in _NOTE_RE.finditer(content):
        actions.append(ResearchAction(op="note", note=m.group(1).strip()))

    if _STOP_RE.search(content):
        actions.append(ResearchAction(op="stop"))

    if actions:
        return actions

    # JSON fallback.
    match = _JSON_BLOCK_RE.search(content)
    if not match:
        return []
    try:
        data = json.loads(match.group(0))
    except (ValueError, TypeError):
        return []
    raw_actions = data.get("actions") if isinstance(data, dict) else None
    if not isinstance(raw_actions, list):
        return []
    for item in raw_actions:
        if not isinstance(item, dict):
            continue
        op = (item.get("op") or "").lower()
        if op == "fetch" and item.get("url"):
            url = str(item["url"]).strip()
            if url not in seen_fetch:
                seen_fetch.add(url)
                actions.append(ResearchAction(op="fetch", url=url))
        elif op == "search" and item.get("domain") and item.get("query"):
            actions.append(
                ResearchAction(
                    op="search",
                    domain=str(item["domain"]).strip(),
                    query=str(item["query"]).strip(),
                )
            )
        elif op == "note" and item.get("note"):
            actions.append(ResearchAction(op="note", note=str(item["note"])))
        elif op == "stop":
            actions.append(ResearchAction(op="stop"))
    return actions
