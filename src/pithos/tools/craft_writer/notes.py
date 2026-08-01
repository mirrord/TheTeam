"""Craft-note retrieval helpers for the CraftWriter tool.

Reads prescriptive craft notes previously produced by the ``craft-notes``
tool (see :mod:`pithos.tools.craft_analyzer`) back out of the shared
:class:`~pithos.tools.memory_tool.MemoryStore` knowledge base, so a
story-writing subagent can be guided by them.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)


def retrieve_craft_notes(
    memory_store: Optional[Any],
    dimensions: list[str],
    query: str,
    per_dimension: int = 5,
    min_relevance: Optional[float] = None,
    source_title: Optional[str] = None,
    category: str = "craft_notes",
) -> dict[str, list[Any]]:
    """Retrieve up to ``per_dimension`` relevant notes for each dimension.

    Args:
        memory_store: A :class:`~pithos.tools.memory_tool.MemoryStore`-like
            object exposing ``retrieve()``. When ``None`` (knowledge base
            unavailable) an empty result set is returned for every dimension.
        dimensions: Craft dimensions to retrieve notes for.
        query: Semantic search text (typically the user's story direction).
        per_dimension: Max notes to retrieve per dimension.
        min_relevance: Optional relevance threshold override.
        source_title: When given, restricts retrieval to notes derived from
            the story with this title (via a ``source_title`` metadata match).
        category: MemoryStore category notes were stored under.

    Returns:
        Dict mapping each dimension to a list of results (whatever
        ``memory_store.retrieve()`` returns, typically
        :class:`~pithos.tools.memory_tool.SearchResult` instances). A
        per-dimension retrieval failure is logged and yields an empty list
        for that dimension rather than aborting the whole run.
    """
    results: dict[str, list[Any]] = {d: [] for d in dimensions}
    if memory_store is None or not query or not query.strip():
        return results

    for dimension in dimensions:
        where: dict[str, Any]
        if source_title:
            where = {
                "$and": [
                    {"dimension": dimension},
                    {"source_title": source_title},
                ]
            }
        else:
            where = {"dimension": dimension}
        try:
            hits = memory_store.retrieve(
                category=category,
                query=query,
                n_results=per_dimension,
                where=where,
                min_relevance=min_relevance,
            )
        except Exception as exc:
            logger.warning(
                "failed to retrieve craft notes [dimension=%s]: %s", dimension, exc
            )
            hits = []
        results[dimension] = list(hits or [])
    return results


def format_notes_for_prompt(notes_by_dimension: dict[str, list[Any]]) -> str:
    """Render retrieved notes (grouped by dimension) for prompt injection.

    Each note is rendered as ``- <note text>`` with an indented evidence
    quote when available. Dimensions with no notes are omitted.
    """
    lines: list[str] = []
    for dimension, hits in notes_by_dimension.items():
        if not hits:
            continue
        heading = dimension.replace("_", " ").title()
        lines.append(f"{heading}:")
        for hit in hits:
            content = getattr(hit, "content", None)
            if content is None and isinstance(hit, dict):
                content = hit.get("content", "")
            content = (content or "").strip()
            if not content:
                continue
            lines.append(f"- {content}")
            metadata = getattr(hit, "metadata", None)
            if metadata is None and isinstance(hit, dict):
                metadata = hit.get("metadata", {})
            evidence = (metadata or {}).get("evidence", "").strip()
            if evidence:
                lines.append(f"  (e.g. {evidence})")
        lines.append("")

    if not lines:
        return "No craft notes are available yet; write with general best practices."
    return "\n".join(lines).strip()
