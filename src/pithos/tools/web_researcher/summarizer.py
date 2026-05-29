"""Synthesize a Markdown report from stored excerpts via the subagent."""

from __future__ import annotations

import logging
from typing import Any, Optional

from .models import Excerpt

logger = logging.getLogger(__name__)


_SUMMARY_SYSTEM_PROMPT = """\
You are a research synthesis assistant.

You will receive an inquiry and a numbered list of excerpts collected from
trusted web sources. Write a clear, well-structured Markdown summary that
addresses the inquiry. Every factual statement should be followed by an
inline citation of the form [N] referring to the excerpt numbers. Do NOT
make up a Sources list - it will be appended automatically.

Prefer concise prose over bullet soup. Note disagreements between
sources explicitly.
"""


def _excerpt_bundle(
    excerpts: list[Excerpt], max_chars: int = 12000
) -> tuple[str, list[str]]:
    """Render excerpts as numbered prompt fodder and return ordered source URLs.

    Excerpts sharing a URL are numbered with the same source index so the
    subagent can cite once per source. The bundle is truncated at
    ``max_chars`` to keep the prompt within reasonable context budgets.
    """
    url_to_idx: dict[str, int] = {}
    sources: list[str] = []
    lines: list[str] = []
    total = 0
    for ex in excerpts:
        url = ex.url or "unknown"
        if url not in url_to_idx:
            url_to_idx[url] = len(sources) + 1
            sources.append(url)
        idx = url_to_idx[url]
        snippet = (ex.text or "").strip()
        if not snippet:
            continue
        header = f"[{idx}] {ex.title or url}"
        block = f"{header}\n{snippet}\n"
        if total + len(block) > max_chars:
            lines.append("\n[... additional excerpts truncated for length ...]")
            break
        lines.append(block)
        total += len(block)
    return "\n".join(lines), sources


def synthesize(
    inquiry: str,
    excerpts: list[Excerpt],
    agent: Any,
    model: Optional[str] = None,
) -> tuple[str, list[str]]:
    """Return ``(summary_markdown, source_urls)`` for the inquiry.

    When ``excerpts`` is empty a graceful "no information found" stub is
    returned without calling the agent.
    """
    if not excerpts:
        return (
            f"No information was collected for the inquiry: **{inquiry}**.",
            [],
        )

    bundle, sources = _excerpt_bundle(excerpts)
    try:
        agent.set_system_prompt(_SUMMARY_SYSTEM_PROMPT)
    except Exception:
        pass

    prompt = (
        f"Inquiry: {inquiry}\n\n"
        "Excerpts (each labelled with its source number):\n\n"
        f"{bundle}\n\n"
        "Write the Markdown summary now. Use [N] inline citations matching the "
        "source numbers above. Do NOT include a Sources section."
    )

    try:
        if model is not None:
            summary = agent.send(prompt, model=model)
        else:
            summary = agent.send(prompt)
    except TypeError:
        # Backends that don't accept the ``model`` kwarg.
        summary = agent.send(prompt)
    except Exception as exc:
        logger.warning("summarizer agent.send failed: %s", exc)
        summary = (
            f"_Synthesis failed: {exc}._\n\n"
            f"{len(excerpts)} excerpts were collected from {len(sources)} sources; "
            "see the Sources section below."
        )

    return (summary or "").strip(), sources
