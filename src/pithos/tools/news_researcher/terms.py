"""Derive technical search terms from an inquiry via a small language model.

Step 1 of the news pipeline: a lightweight model turns a free-form inquiry
into a short list of focused technical search terms (e.g. "machine
learning", "transformer", "cache quantization"). These terms drive both
feed filtering and the search fallback.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Optional

logger = logging.getLogger(__name__)


_TERMS_SYSTEM_PROMPT = """\
You extract technical search terms from a research inquiry.

Given an inquiry, reply with a comma-separated list of the most relevant
technical search terms or key phrases (for example: machine learning,
transformer, cache quantization). Rules:
- Output ONLY the comma-separated terms on a single line.
- Prefer concrete technical nouns and named techniques over generic words.
- Keep each term short (1-4 words). Do not number them or add commentary.
"""


def _parse_terms(reply: str, max_terms: int) -> list[str]:
    """Parse a model reply into a de-duplicated list of terms."""
    if not reply:
        return []
    # Take the first non-empty line to avoid trailing prose.
    line = ""
    for candidate in reply.splitlines():
        if candidate.strip():
            line = candidate.strip()
            break
    if not line:
        line = reply.strip()

    # Strip a leading label like "Terms:".
    line = re.sub(r"^\s*(terms|keywords|search terms)\s*[:\-]\s*", "", line, flags=re.I)

    raw_parts = re.split(r"[,;\n]", line)
    terms: list[str] = []
    seen: set[str] = set()
    for part in raw_parts:
        term = re.sub(r"^[\s\-\*\d\.\)]+", "", part).strip().strip("\"'")
        if not term:
            continue
        key = term.lower()
        if key in seen:
            continue
        seen.add(key)
        terms.append(term)
        if len(terms) >= max_terms:
            break
    return terms


def extract_terms(
    inquiry: str,
    agent: Any,
    max_terms: int = 6,
    model: Optional[str] = None,
) -> list[str]:
    """Return up to ``max_terms`` technical search terms for ``inquiry``.

    Falls back to a naive keyword split of the inquiry itself when the
    model call fails or yields nothing, so the pipeline never stalls on
    an empty term list.
    """
    inquiry = (inquiry or "").strip()
    if not inquiry:
        return []

    try:
        agent.set_system_prompt(_TERMS_SYSTEM_PROMPT)
    except Exception:
        pass

    prompt = (
        f"Inquiry: {inquiry}\n\n"
        "List the technical search terms now, comma-separated on one line."
    )
    reply = ""
    try:
        if model is not None:
            reply = agent.send(prompt, model=model)
        else:
            reply = agent.send(prompt)
    except TypeError:
        try:
            reply = agent.send(prompt)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("term extraction failed: %s", exc)
    except Exception as exc:
        logger.warning("term extraction failed: %s", exc)

    terms = _parse_terms(reply or "", max_terms)
    if terms:
        return terms

    # Fallback: keep the most informative words from the inquiry itself.
    words = re.findall(r"[A-Za-z0-9][A-Za-z0-9\-]{2,}", inquiry)
    fallback: list[str] = []
    seen: set[str] = set()
    for w in words:
        key = w.lower()
        if key in _STOPWORDS or key in seen:
            continue
        seen.add(key)
        fallback.append(w)
        if len(fallback) >= max_terms:
            break
    return fallback or [inquiry]


_STOPWORDS = {
    "the", "and", "for", "with", "what", "which", "how", "why", "does",
    "are", "was", "were", "that", "this", "from", "about", "into", "over",
    "recent", "latest", "news", "new", "using", "use", "used",
}
