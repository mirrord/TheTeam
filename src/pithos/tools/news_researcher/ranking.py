"""Cheap relevance ranking used to pre-filter articles before the LLM stage.

Summarising and judging an article is the most expensive part of the
pipeline, so we avoid spending it on candidates that are obviously off-topic.
:func:`rank_articles` scores each downloaded article against the inquiry and
its derived terms using term overlap alone (no model call, no network), which
is fast, deterministic and dependency-free. The caller then keeps only the
top-K highest-scoring articles for full assessment.
"""

from __future__ import annotations

import re
from typing import Iterable

from .models import NewsArticle
from .terms import _STOPWORDS

_WORD_RE = re.compile(r"[a-z0-9][a-z0-9\-]{2,}")


def _keywords(text: str) -> set[str]:
    """Return informative lowercase keywords from ``text`` (stopwords removed)."""
    return {w for w in _WORD_RE.findall((text or "").lower()) if w not in _STOPWORDS}


def score_article(inquiry: str, terms: Iterable[str], article: NewsArticle) -> float:
    """Return a cheap relevance score for ``article`` against the inquiry.

    Higher is more relevant. Combines three signals: how many search terms
    appear in the article, how much of the inquiry's vocabulary it covers, and
    a small bonus for having a known publication date.
    """
    haystack = f"{article.title or ''} {article.text or ''}".lower()
    if not haystack.strip():
        return 0.0

    score = 0.0
    for term in terms:
        t = (term or "").strip().lower()
        if t and t in haystack:
            score += 2.0

    inquiry_words = _keywords(inquiry)
    if inquiry_words:
        hits = sum(1 for w in inquiry_words if w in haystack)
        score += hits / len(inquiry_words)

    if article.published is not None:
        score += 0.5

    return score


def rank_articles(
    inquiry: str, terms: Iterable[str], articles: list[NewsArticle]
) -> list[NewsArticle]:
    """Return ``articles`` sorted by descending relevance score (stable)."""
    terms = list(terms)
    scored = [(score_article(inquiry, terms, a), i, a) for i, a in enumerate(articles)]
    scored.sort(key=lambda t: (-t[0], t[1]))
    return [a for _, _, a in scored]
