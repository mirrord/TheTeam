"""Per-article summarisation and relevance judgement (subagent stages).

Steps 3-5 of the pipeline. For each downloaded article the subagent:

1. Produces a concise summary, which is stored in the knowledge base with a
   reference back to the source article.
2. Judges whether the article is relevant to the original inquiry.

The two prompts are deliberately separate so the relevance decision is made
against the model's own summary plus the inquiry, matching the requested
flow.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Optional

from .models import ArticleAssessment, NewsArticle, NewsResearchConfig

logger = logging.getLogger(__name__)


_SUMMARY_SYSTEM_PROMPT = """\
You are a technical news summariser. Given a single article, write a concise
summary (3-6 sentences) capturing the key technical points, findings, and
any named methods, models, or results. Write plain prose only - no preamble,
no headings, no bullet lists.
"""

_RELEVANCE_SYSTEM_PROMPT = """\
You decide whether a news article is relevant to a research inquiry.

You are given the inquiry and a summary of one article. Reply with EXACTLY
one line in the form:

  <VERDICT>: <one short sentence of reasoning>

Where VERDICT is one of:
  RELEVANT     - the article materially addresses the inquiry
  NOT RELEVANT - the article does not address the inquiry

Output nothing else.
"""

_ARTICLE_CHAR_CAP = 8000


def _send(agent: Any, prompt: str, model: Optional[str]) -> str:
    """Send a prompt to the agent, tolerating backends without ``model``."""
    try:
        if model is not None:
            return agent.send(prompt, model=model) or ""
        return agent.send(prompt) or ""
    except TypeError:
        return agent.send(prompt) or ""


def summarize_article(
    article: NewsArticle,
    agent: Any,
    model: Optional[str] = None,
) -> str:
    """Return a concise summary of ``article`` produced by the subagent."""
    body = (article.text or "").strip()[:_ARTICLE_CHAR_CAP]
    if not body:
        return ""
    try:
        agent.set_system_prompt(_SUMMARY_SYSTEM_PROMPT)
    except Exception:
        pass
    prompt = (
        f"Article title: {article.title or article.url}\n"
        f"Source: {article.url}\n\n"
        f"Article text:\n{body}\n\n"
        "Write the summary now."
    )
    try:
        return _send(agent, prompt, model).strip()
    except Exception as exc:
        logger.warning("article summarisation failed [%s]: %s", article.url, exc)
        return ""


def judge_relevance(
    inquiry: str,
    article: NewsArticle,
    summary: str,
    agent: Any,
    model: Optional[str] = None,
) -> tuple[bool, str]:
    """Return ``(relevant, reason)`` for ``article`` against ``inquiry``."""
    try:
        agent.set_system_prompt(_RELEVANCE_SYSTEM_PROMPT)
    except Exception:
        pass
    prompt = (
        f"Inquiry: {inquiry}\n\n"
        f"Article title: {article.title or article.url}\n"
        f"Article summary:\n{summary or '(no summary available)'}\n\n"
        "Is this article relevant to the inquiry? Reply with the verdict line."
    )
    try:
        reply = _send(agent, prompt, model).strip()
    except Exception as exc:
        logger.warning("relevance judgement failed [%s]: %s", article.url, exc)
        return False, f"relevance check failed: {exc}"
    return _parse_verdict(reply)


def _parse_verdict(reply: str) -> tuple[bool, str]:
    """Parse a ``VERDICT: reason`` line into ``(relevant, reason)``."""
    if not reply:
        return False, "no verdict returned"
    first = ""
    for line in reply.splitlines():
        if line.strip():
            first = line.strip()
            break
    verdict_part, _, reason = first.partition(":")
    reason = reason.strip()
    verdict = verdict_part.strip().lower()
    # Robust to "RELEVANT", "not relevant", "irrelevant" phrasings.
    if re.search(r"\bnot\b|\birrelevant\b", verdict):
        return False, reason or first
    if "relevant" in verdict:
        return True, reason or first
    # Fall back to scanning the whole reply.
    low = reply.lower()
    if re.search(r"\bnot relevant\b|\birrelevant\b", low):
        return False, reason or first
    if "relevant" in low:
        return True, reason or first
    return False, reason or first


def assess_articles(
    inquiry: str,
    articles: list[NewsArticle],
    agent: Any,
    config: NewsResearchConfig,
    memory_store: Optional[Any] = None,
    errors: Optional[list[str]] = None,
) -> list[ArticleAssessment]:
    """Summarise and judge each article, storing summaries in the KB.

    Returns one :class:`ArticleAssessment` per article, in input order.
    """
    errors = errors if errors is not None else []
    assessments: list[ArticleAssessment] = []
    model = config.subagent_model

    for article in articles:
        summary = summarize_article(article, agent, model)
        summary_entry_id = _store_summary(
            article, summary, inquiry, config, memory_store, errors
        )
        relevant, reason = judge_relevance(inquiry, article, summary, agent, model)
        assessments.append(
            ArticleAssessment(
                url=article.url,
                title=article.title,
                summary=summary,
                relevant=relevant,
                reason=reason,
                published_iso=article.published_iso,
                source_host=article.source_host,
                article_entry_id=article.metadata.get("article_entry_id"),
                summary_entry_id=summary_entry_id,
            )
        )
    return assessments


def _store_summary(
    article: NewsArticle,
    summary: str,
    inquiry: str,
    config: NewsResearchConfig,
    memory_store: Optional[Any],
    errors: list[str],
) -> Optional[str]:
    """Persist the article summary in the knowledge base with a source ref."""
    if memory_store is None or not summary.strip():
        return None
    metadata = {
        "url": article.url,
        "title": article.title or "",
        "published": article.published_iso,
        "source_host": article.source_host,
        "inquiry": inquiry,
        "article_entry_id": article.metadata.get("article_entry_id") or "",
        "kind": "news_summary",
    }
    try:
        return memory_store.store(config.summary_category, summary, metadata)
    except Exception as exc:
        errors.append(f"summary store failed [{article.url}]: {exc}")
        return None
