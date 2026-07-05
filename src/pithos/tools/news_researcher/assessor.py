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
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Optional

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

_COMBINED_SYSTEM_PROMPT = """\
You are a technical news analyst. Given a research inquiry and one article,
do BOTH of the following in a single response:

1. Write a concise summary (3-6 sentences) capturing the key technical
   points, findings, and any named methods, models, or results.
2. Decide whether the article is relevant to the inquiry.

Reply in EXACTLY this format and nothing else:

  SUMMARY: <the summary as a single plain-prose paragraph>
  VERDICT: <RELEVANT or NOT RELEVANT> - <one short sentence of reasoning>

Where RELEVANT means the article materially addresses the inquiry and NOT
RELEVANT means it does not.
"""

_ARTICLE_CHAR_CAP = 8000


def _truncate_body(text: str, char_cap: int) -> str:
    """Trim article text to ``char_cap`` chars using a head+tail slice.

    Keeping both the opening and the closing of an article preserves the
    lede and the conclusions while cutting the (usually less informative)
    middle, giving the model a shorter prompt for faster inference.
    """
    body = (text or "").strip()
    if char_cap <= 0 or len(body) <= char_cap:
        return body
    head = (char_cap * 2) // 3
    tail = char_cap - head
    return body[:head].rstrip() + "\n...\n" + body[-tail:].lstrip()


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
    char_cap: int = _ARTICLE_CHAR_CAP,
) -> str:
    """Return a concise summary of ``article`` produced by the subagent."""
    body = _truncate_body(article.text, char_cap)
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


def summarize_and_judge(
    inquiry: str,
    article: NewsArticle,
    agent: Any,
    model: Optional[str] = None,
    char_cap: int = _ARTICLE_CHAR_CAP,
) -> tuple[str, bool, str]:
    """Summarise ``article`` and judge its relevance in a single LLM call.

    Returns ``(summary, relevant, reason)``. Halves the per-article LLM cost
    versus separate summarise + judge calls.
    """
    body = _truncate_body(article.text, char_cap)
    if not body:
        return "", False, "no article text"
    try:
        agent.set_system_prompt(_COMBINED_SYSTEM_PROMPT)
    except Exception:
        pass
    prompt = (
        f"Inquiry: {inquiry}\n\n"
        f"Article title: {article.title or article.url}\n"
        f"Source: {article.url}\n\n"
        f"Article text:\n{body}\n\n"
        "Respond with the SUMMARY and VERDICT lines now."
    )
    try:
        reply = _send(agent, prompt, model).strip()
    except Exception as exc:
        logger.warning("combined assessment failed [%s]: %s", article.url, exc)
        return "", False, f"assessment failed: {exc}"
    return _parse_combined(reply)


def _parse_combined(reply: str) -> tuple[str, bool, str]:
    """Parse a ``SUMMARY: ... / VERDICT: ...`` reply.

    Robust to the model omitting labels: when no ``VERDICT`` marker is
    present the whole reply is treated as the summary and relevance is
    inferred by scanning the text.
    """
    if not reply:
        return "", False, "no response returned"
    # Locate the VERDICT marker (case-insensitive).
    m = re.search(r"(?im)^\s*verdict\s*[:\-]\s*(.+)$", reply)
    if m:
        verdict_text = m.group(1).strip()
        summary = reply[: m.start()].strip()
    else:
        # No explicit verdict line; infer from the whole reply.
        verdict_text = reply
        summary = reply
    # Strip a leading "SUMMARY:" label from the summary block.
    summary = re.sub(r"(?is)^\s*summary\s*[:\-]\s*", "", summary).strip()
    relevant, reason = _parse_verdict(verdict_text)
    return summary, relevant, reason


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
    agent_factory: Optional[Callable[[], Any]] = None,
) -> list[ArticleAssessment]:
    """Summarise and judge each article, storing summaries in the KB.

    Returns one :class:`ArticleAssessment` per article, in input order.

    When ``config.assess_concurrency`` > 1 and ``agent_factory`` is provided,
    articles are assessed in parallel with one agent per worker thread. When
    ``config.reuse_cached_articles`` is set and a summary for the article's URL
    already exists in the knowledge base, that summary is reused (the article
    is only re-judged against the current inquiry).
    """
    errors = errors if errors is not None else []
    if not articles:
        return []

    def _assess_one(article: NewsArticle, ag: Any) -> ArticleAssessment:
        return _assess_single(inquiry, article, ag, config, memory_store, errors)

    concurrency = max(1, int(config.assess_concurrency or 1))
    if concurrency > 1 and agent_factory is not None and len(articles) > 1:
        assessments = _assess_parallel(
            articles, _assess_one, agent_factory, concurrency
        )
    else:
        assessments = [_assess_one(a, agent) for a in articles]
    return assessments


def _assess_single(
    inquiry: str,
    article: NewsArticle,
    agent: Any,
    config: NewsResearchConfig,
    memory_store: Optional[Any],
    errors: list[str],
) -> ArticleAssessment:
    """Produce a single :class:`ArticleAssessment` for ``article``."""
    model = config.subagent_model
    cap = config.summary_char_cap
    summary_entry_id: Optional[str] = None

    cached_summary, cached_id = (None, None)
    if config.reuse_cached_articles:
        cached_summary, cached_id = _lookup_cached_summary(
            article, config, memory_store
        )

    if cached_summary:
        # Summary is inquiry-independent and already stored: reuse it and only
        # re-judge relevance against the current inquiry (a short call).
        summary = cached_summary
        summary_entry_id = cached_id
        relevant, reason = judge_relevance(inquiry, article, summary, agent, model)
    elif config.combine_summary_and_judgement:
        summary, relevant, reason = summarize_and_judge(
            inquiry, article, agent, model, cap
        )
        summary_entry_id = _store_summary(
            article, summary, inquiry, config, memory_store, errors
        )
    else:
        summary = summarize_article(article, agent, model, cap)
        summary_entry_id = _store_summary(
            article, summary, inquiry, config, memory_store, errors
        )
        relevant, reason = judge_relevance(inquiry, article, summary, agent, model)

    return ArticleAssessment(
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


def _assess_parallel(
    articles: list[NewsArticle],
    assess_one: Callable[[NewsArticle, Any], ArticleAssessment],
    agent_factory: Callable[[], Any],
    concurrency: int,
) -> list[ArticleAssessment]:
    """Assess ``articles`` across a thread pool, one agent per worker thread."""
    local = threading.local()

    def worker(article: NewsArticle) -> ArticleAssessment:
        ag = getattr(local, "agent", None)
        if ag is None:
            ag = agent_factory()
            local.agent = ag
        return assess_one(article, ag)

    results: list[Optional[ArticleAssessment]] = [None] * len(articles)
    workers = min(concurrency, len(articles))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(worker, a): i for i, a in enumerate(articles)}
        for fut in as_completed(futures):
            results[futures[fut]] = fut.result()
    return [r for r in results if r is not None]


def _lookup_cached_summary(
    article: NewsArticle,
    config: NewsResearchConfig,
    memory_store: Optional[Any],
) -> tuple[Optional[str], Optional[str]]:
    """Return ``(summary, entry_id)`` for a previously stored summary, if any."""
    if memory_store is None or not hasattr(memory_store, "retrieve"):
        return None, None
    try:
        results = memory_store.retrieve(
            config.summary_category,
            article.title or article.url,
            n_results=3,
            where={"url": article.url},
            min_relevance=0.0,
        )
    except Exception:
        return None, None
    for r in results or []:
        meta = getattr(r, "metadata", None) or {}
        content = getattr(r, "content", "") or ""
        if meta.get("url") == article.url and content.strip():
            return content, getattr(r, "id", None)
    return None, None


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
