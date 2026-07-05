"""Data models for the NewsResearcher tool."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional


def _content_hash(text: str) -> str:
    """Return a stable SHA-1 hash of normalised article text.

    Whitespace is collapsed and case-folded before hashing so trivially
    different copies of the same article collide.
    """
    normalised = re.sub(r"\s+", " ", text or "").strip().lower()
    return hashlib.sha1(normalised.encode("utf-8"), usedforsecurity=False).hexdigest()


@dataclass
class NewsArticle:
    """A single news article downloaded during a research run.

    Attributes:
        url: Canonical article URL.
        title: Article headline (best-effort).
        text: Extracted main body text.
        published: Publication timestamp (``None`` when undated).
        source_host: Normalised host the article came from.
        terms_matched: Search terms that surfaced this article.
        content_hash: Stable hash for dedup; computed from ``text``.
        metadata: Free-form metadata stored alongside the article.
    """

    url: str
    title: str = ""
    text: str = ""
    published: Optional[datetime] = None
    source_host: str = ""
    terms_matched: list[str] = field(default_factory=list)
    content_hash: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.content_hash and self.text:
            self.content_hash = _content_hash(self.text)

    @property
    def published_iso(self) -> str:
        """Return the publication date as an ISO string (empty if undated)."""
        return self.published.isoformat() if self.published else ""


@dataclass
class ArticleAssessment:
    """Per-article summary + relevance judgement produced by the subagent."""

    url: str
    title: str
    summary: str
    relevant: bool
    reason: str = ""
    published_iso: str = ""
    source_host: str = ""
    article_entry_id: Optional[str] = None
    summary_entry_id: Optional[str] = None


@dataclass
class NewsResearchConfig:
    """Runtime configuration for a news research run.

    All durations are in seconds; sizes in bytes.
    """

    # --- Sources -----------------------------------------------------------
    domains: list[str] = field(default_factory=list)  # whitelist
    feeds: list[str] = field(default_factory=list)  # RSS/Atom feed URLs
    recency_days: int = 14  # only keep articles newer than this
    skip_undated: bool = True  # drop articles without a parseable date

    # --- Discovery budget --------------------------------------------------
    max_articles: int = 15  # hard cap on articles processed per run
    max_articles_per_source: int = 5
    search_fallback: bool = True  # search whitelisted domains when feeds are thin
    search_results_per_domain: int = 5

    # --- HTTP behaviour ----------------------------------------------------
    request_timeout: float = 15.0
    per_domain_rps: float = 1.0
    max_page_bytes: int = 2_000_000
    user_agent: str = "TheTeam-NewsResearcher/1.0 (+https://github.com/mirrord/theteam)"
    respect_robots: bool = True

    # --- Term extraction (small language model) ----------------------------
    term_model: Optional[str] = None  # small model used to derive search terms
    term_config_name: Optional[str] = None  # optional agent config for terms
    max_terms: int = 6

    # --- Summarise + relevance subagent ------------------------------------
    subagent_config_name: str = "news_researcher"
    subagent_model: Optional[str] = None

    # --- Knowledge base (persistent MemoryStore) ---------------------------
    article_category: str = "news_articles"
    summary_category: str = "news_summaries"
    memory_persist_directory: Optional[str] = None  # None = MemoryStore default

    # --- Output ------------------------------------------------------------
    output_dir: str = "./data/research/news"
    write_document: bool = True

    enabled: bool = True

    @classmethod
    def from_dict(cls, data: Optional[dict[str, Any]]) -> "NewsResearchConfig":
        """Build a config from a (possibly partial) dict, applying defaults."""
        if not data:
            return cls()
        known = {f for f in cls.__dataclass_fields__}
        kwargs = {k: v for k, v in data.items() if k in known}
        return cls(**kwargs)


@dataclass
class NewsResearchRequest:
    """A single news research request."""

    inquiry: str
    domains_override: Optional[list[str]] = None
    feeds_override: Optional[list[str]] = None
    recency_days_override: Optional[int] = None


@dataclass
class NewsReport:
    """Output of a news research run."""

    inquiry: str
    terms: list[str]
    assessments: list[ArticleAssessment]
    document_path: Optional[str] = None
    stats: dict[str, Any] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)

    @property
    def relevant(self) -> list[ArticleAssessment]:
        """Return only the assessments judged relevant to the inquiry."""
        return [a for a in self.assessments if a.relevant]

    def to_markdown(self) -> str:
        """Render the report as a Markdown document.

        Lists every article judged relevant along with its summary and a
        reference to the source, followed by search terms, errors and stats.
        """
        relevant = self.relevant
        parts = [f"# News research report: {self.inquiry}", ""]
        if self.terms:
            parts.append(f"**Search terms:** {', '.join(self.terms)}")
            parts.append("")

        if relevant:
            parts.append(f"## Relevant articles ({len(relevant)})")
            parts.append("")
            for i, a in enumerate(relevant, 1):
                heading = a.title or a.url
                parts.append(f"### {i}. {heading}")
                parts.append("")
                meta = f"Source: {a.url}"
                if a.published_iso:
                    meta += f" · Published: {a.published_iso}"
                parts.append(meta)
                parts.append("")
                parts.append(a.summary.strip() or "_No summary produced._")
                parts.append("")
        else:
            parts.append("## Relevant articles (0)")
            parts.append("")
            parts.append("_No articles were judged relevant to the inquiry._")
            parts.append("")

        irrelevant = [a for a in self.assessments if not a.relevant]
        if irrelevant:
            parts.append(f"## Other articles reviewed ({len(irrelevant)})")
            parts.append("")
            for a in irrelevant:
                heading = a.title or a.url
                reason = f" — {a.reason}" if a.reason else ""
                parts.append(f"- {heading} ({a.url}){reason}")
            parts.append("")

        if self.errors:
            parts.append("## Errors")
            parts.append("")
            for err in self.errors:
                parts.append(f"- {err}")
            parts.append("")

        if self.stats:
            parts.append("## Stats")
            parts.append("")
            for k, v in self.stats.items():
                parts.append(f"- **{k}**: {v}")
        return "\n".join(parts)
