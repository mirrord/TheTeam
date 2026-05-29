"""Data models for the WebResearcher tool."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Any, Optional


def _content_hash(text: str) -> str:
    """Return a stable SHA-1 hash of normalised excerpt text.

    Whitespace is collapsed and case-folded before hashing so trivially
    different copies of the same paragraph collide.
    """
    normalised = re.sub(r"\s+", " ", text or "").strip().lower()
    return hashlib.sha1(normalised.encode("utf-8"), usedforsecurity=False).hexdigest()


@dataclass
class Excerpt:
    """A single text excerpt stored during research.

    Attributes:
        url: Source URL the excerpt was extracted from.
        title: Page title (best-effort).
        text: The excerpt body.
        relevance: Optional 0.0-1.0 score (set by the subagent or heuristics).
        content_hash: Stable hash for fast dedup; computed from ``text``.
        metadata: Free-form metadata stored alongside the excerpt.
    """

    url: str
    title: str
    text: str
    relevance: Optional[float] = None
    content_hash: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.content_hash:
            self.content_hash = _content_hash(self.text)


@dataclass
class WebResearchConfig:
    """Runtime configuration for a research run.

    All durations are in seconds; sizes in bytes.
    """

    domains: list[str] = field(default_factory=list)
    max_pages: int = 20
    max_iterations: int = 8
    request_timeout: float = 15.0
    per_domain_rps: float = 1.0  # max requests per second per domain
    max_page_bytes: int = 2_000_000
    user_agent: str = "TheTeam-WebResearcher/1.0 (+https://github.com/mirrord/theteam)"
    respect_robots: bool = True
    dedup_similarity: float = 0.92  # cosine-distance dedup threshold
    chunk_size: int = 600  # approx chars per excerpt
    chunk_overlap: int = 80
    persist_directory: str = "./data/research"
    keep_collection: bool = False
    subagent_config_name: str = "web_researcher"
    subagent_model: Optional[str] = None  # overrides config if set
    summarizer_model: Optional[str] = None
    search_results_per_domain: int = 5
    enabled: bool = True

    @classmethod
    def from_dict(cls, data: Optional[dict[str, Any]]) -> "WebResearchConfig":
        """Build a config from a (possibly partial) dict, applying defaults."""
        if not data:
            return cls()
        known = {f for f in cls.__dataclass_fields__}
        kwargs = {k: v for k, v in data.items() if k in known}
        return cls(**kwargs)


@dataclass
class WebResearchRequest:
    """A single research request."""

    inquiry: str
    domains_override: Optional[list[str]] = None  # restrict whitelist for this run
    extra_seed_urls: list[str] = field(default_factory=list)


@dataclass
class ResearchReport:
    """Output of a research run."""

    inquiry: str
    summary: str
    excerpts: list[Excerpt]
    sources: list[str]
    stats: dict[str, Any] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)

    def to_markdown(self) -> str:
        """Render the report as a Markdown document with a Sources section."""
        parts = [f"# Research report: {self.inquiry}", "", self.summary.strip(), ""]
        if self.sources:
            parts.append("## Sources")
            parts.append("")
            for i, url in enumerate(self.sources, 1):
                parts.append(f"{i}. {url}")
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
