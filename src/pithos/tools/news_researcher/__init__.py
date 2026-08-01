"""NewsResearcher tool - recent-news collection, summarisation and relevance.

Given an inquiry, a small language model derives technical search terms, a
scraper downloads recent (configurable age) articles from a whitelist of
domains via RSS/Atom feeds and a search fallback, and a subagent summarises
each article and judges its relevance. Articles and summaries are persisted
to the knowledge base (a ChromaDB-backed :class:`MemoryStore`); relevant
articles are collected into a Markdown document and returned.

The HTTP/parsing stack (``requests``, ``beautifulsoup4``, ``trafilatura``)
is gated behind the optional ``web`` extra and shared with the
``web-research`` tool. Importing this package never fails when those deps
are missing; consumers should check ``NEWS_RESEARCH_AVAILABLE`` before
instantiating :class:`NewsResearcher`.
"""

from .models import (
    ArticleAssessment,
    NewsArticle,
    NewsReport,
    NewsResearchConfig,
    NewsResearchRequest,
)

try:
    import requests  # noqa: F401
    import bs4  # noqa: F401
    import trafilatura  # noqa: F401

    NEWS_RESEARCH_AVAILABLE = True
except ImportError:
    NEWS_RESEARCH_AVAILABLE = False
    print(
        "News research tool unavailable: missing dependencies. "
        "Install with: pip install -e .[web]"
    )


def __getattr__(name):  # pragma: no cover - thin lazy-import shim
    if name == "NewsResearcher":
        from .researcher import NewsResearcher

        return NewsResearcher
    if name == "NewsResearcherToolExecutor":
        from .researcher import NewsResearcherToolExecutor

        return NewsResearcherToolExecutor
    if name == "NewsScraper":
        from .scraper import NewsScraper

        return NewsScraper
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "ArticleAssessment",
    "NewsArticle",
    "NewsReport",
    "NewsResearchConfig",
    "NewsResearchRequest",
    "NEWS_RESEARCH_AVAILABLE",
    "NewsResearcher",
    "NewsResearcherToolExecutor",
    "NewsScraper",
]
