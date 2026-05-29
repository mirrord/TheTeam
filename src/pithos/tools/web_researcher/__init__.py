"""WebResearcher tool - subagent-controlled web research with deduplicated excerpts.

Given an inquiry, performs per-domain search on a configurable whitelist of
domains, fetches pages, extracts main text, stores deduplicated excerpts in a
per-run ChromaDB collection, then uses a subagent to synthesize a summary
report citing every contributing source URL.

The HTTP/parsing stack (``requests``, ``beautifulsoup4``, ``trafilatura``) is
gated behind the optional ``web`` extra. Importing this package never fails
when those deps are missing; consumers should check
``WEB_RESEARCH_AVAILABLE`` before instantiating :class:`WebResearcher`.
"""

from .models import (
    Excerpt,
    ResearchReport,
    WebResearchConfig,
    WebResearchRequest,
)

try:
    import requests  # noqa: F401
    import bs4  # noqa: F401
    import trafilatura  # noqa: F401

    WEB_RESEARCH_AVAILABLE = True
except ImportError:
    WEB_RESEARCH_AVAILABLE = False


# Lazy imports for the heavy components - they pull in requests/bs4/trafilatura.
def __getattr__(name):  # pragma: no cover - thin lazy-import shim
    if name == "WebResearcher":
        from .researcher import WebResearcher

        return WebResearcher
    if name == "WebResearcherToolExecutor":
        from .researcher import WebResearcherToolExecutor

        return WebResearcherToolExecutor
    if name == "ResearchLoop":
        from .agent_loop import ResearchLoop

        return ResearchLoop
    if name == "Fetcher":
        from .fetcher import Fetcher

        return Fetcher
    if name == "ExcerptStore":
        from .store import ExcerptStore

        return ExcerptStore
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "Excerpt",
    "ResearchReport",
    "WebResearchConfig",
    "WebResearchRequest",
    "WEB_RESEARCH_AVAILABLE",
    "WebResearcher",
    "WebResearcherToolExecutor",
    "ResearchLoop",
    "Fetcher",
    "ExcerptStore",
]
