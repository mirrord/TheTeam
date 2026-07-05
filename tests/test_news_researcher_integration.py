"""Integration tests for the NewsResearcher scraper, assessor and facade."""

from __future__ import annotations

import os
from email.utils import format_datetime
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

import pithos.tools.news_researcher.researcher as researcher_mod
from pithos.tools.news_researcher.assessor import assess_articles
from pithos.tools.news_researcher.models import (
    NewsArticle,
    NewsResearchConfig,
    NewsResearchRequest,
)
from pithos.tools.news_researcher.researcher import NewsResearcher
from pithos.tools.news_researcher.scraper import NewsScraper, _normalize_url
from pithos.tools.web_researcher.fetcher import FetchResult, normalize_domain

# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class FakeFetcher:
    """Minimal fetcher returning canned pages from a URL -> html map."""

    def __init__(self, pages: dict, whitelist: list) -> None:
        self.pages = pages
        self.whitelist = [normalize_domain(d) for d in whitelist]

    def fetch(self, url, **kwargs) -> FetchResult:
        html = self.pages.get(url)
        if html is None:
            return FetchResult(url, url, 404, "", "", error="not found")
        return FetchResult(url, url, 200, "text/html", html)


class FakeMemoryStore:
    def __init__(self) -> None:
        self.calls: list[tuple] = []
        self._n = 0

    def store(self, category, content, metadata=None) -> str:
        self._n += 1
        self.calls.append((category, content, metadata or {}))
        return f"id_{self._n}"


class FakeAgent:
    """Agent whose replies depend on the currently-set system prompt."""

    def __init__(self) -> None:
        self.system = ""

    def set_system_prompt(self, p: str) -> None:
        self.system = p

    def send(self, prompt: str, model=None) -> str:
        s = self.system.lower()
        if "search terms" in s:
            return "term one, term two"
        if "summariser" in s or "summarise" in s:
            return "A concise technical summary."
        if "relevant" in s:
            return "RELEVANT: on topic" if "KEEP" in prompt else "NOT RELEVANT: off"
        return ""


def _recent_rfc822(days_ago: int = 1) -> str:
    return format_datetime(datetime.now(timezone.utc) - timedelta(days=days_ago))


def _rss(items: list[tuple[str, str, str]]) -> str:
    body = "".join(
        f"<item><title>{t}</title><link>{u}</link><pubDate>{d}</pubDate></item>"
        for t, u, d in items
    )
    return f'<?xml version="1.0"?><rss version="2.0"><channel>{body}</channel></rss>'


def _article_html(title: str, body: str) -> str:
    return f"<html><head><title>{title}</title></head><body><article>{body}</article></body></html>"


# ---------------------------------------------------------------------------
# NewsScraper
# ---------------------------------------------------------------------------


class TestNewsScraper:
    def _config(self, **overrides) -> NewsResearchConfig:
        base = dict(
            domains=["example.com"],
            feeds=["https://feeds.example.com/rss"],
            search_fallback=False,
            recency_days=14,
        )
        base.update(overrides)
        return NewsResearchConfig(**base)

    def test_gathers_recent_feed_articles_and_stores_them(self) -> None:
        feed = _rss(
            [
                ("Alpha", "https://example.com/a1", _recent_rfc822(1)),
                ("Beta", "https://example.com/a2", _recent_rfc822(2)),
            ]
        )
        pages = {
            "https://feeds.example.com/rss": feed,
            "https://example.com/a1": _article_html("Alpha", "Content about term one."),
            "https://example.com/a2": _article_html("Beta", "Different content here."),
        }
        store = FakeMemoryStore()
        scraper = NewsScraper(
            self._config(), FakeFetcher(pages, ["example.com"]), None, store
        )
        articles = scraper.gather("inquiry", ["term one"])
        assert len(articles) == 2
        assert all(a.published is not None for a in articles)
        # Both article bodies stored in the KB under the article category.
        assert len(store.calls) == 2
        assert all(c[0] == "news_articles" for c in store.calls)

    def test_old_articles_are_filtered_out(self) -> None:
        feed = _rss([("Old", "https://example.com/old", _recent_rfc822(60))])
        pages = {
            "https://feeds.example.com/rss": feed,
            "https://example.com/old": _article_html("Old", "stale"),
        }
        scraper = NewsScraper(
            self._config(), FakeFetcher(pages, ["example.com"]), None, FakeMemoryStore()
        )
        assert scraper.gather("q", ["x"]) == []

    def test_duplicate_bodies_are_deduped(self) -> None:
        feed = _rss(
            [
                ("One", "https://example.com/1", _recent_rfc822(1)),
                ("Two", "https://example.com/2", _recent_rfc822(1)),
            ]
        )
        pages = {
            "https://feeds.example.com/rss": feed,
            "https://example.com/1": _article_html("Same", "identical body text"),
            "https://example.com/2": _article_html("Same", "identical body text"),
        }
        scraper = NewsScraper(
            self._config(), FakeFetcher(pages, ["example.com"]), None, FakeMemoryStore()
        )
        articles = scraper.gather("q", ["x"])
        assert len(articles) == 1

    def test_off_whitelist_feed_links_ignored(self) -> None:
        feed = _rss([("Bad", "https://evil.com/x", _recent_rfc822(1))])
        pages = {"https://feeds.example.com/rss": feed}
        scraper = NewsScraper(
            self._config(), FakeFetcher(pages, ["example.com"]), None, FakeMemoryStore()
        )
        assert scraper.gather("q", ["x"]) == []

    def test_search_path_uses_html_date_and_skip_undated(self) -> None:
        class FakeSearch:
            def query(self, domain, query, n=None):
                return ["https://example.com/s1", "https://example.com/s2"]

        dated = (
            '<html><head><meta property="article:published_time" '
            f'content="{(datetime.now(timezone.utc) - timedelta(days=2)).isoformat()}">'
            "</head><body><article>fresh news body</article></body></html>"
        )
        undated = _article_html("No date", "no date body")
        pages = {
            "https://example.com/s1": dated,
            "https://example.com/s2": undated,
        }
        cfg = self._config(feeds=[], search_fallback=True, skip_undated=True)
        scraper = NewsScraper(
            cfg, FakeFetcher(pages, ["example.com"]), FakeSearch(), FakeMemoryStore()
        )
        articles = scraper.gather("q", ["news"])
        # Only the dated article survives skip_undated.
        assert [a.url for a in articles] == ["https://example.com/s1"]


# ---------------------------------------------------------------------------
# assessor
# ---------------------------------------------------------------------------


class TestAssessor:
    def test_summarizes_and_judges_and_stores_summaries(self) -> None:
        articles = [
            NewsArticle(url="https://a/keep", title="KEEP me", text="body one"),
            NewsArticle(url="https://a/drop", title="ignore", text="body two"),
        ]
        store = FakeMemoryStore()
        cfg = NewsResearchConfig()
        assessments = assess_articles(
            "my inquiry", articles, FakeAgent(), cfg, store, errors=[]
        )
        assert [a.relevant for a in assessments] == [True, False]
        assert all(a.summary for a in assessments)
        # Two summaries stored under the summary category.
        assert len(store.calls) == 2
        assert all(c[0] == "news_summaries" for c in store.calls)
        assert all(a.summary_entry_id for a in assessments)


# ---------------------------------------------------------------------------
# facade
# ---------------------------------------------------------------------------


class _FakeScraper:
    articles: list = []

    def __init__(self, config, fetcher, search, memory_store) -> None:
        self.errors: list = []

    def gather(self, inquiry, terms):
        return list(_FakeScraper.articles)


class TestNewsResearcherFacade:
    def test_no_domains_returns_error_report(self) -> None:
        r = NewsResearcher(MagicMock(), config=NewsResearchConfig(domains=[], feeds=[]))
        report = r.research("anything")
        assert report.errors
        assert report.assessments == []

    def test_full_pipeline_writes_document(self, tmp_path, monkeypatch) -> None:
        _FakeScraper.articles = [
            NewsArticle(
                url="https://example.com/keep",
                title="KEEP this",
                text="body",
                published=datetime.now(timezone.utc),
                source_host="example.com",
            )
        ]
        monkeypatch.setattr(researcher_mod, "NewsScraper", _FakeScraper)

        cfg = NewsResearchConfig(
            domains=["example.com"],
            feeds=[],
            output_dir=str(tmp_path),
            search_fallback=True,
        )
        store = FakeMemoryStore()
        r = NewsResearcher(
            MagicMock(),
            config=cfg,
            agent_factory=lambda: FakeAgent(),
            term_agent_factory=lambda: FakeAgent(),
            memory_store=store,
        )
        report = r.research(NewsResearchRequest(inquiry="cache quantization"))

        assert report.terms == ["term one", "term two"]
        assert len(report.relevant) == 1
        assert report.document_path and os.path.exists(report.document_path)
        with open(report.document_path, encoding="utf-8") as fh:
            content = fh.read()
        assert "KEEP this" in content
        assert report.stats["articles_relevant"] == 1


# ---------------------------------------------------------------------------
# arXiv URL normalisation
# ---------------------------------------------------------------------------


class TestNormalizeUrl:
    def test_pdf_redirected_to_abs(self) -> None:
        assert (
            _normalize_url("https://arxiv.org/pdf/2301.12345")
            == "https://arxiv.org/abs/2301.12345"
        )

    def test_html_redirected_to_abs(self) -> None:
        assert (
            _normalize_url("https://arxiv.org/html/2301.12345v2")
            == "https://arxiv.org/abs/2301.12345v2"
        )

    def test_eprint_redirected_to_abs(self) -> None:
        assert (
            _normalize_url("https://arxiv.org/e-print/2301.12345")
            == "https://arxiv.org/abs/2301.12345"
        )

    def test_abs_unchanged(self) -> None:
        url = "https://arxiv.org/abs/2301.12345"
        assert _normalize_url(url) == url

    def test_non_arxiv_unchanged(self) -> None:
        url = "https://example.com/article/123"
        assert _normalize_url(url) == url

    def test_arxiv_pdf_url_fetched_as_abs(self) -> None:
        """End-to-end: an arXiv PDF link is re-routed to the abstract page."""
        recent = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()
        feed = (
            '<?xml version="1.0"?><rss version="2.0"><channel>'
            f"<item><title>Paper</title>"
            f"<link>https://arxiv.org/pdf/2506.12345</link>"
            f"<pubDate>{_recent_rfc822(1)}</pubDate>"
            "</item></channel></rss>"
        )
        abs_html = _article_html("Paper", "abstract body text about transformers")
        pages = {
            "https://feeds.example.com/rss": feed,
            # Only the /abs/ URL is served — PDF would return nothing
            "https://arxiv.org/abs/2506.12345": abs_html,
        }
        cfg = NewsResearchConfig(
            domains=["arxiv.org"],
            feeds=["https://feeds.example.com/rss"],
            search_fallback=False,
            recency_days=14,
        )
        store = FakeMemoryStore()
        scraper = NewsScraper(cfg, FakeFetcher(pages, ["arxiv.org"]), None, store)
        articles = scraper.gather("transformer research", ["transformer"])
        assert len(articles) == 1
        assert "arxiv.org/abs/" in articles[0].url
