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
from pithos.tools.news_researcher.researcher import (
    NewsResearcher,
    NewsResearcherToolExecutor,
)
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
        self.calls = 0

    def set_system_prompt(self, p: str) -> None:
        self.system = p

    def send(self, prompt: str, model=None) -> str:
        self.calls += 1
        s = self.system.lower()
        if "search terms" in s:
            return "term one, term two"
        # Combined summarise + judge (single call).
        if "analyst" in s:
            verdict = "RELEVANT: on topic" if "KEEP" in prompt else "NOT RELEVANT: off"
            return f"SUMMARY: A concise technical summary.\nVERDICT: {verdict}"
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

    def test_combined_path_uses_one_call_per_article(self) -> None:
        agent = FakeAgent()
        articles = [
            NewsArticle(url="https://a/keep", title="KEEP me", text="body one"),
            NewsArticle(url="https://a/drop", title="ignore", text="body two"),
        ]
        cfg = NewsResearchConfig(combine_summary_and_judgement=True)
        assess_articles("inq", articles, agent, cfg, FakeMemoryStore(), errors=[])
        # One combined LLM call per article (not two).
        assert agent.calls == 2

    def test_separate_path_uses_two_calls_per_article(self) -> None:
        agent = FakeAgent()
        articles = [NewsArticle(url="https://a/x", title="KEEP", text="body")]
        cfg = NewsResearchConfig(combine_summary_and_judgement=False)
        assess_articles("inq", articles, agent, cfg, FakeMemoryStore(), errors=[])
        assert agent.calls == 2  # summarise + judge


class _Result:
    """Minimal stand-in for MemoryStore SearchResult."""

    def __init__(self, id_: str, content: str, metadata: dict) -> None:
        self.id = id_
        self.content = content
        self.metadata = metadata


class CachingMemoryStore(FakeMemoryStore):
    """Memory store that can return canned retrieve() hits keyed by url."""

    def __init__(self, hits: dict | None = None) -> None:
        super().__init__()
        # {(category, url): _Result}
        self.hits = hits or {}
        self.retrieve_calls: list = []

    def retrieve(self, category, query, n_results=None, where=None, min_relevance=None):
        self.retrieve_calls.append((category, query, where))
        url = (where or {}).get("url")
        hit = self.hits.get((category, url))
        return [hit] if hit is not None else []


class TestAssessorCache:
    def test_reuses_cached_summary_and_skips_store(self) -> None:
        cfg = NewsResearchConfig(reuse_cached_articles=True)
        article = NewsArticle(url="https://a/keep", title="KEEP", text="body")
        store = CachingMemoryStore(
            {
                (cfg.summary_category, "https://a/keep"): _Result(
                    "sum_1", "cached summary text", {"url": "https://a/keep"}
                )
            }
        )
        agent = FakeAgent()
        assessments = assess_articles("inq", [article], agent, cfg, store, errors=[])
        assert assessments[0].summary == "cached summary text"
        assert assessments[0].summary_entry_id == "sum_1"
        # Reused summary => no new store() write; only a relevance judgement call.
        assert store.calls == []
        assert agent.calls == 1

    def test_no_reuse_when_disabled(self) -> None:
        cfg = NewsResearchConfig(reuse_cached_articles=False)
        article = NewsArticle(url="https://a/keep", title="KEEP", text="body")
        store = CachingMemoryStore(
            {
                (cfg.summary_category, "https://a/keep"): _Result(
                    "sum_1", "cached summary text", {"url": "https://a/keep"}
                )
            }
        )
        assess_articles("inq", [article], FakeAgent(), cfg, store, errors=[])
        # Fresh summary stored despite a cache hit being available.
        assert len(store.calls) == 1


class TestAssessConcurrency:
    def test_parallel_assessment_uses_factory_and_covers_all(self) -> None:
        cfg = NewsResearchConfig(assess_concurrency=3)
        articles = [
            NewsArticle(url=f"https://a/{i}", title="KEEP", text=f"body {i}")
            for i in range(6)
        ]
        made = []

        def factory():
            ag = FakeAgent()
            made.append(ag)
            return ag

        assessments = assess_articles(
            "inq",
            articles,
            FakeAgent(),
            cfg,
            FakeMemoryStore(),
            errors=[],
            agent_factory=factory,
        )
        assert len(assessments) == 6
        assert all(a.relevant for a in assessments)
        # Worker agents came from the factory (not the single serial agent).
        assert made


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

    def test_executor_populates_report_paths_when_document_written(
        self, tmp_path, monkeypatch
    ) -> None:
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
        r = NewsResearcher(
            MagicMock(),
            config=cfg,
            agent_factory=lambda: FakeAgent(),
            term_agent_factory=lambda: FakeAgent(),
            memory_store=FakeMemoryStore(),
        )
        executor = NewsResearcherToolExecutor(config_manager=None, researcher=r)
        result = executor.run("cache quantization")

        assert result.success is True
        assert len(result.report_paths) == 1
        assert result.report_paths[0].startswith(str(tmp_path))

    def test_prefilter_limits_assessed_articles(self, monkeypatch) -> None:
        # Six downloaded articles; only the top-2 by term overlap get assessed.
        _FakeScraper.articles = [
            NewsArticle(
                url="https://example.com/keep-a",
                title="cache quantization KEEP",
                text="transformer cache quantization body",
                published=datetime.now(timezone.utc),
                source_host="example.com",
            ),
            NewsArticle(
                url="https://example.com/keep-b",
                title="quantization KEEP",
                text="cache quantization details",
                published=datetime.now(timezone.utc),
                source_host="example.com",
            ),
        ] + [
            NewsArticle(
                url=f"https://example.com/misc{i}",
                title="unrelated gadget",
                text="nothing on topic here",
                published=datetime.now(timezone.utc),
                source_host="example.com",
            )
            for i in range(4)
        ]
        monkeypatch.setattr(researcher_mod, "NewsScraper", _FakeScraper)

        cfg = NewsResearchConfig(
            domains=["example.com"],
            feeds=[],
            write_document=False,
            search_fallback=False,
            prefilter_top_k=2,
        )
        r = NewsResearcher(
            MagicMock(),
            config=cfg,
            agent_factory=lambda: FakeAgent(),
            term_agent_factory=lambda: FakeAgent(),
            memory_store=FakeMemoryStore(),
        )
        report = r.research(NewsResearchRequest(inquiry="cache quantization"))

        assert report.stats["articles_downloaded"] == 6
        assert report.stats["articles_assessed"] == 2
        # All six still appear in the report (4 recorded as not-assessed).
        assert len(report.assessments) == 6
        not_assessed = [
            a for a in report.assessments if "not assessed" in (a.reason or "")
        ]
        assert len(not_assessed) == 4


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


# ---------------------------------------------------------------------------
# Scraper: cross-run cache reuse + parallel download
# ---------------------------------------------------------------------------


class TestScraperCache:
    def _config(self, **overrides) -> NewsResearchConfig:
        base = dict(
            domains=["example.com"],
            feeds=["https://feeds.example.com/rss"],
            search_fallback=False,
            recency_days=14,
            reuse_cached_articles=True,
        )
        base.update(overrides)
        return NewsResearchConfig(**base)

    def test_cached_article_reused_without_fetch(self) -> None:
        cfg = self._config()
        recent_iso = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()
        feed = _rss([("Cached", "https://example.com/c1", _recent_rfc822(1))])
        # The feed is served, but the article page itself is NOT — so a reuse
        # miss would surface as a fetch failure rather than a returned article.
        pages = {"https://feeds.example.com/rss": feed}
        store = CachingMemoryStore(
            {
                (cfg.article_category, "https://example.com/c1"): _Result(
                    "art_1",
                    "previously stored body text",
                    {
                        "url": "https://example.com/c1",
                        "title": "Cached",
                        "published": recent_iso,
                        "source_host": "example.com",
                    },
                )
            }
        )
        scraper = NewsScraper(cfg, FakeFetcher(pages, ["example.com"]), None, store)
        articles = scraper.gather("q", ["x"])
        assert len(articles) == 1
        assert articles[0].text == "previously stored body text"
        assert articles[0].metadata.get("cached") is True
        # Reused article is not re-stored.
        assert store.calls == []

    def test_stale_cached_article_skipped(self) -> None:
        cfg = self._config()
        old_iso = (datetime.now(timezone.utc) - timedelta(days=60)).isoformat()
        feed = _rss([("Old", "https://example.com/old", _recent_rfc822(1))])
        pages = {"https://feeds.example.com/rss": feed}
        store = CachingMemoryStore(
            {
                (cfg.article_category, "https://example.com/old"): _Result(
                    "art_old",
                    "stale body",
                    {
                        "url": "https://example.com/old",
                        "published": old_iso,
                        "source_host": "example.com",
                    },
                )
            }
        )
        scraper = NewsScraper(cfg, FakeFetcher(pages, ["example.com"]), None, store)
        assert scraper.gather("q", ["x"]) == []


class TestScraperParallel:
    def test_parallel_download_preserves_order_and_count(self) -> None:
        items = [
            (f"Item {i}", f"https://example.com/a{i}", _recent_rfc822(i + 1))
            for i in range(5)
        ]
        feed = _rss(items)
        pages = {"https://feeds.example.com/rss": feed}
        for i in range(5):
            pages[f"https://example.com/a{i}"] = _article_html(
                f"Item {i}", f"unique body number {i}"
            )
        cfg = NewsResearchConfig(
            domains=["example.com"],
            feeds=["https://feeds.example.com/rss"],
            search_fallback=False,
            recency_days=30,
            download_concurrency=4,
            max_articles_per_source=10,
        )
        store = FakeMemoryStore()
        scraper = NewsScraper(cfg, FakeFetcher(pages, ["example.com"]), None, store)
        articles = scraper.gather("q", ["item"])
        assert len(articles) == 5
        # Feed order is preserved despite concurrent fetching.
        assert [a.url for a in articles] == [
            f"https://example.com/a{i}" for i in range(5)
        ]

    def test_candidate_cap_limits_downloads(self) -> None:
        items = [
            (f"Item {i}", f"https://example.com/a{i}", _recent_rfc822(1))
            for i in range(20)
        ]
        feed = _rss(items)
        pages = {"https://feeds.example.com/rss": feed}
        for i in range(20):
            pages[f"https://example.com/a{i}"] = _article_html(f"Item {i}", f"body {i}")
        cfg = NewsResearchConfig(
            domains=["example.com"],
            feeds=["https://feeds.example.com/rss"],
            search_fallback=False,
            recency_days=30,
            max_candidates=5,
            max_articles=100,
            max_articles_per_source=100,
        )
        store = FakeMemoryStore()
        scraper = NewsScraper(cfg, FakeFetcher(pages, ["example.com"]), None, store)
        articles = scraper.gather("q", ["item"])
        # Never fetch more than the candidate cap.
        assert len(articles) <= 5
