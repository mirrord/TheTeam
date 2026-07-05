"""Unit tests for NewsResearcher models, dates, feeds, terms and report."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from pithos.tools.news_researcher.dates import (
    is_recent,
    parse_feed_date,
    parse_html_date,
    parse_iso_date,
)
from pithos.tools.news_researcher.feeds import parse_feed, _clean_title
from pithos.tools.news_researcher.models import (
    ArticleAssessment,
    NewsArticle,
    NewsReport,
    NewsResearchConfig,
)
from pithos.tools.news_researcher.terms import _parse_terms, extract_terms

# ---------------------------------------------------------------------------
# models
# ---------------------------------------------------------------------------


class TestNewsArticle:
    def test_content_hash_stable_across_whitespace(self) -> None:
        a = NewsArticle(url="u", text="Hello   World")
        b = NewsArticle(url="u", text="hello world")
        assert a.content_hash == b.content_hash and a.content_hash

    def test_published_iso_empty_when_undated(self) -> None:
        a = NewsArticle(url="u", text="x")
        assert a.published_iso == ""

    def test_published_iso_serialises(self) -> None:
        dt = datetime(2025, 7, 1, tzinfo=timezone.utc)
        a = NewsArticle(url="u", text="x", published=dt)
        assert a.published_iso.startswith("2025-07-01")


class TestNewsResearchConfig:
    def test_from_dict_filters_unknown_keys(self) -> None:
        cfg = NewsResearchConfig.from_dict(
            {"recency_days": 5, "garbage": 1, "domains": ["a.com"]}
        )
        assert cfg.recency_days == 5
        assert cfg.domains == ["a.com"]
        assert not hasattr(cfg, "garbage")

    def test_from_dict_none_returns_defaults(self) -> None:
        cfg = NewsResearchConfig.from_dict(None)
        assert cfg.recency_days > 0
        assert cfg.max_articles > 0


class TestNewsReport:
    def _report(self) -> NewsReport:
        return NewsReport(
            inquiry="cache quantization",
            terms=["cache quantization", "transformer"],
            assessments=[
                ArticleAssessment(
                    url="https://a/1",
                    title="KV cache tricks",
                    summary="A summary about quantizing the KV cache.",
                    relevant=True,
                    reason="directly on topic",
                    published_iso="2025-07-01T00:00:00+00:00",
                ),
                ArticleAssessment(
                    url="https://a/2",
                    title="Unrelated",
                    summary="About something else.",
                    relevant=False,
                    reason="off topic",
                ),
            ],
        )

    def test_relevant_property_filters(self) -> None:
        rep = self._report()
        assert len(rep.relevant) == 1
        assert rep.relevant[0].url == "https://a/1"

    def test_to_markdown_lists_relevant_with_summaries_and_source(self) -> None:
        md = self._report().to_markdown()
        assert "# News research report: cache quantization" in md
        assert "cache quantization, transformer" in md
        assert "KV cache tricks" in md
        assert "https://a/1" in md
        assert "quantizing the KV cache" in md
        assert "## Other articles reviewed (1)" in md
        assert "off topic" in md

    def test_to_markdown_handles_no_relevant(self) -> None:
        rep = NewsReport(inquiry="q", terms=[], assessments=[])
        md = rep.to_markdown()
        assert "Relevant articles (0)" in md
        assert "No articles were judged relevant" in md


# ---------------------------------------------------------------------------
# dates
# ---------------------------------------------------------------------------


class TestDates:
    def test_parse_rss_pubdate(self) -> None:
        dt = parse_feed_date("Tue, 01 Jul 2025 13:45:00 +0000")
        assert dt is not None and dt.year == 2025 and dt.month == 7

    def test_parse_atom_iso(self) -> None:
        dt = parse_feed_date("2025-07-01T13:45:00Z")
        assert dt is not None and dt.tzinfo is not None

    def test_parse_iso_z_suffix(self) -> None:
        assert parse_iso_date("2025-07-01T00:00:00Z") is not None

    def test_parse_iso_date_only(self) -> None:
        assert parse_iso_date("2025-07-01") is not None

    def test_parse_bad_returns_none(self) -> None:
        assert parse_feed_date("not a date") is None
        assert parse_feed_date("") is None
        assert parse_feed_date(None) is None

    def test_is_recent_true_for_recent(self) -> None:
        recent = datetime.now(timezone.utc) - timedelta(days=3)
        assert is_recent(recent, 14) is True

    def test_is_recent_false_for_old(self) -> None:
        old = datetime.now(timezone.utc) - timedelta(days=30)
        assert is_recent(old, 14) is False

    def test_is_recent_none_is_false(self) -> None:
        assert is_recent(None, 14) is False

    def test_parse_html_meta_date(self) -> None:
        html = (
            '<html><head><meta property="article:published_time" '
            'content="2025-07-01T10:00:00Z"></head><body>x</body></html>'
        )
        dt = parse_html_date(html)
        assert dt is not None and dt.year == 2025

    def test_parse_html_time_tag(self) -> None:
        html = '<article><time datetime="2025-06-15">June</time></article>'
        dt = parse_html_date(html)
        assert dt is not None and dt.month == 6

    def test_parse_html_jsonld(self) -> None:
        html = (
            '<script type="application/ld+json">'
            '{"@type":"NewsArticle","datePublished":"2025-05-20T08:00:00Z"}'
            "</script>"
        )
        dt = parse_html_date(html)
        assert dt is not None and dt.month == 5

    def test_parse_html_none_when_absent(self) -> None:
        assert parse_html_date("<html><body>no date</body></html>") is None

    def test_parse_iso_slash_date(self) -> None:
        """arxiv citation_date uses YYYY/MM/DD format."""
        dt = parse_iso_date("2025/06/15")
        assert dt is not None and dt.year == 2025 and dt.month == 6 and dt.day == 15

    def test_parse_html_citation_date_meta(self) -> None:
        """arXiv abstract pages carry <meta name="citation_date" content="YYYY/MM/DD">."""
        html = (
            "<html><head>"
            '<meta name="citation_date" content="2025/06/20">'
            "</head><body>x</body></html>"
        )
        dt = parse_html_date(html)
        assert dt is not None and dt.year == 2025 and dt.month == 6 and dt.day == 20

    def test_parse_html_arxiv_submission_history(self) -> None:
        """arXiv abstract pages embed the submission date in [v1] history text."""
        html = (
            '<div class="submission-history">'
            "From: Author [view email]<br>"
            "<b>[v1]</b> Mon, 16 Jun 2025 12:34:56 UTC  (1234kb, 12pp)<br>"
            "</div>"
        )
        dt = parse_html_date(html)
        assert dt is not None and dt.year == 2025 and dt.month == 6 and dt.day == 16

    def test_parse_html_arxiv_submission_history_gmt(self) -> None:
        html = "<b>[v1]</b> Tue, 01 Jul 2025 08:00:00 GMT  (512kb)"
        dt = parse_html_date(html)
        assert dt is not None and dt.year == 2025 and dt.month == 7


# ---------------------------------------------------------------------------
# feeds
# ---------------------------------------------------------------------------


RSS_SAMPLE = """<?xml version="1.0"?>
<rss version="2.0"><channel>
  <title>Example</title>
  <item>
    <title>First article</title>
    <link>https://example.com/first</link>
    <pubDate>Tue, 01 Jul 2025 13:45:00 +0000</pubDate>
  </item>
  <item>
    <title>Second article</title>
    <link>https://example.com/second</link>
    <pubDate>Mon, 30 Jun 2025 09:00:00 +0000</pubDate>
  </item>
</channel></rss>"""

ATOM_SAMPLE = """<?xml version="1.0"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <title>Example</title>
  <entry>
    <title>Atom one</title>
    <link rel="alternate" href="https://example.com/atom-one"/>
    <published>2025-07-01T12:00:00Z</published>
  </entry>
</feed>"""


class TestFeeds:
    def test_parse_rss(self) -> None:
        entries = parse_feed(RSS_SAMPLE)
        assert len(entries) == 2
        assert entries[0].url == "https://example.com/first"
        assert entries[0].title == "First article"
        assert entries[0].published is not None

    def test_parse_atom(self) -> None:
        entries = parse_feed(ATOM_SAMPLE)
        assert len(entries) == 1
        assert entries[0].url == "https://example.com/atom-one"
        assert entries[0].published is not None

    def test_parse_garbage_returns_empty(self) -> None:
        assert parse_feed("not xml") == []
        assert parse_feed("") == []

    def test_arxiv_rss_title_suffix_stripped(self) -> None:
        """arXiv RSS titles include the paper ID; it should be removed."""
        rss = (
            '<?xml version="1.0"?><rss version="2.0"><channel>'
            "<item>"
            "<title>Efficient KV Cache Quantization. (arXiv:2506.12345v1 [cs.LG])</title>"
            "<link>https://arxiv.org/abs/2506.12345</link>"
            "<pubDate>Mon, 16 Jun 2025 00:00:00 +0000</pubDate>"
            "</item>"
            "</channel></rss>"
        )
        entries = parse_feed(rss)
        assert len(entries) == 1
        assert entries[0].title == "Efficient KV Cache Quantization"
        assert entries[0].published is not None

    def test_clean_title_leaves_normal_titles_unchanged(self) -> None:
        assert _clean_title("Normal article title") == "Normal article title"

    def test_clean_title_strips_arxiv_suffix(self) -> None:
        assert (
            _clean_title("Some Paper Title. (arXiv:2301.12345v2 [cs.AI])")
            == "Some Paper Title"
        )


# ---------------------------------------------------------------------------
# terms
# ---------------------------------------------------------------------------


class _TermAgent:
    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.system = ""

    def set_system_prompt(self, p: str) -> None:
        self.system = p

    def send(self, prompt: str, model=None) -> str:
        return self.reply


class TestTerms:
    def test_parse_terms_comma_separated(self) -> None:
        terms = _parse_terms("machine learning, transformer, cache quantization", 6)
        assert terms == ["machine learning", "transformer", "cache quantization"]

    def test_parse_terms_dedup_and_cap(self) -> None:
        terms = _parse_terms("a, a, b, c, d", 3)
        assert terms == ["a", "b", "c"]

    def test_parse_terms_strips_label_and_bullets(self) -> None:
        terms = _parse_terms("Terms: - one, - two", 6)
        assert terms == ["one", "two"]

    def test_extract_terms_uses_model(self) -> None:
        agent = _TermAgent("transformer, attention")
        terms = extract_terms("how do transformers work", agent, max_terms=6)
        assert terms == ["transformer", "attention"]

    def test_extract_terms_falls_back_on_empty(self) -> None:
        agent = _TermAgent("")
        terms = extract_terms("quantization of neural networks", agent, max_terms=4)
        assert terms  # non-empty fallback derived from the inquiry
        assert "the" not in [t.lower() for t in terms]

    def test_extract_terms_empty_inquiry(self) -> None:
        assert extract_terms("", _TermAgent("x")) == []
