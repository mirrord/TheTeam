"""Unit tests for NewsResearcher models, dates, feeds, terms and report."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from pithos.tools.news_researcher.dates import (
    is_recent,
    parse_feed_date,
    parse_html_date,
    parse_iso_date,
)
from pithos.tools.news_researcher.assessor import (
    _parse_combined,
    _truncate_body,
    summarize_and_judge,
)
from pithos.tools.news_researcher.feeds import parse_feed, _clean_title
from pithos.tools.news_researcher.models import (
    ArticleAssessment,
    NewsArticle,
    NewsReport,
    NewsResearchConfig,
)
from pithos.tools.news_researcher.ranking import rank_articles, score_article
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


# ---------------------------------------------------------------------------
# assessor: combined parse + truncation + single-call assessment
# ---------------------------------------------------------------------------


class TestTruncateBody:
    def test_short_text_unchanged(self) -> None:
        assert _truncate_body("hello world", 100) == "hello world"

    def test_long_text_head_and_tail_kept(self) -> None:
        body = "START " + ("x" * 5000) + " END"
        out = _truncate_body(body, 200)
        assert len(out) <= 220  # cap + ellipsis marker
        assert out.startswith("START")
        assert out.endswith("END")
        assert "..." in out

    def test_zero_cap_disables_truncation(self) -> None:
        body = "a" * 500
        assert _truncate_body(body, 0) == body


class TestParseCombined:
    def test_parses_summary_and_verdict(self) -> None:
        reply = (
            "SUMMARY: The paper introduces a new KV cache scheme.\n"
            "VERDICT: RELEVANT - directly on topic"
        )
        summary, relevant, reason = _parse_combined(reply)
        assert summary == "The paper introduces a new KV cache scheme."
        assert relevant is True
        assert "on topic" in reason

    def test_parses_not_relevant(self) -> None:
        reply = "SUMMARY: Unrelated gadget review.\nVERDICT: NOT RELEVANT - off topic"
        summary, relevant, reason = _parse_combined(reply)
        assert summary == "Unrelated gadget review."
        assert relevant is False

    def test_missing_verdict_marker_infers(self) -> None:
        reply = "This is clearly RELEVANT to the inquiry."
        summary, relevant, _ = _parse_combined(reply)
        assert relevant is True
        assert summary  # whole reply retained as summary

    def test_empty_reply(self) -> None:
        summary, relevant, reason = _parse_combined("")
        assert summary == "" and relevant is False and reason


class _CombinedAgent:
    """Agent that returns a combined SUMMARY/VERDICT and counts calls."""

    def __init__(self, relevant: bool = True) -> None:
        self.system = ""
        self.calls = 0
        self._relevant = relevant

    def set_system_prompt(self, p: str) -> None:
        self.system = p

    def send(self, prompt: str, model=None) -> str:
        self.calls += 1
        verdict = "RELEVANT - yes" if self._relevant else "NOT RELEVANT - no"
        return f"SUMMARY: A summary.\nVERDICT: {verdict}"


class TestSummarizeAndJudge:
    def test_single_call_returns_all_three(self) -> None:
        agent = _CombinedAgent(relevant=True)
        article = NewsArticle(url="u", title="t", text="body text")
        summary, relevant, reason = summarize_and_judge("inq", article, agent)
        assert agent.calls == 1  # one LLM call, not two
        assert summary == "A summary."
        assert relevant is True

    def test_empty_body_no_call(self) -> None:
        agent = _CombinedAgent()
        article = NewsArticle(url="u", text="")
        summary, relevant, reason = summarize_and_judge("inq", article, agent)
        assert agent.calls == 0
        assert summary == "" and relevant is False


# ---------------------------------------------------------------------------
# ranking (pre-filter)
# ---------------------------------------------------------------------------


class TestRanking:
    def test_score_rewards_term_matches(self) -> None:
        hit = NewsArticle(url="a", title="Transformer cache", text="quantization work")
        miss = NewsArticle(url="b", title="Cooking recipes", text="how to bake bread")
        s_hit = score_article(
            "cache quantization", ["transformer", "quantization"], hit
        )
        s_miss = score_article(
            "cache quantization", ["transformer", "quantization"], miss
        )
        assert s_hit > s_miss

    def test_rank_orders_by_relevance(self) -> None:
        a = NewsArticle(url="a", title="unrelated", text="nothing here")
        b = NewsArticle(url="b", title="transformer quantization", text="cache methods")
        c = NewsArticle(url="c", title="quantization", text="some cache detail")
        ranked = rank_articles(
            "cache quantization", ["transformer", "quantization"], [a, b, c]
        )
        assert ranked[0].url == "b"
        assert ranked[-1].url == "a"

    def test_rank_stable_for_ties(self) -> None:
        arts = [NewsArticle(url=str(i), text="zzz") for i in range(4)]
        ranked = rank_articles("x", [], arts)
        assert [a.url for a in ranked] == ["0", "1", "2", "3"]
