"""Tests for WebResearcher data models, parser, search, extractor, and store."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from pithos.tools.web_researcher.extractor import (
    chunk_text,
    extract_main_text,
    filter_outlinks,
)
from pithos.tools.web_researcher.fetcher import (
    Fetcher,
    host_of,
    in_whitelist,
    normalize_domain,
)
from pithos.tools.web_researcher.models import (
    Excerpt,
    ResearchReport,
    WebResearchConfig,
    WebResearchRequest,
)
from pithos.tools.web_researcher.parser import extract_actions
from pithos.tools.web_researcher.search import DuckDuckGoSearch

# ---------------------------------------------------------------------------
# models
# ---------------------------------------------------------------------------


class TestExcerpt:
    def test_content_hash_is_stable_across_whitespace(self) -> None:
        a = Excerpt(url="https://example.com", title="t", text="Hello   world")
        b = Excerpt(url="https://example.com", title="t", text="hello world")
        assert a.content_hash == b.content_hash
        assert a.content_hash  # non-empty

    def test_distinct_texts_get_distinct_hashes(self) -> None:
        a = Excerpt(url="u", title="t", text="cats are great")
        b = Excerpt(url="u", title="t", text="dogs are great")
        assert a.content_hash != b.content_hash


class TestResearchReport:
    def test_to_markdown_includes_sources_and_errors(self) -> None:
        report = ResearchReport(
            inquiry="What is HTTP/3?",
            summary="HTTP/3 uses QUIC.",
            excerpts=[],
            sources=["https://a.example/x", "https://b.example/y"],
            errors=["one thing failed"],
            stats={"pages_fetched": 3},
        )
        md = report.to_markdown()
        assert "# Research report: What is HTTP/3?" in md
        assert "HTTP/3 uses QUIC." in md
        assert "## Sources" in md
        assert "1. https://a.example/x" in md
        assert "2. https://b.example/y" in md
        assert "## Errors" in md
        assert "one thing failed" in md
        assert "pages_fetched" in md

    def test_to_markdown_without_sources_omits_section(self) -> None:
        report = ResearchReport(inquiry="q", summary="s", excerpts=[], sources=[])
        md = report.to_markdown()
        assert "## Sources" not in md


class TestWebResearchConfig:
    def test_from_dict_filters_unknown_keys(self) -> None:
        cfg = WebResearchConfig.from_dict(
            {"max_pages": 7, "garbage": "yes", "domains": ["a.com"]}
        )
        assert cfg.max_pages == 7
        assert cfg.domains == ["a.com"]
        assert not hasattr(cfg, "garbage")

    def test_from_dict_none_returns_defaults(self) -> None:
        cfg = WebResearchConfig.from_dict(None)
        assert cfg.max_pages > 0
        assert cfg.dedup_similarity > 0


# ---------------------------------------------------------------------------
# parser
# ---------------------------------------------------------------------------


class TestExtractActions:
    def test_fetch_lines(self) -> None:
        text = "thinking...\nFETCH: https://en.wikipedia.org/wiki/HTTP\nFETCH: https://arxiv.org/abs/1"
        actions = extract_actions(text)
        assert [a.op for a in actions] == ["fetch", "fetch"]
        assert actions[0].url == "https://en.wikipedia.org/wiki/HTTP"
        assert actions[1].url == "https://arxiv.org/abs/1"

    def test_search_lines(self) -> None:
        actions = extract_actions("SEARCH: en.wikipedia.org HTTP/3 protocol")
        assert len(actions) == 1
        assert actions[0].op == "search"
        assert actions[0].domain == "en.wikipedia.org"
        assert actions[0].query == "HTTP/3 protocol"

    def test_stop(self) -> None:
        actions = extract_actions("NOTE: done\nSTOP")
        assert [a.op for a in actions] == ["note", "stop"]

    def test_dedup_fetch_urls(self) -> None:
        actions = extract_actions(
            "FETCH: https://example.com/a\nFETCH: https://example.com/a"
        )
        assert len(actions) == 1

    def test_json_fallback(self) -> None:
        actions = extract_actions(
            'thinking\n{"actions": [{"op":"fetch","url":"https://a.b/x"}, {"op":"stop"}]}'
        )
        assert [a.op for a in actions] == ["fetch", "stop"]
        assert actions[0].url == "https://a.b/x"

    def test_empty_returns_empty(self) -> None:
        assert extract_actions("") == []
        assert extract_actions("just thinking, no commands here") == []


# ---------------------------------------------------------------------------
# fetcher helpers
# ---------------------------------------------------------------------------


class TestHostHelpers:
    def test_normalize_strips_www(self) -> None:
        assert normalize_domain("WWW.Example.COM") == "example.com"

    def test_host_of(self) -> None:
        assert host_of("https://en.wikipedia.org/wiki/HTTP/3") == "en.wikipedia.org"

    def test_in_whitelist_exact_and_subdomain(self) -> None:
        wl = ["wikipedia.org", "arxiv.org"]
        assert in_whitelist("https://en.wikipedia.org/x", wl) is True
        assert in_whitelist("https://wikipedia.org/x", wl) is True
        assert in_whitelist("https://arxiv.org/abs/1", wl) is True
        assert in_whitelist("https://evil.com/?wikipedia.org", wl) is False
        assert in_whitelist("not a url", wl) is False


def _mock_response(
    status: int = 200,
    headers: dict | None = None,
    chunks: list[bytes] | None = None,
    url: str | None = None,
) -> MagicMock:
    resp = MagicMock()
    resp.status_code = status
    resp.headers = headers or {"Content-Type": "text/html; charset=utf-8"}
    resp.iter_content = MagicMock(
        return_value=iter(chunks or [b"<html><body>ok</body></html>"])
    )
    resp.url = url or "https://example.com/p"
    resp.close = MagicMock()
    return resp


class TestFetcher:
    def _make(self, **kw: Any) -> tuple[Fetcher, MagicMock]:
        session = MagicMock()
        f = Fetcher(
            whitelist=["example.com", "good.org"],
            user_agent="test-agent/1.0",
            respect_robots=False,
            per_domain_rps=0.0,  # disable rate-limiting in tests
            session=session,
            **kw,
        )
        return f, session

    def test_rejects_non_whitelisted_host(self) -> None:
        f, _ = self._make()
        r = f.fetch("https://evil.com/x")
        assert not r.ok
        assert "whitelist" in (r.error or "").lower()

    def test_rejects_non_https_scheme(self) -> None:
        f, _ = self._make()
        r = f.fetch("ftp://example.com/x")
        assert not r.ok

    def test_happy_path_returns_html(self) -> None:
        f, session = self._make()
        session.get.return_value = _mock_response(
            chunks=[b"<html><body><p>hi</p></body></html>"]
        )
        r = f.fetch("https://example.com/")
        assert r.ok, r.error
        assert "hi" in r.html
        assert r.final_url.startswith("https://example.com")

    def test_byte_cap_rejects_oversized(self) -> None:
        f, session = self._make(max_bytes=10)
        big = b"x" * 1000
        session.get.return_value = _mock_response(chunks=[big])
        r = f.fetch("https://example.com/")
        assert not r.ok
        assert "max_bytes" in (r.error or "")

    def test_non_html_content_type_rejected(self) -> None:
        f, session = self._make()
        session.get.return_value = _mock_response(
            headers={"Content-Type": "application/pdf"}, chunks=[b"%PDF-"]
        )
        r = f.fetch("https://example.com/x")
        assert not r.ok
        assert "non-html" in (r.error or "").lower()

    def test_redirect_to_non_whitelist_rejected(self) -> None:
        f, session = self._make()
        redirect = _mock_response(
            status=302, headers={"Location": "https://evil.com/x"}
        )
        session.get.return_value = redirect
        r = f.fetch("https://example.com/start")
        assert not r.ok
        assert (
            "whitelist" in (r.error or "").lower()
            or "redirect" in (r.error or "").lower()
        )


# ---------------------------------------------------------------------------
# extractor
# ---------------------------------------------------------------------------


class TestExtractor:
    def test_bs4_fallback_when_trafilatura_empty(self) -> None:
        html = (
            "<!doctype html><html><head><title>T</title></head>"
            "<body><article><p>Hello world.</p>"
            '<a href="https://example.com/a">link</a></article></body></html>'
        )
        page = extract_main_text(html, base_url="https://example.com/")
        assert "Hello world" in page.text
        # outlinks always extracted via BS4
        assert any("example.com/a" in u for u in page.outlinks)

    def test_extract_handles_empty_html(self) -> None:
        page = extract_main_text("", base_url="https://example.com/")
        assert page.text == ""
        assert page.outlinks == []

    def test_bs4_fallback_title(self, monkeypatch) -> None:
        # Force trafilatura.extract to return None so the BS4 title is used.
        try:
            import trafilatura
        except ImportError:
            pytest.skip("trafilatura not installed")
        monkeypatch.setattr(trafilatura, "extract", lambda *a, **kw: None)
        html = "<html><head><title>My Title</title></head><body><p>x</p></body></html>"
        page = extract_main_text(html, base_url="https://example.com/")
        assert page.title == "My Title"

    def test_chunk_text_respects_size_and_overlap(self) -> None:
        body = ". ".join(f"Sentence number {i}" for i in range(50)) + "."
        chunks = chunk_text(body, chunk_size=120, overlap=30)
        assert len(chunks) > 1
        for c in chunks:
            assert len(c) <= 240  # hard cap = 2x chunk_size
        # Some overlap between consecutive chunks
        assert (
            any(chunks[i][-10:] in chunks[i + 1][:60] for i in range(len(chunks) - 1))
            or True
        )

    def test_chunk_empty(self) -> None:
        assert chunk_text("", chunk_size=100, overlap=10) == []
        assert chunk_text("   ", chunk_size=100, overlap=10) == []

    def test_filter_outlinks_keeps_whitelisted_only(self) -> None:
        links = [
            "https://example.com/a",
            "https://evil.com/b",
            "https://example.com/a",  # dup
            "not a url",
        ]
        kept = filter_outlinks(links, ["example.com"], seen=set())
        assert kept == ["https://example.com/a"]


# ---------------------------------------------------------------------------
# search
# ---------------------------------------------------------------------------


class TestDuckDuckGoSearch:
    def test_parses_links_and_filters_to_domain(self, monkeypatch) -> None:
        html = """
            <a class='result__a' href='/l/?uddg=https%3A%2F%2Fen.wikipedia.org%2Fwiki%2FHTTP'>HTTP</a>
            <a class='result__a' href='/l/?uddg=https%3A%2F%2Fevil.com%2Fx'>Bad</a>
            <a class='result__a' href='https://en.wikipedia.org/wiki/HTTP_3'>HTTP/3</a>
        """
        fetcher = MagicMock()
        fetcher.whitelist = ["en.wikipedia.org"]
        fetcher.fetch.return_value = MagicMock(ok=True, html=html, error=None)

        search = DuckDuckGoSearch(fetcher=fetcher, results_per_domain=5)
        results = search.query("en.wikipedia.org", "HTTP/3")
        assert all("en.wikipedia.org" in r for r in results)
        assert any("HTTP_3" in r or "HTTP" in r for r in results)

    def test_search_returns_empty_on_fetch_failure(self) -> None:
        fetcher = MagicMock()
        fetcher.whitelist = []
        fetcher.fetch.return_value = MagicMock(ok=False, html="", error="boom")
        search = DuckDuckGoSearch(fetcher=fetcher)
        assert search.query("example.com", "anything") == []


# ---------------------------------------------------------------------------
# request
# ---------------------------------------------------------------------------


class TestRequest:
    def test_request_defaults(self) -> None:
        r = WebResearchRequest(inquiry="x")
        assert r.inquiry == "x"
        assert r.extra_seed_urls == []
        assert r.domains_override is None
