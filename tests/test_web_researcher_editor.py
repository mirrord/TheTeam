"""Unit tests for the WebResearcher editor (citation verifier + rewriter)."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

from pithos.tools.web_researcher.editor import (
    edit_summary,
    extract_citations,
    verify_citation,
    verify_citations,
    verify_sources,
)
from pithos.tools.web_researcher.models import (
    CitationCheck,
    Excerpt,
    SourceStatus,
)

# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class _ScriptedAgent:
    """Agent stub that returns a queue of canned replies."""

    def __init__(self, replies: list[str]) -> None:
        self._replies = list(replies)
        self.sent: list[str] = []
        self.system_prompts: list[str] = []

    def send(self, prompt: str, **kw: Any) -> str:
        self.sent.append(prompt)
        if not self._replies:
            return ""
        return self._replies.pop(0)

    def set_system_prompt(self, p: str) -> None:
        self.system_prompts.append(p)


class _FailingAgent:
    def send(self, prompt: str, **kw: Any) -> str:
        raise RuntimeError("agent down")

    def set_system_prompt(self, p: str) -> None:
        pass


class _FakeStore:
    def __init__(self, urls: list[str]) -> None:
        self._urls = list(urls)

    def sources(self) -> list[str]:
        return list(self._urls)


# ---------------------------------------------------------------------------
# extract_citations
# ---------------------------------------------------------------------------


class TestExtractCitations:
    def test_empty_summary_returns_empty(self) -> None:
        assert extract_citations("") == []
        assert extract_citations("no citations here.") == []

    def test_single_citation_at_end_of_sentence(self) -> None:
        text = "Python is a programming language [1]. It is widely used."
        out = extract_citations(text)
        assert len(out) == 1
        idx, _, claim = out[0]
        assert idx == 1
        assert "Python is a programming language" in claim
        assert "widely used" not in claim

    def test_multiple_citations_one_sentence(self) -> None:
        text = "Foo is fast [1] and safe [2]. Bar is slow [3]."
        out = extract_citations(text)
        assert [i for i, _, _ in out] == [1, 2, 3]
        assert out[0][2] == out[1][2]  # same claim sentence
        assert "Bar is slow" in out[2][2]

    def test_multi_digit_index(self) -> None:
        text = "Big claim [42]."
        out = extract_citations(text)
        assert out[0][0] == 42


# ---------------------------------------------------------------------------
# verify_sources
# ---------------------------------------------------------------------------


class TestVerifySources:
    def test_store_hit_shortcut(self) -> None:
        store = _FakeStore(["https://a/x"])
        statuses = verify_sources(["https://a/x"], store, fetcher=None)
        assert len(statuses) == 1
        assert statuses[0].exists is True
        assert statuses[0].status_code == 200

    def test_uncached_without_fetcher_unverified(self) -> None:
        store = _FakeStore([])
        statuses = verify_sources(["https://a/x"], store, fetcher=None)
        assert statuses[0].exists is False
        assert "no fetcher" in (statuses[0].error or "")

    def test_uncached_uses_fetcher(self) -> None:
        store = _FakeStore([])
        fetcher = MagicMock()
        fetcher.verify_url.return_value = (True, 200, None)
        statuses = verify_sources(["https://b/y"], store, fetcher=fetcher)
        fetcher.verify_url.assert_called_once_with("https://b/y")
        assert statuses[0].exists is True
        assert statuses[0].status_code == 200

    def test_fetcher_failure_recorded(self) -> None:
        store = _FakeStore([])
        fetcher = MagicMock()
        fetcher.verify_url.return_value = (False, 404, "HTTP 404")
        statuses = verify_sources(["https://b/y"], store, fetcher=fetcher)
        assert statuses[0].exists is False
        assert statuses[0].status_code == 404

    def test_fetcher_exception_caught(self) -> None:
        store = _FakeStore([])
        fetcher = MagicMock()
        fetcher.verify_url.side_effect = RuntimeError("boom")
        statuses = verify_sources(["https://b/y"], store, fetcher=fetcher)
        assert statuses[0].exists is False
        assert "boom" in (statuses[0].error or "")

    def test_empty_url(self) -> None:
        store = _FakeStore([])
        statuses = verify_sources([""], store, fetcher=None)
        assert statuses[0].exists is False


# ---------------------------------------------------------------------------
# verify_citation / verify_citations
# ---------------------------------------------------------------------------


class TestVerifyCitation:
    def test_supported_verdict(self) -> None:
        agent = _ScriptedAgent(["SUPPORTED: directly stated in excerpt"])
        ex = [Excerpt(url="u", title="t", text="The sky is blue.")]
        check = verify_citation("the sky is blue", "u", ex, agent, index=1)
        assert check.verdict == "supported"
        assert "directly stated" in check.reason

    def test_partial_verdict(self) -> None:
        agent = _ScriptedAgent(["PARTIAL: backs half"])
        ex = [Excerpt(url="u", title="t", text="some text")]
        check = verify_citation("claim", "u", ex, agent, index=2)
        assert check.verdict == "partial"

    def test_unsupported_verdict(self) -> None:
        agent = _ScriptedAgent(["UNSUPPORTED: nothing about it"])
        ex = [Excerpt(url="u", title="t", text="some text")]
        check = verify_citation("claim", "u", ex, agent, index=3)
        assert check.verdict == "unsupported"

    def test_unparseable_defaults_unsupported(self) -> None:
        agent = _ScriptedAgent(["maybe? not sure honestly"])
        ex = [Excerpt(url="u", title="t", text="text")]
        check = verify_citation("claim", "u", ex, agent, index=4)
        assert check.verdict == "unsupported"
        assert "unparseable" in check.reason

    def test_no_excerpts_unsupported(self) -> None:
        agent = _ScriptedAgent(["SUPPORTED: ..."])  # should not be called
        check = verify_citation("claim", "u", [], agent, index=5)
        assert check.verdict == "unsupported"
        assert agent.sent == []

    def test_agent_error_unsupported(self) -> None:
        agent = _FailingAgent()
        ex = [Excerpt(url="u", title="t", text="text")]
        check = verify_citation("claim", "u", ex, agent, index=6)
        assert check.verdict == "unsupported"
        assert "agent error" in check.reason

    def test_verdict_with_leading_text(self) -> None:
        # Loose match: tag appears in the first line.
        agent = _ScriptedAgent(["Final answer: SUPPORTED because excerpt matches"])
        ex = [Excerpt(url="u", title="t", text="text")]
        check = verify_citation("claim", "u", ex, agent, index=1)
        assert check.verdict == "supported"


class TestVerifyCitationsBatch:
    def test_runs_one_per_marker(self) -> None:
        agent = _ScriptedAgent(
            [
                "SUPPORTED: ok",
                "UNSUPPORTED: missing",
            ]
        )
        summary = "Claim one [1]. Claim two [2]."
        sources = ["https://a", "https://b"]
        excerpts = [
            Excerpt(url="https://a", title="t", text="claim one is true"),
            Excerpt(url="https://b", title="t", text="unrelated text"),
        ]
        checks = verify_citations(summary, sources, excerpts, agent)
        assert [c.index for c in checks] == [1, 2]
        assert checks[0].verdict == "supported"
        assert checks[1].verdict == "unsupported"

    def test_out_of_range_index_unsupported(self) -> None:
        agent = _ScriptedAgent([])
        summary = "Foo [9]."
        checks = verify_citations(
            summary, sources=["https://a"], excerpts=[], agent=agent
        )
        assert checks[0].verdict == "unsupported"
        assert "no matching source" in checks[0].reason


# ---------------------------------------------------------------------------
# edit_summary
# ---------------------------------------------------------------------------


class TestEditSummary:
    def test_noop_when_all_supported(self) -> None:
        agent = _ScriptedAgent(["should not be called"])
        summary = "All good [1]."
        checks = [
            CitationCheck(
                index=1, source_url="u", claim="all good", verdict="supported"
            )
        ]
        statuses = [SourceStatus(url="u", exists=True, status_code=200)]
        out = edit_summary("q", summary, checks, statuses, agent)
        assert out == summary
        assert agent.sent == []

    def test_rewrite_called_when_unsupported(self) -> None:
        agent = _ScriptedAgent(["Rewritten summary without bad claim."])
        summary = "Good [1]. Bad [2]."
        checks = [
            CitationCheck(index=1, source_url="u1", claim="good", verdict="supported"),
            CitationCheck(index=2, source_url="u2", claim="bad", verdict="unsupported"),
        ]
        statuses = [
            SourceStatus(url="u1", exists=True, status_code=200),
            SourceStatus(url="u2", exists=True, status_code=200),
        ]
        out = edit_summary("q", summary, checks, statuses, agent)
        assert out == "Rewritten summary without bad claim."
        assert len(agent.sent) == 1
        assert "[2]" in agent.sent[0]

    def test_rewrite_called_when_dead_source(self) -> None:
        agent = _ScriptedAgent(["Cleaned."])
        summary = "Stale [1]."
        checks = [
            CitationCheck(index=1, source_url="u", claim="stale", verdict="supported")
        ]
        statuses = [SourceStatus(url="u", exists=False, error="HTTP 404")]
        out = edit_summary("q", summary, checks, statuses, agent)
        assert out == "Cleaned."

    def test_agent_failure_returns_original(self) -> None:
        agent = _FailingAgent()
        summary = "Bad [1]."
        checks = [
            CitationCheck(index=1, source_url="u", claim="bad", verdict="unsupported")
        ]
        statuses = [SourceStatus(url="u", exists=True, status_code=200)]
        out = edit_summary("q", summary, checks, statuses, agent)
        assert out == summary

    def test_empty_rewrite_returns_original(self) -> None:
        agent = _ScriptedAgent(["   "])  # whitespace-only
        summary = "Bad [1]."
        checks = [
            CitationCheck(index=1, source_url="u", claim="bad", verdict="unsupported")
        ]
        statuses = [SourceStatus(url="u", exists=True, status_code=200)]
        out = edit_summary("q", summary, checks, statuses, agent)
        assert out == summary


# ---------------------------------------------------------------------------
# End-to-end research with verify_citations
# ---------------------------------------------------------------------------


class TestResearcherCitationVerification:
    def _make_researcher_with_stub_loop(
        self, monkeypatch, verify: bool, editor_replies
    ):
        """Patch the heavy bits so research() exercises the editor stage."""
        from pithos.tools.web_researcher import researcher as researcher_mod
        from pithos.tools.web_researcher.models import (
            Excerpt as _Excerpt,
            WebResearchConfig,
        )

        cm = MagicMock()
        cm.get_config.return_value = None

        cfg = WebResearchConfig(
            domains=["example.com"],
            verify_citations=verify,
        )

        # Patch fetcher / search / store creation.
        monkeypatch.setattr(researcher_mod, "Fetcher", MagicMock())
        monkeypatch.setattr(researcher_mod, "DuckDuckGoSearch", MagicMock())

        fake_store = MagicMock()
        fake_store.all.return_value = [
            _Excerpt(url="https://example.com/a", title="A", text="alpha fact"),
            _Excerpt(url="https://example.com/b", title="B", text="beta fact"),
        ]
        fake_store.sources.return_value = [
            "https://example.com/a",
            "https://example.com/b",
        ]
        monkeypatch.setattr(researcher_mod, "ExcerptStore", lambda *a, **kw: fake_store)

        # Stub the loop to be a no-op.
        class _Loop:
            def __init__(self, **kw):
                self.candidates = []
                self.errors: list[str] = []
                self.pages_fetched = 2
                self.notes: list[str] = []

            def run(self, inquiry):
                return None

        monkeypatch.setattr(researcher_mod, "ResearchLoop", _Loop)
        monkeypatch.setattr(researcher_mod, "per_domain_stats", lambda loop: {})

        # Summarizer returns a fixed summary + sources.
        monkeypatch.setattr(
            researcher_mod,
            "synthesize",
            lambda inquiry, excerpts, agent, model: (
                "Alpha is true [1]. Beta is false [2].",
                ["https://example.com/a", "https://example.com/b"],
            ),
        )

        editor_agent = _ScriptedAgent(editor_replies)
        summarizer_agent = _ScriptedAgent([])
        wr = researcher_mod.WebResearcher(
            cm,
            config=cfg,
            agent_factory=lambda: summarizer_agent,
            editor_agent_factory=lambda: editor_agent,
        )
        return wr, editor_agent

    def test_verification_populates_report(self, monkeypatch) -> None:
        wr, editor = self._make_researcher_with_stub_loop(
            monkeypatch,
            verify=True,
            editor_replies=[
                "SUPPORTED: alpha is in excerpt",
                "UNSUPPORTED: nothing about beta being false",
                "Alpha is true [1].",  # rewritten summary
            ],
        )
        report = wr.research("inquiry")
        assert len(report.citation_checks) == 2
        assert report.citation_checks[0].verdict == "supported"
        assert report.citation_checks[1].verdict == "unsupported"
        assert len(report.source_statuses) == 2
        assert all(s.exists for s in report.source_statuses)
        assert report.original_summary is not None
        assert "[2]" in report.original_summary
        assert report.summary == "Alpha is true [1]."
        assert report.stats["citations_total"] == 2
        assert report.stats["citations_unsupported"] == 1
        assert report.stats["dead_sources"] == 0
        assert report.stats["editor_rewrote"] is True

    def test_disabling_verification_skips_editor(self, monkeypatch) -> None:
        wr, editor = self._make_researcher_with_stub_loop(
            monkeypatch, verify=False, editor_replies=[]
        )
        report = wr.research("inquiry")
        assert report.summary == "Alpha is true [1]. Beta is false [2]."
        assert report.citation_checks == []
        assert report.source_statuses == []
        assert report.original_summary is None
        assert "citations_total" not in report.stats
        assert editor.sent == []

    def test_all_supported_no_rewrite(self, monkeypatch) -> None:
        wr, _ = self._make_researcher_with_stub_loop(
            monkeypatch,
            verify=True,
            editor_replies=[
                "SUPPORTED: ok",
                "SUPPORTED: ok",
            ],
        )
        report = wr.research("inquiry")
        assert report.original_summary is None
        assert report.summary == "Alpha is true [1]. Beta is false [2]."
        assert report.stats["editor_rewrote"] is False
        assert report.stats["citations_unsupported"] == 0
