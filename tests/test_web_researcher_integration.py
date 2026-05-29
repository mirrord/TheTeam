"""Integration-ish tests for ExcerptStore, ResearchLoop, WebResearcher, and CLI."""

from __future__ import annotations

import io
import sys
from typing import Any
from unittest.mock import MagicMock

import pytest

from pithos.tools.web_researcher.agent_loop import ResearchLoop
from pithos.tools.web_researcher.models import (
    Excerpt,
    WebResearchConfig,
    WebResearchRequest,
)
from pithos.tools.web_researcher.store import ExcerptStore, _sanitize_collection_name

# ---------------------------------------------------------------------------
# Fake ChromaDB collection + client for ExcerptStore tests
# ---------------------------------------------------------------------------


class _FakeCollection:
    """Bare-bones ChromaDB collection mock recording add() calls."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.ids: list[str] = []
        self.docs: list[str] = []
        self.metadatas: list[dict] = []
        # ``next_distance`` controls what query() returns for the next call.
        self.next_distance: float | None = None

    def add(self, ids: list[str], documents: list[str], metadatas: list[dict]) -> None:
        self.ids.extend(ids)
        self.docs.extend(documents)
        self.metadatas.extend(metadatas)

    def query(self, query_texts: list[str], n_results: int = 1) -> dict:
        d = self.next_distance if self.next_distance is not None else 0.99
        # Reset so each call uses an explicit value.
        return {
            "ids": [self.ids[:n_results] or ["__missing__"]],
            "documents": [self.docs[:n_results]],
            "distances": [[d] * max(1, min(n_results, len(self.ids) or 1))],
        }


class _FakeClient:
    def __init__(self) -> None:
        self.collections: dict[str, _FakeCollection] = {}

    def get_or_create_collection(
        self, name: str, metadata: dict | None = None
    ) -> _FakeCollection:
        if name not in self.collections:
            self.collections[name] = _FakeCollection(name)
        return self.collections[name]

    def delete_collection(self, name: str) -> None:
        self.collections.pop(name, None)


@pytest.fixture
def fake_client() -> _FakeClient:
    return _FakeClient()


# ---------------------------------------------------------------------------
# ExcerptStore
# ---------------------------------------------------------------------------


class TestExcerptStore:
    def test_sanitize_collection_name(self) -> None:
        assert _sanitize_collection_name("My Inquiry?!").startswith("My_Inquiry")
        assert len(_sanitize_collection_name("a")) >= 3

    def test_add_first_succeeds(self, fake_client: _FakeClient) -> None:
        store = ExcerptStore("test", similarity_threshold=0.92, client=fake_client)
        ex = Excerpt(url="https://a/x", title="t", text="hello world")
        assert store.add(ex) is True
        assert len(store) == 1
        assert store.sources() == ["https://a/x"]

    def test_add_rejects_hash_duplicate(self, fake_client: _FakeClient) -> None:
        store = ExcerptStore("test", client=fake_client)
        ex1 = Excerpt(url="https://a/x", title="t", text="hello world")
        ex2 = Excerpt(url="https://b/y", title="t", text="HELLO   WORLD")  # same hash
        assert store.add(ex1) is True
        assert store.add(ex2) is False
        assert len(store) == 1

    def test_add_rejects_empty(self, fake_client: _FakeClient) -> None:
        store = ExcerptStore("test", client=fake_client)
        assert store.add(Excerpt(url="u", title="t", text="   ")) is False

    def test_semantic_dedup_via_distance(self, fake_client: _FakeClient) -> None:
        store = ExcerptStore("test", similarity_threshold=0.92, client=fake_client)
        a = Excerpt(url="u1", title="t", text="alpha beta gamma")
        b = Excerpt(url="u2", title="t", text="alpha beta delta epsilon")
        assert store.add(a) is True
        # Next query() will report a near-zero distance => duplicate.
        store._collection.next_distance = 0.01
        assert store.add(b) is False
        # And with a large distance, it passes.
        c = Excerpt(url="u3", title="t", text="zeta eta theta")
        store._collection.next_distance = 0.9
        assert store.add(c) is True

    def test_sources_dedup(self, fake_client: _FakeClient) -> None:
        store = ExcerptStore("test", client=fake_client)
        store.add(Excerpt(url="u1", title="t", text="one fact"))
        # Distinct hash; pretend semantic distance is large.
        store._collection.next_distance = 0.9
        store.add(Excerpt(url="u1", title="t", text="another fact"))
        assert store.sources() == ["u1"]
        assert len(store) == 2

    def test_cleanup_removes_collection(self, fake_client: _FakeClient) -> None:
        store = ExcerptStore("test", client=fake_client)
        store.add(Excerpt(url="u", title="t", text="abc"))
        store.cleanup()
        assert "test" not in [
            _sanitize_collection_name(n) for n in fake_client.collections
        ]

    def test_invalid_similarity_threshold(self, fake_client: _FakeClient) -> None:
        with pytest.raises(ValueError):
            ExcerptStore("t", similarity_threshold=0.0, client=fake_client)
        with pytest.raises(ValueError):
            ExcerptStore("t", similarity_threshold=1.5, client=fake_client)


# ---------------------------------------------------------------------------
# ResearchLoop
# ---------------------------------------------------------------------------


class _ScriptedAgent:
    """Agent stub that returns a queue of canned replies."""

    def __init__(self, replies: list[str]) -> None:
        self._replies = list(replies)
        self.sent: list[str] = []
        self.system_prompts: list[str] = []

    def send(self, prompt: str, **kw: Any) -> str:
        self.sent.append(prompt)
        return self._replies.pop(0) if self._replies else "STOP"

    def set_system_prompt(self, p: str) -> None:
        self.system_prompts.append(p)


def _stub_fetcher(whitelist: list[str], html_by_url: dict[str, str]):
    f = MagicMock()
    f.whitelist = whitelist

    def fake_fetch(url: str):
        if url in html_by_url:
            return MagicMock(
                ok=True,
                html=html_by_url[url],
                final_url=url,
                error=None,
                status=200,
                content_type="text/html",
            )
        return MagicMock(
            ok=False, html="", final_url=url, error="not found", status=404
        )

    f.fetch.side_effect = fake_fetch
    return f


class TestResearchLoop:
    def test_loop_fetches_searches_and_stops(self, fake_client: _FakeClient) -> None:
        whitelist = ["example.com"]
        html_by_url = {
            "https://example.com/a": (
                "<html><head><title>A</title></head><body>"
                "<article><p>First fact about A.</p></article></body></html>"
            ),
            "https://example.com/b": (
                "<html><head><title>B</title></head><body>"
                "<article><p>Second fact about B.</p></article></body></html>"
            ),
        }
        fetcher = _stub_fetcher(whitelist, html_by_url)
        search = MagicMock()
        search.query.return_value = ["https://example.com/a"]

        store = ExcerptStore("loop_test", client=fake_client)
        # Semantic dedup always far apart.
        store._collection.next_distance = 0.9

        cfg = WebResearchConfig(
            domains=whitelist,
            max_pages=3,
            max_iterations=4,
            search_results_per_domain=1,
        )

        agent = _ScriptedAgent(
            [
                "FETCH: https://example.com/a",
                "FETCH: https://example.com/b",
                "STOP",
            ]
        )

        loop = ResearchLoop(
            config=cfg,
            agent=agent,
            fetcher=fetcher,
            search=search,
            store=store,
        )
        loop.run("an inquiry")

        assert loop.pages_fetched == 2
        assert len(store) >= 2
        # Subagent was primed with a system prompt.
        assert agent.system_prompts, "expected loop to prime subagent"
        # Visited set covers both fetched URLs.
        assert "https://example.com/a" in loop.visited
        assert "https://example.com/b" in loop.visited

    def test_loop_rejects_non_whitelisted_fetch(self, fake_client: _FakeClient) -> None:
        fetcher = _stub_fetcher(["example.com"], {})
        search = MagicMock()
        search.query.return_value = []
        store = ExcerptStore("loop_reject", client=fake_client)

        cfg = WebResearchConfig(domains=["example.com"], max_pages=2, max_iterations=2)
        agent = _ScriptedAgent(["FETCH: https://evil.com/x\nSTOP"])
        loop = ResearchLoop(
            config=cfg, agent=agent, fetcher=fetcher, search=search, store=store
        )
        loop.run("inquiry")
        assert any("whitelist" in e for e in loop.errors)
        assert loop.pages_fetched == 0

    def test_loop_falls_back_to_candidate_when_no_actions(
        self, fake_client: _FakeClient
    ) -> None:
        fetcher = _stub_fetcher(
            ["example.com"],
            {
                "https://example.com/seed": (
                    "<html><body><article>Seed body.</article></body></html>"
                )
            },
        )
        search = MagicMock()
        search.query.return_value = ["https://example.com/seed"]
        store = ExcerptStore("loop_fb", client=fake_client)
        store._collection.next_distance = 0.9

        cfg = WebResearchConfig(
            domains=["example.com"],
            max_pages=1,
            max_iterations=2,
            search_results_per_domain=1,
        )
        # Agent yields no parseable actions -> loop should fetch the seed candidate.
        agent = _ScriptedAgent(["I dunno"])
        loop = ResearchLoop(
            config=cfg, agent=agent, fetcher=fetcher, search=search, store=store
        )
        loop.run("inquiry")
        assert loop.pages_fetched == 1


# ---------------------------------------------------------------------------
# WebResearcher facade
# ---------------------------------------------------------------------------


class TestWebResearcherFacade:
    def test_empty_inquiry_raises(self) -> None:
        from pithos.tools.web_researcher.researcher import WebResearcher

        cm = MagicMock()
        cm.get_config.return_value = None
        wr = WebResearcher(cm, agent_factory=lambda: _ScriptedAgent(["STOP"]))
        with pytest.raises(ValueError):
            wr.research("")

    def test_no_domains_returns_graceful_report(self) -> None:
        from pithos.tools.web_researcher.researcher import WebResearcher

        cm = MagicMock()
        cm.get_config.return_value = None  # config defaults: no domains
        wr = WebResearcher(cm, agent_factory=lambda: _ScriptedAgent(["STOP"]))
        report = wr.research("anything")
        assert "No domains" in report.summary or "no whitelisted" in " ".join(
            report.errors
        )
        assert report.sources == []


# ---------------------------------------------------------------------------
# CLI smoke test (monkey-patch researcher)
# ---------------------------------------------------------------------------


class TestCLI:
    def test_cli_prints_markdown(self, monkeypatch, capsys) -> None:
        from pithos.tools.web_researcher import cli as cli_mod
        from pithos.tools.web_researcher.models import ResearchReport

        fake_report = ResearchReport(
            inquiry="q",
            summary="hello",
            excerpts=[],
            sources=["https://example.com/x"],
        )

        class _FakeResearcher:
            def __init__(self, *a, **kw): ...

            def research(self, request_or_inquiry):
                return fake_report

        monkeypatch.setattr(cli_mod, "WebResearcher", _FakeResearcher)
        import pithos.tools.web_researcher as wr_pkg

        monkeypatch.setattr(wr_pkg, "WEB_RESEARCH_AVAILABLE", True, raising=False)
        rc = cli_mod.main(["test", "inquiry"])
        out = capsys.readouterr().out
        assert rc == 0
        assert "# Research report: q" in out
        assert "https://example.com/x" in out

    def test_cli_json_mode(self, monkeypatch, capsys) -> None:
        import json as _json

        from pithos.tools.web_researcher import cli as cli_mod
        from pithos.tools.web_researcher.models import ResearchReport

        fake = ResearchReport(
            inquiry="q", summary="s", excerpts=[], sources=["https://a/b"]
        )

        class _R:
            def __init__(self, *a, **kw): ...

            def research(self, request_or_inquiry):
                return fake

        monkeypatch.setattr(cli_mod, "WebResearcher", _R)
        import pithos.tools.web_researcher as wr_pkg

        monkeypatch.setattr(wr_pkg, "WEB_RESEARCH_AVAILABLE", True, raising=False)
        rc = cli_mod.main(["--json", "topic"])
        out = capsys.readouterr().out
        assert rc == 0
        data = _json.loads(out)
        assert data["sources"] == ["https://a/b"]
        assert data["inquiry"] == "q"
