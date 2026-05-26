"""Subagent-driven research loop: search → fetch → extract → store → repeat."""

from __future__ import annotations

import logging
from typing import Any

from .extractor import chunk_text, extract_main_text, filter_outlinks
from .fetcher import Fetcher, host_of, in_whitelist
from .models import Excerpt, WebResearchConfig
from .parser import ResearchAction, extract_actions
from .search import DuckDuckGoSearch
from .store import ExcerptStore

logger = logging.getLogger(__name__)


_LOOP_SYSTEM_PROMPT = """\
You are a web research assistant controlling a crawler.

You receive an inquiry plus a status report each round (pages already fetched
and a short summary of stored excerpts). Pick the next actions using ONLY
the following grammar, one action per line:

  FETCH: <absolute https url>
  SEARCH: <domain> <search query>
  NOTE: <short reasoning note>
  STOP

Rules:
- Only request URLs whose host is in the configured whitelist.
- Prefer canonical, high-quality pages (overview, reference, encyclopedia).
- Issue STOP when you have enough information to write a thorough summary
  or when no productive actions remain.
- Emit at most 5 actions per round.
- Do not produce prose outside the grammar.
"""


class ResearchLoop:
    """Drives the search/fetch/extract/store loop with a subagent in charge."""

    def __init__(
        self,
        config: WebResearchConfig,
        agent: Any,
        fetcher: Fetcher,
        search: DuckDuckGoSearch,
        store: ExcerptStore,
    ) -> None:
        self.config = config
        self.agent = agent
        self.fetcher = fetcher
        self.search = search
        self.store = store
        self.pages_fetched: int = 0
        self.errors: list[str] = []
        self.visited: set[str] = set()
        self.candidates: list[str] = []  # discovered, not yet fetched
        self.notes: list[str] = []
        self._per_domain_pages: dict[str, int] = {}

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------

    def run(self, inquiry: str) -> ExcerptStore:
        """Execute the loop and return the populated :class:`ExcerptStore`."""
        self._seed(inquiry)
        self._prime_subagent(inquiry)

        for round_idx in range(self.config.max_iterations):
            if self.pages_fetched >= self.config.max_pages:
                logger.debug("page budget exhausted at round %d", round_idx)
                break

            status = self._status_report()
            prompt = (
                f"Inquiry: {inquiry}\n\n"
                f"Round {round_idx + 1} / {self.config.max_iterations}.\n"
                f"{status}\n\n"
                "Reply with FETCH / SEARCH / NOTE / STOP actions."
            )
            try:
                reply = self.agent.send(prompt)
            except Exception as exc:
                self.errors.append(f"subagent send failed: {exc}")
                break

            actions = extract_actions(reply or "")
            if not actions:
                # No parseable actions - greedy fallback: fetch next candidate.
                if self.candidates:
                    actions = [ResearchAction(op="fetch", url=self.candidates[0])]
                else:
                    self.errors.append(
                        "subagent returned no actions and no candidates remain"
                    )
                    break

            stop_requested = False
            for action in actions:
                if self.pages_fetched >= self.config.max_pages:
                    break
                if action.op == "fetch" and action.url:
                    self._do_fetch(action.url)
                elif action.op == "search" and action.domain and action.query:
                    self._do_search(action.domain, action.query)
                elif action.op == "note" and action.note:
                    self.notes.append(action.note)
                elif action.op == "stop":
                    stop_requested = True
            if stop_requested:
                break

        return self.store

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _seed(self, inquiry: str) -> None:
        """Seed candidate URLs with one per-domain search."""
        for domain in self.config.domains:
            try:
                results = self.search.query(
                    domain, inquiry, n=self.config.search_results_per_domain
                )
            except Exception as exc:
                self.errors.append(f"seed search failed for {domain}: {exc}")
                continue
            for url in results:
                if url not in self.visited and url not in self.candidates:
                    self.candidates.append(url)

    def _prime_subagent(self, inquiry: str) -> None:
        """Install the loop system prompt on the subagent's current context."""
        whitelist_str = ", ".join(self.config.domains) or "(none)"
        primer = (
            _LOOP_SYSTEM_PROMPT
            + f"\n\nWhitelisted domains: {whitelist_str}\nInquiry: {inquiry}\n"
        )
        try:
            self.agent.set_system_prompt(primer)
        except Exception:
            # Some agent stubs may not implement set_system_prompt; fall back
            # to prepending the primer as a regular send (best effort).
            try:
                self.agent.send(primer)
            except Exception as exc:
                logger.debug("subagent priming failed: %s", exc)

    def _status_report(self) -> str:
        sources = self.store.sources()
        lines = [
            f"Pages fetched: {self.pages_fetched} / {self.config.max_pages}",
            f"Excerpts stored: {len(self.store)}",
            f"Sources so far: {len(sources)}",
        ]
        if sources:
            recent = sources[-5:]
            lines.append("Recent sources:")
            for s in recent:
                lines.append(f"  - {s}")
        if self.candidates:
            preview = self.candidates[:8]
            lines.append("Candidate URLs (not yet fetched):")
            for u in preview:
                lines.append(f"  - {u}")
            if len(self.candidates) > len(preview):
                lines.append(f"  ... and {len(self.candidates) - len(preview)} more")
        else:
            lines.append("No outstanding candidate URLs.")
        return "\n".join(lines)

    def _do_search(self, domain: str, query: str) -> None:
        # Only allow searches against whitelisted domains.
        if not any(domain.lower().endswith(d.lower()) for d in self.config.domains):
            self.errors.append(f"search rejected (domain not whitelisted): {domain}")
            return
        try:
            results = self.search.query(domain, query)
        except Exception as exc:
            self.errors.append(f"search failed for {domain}: {exc}")
            return
        for url in results:
            if url not in self.visited and url not in self.candidates:
                self.candidates.append(url)

    def _do_fetch(self, url: str) -> None:
        if url in self.visited:
            return
        self.visited.add(url)
        if url in self.candidates:
            self.candidates.remove(url)

        if not in_whitelist(url, self.fetcher.whitelist):
            self.errors.append(f"fetch rejected (not in whitelist): {url}")
            return

        result = self.fetcher.fetch(url)
        if not result.ok:
            self.errors.append(f"fetch failed [{url}]: {result.error}")
            return

        self.pages_fetched += 1
        host = host_of(result.final_url)
        self._per_domain_pages[host] = self._per_domain_pages.get(host, 0) + 1

        page = extract_main_text(result.html, base_url=result.final_url)
        if not page.text:
            self.errors.append(f"no main text extracted: {url}")
            return

        chunks = chunk_text(
            page.text,
            chunk_size=self.config.chunk_size,
            overlap=self.config.chunk_overlap,
        )
        for chunk in chunks:
            excerpt = Excerpt(
                url=result.final_url,
                title=page.title,
                text=chunk,
                metadata={"source_host": host},
            )
            self.store.add(excerpt)

        # Discover new candidates (filtered to whitelist, capped to avoid blow-up).
        new_links = filter_outlinks(
            page.outlinks, self.fetcher.whitelist, seen=self.visited
        )
        for link in new_links[:25]:
            if link not in self.candidates:
                self.candidates.append(link)


def _summary_status_report(loop: ResearchLoop) -> str:
    """Public helper used by the summarizer for stats."""
    return loop._status_report()


def per_domain_stats(loop: ResearchLoop) -> dict[str, int]:
    """Return a copy of per-domain page counts."""
    return dict(loop._per_domain_pages)
