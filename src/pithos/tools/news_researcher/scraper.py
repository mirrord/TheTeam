"""Article discovery + download for the news pipeline.

Step 2 of the pipeline: gather candidate article URLs from RSS/Atom feeds
(preferred, because entries are dated) and, as a fallback, from a search
of the whitelisted domains. Each candidate is fetched, its main text is
extracted, and articles newer than ``recency_days`` are saved to the
persistent knowledge base (a :class:`MemoryStore` category).
"""

from __future__ import annotations

import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Any, Optional

from ..web_researcher.extractor import extract_main_text
from ..web_researcher.fetcher import Fetcher, host_of, in_whitelist
from .dates import is_recent, parse_html_date, parse_feed_date, parse_iso_date
from .feeds import fetch_feed
from .models import NewsArticle, NewsResearchConfig

logger = logging.getLogger(__name__)

# Redirect arxiv PDF/HTML/e-print URLs to the human-readable abstract page so
# the scraper receives HTML rather than a binary PDF or rendered HTML diff.
_ARXIV_URL_RE = re.compile(
    r"(https?://arxiv\.org/)(?:pdf|html|e-print)/([^?#\s]+)",
    re.IGNORECASE,
)


def _normalize_url(url: str) -> str:
    """Redirect non-abstract arXiv paths to the ``/abs/`` page."""
    m = _ARXIV_URL_RE.match(url)
    return m.group(1) + "abs/" + m.group(2) if m else url


class NewsScraper:
    """Discovers, downloads and stores recent articles for an inquiry."""

    def __init__(
        self,
        config: NewsResearchConfig,
        fetcher: Fetcher,
        search: Optional[Any] = None,
        memory_store: Optional[Any] = None,
    ) -> None:
        self.config = config
        self.fetcher = fetcher
        self.search = search
        self.memory_store = memory_store
        self.errors: list[str] = []
        self._seen_hashes: set[str] = set()
        self._per_source: dict[str, int] = {}

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------

    def gather(self, inquiry: str, terms: list[str]) -> list[NewsArticle]:
        """Return recent, deduplicated articles saved to the knowledge base.

        Discovery is cheap and sequential; the slow per-article HTTP fetch and
        text extraction run in a thread pool (``download_concurrency``). The
        shared dedup / per-source / budget decisions and knowledge-base writes
        are then applied single-threaded to keep them race-free.
        """
        feed_candidates = self._from_feeds(terms)
        search_candidates = self._from_search(terms)
        # Feed candidates carry a known date, so they lead the ordering.
        candidates = self._dedup_and_cap_candidates(feed_candidates + search_candidates)

        fetched = self._download_all(candidates, terms, inquiry)

        articles: list[NewsArticle] = []
        for article in fetched:
            if article is None:
                continue
            if len(articles) >= self.config.max_articles:
                break
            host = article.source_host or host_of(article.url)
            if self._per_source.get(host, 0) >= self.config.max_articles_per_source:
                continue
            if article.content_hash in self._seen_hashes:
                continue
            self._seen_hashes.add(article.content_hash)
            # Articles reused from the knowledge base are already stored.
            if not article.metadata.get("cached"):
                self._store_article(article, inquiry)
            self._per_source[host] = self._per_source.get(host, 0) + 1
            articles.append(article)
        return articles

    def _download_all(
        self,
        candidates: list[tuple[str, str, Optional[datetime]]],
        terms: list[str],
        inquiry: str,
    ) -> list[Optional[NewsArticle]]:
        """Fetch every candidate, optionally in parallel, preserving order."""
        workers = max(1, min(self.config.download_concurrency, len(candidates)))
        if workers <= 1 or len(candidates) <= 1:
            return [
                self._fetch_one(url, title, published, terms)
                for url, title, published in candidates
            ]

        results: list[Optional[NewsArticle]] = [None] * len(candidates)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(self._fetch_one, url, title, published, terms): i
                for i, (url, title, published) in enumerate(candidates)
            }
            for fut in as_completed(futures):
                idx = futures[fut]
                try:
                    results[idx] = fut.result()
                except Exception as exc:  # pragma: no cover - defensive
                    self.errors.append(f"download crashed: {exc}")
        return results

    # ------------------------------------------------------------------
    # Discovery
    # ------------------------------------------------------------------

    def _from_feeds(
        self, terms: list[str]
    ) -> list[tuple[str, str, Optional[datetime]]]:
        """Collect recent, whitelisted entries from configured feeds."""
        out: list[tuple[str, str, Optional[datetime]]] = []
        term_hits: list[tuple[str, str, Optional[datetime]]] = []
        rest: list[tuple[str, str, Optional[datetime]]] = []
        lowered = [t.lower() for t in terms]

        for feed_url in self.config.feeds:
            try:
                entries = fetch_feed(self.fetcher, feed_url)
            except Exception as exc:
                self.errors.append(f"feed fetch failed [{feed_url}]: {exc}")
                continue
            for entry in entries:
                if not in_whitelist(entry.url, self.fetcher.whitelist):
                    continue
                if not is_recent(entry.published, self.config.recency_days):
                    continue
                item = (entry.url, entry.title, entry.published)
                if lowered and any(t in (entry.title or "").lower() for t in lowered):
                    term_hits.append(item)
                else:
                    rest.append(item)

        # Term-matching entries first, then other recent entries.
        out.extend(term_hits)
        out.extend(rest)
        return out

    def _from_search(
        self, terms: list[str]
    ) -> list[tuple[str, str, Optional[datetime]]]:
        """Collect candidate URLs by searching whitelisted domains.

        Search results are undated at this stage; the publication date is
        resolved from each page's HTML during download.
        """
        if not self.config.search_fallback or self.search is None:
            return []
        out: list[tuple[str, str, Optional[datetime]]] = []
        seen: set[str] = set()
        query = " ".join(terms) if terms else ""
        if not query:
            return []
        for domain in self.config.domains:
            try:
                results = self.search.query(
                    domain, query, n=self.config.search_results_per_domain
                )
            except Exception as exc:
                self.errors.append(f"search failed for {domain}: {exc}")
                continue
            for url in results:
                if url not in seen:
                    seen.add(url)
                    out.append((url, "", None))
        return out

    # ------------------------------------------------------------------
    # Candidate selection
    # ------------------------------------------------------------------

    def _dedup_and_cap_candidates(
        self, candidates: list[tuple[str, str, Optional[datetime]]]
    ) -> list[tuple[str, str, Optional[datetime]]]:
        """De-duplicate candidate URLs and cap the total / per-source counts.

        Bounds the work before the (parallel) download stage so an inquiry
        against a busy feed source (e.g. arXiv's daily category feeds) cannot
        balloon into hundreds of fetches.
        """
        seen: set[str] = set()
        per_host: dict[str, int] = {}
        # Allow a little slack per host so undated/stale search hits that get
        # dropped during download can be replaced by others from the same host.
        host_cap = max(1, self.config.max_articles_per_source * 2)
        out: list[tuple[str, str, Optional[datetime]]] = []
        for url, title, published in candidates:
            norm = _normalize_url(url)
            if norm in seen:
                continue
            host = host_of(norm)
            if per_host.get(host, 0) >= host_cap:
                continue
            seen.add(norm)
            per_host[host] = per_host.get(host, 0) + 1
            out.append((norm, title, published))
            if len(out) >= self.config.max_candidates:
                break
        return out

    # ------------------------------------------------------------------
    # Download + store
    # ------------------------------------------------------------------

    def _fetch_one(
        self,
        url: str,
        title: str,
        published: Optional[datetime],
        terms: list[str],
    ) -> Optional[NewsArticle]:
        """Fetch + date-filter a single article (no shared-state mutation).

        Safe to run concurrently: dedup, per-source caps, budget enforcement
        and knowledge-base writes are handled by the caller in :meth:`gather`.
        Returns a :class:`NewsArticle` (possibly reused from the knowledge
        base) or ``None`` when the URL is off-whitelist, stale or unusable.
        """
        url = _normalize_url(url)
        if not in_whitelist(url, self.fetcher.whitelist):
            return None

        cached = self._cached_article(url, terms)
        if cached is not None:
            return cached

        result = self.fetcher.fetch(url)
        if not result.ok:
            self.errors.append(f"fetch failed [{url}]: {result.error}")
            return None

        page = extract_main_text(result.html, base_url=result.final_url)
        if not page.text:
            self.errors.append(f"no main text extracted: {url}")
            return None

        # Resolve the publication date: feed date wins, else parse the page.
        if published is None:
            published = parse_html_date(result.html)

        if published is None:
            if self.config.skip_undated:
                self.errors.append(f"skipped (no date): {url}")
                return None
        elif not is_recent(published, self.config.recency_days):
            return None

        return NewsArticle(
            url=result.final_url,
            title=title or page.title,
            text=page.text,
            published=published,
            source_host=host_of(result.final_url),
            terms_matched=[t for t in terms if t.lower() in page.text.lower()],
        )

    def _cached_article(self, url: str, terms: list[str]) -> Optional[NewsArticle]:
        """Return a previously stored article for ``url``, if reuse is enabled.

        Skips the HTTP fetch and text extraction entirely on repeat runs. The
        cached article still passes the recency / skip-undated policy so stale
        entries are not resurfaced.
        """
        ms = self.memory_store
        if (
            not self.config.reuse_cached_articles
            or ms is None
            or not hasattr(ms, "retrieve")
        ):
            return None
        try:
            results = ms.retrieve(
                self.config.article_category,
                url,
                n_results=3,
                where={"url": url},
                min_relevance=0.0,
            )
        except Exception:
            return None
        for r in results or []:
            meta = getattr(r, "metadata", None) or {}
            content = getattr(r, "content", "") or ""
            if meta.get("url") != url or not content.strip():
                continue
            published = parse_iso_date(meta.get("published") or "") or parse_feed_date(
                meta.get("published") or ""
            )
            if published is None:
                if self.config.skip_undated:
                    return None
            elif not is_recent(published, self.config.recency_days):
                return None
            article = NewsArticle(
                url=url,
                title=meta.get("title") or "",
                text=content,
                published=published,
                source_host=meta.get("source_host") or host_of(url),
                terms_matched=[t for t in terms if t.lower() in content.lower()],
            )
            article.metadata["article_entry_id"] = getattr(r, "id", None)
            article.metadata["cached"] = True
            return article
        return None

    def _store_article(self, article: NewsArticle, inquiry: str) -> None:
        """Persist the article body in the knowledge base."""
        if self.memory_store is None:
            return
        metadata = {
            "url": article.url,
            "title": article.title or "",
            "published": article.published_iso,
            "source_host": article.source_host,
            "inquiry": inquiry,
            "terms": ", ".join(article.terms_matched),
            "kind": "news_article",
        }
        try:
            entry_id = self.memory_store.store(
                self.config.article_category, article.text, metadata
            )
            article.metadata["article_entry_id"] = entry_id
        except Exception as exc:
            self.errors.append(f"knowledge-base store failed [{article.url}]: {exc}")
