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
from datetime import datetime
from typing import Any, Optional

from ..web_researcher.extractor import extract_main_text
from ..web_researcher.fetcher import Fetcher, host_of, in_whitelist
from .dates import is_recent, parse_html_date
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
        """Return recent, deduplicated articles saved to the knowledge base."""
        feed_candidates = self._from_feeds(terms)
        search_candidates = self._from_search(terms)

        articles: list[NewsArticle] = []
        # Feed candidates carry a known date, so process them first.
        for url, title, published in feed_candidates + search_candidates:
            if len(articles) >= self.config.max_articles:
                break
            article = self._download(url, title, published, terms, inquiry)
            if article is not None:
                articles.append(article)
        return articles

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
    # Download + store
    # ------------------------------------------------------------------

    def _download(
        self,
        url: str,
        title: str,
        published: Optional[datetime],
        terms: list[str],
        inquiry: str,
    ) -> Optional[NewsArticle]:
        """Fetch, date-filter, dedup and persist a single article."""
        url = _normalize_url(url)
        if not in_whitelist(url, self.fetcher.whitelist):
            return None

        host = host_of(url)
        if self._per_source.get(host, 0) >= self.config.max_articles_per_source:
            return None

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

        article = NewsArticle(
            url=result.final_url,
            title=title or page.title,
            text=page.text,
            published=published,
            source_host=host_of(result.final_url),
            terms_matched=[t for t in terms if t.lower() in page.text.lower()],
        )

        if article.content_hash in self._seen_hashes:
            return None
        self._seen_hashes.add(article.content_hash)

        self._store_article(article, inquiry)
        self._per_source[host] = self._per_source.get(host, 0) + 1
        return article

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
