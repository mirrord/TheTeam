"""Domain-restricted search via DuckDuckGo's HTML endpoint (no API key)."""

from __future__ import annotations

import logging
import re
from typing import Optional
from urllib.parse import parse_qs, unquote, urlparse

logger = logging.getLogger(__name__)


_DDG_HTML_URL = "https://html.duckduckgo.com/html/"
_RESULT_HREF_RE = re.compile(
    r'<a[^>]+class=["\']\s*[^"\']*result__a[^"\']*\s*["\'][^>]+href=["\']([^"\']+)["\']',
    re.IGNORECASE,
)
# Fallback regex - DDG sometimes ships variants without the result__a class.
_FALLBACK_HREF_RE = re.compile(
    r'<a[^>]+href=["\'](/l/\?[^"\']+|https?://[^"\']+)["\']',
    re.IGNORECASE,
)


def _unwrap_ddg_link(href: str) -> str:
    """DuckDuckGo wraps real URLs in /l/?uddg=<encoded>; unwrap that."""
    if href.startswith("/l/") or "duckduckgo.com/l/" in href:
        try:
            qs = parse_qs(urlparse(href).query)
            if "uddg" in qs and qs["uddg"]:
                return unquote(qs["uddg"][0])
        except (ValueError, KeyError):
            return href
    return href


class DuckDuckGoSearch:
    """Issue ``site:<domain> <query>`` searches against DDG's HTML endpoint.

    The fetcher is reused so the same whitelist / rate-limit / robots policy
    applies to the search engine itself. DDG is implicitly added to the
    whitelist for the duration of a search to avoid surprising the user.
    """

    def __init__(self, fetcher, results_per_domain: int = 5) -> None:
        self._fetcher = fetcher
        self.results_per_domain = max(1, results_per_domain)
        # Make sure DDG itself is fetchable regardless of user whitelist.
        if "duckduckgo.com" not in [d.lower() for d in fetcher.whitelist]:
            fetcher.whitelist = list(fetcher.whitelist) + ["duckduckgo.com"]

    def query(self, domain: str, query: str, n: Optional[int] = None) -> list[str]:
        """Return up to ``n`` URLs from ``domain`` matching ``query``.

        Routing: domains with a native backend (Wikipedia API, MDN, etc.)
        use that backend first because it's much more reliable than DDG's
        HTML endpoint, which aggressively anomaly-blocks automated
        clients. DDG is used as a universal fallback for everything else
        and as a secondary source when a native backend returns nothing.
        """
        from .native_search import native_search

        n = n or self.results_per_domain

        # 1) Try the native backend for this domain (if any).
        native = native_search(domain, query, self._fetcher, n)
        if native:
            return native[:n]

        # 2) Fall back to DDG. POST + bypass robots: the HTML endpoint's
        # robots.txt is Disallow:/ and the endpoint only returns useful
        # results to form-encoded POSTs. Robots is still enforced for
        # every content page we actually scrape.
        q = f"site:{domain} {query}".strip()
        result = self._fetcher.fetch(
            _DDG_HTML_URL,
            bypass_robots=True,
            method="POST",
            data={"q": q},
        )
        if not result.ok:
            logger.debug("DDG search failed: %s", result.error)
            return []
        return self._parse(result.html, domain=domain)[:n]

    def _parse(self, html: str, domain: str) -> list[str]:
        """Extract result URLs from a DDG HTML page, restricted to ``domain``."""
        from .fetcher import normalize_domain

        target = normalize_domain(domain)
        urls: list[str] = []
        seen: set[str] = set()

        candidates = _RESULT_HREF_RE.findall(html)
        if not candidates:
            candidates = _FALLBACK_HREF_RE.findall(html)

        for href in candidates:
            real = _unwrap_ddg_link(href)
            if not real.startswith(("http://", "https://")):
                continue
            try:
                host = urlparse(real).netloc.lower()
            except (ValueError, AttributeError):
                continue
            if host.startswith("www."):
                host = host[4:]
            if not (host == target or host.endswith("." + target)):
                continue
            if real in seen:
                continue
            seen.add(real)
            urls.append(real)
        return urls
