"""Per-domain native search backends.

DuckDuckGo's HTML/lite endpoints aggressively anomaly-block automated
clients (returning a 202 + JS challenge page), which makes them an
unreliable single source of seed URLs. For domains with stable, public
search APIs we query those directly; everything else falls back to DDG.

Each backend is a tiny stateless function that takes a domain, a query
string, an HTTP fetcher (so robots/rate-limit/timeouts are still honoured
for non-API calls), and a result cap. Backends MUST return only URLs
whose host is on the configured domain (or a subdomain).
"""

from __future__ import annotations

import json
import logging
from typing import Callable, Optional
from urllib.parse import quote, urlencode

logger = logging.getLogger(__name__)


def _wikipedia_search(domain: str, query: str, fetcher, n: int) -> list[str]:
    """Use MediaWiki's full-text search API (no auth, very reliable).

    OpenSearch (``action=opensearch``) is phrase-prefix only and returns
    nothing for non-prefix queries, so we use ``action=query&list=search``
    which does relevance ranking like the on-site search box.
    """
    # domain is e.g. 'en.wikipedia.org'; the API lives at /w/api.php.
    params = urlencode(
        {
            "action": "query",
            "list": "search",
            "srsearch": query,
            "srlimit": str(max(1, n)),
            "srnamespace": "0",
            "srprop": "",  # we only need the title
            "format": "json",
        }
    )
    url = f"https://{domain}/w/api.php?{params}"
    result = fetcher.fetch(url, bypass_robots=True, allow_non_html=True)
    if not result.ok:
        logger.debug("wikipedia search failed: %s", result.error)
        return []
    try:
        data = json.loads(result.html)
    except (ValueError, TypeError) as exc:
        logger.debug("wikipedia search returned non-JSON: %s", exc)
        return []
    hits = (data.get("query") or {}).get("search") or []
    if not isinstance(hits, list):
        return []
    out: list[str] = []
    for hit in hits[:n]:
        title = hit.get("title") if isinstance(hit, dict) else None
        if not isinstance(title, str):
            continue
        out.append(f"https://{domain}/wiki/{quote(title.replace(' ', '_'))}")
    return out


def _mdn_search(domain: str, query: str, fetcher, n: int) -> list[str]:
    """Use MDN's documents search JSON API (`/api/v1/search`)."""
    url = f"https://developer.mozilla.org/api/v1/search?q={quote(query)}&locale=en-US"
    result = fetcher.fetch(url, bypass_robots=True, allow_non_html=True)
    if not result.ok:
        logger.debug("mdn search failed: %s", result.error)
        return []
    try:
        data = json.loads(result.html)
    except (ValueError, TypeError):
        return []
    docs = data.get("documents") if isinstance(data, dict) else None
    if not isinstance(docs, list):
        return []
    out: list[str] = []
    for d in docs[:n]:
        if not isinstance(d, dict):
            continue
        slug = d.get("mdn_url") or d.get("slug")
        if isinstance(slug, str):
            if slug.startswith("/"):
                out.append(f"https://developer.mozilla.org{slug}")
            elif slug.startswith("http"):
                out.append(slug)
    return out


# Registry: domain suffix -> handler. Longest match wins.
_NATIVE_BACKENDS: list[tuple[str, Callable[[str, str, object, int], list[str]]]] = [
    ("wikipedia.org", _wikipedia_search),
    ("developer.mozilla.org", _mdn_search),
]


def native_search(domain: str, query: str, fetcher, n: int) -> Optional[list[str]]:
    """Return a list of URLs from the native backend for ``domain`` if any.

    Returns ``None`` when no native backend is registered (signalling to
    the caller that it should fall back to a generic search engine).
    Returns ``[]`` when a backend is registered but produced no results.
    """
    d = (domain or "").lower().lstrip(".")
    # Longest suffix match wins so 'en.wikipedia.org' picks up
    # the 'wikipedia.org' handler.
    best: Optional[tuple[int, Callable]] = None
    for suffix, handler in _NATIVE_BACKENDS:
        if d == suffix or d.endswith("." + suffix):
            if best is None or len(suffix) > best[0]:
                best = (len(suffix), handler)
    if best is None:
        return None
    try:
        return best[1](domain, query, fetcher, n)
    except Exception as exc:
        logger.debug("native search backend crashed: %s", exc)
        return []
