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
import re
import xml.etree.ElementTree as ET
from typing import Callable, Optional
from urllib.parse import quote, urlencode

logger = logging.getLogger(__name__)


# arxiv abs URL pattern, used by both the search backend and the
# /pdf/ -> /html/ rewriter below.
_ARXIV_ABS_RE = re.compile(
    r"^https?://(?:www\.)?arxiv\.org/abs/(?P<id>[^?#\s]+)", re.IGNORECASE
)
_ARXIV_PDF_RE = re.compile(
    r"^(https?://(?:www\.)?arxiv\.org)/pdf/(?P<id>[^?#\s]+?)(?:\.pdf)?(?P<tail>[?#].*)?$",
    re.IGNORECASE,
)


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


def _arxiv_search(domain: str, query: str, fetcher, n: int) -> list[str]:
    """Use the arxiv Atom export API (no auth, no anti-bot).

    The API lives at ``export.arxiv.org/api/query`` and returns Atom XML.
    Each ``<entry>`` has an ``<id>`` of the form
    ``http://arxiv.org/abs/<paper_id>v<version>``. We return the matching
    ``https://arxiv.org/html/<paper_id>v<version>`` URLs so the crawler
    grabs the HTML rendition of the paper rather than the PDF (PDF
    parsing is intentionally out of scope for this tool).
    """
    params = urlencode(
        {
            "search_query": f"all:{query}",
            "start": "0",
            "max_results": str(max(1, n)),
            "sortBy": "relevance",
            "sortOrder": "descending",
        }
    )
    # The export host is the documented public API; the user-facing
    # arxiv.org host is rate-limited far more aggressively. We have to
    # add it to the whitelist temporarily because Fetcher's whitelist
    # gate runs before any per-host policy.
    api_host = "export.arxiv.org"
    added = False
    if api_host not in [h.lower() for h in getattr(fetcher, "whitelist", [])]:
        fetcher.whitelist = list(fetcher.whitelist) + [api_host]
        added = True
    try:
        url = f"https://{api_host}/api/query?{params}"
        result = fetcher.fetch(url, bypass_robots=True, allow_non_html=True)
    finally:
        if added:
            fetcher.whitelist = [h for h in fetcher.whitelist if h.lower() != api_host]
    if not result.ok:
        logger.debug("arxiv search failed: %s", result.error)
        return []

    try:
        root = ET.fromstring(result.html)
    except ET.ParseError as exc:
        logger.debug("arxiv search returned non-XML: %s", exc)
        return []

    # Atom namespace; ElementTree requires the full URI in tag names.
    ns = "{http://www.w3.org/2005/Atom}"
    out: list[str] = []
    for entry in root.findall(f"{ns}entry"):
        id_el = entry.find(f"{ns}id")
        if id_el is None or not id_el.text:
            continue
        m = _ARXIV_ABS_RE.match(id_el.text.strip())
        if not m:
            continue
        paper_id = m.group("id")
        out.append(f"https://arxiv.org/html/{paper_id}")
        if len(out) >= n:
            break
    return out


def arxiv_pdf_to_html(url: str) -> str:
    """Rewrite ``https://arxiv.org/pdf/<id>`` -> ``.../html/<id>``.

    Returns ``url`` unchanged when it isn't an arxiv PDF URL. The query
    string / fragment is preserved. This is a pure string transform so
    it's safe to call on every URL before fetch.
    """
    m = _ARXIV_PDF_RE.match(url or "")
    if not m:
        return url
    base = m.group(1)
    paper_id = m.group("id")
    tail = m.group("tail") or ""
    return f"{base}/html/{paper_id}{tail}"


# Registry: domain suffix -> handler. Longest match wins.
_NATIVE_BACKENDS: list[tuple[str, Callable[[str, str, object, int], list[str]]]] = [
    ("wikipedia.org", _wikipedia_search),
    ("developer.mozilla.org", _mdn_search),
    ("arxiv.org", _arxiv_search),
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
