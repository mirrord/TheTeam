"""Publication-date parsing helpers for feeds and HTML pages.

All parsers return timezone-aware :class:`datetime` objects normalised to
UTC, or ``None`` when no usable date can be found. Keeping everything in
UTC lets the scraper compare article ages against a single ``now`` without
worrying about mixed offsets.
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Optional

logger = logging.getLogger(__name__)


def _to_utc(dt: Optional[datetime]) -> Optional[datetime]:
    """Normalise a datetime to timezone-aware UTC (assume UTC if naive)."""
    if dt is None:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def parse_feed_date(value: Optional[str]) -> Optional[datetime]:
    """Parse an RSS (RFC 822) or Atom (ISO 8601) date string.

    Tries RFC 822 first (RSS ``pubDate``) then ISO 8601 (Atom
    ``published`` / ``updated``). Returns ``None`` on failure.
    """
    if not value or not value.strip():
        return None
    text = value.strip()

    # RSS pubDate: e.g. "Tue, 01 Jul 2025 13:45:00 +0000".
    try:
        dt = parsedate_to_datetime(text)
        if dt is not None:
            return _to_utc(dt)
    except (TypeError, ValueError, IndexError):
        pass

    return parse_iso_date(text)


def parse_iso_date(value: Optional[str]) -> Optional[datetime]:
    """Parse an ISO 8601 datetime string (tolerating a trailing ``Z``)."""
    if not value or not value.strip():
        return None
    text = value.strip()
    # Python's fromisoformat historically rejects the ``Z`` suffix.
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    # Normalise slash-separated dates: YYYY/MM/DD → YYYY-MM-DD.
    # arXiv uses this format for ``citation_date`` meta content.
    if re.match(r"\d{4}/\d{2}/\d{2}", text):
        text = text.replace("/", "-", 2)
    # Date-only strings are acceptable (midnight UTC).
    try:
        return _to_utc(datetime.fromisoformat(text))
    except ValueError:
        pass
    # Last resort: a bare YYYY-MM-DD embedded in a larger string.
    m = re.search(r"\d{4}-\d{2}-\d{2}", text)
    if m:
        try:
            return _to_utc(datetime.fromisoformat(m.group(0)))
        except ValueError:
            return None
    return None


# --- HTML page date extraction --------------------------------------------

_META_DATE_PATTERNS = [
    # <meta property="article:published_time" content="...">
    r'<meta[^>]+(?:property|name)=["\'](?:article:published_time|article:published|'
    r"og:article:published_time|datePublished|date|pubdate|publish-date|"
    # citation_* used by arXiv and academic publishers (content may be YYYY/MM/DD)
    r"citation_date|citation_online_date|citation_publication_date|"
    r'dc\.date|dc\.date\.issued|sailthru\.date)["\'][^>]+content=["\']([^"\']+)["\']',
    # content-first ordering (attribute order varies by CMS)
    r'<meta[^>]+content=["\']([^"\']+)["\'][^>]+(?:property|name)=["\']'
    r'(?:article:published_time|datePublished|date|pubdate|citation_date)["\']',
]

# arXiv submission history line: "<b>[v1]</b> Mon, 15 Jun 2025 12:34:56 UTC  (...)"
# The [v1] marker may be inside an HTML tag; [^,]{0,15} skips the closing tag
# and any whitespace before the RFC-822-style date (weekday always ends with ',').
_ARXIV_SUBMISSION_RE = re.compile(
    r"\[v1\][^,]{0,15}(\w{3},\s+\d{1,2}\s+\w{3}\s+\d{4}\s+\d{2}:\d{2}:\d{2})"
    r"(\s+(?:UTC|GMT|[+-]\d{4}))",
    re.IGNORECASE,
)

_TIME_TAG_RE = re.compile(r'<time[^>]+datetime=["\']([^"\']+)["\']', re.IGNORECASE)
_JSONLD_RE = re.compile(
    r'<script[^>]+type=["\']application/ld\+json["\'][^>]*>(.*?)</script>',
    re.IGNORECASE | re.DOTALL,
)


def parse_html_date(html: str) -> Optional[datetime]:
    """Best-effort publication date extraction from raw HTML.

    Checks, in order: ``<meta>`` publication tags, JSON-LD
    ``datePublished``, and the first ``<time datetime=...>`` element.
    """
    if not html:
        return None

    for pattern in _META_DATE_PATTERNS:
        m = re.search(pattern, html, re.IGNORECASE)
        if m:
            dt = parse_iso_date(m.group(1)) or parse_feed_date(m.group(1))
            if dt:
                return dt

    dt = _jsonld_date(html)
    if dt:
        return dt

    m = _TIME_TAG_RE.search(html)
    if m:
        dt = parse_iso_date(m.group(1)) or parse_feed_date(m.group(1))
        if dt:
            return dt

    # arXiv submission history: "[v1] Mon, 15 Jun 2025 12:34:56 UTC  (...)"
    m = _ARXIV_SUBMISSION_RE.search(html)
    if m:
        tz = m.group(2).strip().upper()
        if tz in ("UTC", "GMT"):
            tz = "+0000"
        dt = parse_feed_date(m.group(1) + " " + tz)
        if dt:
            return dt

    return None


def _jsonld_date(html: str) -> Optional[datetime]:
    """Pull ``datePublished`` from any JSON-LD block in the page."""
    for block in _JSONLD_RE.findall(html):
        try:
            data = json.loads(block.strip())
        except (ValueError, TypeError):
            continue
        for candidate in _iter_jsonld_objects(data):
            if isinstance(candidate, dict):
                raw = (
                    candidate.get("datePublished")
                    or candidate.get("dateCreated")
                    or candidate.get("dateModified")
                )
                if isinstance(raw, str):
                    dt = parse_iso_date(raw) or parse_feed_date(raw)
                    if dt:
                        return dt
    return None


def _iter_jsonld_objects(data):
    """Yield dicts from an arbitrarily nested JSON-LD document."""
    if isinstance(data, dict):
        yield data
        graph = data.get("@graph")
        if isinstance(graph, list):
            for item in graph:
                yield from _iter_jsonld_objects(item)
    elif isinstance(data, list):
        for item in data:
            yield from _iter_jsonld_objects(item)


def is_recent(published: Optional[datetime], recency_days: int) -> bool:
    """Return True iff ``published`` is within ``recency_days`` of now (UTC).

    ``None`` publication dates are treated as *not* recent so callers can
    apply their own ``skip_undated`` policy explicitly.
    """
    if published is None:
        return False
    if recency_days <= 0:
        return True
    now = datetime.now(timezone.utc)
    age = now - _to_utc(published)
    return age.total_seconds() <= recency_days * 86400
