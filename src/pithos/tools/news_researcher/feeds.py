"""RSS 2.0 / Atom feed discovery and parsing (stdlib only).

News sites almost universally publish an RSS or Atom feed, which is the
most reliable source of *dated* article links. We parse both formats with
:mod:`xml.etree.ElementTree` to avoid pulling in a third-party feed parser.
Only the fields we need (link, title, publication date) are extracted.
"""

from __future__ import annotations

import logging
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime
from typing import Optional
from urllib.parse import urlparse

from .dates import parse_feed_date

logger = logging.getLogger(__name__)

_ATOM_NS = "{http://www.w3.org/2005/Atom}"

# arXiv RSS titles append a paper-ID suffix: "Title. (arXiv:2301.12345v1 [cs.AI])"
_ARXIV_TITLE_SUFFIX_RE = re.compile(r"\.?\s*\(arXiv:[^)]+\)\s*$", re.IGNORECASE)


@dataclass
class FeedEntry:
    """A single dated entry parsed from a feed."""

    url: str
    title: str = ""
    published: Optional[datetime] = None


def _clean_title(title: str) -> str:
    """Strip arXiv paper-ID suffixes like ``(arXiv:2301.12345v1 [cs.AI])``."""
    return _ARXIV_TITLE_SUFFIX_RE.sub("", title).strip()


def _text(elem: Optional[ET.Element]) -> str:
    return (elem.text or "").strip() if elem is not None else ""


def _parse_rss(root: ET.Element) -> list[FeedEntry]:
    """Parse an RSS 2.0 document (``<rss><channel><item>...``)."""
    entries: list[FeedEntry] = []
    channel = root.find("channel")
    items = channel.findall("item") if channel is not None else root.findall("item")
    for item in items:
        link = _text(item.find("link"))
        title = _clean_title(_text(item.find("title")))
        # pubDate is standard RSS; dc:date is a common Dublin Core variant.
        pub = _text(item.find("pubDate"))
        if not pub:
            for child in item:
                if child.tag.lower().endswith("date"):
                    pub = (child.text or "").strip()
                    if pub:
                        break
        if not link:
            continue
        entries.append(FeedEntry(url=link, title=title, published=parse_feed_date(pub)))
    return entries


def _parse_atom(root: ET.Element) -> list[FeedEntry]:
    """Parse an Atom document (``<feed><entry>...``)."""
    entries: list[FeedEntry] = []
    for entry in root.findall(f"{_ATOM_NS}entry"):
        title = _text(entry.find(f"{_ATOM_NS}title"))
        # Prefer the alternate/self link with an http href.
        link = ""
        for link_el in entry.findall(f"{_ATOM_NS}link"):
            rel = link_el.get("rel", "alternate")
            href = link_el.get("href", "")
            if href.startswith(("http://", "https://")) and rel in ("alternate", ""):
                link = href
                break
            if href.startswith(("http://", "https://")) and not link:
                link = href
        pub = _text(entry.find(f"{_ATOM_NS}published")) or _text(
            entry.find(f"{_ATOM_NS}updated")
        )
        if not link:
            continue
        entries.append(FeedEntry(url=link, title=title, published=parse_feed_date(pub)))
    return entries


def parse_feed(xml_text: str) -> list[FeedEntry]:
    """Parse an RSS or Atom feed document into :class:`FeedEntry` items.

    Returns an empty list when the document is not well-formed XML or is
    not a recognised feed format.
    """
    if not xml_text or not xml_text.strip():
        return []
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as exc:
        logger.debug("feed parse failed: %s", exc)
        return []

    tag = root.tag.lower()
    if tag.endswith("rss") or tag.endswith("rdf"):
        return _parse_rss(root)
    if tag.endswith("feed"):
        return _parse_atom(root)
    # Some feeds omit the <rss> wrapper; fall back on presence of <item>.
    if root.find("channel") is not None or root.find("item") is not None:
        return _parse_rss(root)
    return []


def fetch_feed(fetcher, url: str) -> list[FeedEntry]:
    """Fetch and parse a feed via ``fetcher``.

    The feed host is temporarily whitelisted (feeds are often served from a
    dedicated host such as ``feeds.example.com``) and ``robots.txt`` is
    bypassed for the feed document itself, exactly as the search backends
    do for their API endpoints. Article pages linked from the feed are
    still fetched through the normal whitelist + robots path.
    """
    host = urlparse(url).netloc.lower()
    added = False
    existing = [h.lower() for h in getattr(fetcher, "whitelist", [])]
    if host and host not in existing:
        fetcher.whitelist = list(fetcher.whitelist) + [host]
        added = True
    try:
        result = fetcher.fetch(url, bypass_robots=True, allow_non_html=True)
    finally:
        if added:
            fetcher.whitelist = [h for h in fetcher.whitelist if h.lower() != host]
    if not result.ok:
        logger.debug("feed fetch failed [%s]: %s", url, result.error)
        return []
    return parse_feed(result.html)
