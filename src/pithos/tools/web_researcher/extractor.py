"""Main-text extraction + outlink discovery from HTML pages."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Optional
from urllib.parse import urljoin, urlparse

logger = logging.getLogger(__name__)


@dataclass
class ExtractedPage:
    """A page after main-text extraction."""

    title: str
    text: str
    outlinks: list[str] = field(default_factory=list)


def _bs4_fallback(html: str, base_url: str) -> ExtractedPage:
    """BS4-only extraction used when trafilatura is unavailable or empty."""
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        # Last-ditch regex strip - returns raw text with no structure.
        text = re.sub(r"<[^>]+>", " ", html or "")
        text = re.sub(r"\s+", " ", text).strip()
        return ExtractedPage(title="", text=text, outlinks=[])

    soup = BeautifulSoup(html or "", "html.parser")
    for bad in soup(["script", "style", "noscript", "template"]):
        bad.decompose()

    title = ""
    if soup.title and soup.title.string:
        title = soup.title.string.strip()

    # Prefer <article> / <main>; fall back to <body>.
    body = soup.find("article") or soup.find("main") or soup.body or soup
    text = body.get_text(separator=" ", strip=True) if body else ""
    text = re.sub(r"\s+", " ", text).strip()

    outlinks: list[str] = []
    for a in soup.find_all("a", href=True):
        href = a.get("href")
        if not href:
            continue
        absolute = urljoin(base_url, href)
        if absolute.startswith(("http://", "https://")):
            outlinks.append(absolute)

    return ExtractedPage(title=title, text=text, outlinks=outlinks)


def extract_main_text(html: str, base_url: str) -> ExtractedPage:
    """Extract main text + title + outlinks from ``html``.

    Primary path uses trafilatura for body text and metadata; outlinks
    always come from BS4 (trafilatura strips them). When trafilatura is
    unavailable or returns nothing, the BS4 fallback handles both.
    """
    title = ""
    main_text = ""

    try:
        import trafilatura

        extracted = trafilatura.extract(
            html or "",
            include_comments=False,
            include_tables=False,
            no_fallback=False,
        )
        if extracted:
            main_text = extracted.strip()
        try:
            meta = trafilatura.extract_metadata(html or "")
            if meta and getattr(meta, "title", None):
                title = (meta.title or "").strip()
        except Exception as exc:  # pragma: no cover - trafilatura metadata quirks
            logger.debug("trafilatura metadata failed: %s", exc)
    except ImportError:
        pass
    except Exception as exc:  # pragma: no cover
        logger.debug("trafilatura extract failed: %s", exc)

    fallback = _bs4_fallback(html, base_url)
    if not main_text:
        main_text = fallback.text
    if not title:
        title = fallback.title

    return ExtractedPage(title=title, text=main_text, outlinks=fallback.outlinks)


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[\.\!\?])\s+(?=[A-Z\(\"'])")


def chunk_text(text: str, chunk_size: int = 600, overlap: int = 80) -> list[str]:
    """Split ``text`` into ~``chunk_size``-char chunks on sentence boundaries.

    Greedy: accumulates whole sentences until the next one would overshoot,
    then emits a chunk and starts the next with up to ``overlap`` trailing
    chars from the previous (to preserve cross-chunk context).
    """
    text = (text or "").strip()
    if not text:
        return []
    if chunk_size <= 0:
        return [text]

    sentences = [s.strip() for s in _SENTENCE_SPLIT_RE.split(text) if s.strip()]
    if not sentences:
        return [text]

    chunks: list[str] = []
    current = ""
    for sent in sentences:
        if not current:
            current = sent
            continue
        if len(current) + 1 + len(sent) <= chunk_size:
            current = f"{current} {sent}"
        else:
            chunks.append(current)
            tail = current[-overlap:] if overlap > 0 and len(current) > overlap else ""
            current = (tail + " " + sent).strip() if tail else sent
    if current:
        chunks.append(current)

    # Any single sentence longer than chunk_size becomes its own oversized chunk.
    # We optionally hard-split very long chunks to keep embedder happy.
    result: list[str] = []
    hard_cap = max(chunk_size * 2, chunk_size + overlap + 100)
    for c in chunks:
        if len(c) <= hard_cap:
            result.append(c)
        else:
            for i in range(0, len(c), chunk_size):
                result.append(c[i : i + chunk_size])
    return result


def filter_outlinks(
    outlinks: list[str], whitelist: list[str], seen: Optional[set[str]] = None
) -> list[str]:
    """Return outlinks whose host is in ``whitelist`` and not already seen."""
    from .fetcher import normalize_domain

    norm_wl = [normalize_domain(d) for d in whitelist if d]
    seen = seen or set()
    out: list[str] = []
    out_seen: set[str] = set()
    for href in outlinks:
        if href in seen or href in out_seen:
            continue
        try:
            host = urlparse(href).netloc.lower()
        except (ValueError, AttributeError):
            continue
        if host.startswith("www."):
            host = host[4:]
        if any(host == d or host.endswith("." + d) for d in norm_wl):
            out.append(href)
            out_seen.add(href)
    return out
