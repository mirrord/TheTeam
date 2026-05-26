"""HTTP fetcher with robots.txt, rate limiting, byte/size caps, and whitelist."""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from typing import Optional
from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser

logger = logging.getLogger(__name__)


@dataclass
class FetchResult:
    """Result of a single HTTP fetch."""

    url: str
    final_url: str
    status: int
    content_type: str
    html: str
    error: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.error is None and 200 <= self.status < 300 and bool(self.html)


def normalize_domain(domain: str) -> str:
    """Normalise a domain string for whitelist comparison.

    Strips scheme, leading ``www.``, trailing slash, and lowercases.
    """
    d = (domain or "").strip().lower()
    if "://" in d:
        d = urlparse(d).netloc or d
    if d.startswith("www."):
        d = d[4:]
    return d.rstrip("/")


def host_of(url: str) -> str:
    """Return the normalised host of ``url`` or '' if unparseable."""
    try:
        netloc = urlparse(url).netloc
    except (ValueError, AttributeError):
        return ""
    return normalize_domain(netloc)


def in_whitelist(url: str, whitelist: list[str]) -> bool:
    """Return True iff ``url``'s host matches any whitelisted domain.

    A whitelist entry matches its exact host and any subdomain
    (e.g. ``wikipedia.org`` matches ``en.wikipedia.org``).
    """
    host = host_of(url)
    if not host:
        return False
    for entry in whitelist:
        e = normalize_domain(entry)
        if not e:
            continue
        if host == e or host.endswith("." + e):
            return True
    return False


class _DomainRateLimiter:
    """Per-domain min-interval limiter (simple, monotonic-clock based)."""

    def __init__(self, rps: float) -> None:
        self.min_interval = 1.0 / rps if rps > 0 else 0.0
        self._last: dict[str, float] = {}
        self._lock = threading.Lock()

    def wait(self, domain: str) -> None:
        if self.min_interval <= 0:
            return
        with self._lock:
            now = time.monotonic()
            prev = self._last.get(domain, 0.0)
            wait_for = self.min_interval - (now - prev)
            if wait_for > 0:
                time.sleep(wait_for)
                now = time.monotonic()
            self._last[domain] = now


class Fetcher:
    """HTTP fetcher with whitelist, robots.txt, rate-limit, and byte caps."""

    def __init__(
        self,
        whitelist: list[str],
        user_agent: str,
        timeout: float = 15.0,
        max_bytes: int = 2_000_000,
        per_domain_rps: float = 1.0,
        respect_robots: bool = True,
        session: Optional[object] = None,
    ) -> None:
        self.whitelist = [normalize_domain(d) for d in whitelist if d]
        self.user_agent = user_agent
        self.timeout = timeout
        self.max_bytes = max_bytes
        self.respect_robots = respect_robots
        self._limiter = _DomainRateLimiter(per_domain_rps)
        self._robots_cache: dict[str, RobotFileParser] = {}
        self._robots_lock = threading.Lock()
        self._session = session  # injectable for testing

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def is_allowed(self, url: str) -> bool:
        """Return True iff ``url`` is in the whitelist and robots.txt permits it."""
        if not in_whitelist(url, self.whitelist):
            return False
        if self.respect_robots and not self._robots_allow(url):
            return False
        return True

    def fetch(self, url: str) -> FetchResult:
        """Fetch ``url`` and return a :class:`FetchResult`.

        Errors (whitelist rejection, robots disallow, redirect-out, timeout,
        non-HTML content, size cap) are reported via :attr:`FetchResult.error`
        rather than raising, so the calling loop can record and continue.
        """
        if not in_whitelist(url, self.whitelist):
            return FetchResult(url, url, 0, "", "", error="rejected: not in whitelist")
        if self.respect_robots and not self._robots_allow(url):
            return FetchResult(url, url, 0, "", "", error="rejected: robots.txt")

        try:
            import requests  # local import: optional dep
        except ImportError:
            return FetchResult(url, url, 0, "", "", error="requests not installed")

        host = host_of(url)
        self._limiter.wait(host)

        session = self._session or requests
        try:
            # Manual redirect handling so we can enforce the whitelist on every hop.
            current_url = url
            for _ in range(6):
                resp = session.get(
                    current_url,
                    headers={"User-Agent": self.user_agent, "Accept": "text/html,*/*"},
                    timeout=self.timeout,
                    allow_redirects=False,
                    stream=True,
                )
                status = getattr(resp, "status_code", 0)
                if status in (301, 302, 303, 307, 308):
                    loc = resp.headers.get("Location") or resp.headers.get("location")
                    if not loc:
                        return FetchResult(
                            url,
                            current_url,
                            status,
                            "",
                            "",
                            error="redirect with no Location",
                        )
                    # Resolve relative redirects.
                    from urllib.parse import urljoin

                    next_url = urljoin(current_url, loc)
                    if not in_whitelist(next_url, self.whitelist):
                        return FetchResult(
                            url,
                            next_url,
                            status,
                            "",
                            "",
                            error=f"redirect leaves whitelist: {next_url}",
                        )
                    if self.respect_robots and not self._robots_allow(next_url):
                        return FetchResult(
                            url,
                            next_url,
                            status,
                            "",
                            "",
                            error="redirect blocked by robots",
                        )
                    current_url = next_url
                    self._limiter.wait(host_of(current_url))
                    continue
                break
            else:
                return FetchResult(
                    url, current_url, 0, "", "", error="too many redirects"
                )

            content_type = (
                (resp.headers.get("Content-Type") or "").split(";")[0].strip().lower()
            )
            if content_type and not (
                content_type.startswith("text/")
                or "html" in content_type
                or "xml" in content_type
            ):
                return FetchResult(
                    url,
                    current_url,
                    status,
                    content_type,
                    "",
                    error=f"non-HTML content: {content_type}",
                )

            # Streamed read with byte cap.
            chunks: list[bytes] = []
            total = 0
            try:
                iterator = resp.iter_content(chunk_size=16_384, decode_unicode=False)
            except TypeError:
                iterator = resp.iter_content(16_384)
            for chunk in iterator:
                if not chunk:
                    continue
                if isinstance(chunk, str):
                    chunk = chunk.encode("utf-8", errors="ignore")
                total += len(chunk)
                if total > self.max_bytes:
                    return FetchResult(
                        url,
                        current_url,
                        status,
                        content_type,
                        "",
                        error=f"page exceeds max_bytes ({self.max_bytes})",
                    )
                chunks.append(chunk)

            encoding = getattr(resp, "encoding", None) or "utf-8"
            try:
                body = b"".join(chunks).decode(encoding, errors="replace")
            except (LookupError, TypeError):
                body = b"".join(chunks).decode("utf-8", errors="replace")

            if not (200 <= status < 300):
                return FetchResult(
                    url,
                    current_url,
                    status,
                    content_type,
                    body,
                    error=f"HTTP {status}",
                )

            return FetchResult(url, current_url, status, content_type, body)
        except Exception as exc:  # network errors, timeouts, etc.
            return FetchResult(url, url, 0, "", "", error=f"fetch failed: {exc}")

    # ------------------------------------------------------------------
    # robots.txt
    # ------------------------------------------------------------------

    def _robots_allow(self, url: str) -> bool:
        try:
            parsed = urlparse(url)
            base = f"{parsed.scheme}://{parsed.netloc}"
        except (ValueError, AttributeError):
            return True
        if not base.startswith(("http://", "https://")):
            return True

        with self._robots_lock:
            rp = self._robots_cache.get(base)
            if rp is None:
                rp = RobotFileParser()
                rp.set_url(base + "/robots.txt")
                try:
                    rp.read()
                except Exception as exc:
                    logger.debug("robots.txt read failed for %s: %s", base, exc)
                    # Fail-open: if robots.txt cannot be fetched, allow.
                self._robots_cache[base] = rp
        try:
            return rp.can_fetch(self.user_agent, url)
        except Exception:
            return True
