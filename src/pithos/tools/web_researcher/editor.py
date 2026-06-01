"""Editor subagent: verify citations and (when needed) rewrite the summary.

Two stages run after :func:`summarizer.synthesize`:

1. **Deterministic** — each cited source URL is checked for reachability.
   URLs already present in the :class:`ExcerptStore` are trusted (they were
   fetched successfully during the research loop); the rest are probed
   with :meth:`Fetcher.verify_url`.
2. **LLM** — for each ``[N]`` marker in the summary, the editor agent is
   asked whether the surrounding claim is supported by that source's
   stored excerpts. Verdicts are SUPPORTED, PARTIAL, or UNSUPPORTED.

If any claim comes back UNSUPPORTED or its source is DEAD, the editor
agent is asked to rewrite the summary, dropping or softening those claims
and removing their ``[N]`` markers. Supported citations survive untouched.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Optional

from .fetcher import Fetcher
from .models import CitationCheck, Excerpt, SourceStatus, Verdict
from .store import ExcerptStore

logger = logging.getLogger(__name__)


_CITATION_RE = re.compile(r"\[(\d+)\]")
# Sentence terminator that respects trailing quotes/brackets.
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])[\s\n]+(?=[A-Z\[\"'(])")


_VERIFY_SYSTEM_PROMPT = """\
You are a citation auditor. Given a CLAIM and one or more SOURCE EXCERPTS,
decide whether the excerpts support the claim.

Reply with EXACTLY one line in the form:
  <VERDICT>: <one short sentence of reasoning>

Where VERDICT is one of:
  SUPPORTED   - the excerpts directly state or clearly entail the claim
  PARTIAL     - the excerpts back part of the claim but not all of it
  UNSUPPORTED - the excerpts do not back the claim (or contradict it)

Do not output anything else. No preamble. No closing remarks.
"""

_EDIT_SYSTEM_PROMPT = """\
You are a research-report editor. You will receive an original Markdown
summary that uses inline [N] citations, plus a list of citations flagged
as problematic (either UNSUPPORTED by the cited source or pointing to a
DEAD source URL).

Rewrite the summary so that:
- Claims attached to flagged citations are removed or softened to remain
  truthful without those references.
- The [N] markers for flagged citations are deleted; markers for
  unflagged citations are preserved exactly.
- Section headings, paragraph structure, and overall length are kept
  close to the original where possible.
- Do NOT invent new citations or new factual content.
- Do NOT add a Sources section; it is appended automatically.

Reply with the rewritten Markdown only - no preamble, no explanation.
"""


# ---------------------------------------------------------------------------
# Citation extraction
# ---------------------------------------------------------------------------


def extract_citations(summary: str) -> list[tuple[int, int, str]]:
    """Return ``(index, char_pos, claim_sentence)`` for each ``[N]`` marker.

    The claim sentence is the sentence containing the marker (split on
    ``.!?`` followed by whitespace + capital-ish character). Markers in
    code fences are intentionally not filtered out (the summarizer is
    instructed not to emit any).
    """
    if not summary:
        return []
    out: list[tuple[int, int, str]] = []
    for m in _CITATION_RE.finditer(summary):
        idx = int(m.group(1))
        pos = m.start()
        sentence = _sentence_for_position(summary, pos)
        out.append((idx, pos, sentence))
    return out


def _sentence_for_position(text: str, pos: int) -> str:
    """Best-effort: return the sentence containing ``pos``."""
    # Find sentence start: walk back to nearest terminator + space.
    start = 0
    for m in _SENT_SPLIT_RE.finditer(text, 0, pos):
        start = m.end()
    # Find sentence end: first terminator at or after pos.
    end_match = _SENT_SPLIT_RE.search(text, pos)
    end = end_match.start() if end_match else len(text)
    return text[start:end].strip()


# ---------------------------------------------------------------------------
# Source reachability
# ---------------------------------------------------------------------------


def verify_sources(
    sources: list[str],
    store: ExcerptStore,
    fetcher: Optional[Fetcher],
) -> list[SourceStatus]:
    """Confirm each source URL is reachable.

    Sources already fetched into ``store`` are marked existing without a
    network call. Remaining URLs are probed via ``fetcher.verify_url``;
    when no fetcher is provided, uncached sources are reported as
    unverified (``exists=False``).
    """
    known = set(store.sources())
    out: list[SourceStatus] = []
    for url in sources:
        if not url:
            out.append(SourceStatus(url=url, exists=False, error="empty url"))
            continue
        if url in known:
            out.append(SourceStatus(url=url, exists=True, status_code=200))
            continue
        if fetcher is None:
            out.append(SourceStatus(url=url, exists=False, error="no fetcher"))
            continue
        try:
            ok, status, err = fetcher.verify_url(url)
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug("verify_url crashed for %s: %s", url, exc)
            ok, status, err = False, None, f"verify_url crashed: {exc}"
        out.append(SourceStatus(url=url, exists=ok, status_code=status, error=err))
    return out


# ---------------------------------------------------------------------------
# Per-citation LLM verification
# ---------------------------------------------------------------------------


def _excerpts_for_url(excerpts: list[Excerpt], url: str) -> list[Excerpt]:
    return [e for e in excerpts if (e.url or "") == url]


def _parse_verdict(reply: str) -> tuple[Verdict, str]:
    text = (reply or "").strip()
    if not text:
        return "unsupported", "empty reply"
    # Take the first non-empty line.
    line = next((ln for ln in text.splitlines() if ln.strip()), "")
    upper = line.upper()
    # Order matters: check UNSUPPORTED before SUPPORTED (substring overlap).
    for tag in ("UNSUPPORTED", "PARTIAL", "SUPPORTED"):
        if upper.startswith(tag):
            reason = line[len(tag) :].lstrip(" :-\t")
            return tag.lower(), reason.strip()  # type: ignore[return-value]
    for tag in ("UNSUPPORTED", "PARTIAL", "SUPPORTED"):
        if tag in upper:
            after = line[upper.index(tag) + len(tag) :]
            reason = after.lstrip(" :-\t")
            return tag.lower(), reason.strip()  # type: ignore[return-value]
    return "unsupported", f"unparseable reply: {line[:120]}"


def _send(agent: Any, prompt: str, model: Optional[str]) -> str:
    try:
        if model is not None:
            return agent.send(prompt, model=model) or ""
    except TypeError:
        pass
    return agent.send(prompt) or ""


def verify_citation(
    claim: str,
    source_url: str,
    source_excerpts: list[Excerpt],
    agent: Any,
    model: Optional[str] = None,
    index: int = 0,
) -> CitationCheck:
    """Ask the editor agent whether ``claim`` is supported by the excerpts."""
    if not source_excerpts:
        return CitationCheck(
            index=index,
            source_url=source_url,
            claim=claim,
            verdict="unsupported",
            reason="no stored excerpts for source",
        )

    try:
        agent.set_system_prompt(_VERIFY_SYSTEM_PROMPT)
    except Exception:
        pass

    bundle_parts: list[str] = []
    total = 0
    for ex in source_excerpts:
        snippet = (ex.text or "").strip()
        if not snippet:
            continue
        if total + len(snippet) > 6000:
            bundle_parts.append("[... excerpts truncated ...]")
            break
        bundle_parts.append(snippet)
        total += len(snippet)
    bundle = "\n---\n".join(bundle_parts)

    prompt = (
        f"CLAIM: {claim}\n\n"
        f"SOURCE URL: {source_url}\n\n"
        "SOURCE EXCERPTS:\n"
        f"{bundle}\n\n"
        "Respond with one line: SUPPORTED|PARTIAL|UNSUPPORTED: <reason>."
    )

    try:
        reply = _send(agent, prompt, model)
    except Exception as exc:
        logger.warning("verify_citation send failed: %s", exc)
        return CitationCheck(
            index=index,
            source_url=source_url,
            claim=claim,
            verdict="unsupported",
            reason=f"agent error: {exc}",
        )

    verdict, reason = _parse_verdict(reply)
    return CitationCheck(
        index=index,
        source_url=source_url,
        claim=claim,
        verdict=verdict,
        reason=reason,
    )


def verify_citations(
    summary: str,
    sources: list[str],
    excerpts: list[Excerpt],
    agent: Any,
    model: Optional[str] = None,
) -> list[CitationCheck]:
    """Run :func:`verify_citation` for every ``[N]`` marker in ``summary``."""
    out: list[CitationCheck] = []
    for index, _pos, claim in extract_citations(summary):
        if index < 1 or index > len(sources):
            out.append(
                CitationCheck(
                    index=index,
                    source_url="",
                    claim=claim,
                    verdict="unsupported",
                    reason=f"citation index {index} has no matching source",
                )
            )
            continue
        url = sources[index - 1]
        out.append(
            verify_citation(
                claim=claim,
                source_url=url,
                source_excerpts=_excerpts_for_url(excerpts, url),
                agent=agent,
                model=model,
                index=index,
            )
        )
    return out


# ---------------------------------------------------------------------------
# Summary rewrite
# ---------------------------------------------------------------------------


def _bad_indices(
    citation_checks: list[CitationCheck],
    source_statuses: list[SourceStatus],
) -> set[int]:
    dead_urls = {s.url for s in source_statuses if not s.exists}
    bad: set[int] = set()
    for c in citation_checks:
        if c.verdict == "unsupported" or c.source_url in dead_urls:
            bad.add(c.index)
    return bad


def edit_summary(
    inquiry: str,
    summary: str,
    citation_checks: list[CitationCheck],
    source_statuses: list[SourceStatus],
    agent: Any,
    model: Optional[str] = None,
) -> str:
    """Rewrite ``summary`` removing unsupported/dead-source claims.

    Returns the original summary when no problems were found or the
    editor agent fails. The caller is responsible for preserving the
    pre-edit text on the report.
    """
    bad = _bad_indices(citation_checks, source_statuses)
    if not bad:
        return summary

    try:
        agent.set_system_prompt(_EDIT_SYSTEM_PROMPT)
    except Exception:
        pass

    dead_urls = {s.url for s in source_statuses if not s.exists}
    by_index = {c.index: c for c in citation_checks}
    problems: list[str] = []
    for idx in sorted(bad):
        c = by_index.get(idx)
        if c is None:
            problems.append(f"- [{idx}] dead source")
            continue
        tag = "DEAD SOURCE" if c.source_url in dead_urls else c.verdict.upper()
        problems.append(
            f"- [{idx}] {tag}: {c.reason or '(no reason)'} (source: {c.source_url})"
        )

    prompt = (
        f"Inquiry: {inquiry}\n\n"
        "ORIGINAL SUMMARY:\n"
        f"{summary}\n\n"
        "FLAGGED CITATIONS (remove their [N] markers and rewrite the surrounding claims):\n"
        f"{chr(10).join(problems)}\n\n"
        "Rewrite the Markdown summary now."
    )

    try:
        rewritten = _send(agent, prompt, model)
    except Exception as exc:
        logger.warning("edit_summary send failed: %s", exc)
        return summary

    rewritten = (rewritten or "").strip()
    return rewritten or summary
