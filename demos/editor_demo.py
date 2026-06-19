"""Demo: Citation Verification (Editor Subagent)

This script demonstrates the two-stage citation-verification pipeline that
runs after every research pass:

1. **Deterministic stage** — each cited source URL is probed for
   reachability.  URLs already in the excerpt store are trusted; uncached
   ones are HEAD-verified.
2. **LLM stage** — for every ``[N]`` marker in the summary the editor
   agent decides whether the surrounding claim is actually supported by the
   stored excerpts from that source. Verdict: SUPPORTED / PARTIAL /
   UNSUPPORTED.
3. **Rewrite** — if any claims fail verification the editor agent rewrites
   the summary, softening or removing unsupported assertions.

Parts:
  A  — Low-level functions on a synthetic summary (no network, no Ollama)
  B  — Full end-to-end research + editor pass with a live Ollama model
  C  — Config knobs: ``verify_citations`` flag and per-model overrides

Run:
    python demos/editor_demo.py

Requirements:
    - ``pip install -e ".[web]"`` (requests, beautifulsoup4, trafilatura)
    - Ollama running locally for Part B (skipped gracefully if absent)
"""

from __future__ import annotations

import sys
import textwrap
import traceback
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from pithos import ConfigManager
from pithos.tools.web_researcher import WEB_RESEARCH_AVAILABLE

# ── pretty output helpers ────────────────────────────────────────────────────

DIVIDER = "-" * 70
BOLD = "\033[1m"
DIM = "\033[2m"
CYAN = "\033[36m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
RED = "\033[31m"
RESET = "\033[0m"


def header(title: str) -> None:
    print(f"\n{BOLD}{CYAN}{DIVIDER}{RESET}")
    print(f"{BOLD}{CYAN}  {title}{RESET}")
    print(f"{BOLD}{CYAN}{DIVIDER}{RESET}")


def step(label: str) -> None:
    print(f"\n{BOLD}{GREEN}> {label}{RESET}")


def info(text: str) -> None:
    for line in textwrap.wrap(text, width=68):
        print(f"  {DIM}{line}{RESET}")


def warn(text: str) -> None:
    print(f"  {YELLOW}! {text}{RESET}")


def fail(text: str) -> None:
    print(f"  {RED}x {text}{RESET}")


def ok(text: str) -> None:
    print(f"  {GREEN}✓ {text}{RESET}")


def ask(prompt: str, default: str) -> str:
    answer = input(f"  {prompt} [{default}]: ").strip()
    return answer if answer else default


def show_text(text: str, indent: int = 4, limit_lines: int = 35) -> None:
    lines = text.splitlines()
    pad = " " * indent
    for line in lines[:limit_lines]:
        print(f"{pad}{line}")
    if len(lines) > limit_lines:
        print(f"{pad}{DIM}... ({len(lines) - limit_lines} more lines){RESET}")


def verdict_colour(v: str) -> str:
    colours = {
        "supported": GREEN,
        "partial": YELLOW,
        "unsupported": RED,
    }
    return f"{colours.get(v, RESET)}{v.upper()}{RESET}"


# ── Part A: low-level functions on a synthetic summary ───────────────────────

SYNTHETIC_SUMMARY = """\
The QUIC protocol was standardised in RFC 9000 (May 2021) and forms the
transport layer for HTTP/3 [1].  QUIC runs over UDP, eliminating the
head-of-line blocking inherent in TCP [2].  According to some reports,
HTTP/3 has completely replaced TCP in all internet traffic since 2022 [3].
Multiplexed streams allow independent delivery of resources without
stalling sibling requests when one packet is lost [1].
"""

SYNTHETIC_EXCERPTS = [
    # Source 1 — Wikipedia QUIC article (real-ish content)
    {
        "url": "https://en.wikipedia.org/wiki/QUIC",
        "title": "QUIC — Wikipedia",
        "text": (
            "QUIC is a general-purpose transport layer network protocol "
            "initially designed by Jim Roskind at Google. It was standardised "
            "by the IETF in RFC 9000 in May 2021. HTTP/3 uses QUIC as its "
            "transport protocol, replacing the TCP/TLS stack used by HTTP/2."
        ),
    },
    # Source 2 — MDN QUIC overview (real-ish content)
    {
        "url": "https://developer.mozilla.org/en-US/docs/Web/HTTP/Overview",
        "title": "HTTP overview — MDN",
        "text": (
            "QUIC was designed to eliminate head-of-line blocking. Because "
            "QUIC runs over UDP, a lost packet on one QUIC stream does not "
            "block data delivery on other streams multiplexed over the same "
            "connection. This contrasts with TCP, where a single dropped "
            "segment stalls the entire connection until the retransmit arrives."
        ),
    },
    # Source 3 is intentionally absent — simulating a dead or unrelated source
]

SOURCES = [
    "https://en.wikipedia.org/wiki/QUIC",
    "https://developer.mozilla.org/en-US/docs/Web/HTTP/Overview",
    "https://fake-stats-blog.example.com/http3-dominates",  # dead / bad source
]


def _make_stub_agent(
    verdict_map: dict[int, str],
    rewritten_summary: str | None = None,
) -> Any:
    """Return a mock agent that returns pre-canned verdicts for each citation.

    ``verdict_map`` keys are 1-based citation indices; values are the raw
    one-line reply the "LLM" would return (e.g. ``"SUPPORTED: it says so"``).
    ``rewritten_summary`` is returned when the agent is asked to rewrite.
    """
    calls: list[str] = []

    class _StubAgent:
        def set_system_prompt(self, _prompt: str) -> None:
            pass

        def send(self, prompt: str, **_kw: Any) -> str:
            calls.append(prompt)
            # Rewrite requests are identified by the ORIGINAL SUMMARY block —
            # check this BEFORE the per-index branch to avoid matching [N]
            # markers embedded in the summary text.
            if "ORIGINAL SUMMARY" in prompt and rewritten_summary is not None:
                return rewritten_summary
            # Detect which citation is being checked from the CLAIM: line.
            for idx, reply in verdict_map.items():
                if f"[{idx}]" in prompt or f"citation index {idx}" in prompt:
                    return reply
            return "SUPPORTED: looks fine"

        @property
        def n_calls(self) -> int:
            return len(calls)

    return _StubAgent()


def demo_low_level() -> None:
    header("Part A — Low-level editor functions (synthetic, no network/Ollama)")
    info(
        "The editor module exposes three building blocks that work "
        "independently of the full WebResearcher pipeline:"
    )
    info("  extract_citations  – pull [N] markers + surrounding sentences")
    info("  verify_sources     – HEAD-check each cited URL")
    info("  verify_citations   – ask an LLM to audit each claim")
    info("  edit_summary       – rewrite the summary when problems exist")

    from pithos.tools.web_researcher.editor import (
        edit_summary,
        extract_citations,
        verify_citations,
        verify_sources,
    )
    from pithos.tools.web_researcher.models import Excerpt

    excerpts = [
        Excerpt(url=e["url"], title=e["title"], text=e["text"])
        for e in SYNTHETIC_EXCERPTS
    ]

    # ── A.1  Citation extraction ──────────────────────────────────────────
    step("A.1  extract_citations")
    info("Scan the summary for [N] markers and capture the host sentence.")

    print()
    show_text(SYNTHETIC_SUMMARY)

    citations = extract_citations(SYNTHETIC_SUMMARY)
    print()
    print(f"  Found {BOLD}{len(citations)}{RESET} citation(s):")
    for idx, _pos, sentence in citations:
        truncated = sentence if len(sentence) < 80 else sentence[:77] + "..."
        print(f"    [{idx}]  {DIM}{truncated}{RESET}")

    # ── A.2  Source reachability (no network — store shortcut) ────────────
    step("A.2  verify_sources — store-shortcut path (no network call)")
    info(
        "When a URL is already in the ExcerptStore it is trusted without a "
        "network request. Only URLs absent from the store are probed via "
        "Fetcher.verify_url. Here we simulate a store that knows about "
        "sources 1 and 2, and a fetcher that returns 404 for source 3."
    )

    # Minimal stub store that reports sources 1+2 as known.
    stub_store = MagicMock()
    stub_store.sources.return_value = SOURCES[:2]

    # Minimal stub fetcher that returns 404 for unknown URLs.
    stub_fetcher = MagicMock()
    stub_fetcher.verify_url.return_value = (False, 404, "Not Found")

    statuses = verify_sources(SOURCES, stub_store, stub_fetcher)
    print()
    for s in statuses:
        flag = f"{GREEN}LIVE{RESET}" if s.exists else f"{RED}DEAD{RESET}"
        code = f"  HTTP {s.status_code}" if s.status_code else ""
        err = f"  ({s.error})" if s.error and not s.exists else ""
        print(f"    {flag}  {s.url}{code}{DIM}{err}{RESET}")

    dead = sum(1 for s in statuses if not s.exists)
    print(f"\n  {dead} dead source(s) detected.")

    # ── A.3  LLM claim verification (stub agent) ──────────────────────────
    step("A.3  verify_citations — per-claim LLM audit (stub agent)")
    info(
        "For each [N] marker the editor agent receives the surrounding "
        "claim and the stored excerpts from that source.  The stub agent "
        "below returns hard-coded verdicts to show the logic without "
        "a real LLM call."
    )

    verdict_map = {
        1: "SUPPORTED: RFC 9000 and HTTP/3 are both mentioned explicitly.",
        2: "SUPPORTED: head-of-line blocking over UDP is explained.",
        3: "UNSUPPORTED: no excerpts exist for this source.",
    }
    stub_agent = _make_stub_agent(verdict_map)

    checks = verify_citations(
        summary=SYNTHETIC_SUMMARY,
        sources=SOURCES,
        excerpts=excerpts,
        agent=stub_agent,
    )

    print()
    for c in checks:
        print(
            f"    [{c.index}] {verdict_colour(c.verdict):<28}"
            f"  {DIM}{c.reason[:60]}{RESET}"
        )

    unsupported = sum(1 for c in checks if c.verdict == "unsupported")
    print(f"\n  {unsupported} unsupported citation(s).")

    # ── A.4  Summary rewrite (stub agent) ────────────────────────────────
    step("A.4  edit_summary — rewrite when problems found")
    info(
        "The editor drops/softens claims attached to UNSUPPORTED "
        "citations or dead-source citations, then re-numbers the "
        "remaining [N] markers.  The original text is preserved on "
        "the report object for comparison."
    )

    rewritten = (
        "The QUIC protocol was standardised in RFC 9000 (May 2021) and "
        "forms the transport layer for HTTP/3 [1].  QUIC runs over UDP, "
        "eliminating the head-of-line blocking inherent in TCP [2].  "
        "Multiplexed streams allow independent delivery of resources "
        "without stalling sibling requests when one packet is lost [1]."
    )
    rewrite_agent = _make_stub_agent(verdict_map, rewritten_summary=rewritten)

    new_summary = edit_summary(
        inquiry="What is QUIC?",
        summary=SYNTHETIC_SUMMARY,
        citation_checks=checks,
        source_statuses=statuses,
        agent=rewrite_agent,
    )

    step("Before (original summary)")
    show_text(SYNTHETIC_SUMMARY)

    step("After (rewritten summary)")
    show_text(new_summary)

    removed = (
        "[3]" not in new_summary
        and "replaced TCP in all internet traffic" not in new_summary
    )
    if removed:
        ok("Unsupported claim and its [3] marker were removed.")
    else:
        warn("Rewrite did not remove the flagged claim — check stub agent.")


# ── Part B: full end-to-end via WebResearcher ────────────────────────────────


def demo_end_to_end(cm: ConfigManager) -> None:
    header("Part B — Full research + editor pass (live Ollama + network)")
    info(
        "WebResearcher.research() runs the crawl, synthesis, and then the "
        "editor stage in one call. The ResearchReport exposes:"
    )
    info("  .citation_checks    — one CitationCheck per [N] marker")
    info("  .source_statuses    — live/dead result per cited URL")
    info("  .original_summary   — pre-rewrite text (set when editor rewrote)")
    info("  .stats              — includes editor timing + rewrite flag")

    if not WEB_RESEARCH_AVAILABLE:
        fail('`[web]` extra not installed — run: pip install -e ".[web]"')
        return

    from pithos.tools.web_researcher import WebResearcher, WebResearchRequest

    model = ask("Ollama model for both subagents", "glm-4.7-flash")

    researcher = WebResearcher(cm)
    inquiry = "What is the QUIC transport protocol and how does it relate to HTTP/3?"
    domains = ["en.wikipedia.org", "developer.mozilla.org"]

    step(f"Inquiry: {inquiry}")
    info(f"Whitelist: {', '.join(domains)}")
    info(f"Model: {model}  |  verify_citations: {researcher.config.verify_citations}")

    request = WebResearchRequest(inquiry=inquiry, domains_override=domains)

    # Override the model used for both research and editor agents.
    researcher.config.subagent_model = model
    researcher.config.editor_model = model

    step("Running research() …  (this may take 30-90 seconds)")
    try:
        report = researcher.research(request)
    except Exception as exc:
        fail(f"research failed: {exc}")
        traceback.print_exc()
        return

    # ── Summary ──────────────────────────────────────────────────────────
    step("Final summary (after editor)")
    show_text(report.summary, limit_lines=30)

    if report.original_summary:
        step("Original summary (before editor rewrote)")
        show_text(report.original_summary, limit_lines=25)
        ok("Editor rewrote the summary.")
    else:
        info(
            "Editor did not rewrite (all citations were supported or no changes needed)."
        )

    # ── Source reachability ───────────────────────────────────────────────
    step("Source reachability")
    if not report.source_statuses:
        warn("No source statuses (verify_citations may be disabled).")
    for s in report.source_statuses:
        flag = f"{GREEN}LIVE{RESET}" if s.exists else f"{RED}DEAD{RESET}"
        print(f"  {flag}  {s.url}")

    # ── Citation checks ───────────────────────────────────────────────────
    step("Citation verdicts")
    if not report.citation_checks:
        warn("No citation checks (no [N] markers in summary or disabled).")
    for c in report.citation_checks:
        print(
            f"  [{c.index}] {verdict_colour(c.verdict):<28}"
            f"  {DIM}{(c.reason or '')[:60]}{RESET}"
        )

    # ── Stats ─────────────────────────────────────────────────────────────
    step("Stats (editor section)")
    editor_keys = {
        "citations_total",
        "citations_unsupported",
        "dead_sources",
        "editor_rewrote",
        "editor_duration_seconds",
        "editor_error",
    }
    stats = report.stats or {}
    for k in editor_keys:
        if k in stats:
            print(f"  {BOLD}{k}{RESET}: {stats[k]}")

    if report.errors:
        step("Errors")
        for err in report.errors:
            print(f"  {YELLOW}-{RESET} {err}")


# ── Part C: config knobs ─────────────────────────────────────────────────────


def demo_config_knobs() -> None:
    header("Part C — Config knobs")
    info(
        "Citation verification is controlled by three keys in "
        "configs/tools/web_research_config.yaml:"
    )
    print()

    rows = [
        (
            "verify_citations",
            "bool",
            "true",
            "Master switch. Set false to skip the editor stage entirely.",
        ),
        (
            "editor_config_name",
            "string",
            '"editor"',
            "Agent YAML in configs/agents/ used as the editor subagent.",
        ),
        (
            "editor_model",
            "string",
            "null",
            "Model override for the editor agent. Null inherits from the agent config.",
        ),
    ]

    col_w = [22, 8, 14, 0]
    header_row = ["Key", "Type", "Default", "Description"]
    fmt = "  {:<{}} {:<{}} {:<{}} {}"
    print(fmt.format(*[v for pair in zip(header_row, col_w) for v in pair], ""))
    print(f"  {'─' * 68}")
    for key, typ, default, desc in rows:
        wrapped = textwrap.wrap(desc, 38)
        print(fmt.format(key, col_w[0], typ, col_w[1], default, col_w[2], wrapped[0]))
        for extra in wrapped[1:]:
            print(fmt.format("", col_w[0], "", col_w[1], "", col_w[2], extra))

    print()
    info(
        "You can also pass editor_agent_factory= to WebResearcher.__init__ "
        "to supply a pre-built agent object, bypassing config loading. This "
        "is useful in tests or when you want the editor to use a different "
        "backend from the research subagent."
    )
    info(
        "The full ResearchReport.to_markdown() output includes a "
        "'## Citation verification' section summarising the verdicts and "
        "source reachability table."
    )

    # Show the markdown section template.
    step("Example markdown output (citation verification section)")
    example_md = """\
## Citation verification

### Source reachability

| URL | Status |
|-----|--------|
| https://en.wikipedia.org/wiki/QUIC | ✅ live |
| https://fake-stats-blog.example.com/http3-dominates | ❌ dead (404) |

### Claim checks

| # | Verdict | Reason | Source |
|---|---------|--------|--------|
| 1 | ✅ supported | RFC 9000 and HTTP/3 are mentioned explicitly | https://en.wikipedia.org/wiki/QUIC |
| 2 | ✅ supported | head-of-line blocking over UDP is explained | https://developer.mozilla.org/... |
| 3 | ❌ unsupported | no excerpts exist for this source | https://fake-stats-blog.example.com/... |
"""
    show_text(example_md)


# ── entry point ──────────────────────────────────────────────────────────────


def main() -> None:
    print(f"\n{BOLD}{CYAN}{'=' * 70}{RESET}")
    print(f"{BOLD}{CYAN}  pithos — Citation Verification (Editor Subagent) Demo{RESET}")
    print(f"{BOLD}{CYAN}  Two-stage: deterministic URL check + LLM claim audit{RESET}")
    print(f"{BOLD}{CYAN}{'=' * 70}{RESET}")

    cm = ConfigManager()

    print()
    if (
        ask("Run Part A (low-level functions, no network/LLM)? (y/n)", "y")
        .lower()
        .startswith("y")
    ):
        try:
            demo_low_level()
        except Exception as exc:
            fail(f"Part A failed: {exc}")
            traceback.print_exc()

    print()
    if (
        ask("Run Part B (full end-to-end, requires Ollama + internet)? (y/n)", "n")
        .lower()
        .startswith("y")
    ):
        try:
            demo_end_to_end(cm)
        except Exception as exc:
            fail(f"Part B failed: {exc}")
            traceback.print_exc()

    print()
    if ask("Run Part C (config knobs overview)? (y/n)", "y").lower().startswith("y"):
        demo_config_knobs()

    print(f"\n{BOLD}{CYAN}Done!{RESET}\n")


if __name__ == "__main__":
    main()
