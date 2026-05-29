"""Demo: Web Researcher — subagent-driven, whitelist-bound web crawler.

This script walks through the `web-research` virtual tool end-to-end:

1. **Configuration & capability check** — show the active config, the
   whitelist of domains, and whether the optional ``[web]`` extra is
   installed.
2. **Direct programmatic usage** — call :class:`WebResearcher.research`
   from Python and render the resulting markdown report (with cited
   sources).
3. **Agent tool call** — let an :class:`OllamaAgent` invoke the tool via
   ``RUN: web-research <inquiry>`` and stream the cited summary back into
   the conversation.
4. **Flowchart node** — execute a tiny in-memory flowchart that wraps a
   ``webresearch`` node and feeds its markdown output into a downstream
   node.

Run:
    python demos/web_researcher_demo.py

Requirements:
    - ``pip install -e ".[web]"`` (requests, beautifulsoup4, trafilatura)
    - Live internet connection for Parts 2-4 (skipped gracefully if absent)
    - Ollama running locally for Part 3 (default model: glm-4.7-flash)
"""

from __future__ import annotations

import sys
import textwrap
import traceback
from pathlib import Path

# Allow running this file directly from the repo without `pip install -e .`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from pithos import ConfigManager, OllamaAgent
from pithos.flownode import create_node
from pithos.tools.web_researcher import (
    WEB_RESEARCH_AVAILABLE,
    WebResearchRequest,
)

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


def ask(prompt: str, default: str) -> str:
    answer = input(f"  {prompt} [{default}]: ").strip()
    return answer if answer else default


def show_markdown(md: str, limit_lines: int = 40) -> None:
    lines = md.splitlines()
    for line in lines[:limit_lines]:
        print(f"    {line}")
    if len(lines) > limit_lines:
        print(f"    {DIM}... ({len(lines) - limit_lines} more lines){RESET}")


# ── Part 1: configuration & capability check ────────────────────────────────


def demo_config(cm: ConfigManager):
    """Return a built WebResearcher if the optional extra is installed, else None."""
    header("Part 1 — Configuration & capability check")
    info(
        "The web-research tool is a virtual tool: the agent invokes it "
        "with `RUN: web-research <inquiry>` and dispatch is handled "
        "in-process. The crawler is restricted to a configurable "
        "whitelist of domains."
    )

    step("Optional `[web]` extra installed?")
    if WEB_RESEARCH_AVAILABLE:
        print(f"  {GREEN}yes{RESET} — requests + beautifulsoup4 + trafilatura present")
    else:
        fail('missing — run: pip install -e ".[web]"')
        warn("Parts 2-4 will be skipped.")
        return None

    # Heavy classes are exposed lazily via the package __getattr__ shim, so
    # only import them once we know the extra is installed.
    from pithos.tools.web_researcher import WebResearcher

    researcher = WebResearcher(cm)
    cfg = researcher.config

    step("Active configuration")
    print(f"  {BOLD}domains{RESET}              : {', '.join(cfg.domains) or '(none)'}")
    print(f"  {BOLD}max_pages{RESET}            : {cfg.max_pages}")
    print(f"  {BOLD}max_iterations{RESET}       : {cfg.max_iterations}")
    print(f"  {BOLD}dedup_similarity{RESET}     : {cfg.dedup_similarity}")
    print(f"  {BOLD}per_domain_rps{RESET}       : {cfg.per_domain_rps}")
    print(f"  {BOLD}respect_robots{RESET}       : {cfg.respect_robots}")
    print(f"  {BOLD}persist_directory{RESET}    : {cfg.persist_directory}")
    print(f"  {BOLD}subagent_config_name{RESET} : {cfg.subagent_config_name}")

    if not cfg.domains:
        warn(
            "No domains are configured. Add some under `domains:` in "
            "configs/tools/web_research_config.yaml or pass "
            "`domains_override` on a request."
        )

    return researcher


# ── Part 2: direct programmatic usage ───────────────────────────────────────


def demo_direct(researcher) -> None:  # noqa: ANN001 - WebResearcher lazy import
    header("Part 2 — Direct programmatic usage")
    info(
        "WebResearcher.research(...) returns a ResearchReport with "
        "deduplicated excerpts, the cited summary, and a Sources "
        "section. Each call builds its own fetcher/store, so concurrent "
        "runs stay isolated."
    )

    inquiry = "What is the QUIC transport protocol, and how does it relate to HTTP/3?"
    # Constrain to a couple of well-behaved encyclopedic/spec sources for a
    # short, predictable demo run.
    domains = ["en.wikipedia.org", "developer.mozilla.org"]

    step(f"Inquiry: {inquiry}")
    info(f"Restricting whitelist to: {', '.join(domains)}")

    request = WebResearchRequest(inquiry=inquiry, domains_override=domains)

    try:
        report = researcher.research(request)
    except Exception as exc:  # network errors, ollama down, etc.
        fail(f"research failed: {exc}")
        traceback.print_exc()
        return

    step("Summary report (markdown, truncated)")
    show_markdown(report.to_markdown(), limit_lines=40)

    step("Sources cited")
    if not report.sources:
        warn("no sources were collected")
    for i, url in enumerate(report.sources, 1):
        print(f"  {i:>2}. {url}")

    step("Stats")
    for k, v in (report.stats or {}).items():
        print(f"  {BOLD}{k}{RESET}: {v}")
    if report.errors:
        step("Errors")
        for err in report.errors:
            print(f"  {YELLOW}-{RESET} {err}")


# ── Part 3: agent-driven tool call ──────────────────────────────────────────


def demo_agent(cm: ConfigManager) -> None:
    header("Part 3 — Agent invokes the tool via `RUN: web-research`")
    info(
        "With tools enabled, an OllamaAgent sees `web-research` listed "
        "alongside CLI tools. When it emits a tool call mid-stream, "
        "dispatch is intercepted, the crawler runs, and the cited "
        "markdown report is injected back into the stream."
    )

    model = ask("Ollama model", "glm-4.7-flash")
    step(f"Creating agent with model: {model}")

    try:
        agent = OllamaAgent(default_model=model, agent_name="web_demo")
        agent.enable_tools(cm)
    except Exception as exc:
        fail(f"failed to construct agent / enable tools: {exc}")
        return

    if getattr(agent, "web_research_executor", None) is None:
        warn(
            "Agent could not wire up the web-research executor — check "
            "that `web_research.enabled: true` is set in "
            "configs/tools/tool_config.yaml and the `[web]` extra is "
            "installed."
        )
        return

    prompt = (
        "Use the web-research tool to find the main advantages of HTTP/3 "
        "over HTTP/2. Reply with a short bulleted summary and cite the "
        "sources the tool returns. Only call the tool once."
    )

    step("Prompt")
    info(prompt)

    step("Streaming response")
    print()
    try:
        for token in agent.stream(prompt):
            print(token, end="", flush=True)
        print("\n")
    except Exception as exc:
        fail(f"agent stream failed: {exc}")
        traceback.print_exc()


# ── Part 4: flowchart node usage ────────────────────────────────────────────


def demo_flowchart(cm: ConfigManager, researcher) -> None:  # noqa: ANN001
    header("Part 4 — `webresearch` flowchart node")
    info(
        "The `webresearch` node lets you embed the tool inside any "
        "pithos flowchart. After it runs, `current_input` holds the "
        "rendered markdown report and the structured result lives at "
        "the configured `save_to` key. Here we execute the node "
        "directly against a hand-built context to show the contract "
        "without wiring up a full flowchart graph."
    )

    inquiry = "What is the BLAKE3 hash function?"
    step(f"Inquiry: {inquiry}")
    info("Restricting to en.wikipedia.org for a fast, predictable run.")

    research_node = create_node(
        "webresearch",
        {
            "inquiry": "{inquiry}",
            "save_to": "research_report",
            "domains": ["en.wikipedia.org"],
            "error_handling": "continue",
        },
    )

    # Hand-built execution context. In a real flowchart the executor
    # populates these keys for you; we set them explicitly here so the
    # demo stays self-contained.
    context: dict = {
        "inquiry": inquiry,
        "web_researcher": researcher,
        "config_manager": cm,
        "current_input": inquiry,
    }

    try:
        # Invoke the node's _execute hook directly — it returns the dict
        # the flowchart executor would normally merge back into context.
        result = research_node._execute(context)
    except Exception as exc:
        fail(f"node execution failed: {exc}")
        traceback.print_exc()
        return

    report = result.get("research_report") or {}
    step("Structured result keys")
    for k in sorted(report.keys()):
        print(f"  - {k}")

    step("Markdown report (truncated)")
    show_markdown(report.get("markdown", "(no markdown produced)"), limit_lines=30)


# ── entry point ──────────────────────────────────────────────────────────────


def main() -> None:
    print(f"\n{BOLD}{CYAN}{'=' * 70}{RESET}")
    print(f"{BOLD}{CYAN}  pithos — Web Researcher Demo{RESET}")
    print(f"{BOLD}{CYAN}  Subagent-driven, whitelist-bound web crawler{RESET}")
    print(f"{BOLD}{CYAN}{'=' * 70}{RESET}")

    cm = ConfigManager()

    researcher = demo_config(cm)
    if researcher is None:
        # Capability check failed; bail out cleanly.
        return

    print()
    if (
        ask("Run Part 2 (direct programmatic crawl)? (y/n)", "y")
        .lower()
        .startswith("y")
    ):
        demo_direct(researcher)

    print()
    if (
        ask("Run Part 3 (agent tool call, requires Ollama)? (y/n)", "n")
        .lower()
        .startswith("y")
    ):
        demo_agent(cm)

    print()
    if ask("Run Part 4 (flowchart node)? (y/n)", "y").lower().startswith("y"):
        demo_flowchart(cm, researcher)

    print(f"\n{BOLD}{CYAN}Done!{RESET}\n")


if __name__ == "__main__":
    main()
