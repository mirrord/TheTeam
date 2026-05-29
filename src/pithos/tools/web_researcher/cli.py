"""Command-line entry point for the WebResearcher tool.

Examples::

    pithos-research "How does HTTP/3 differ from HTTP/2?"
    pithos-research --json "Notable open-source vector databases"
    pithos-research --domains en.wikipedia.org --domains arxiv.org "BERT pretraining"
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from typing import Optional

from ...config_manager import ConfigManager
from .researcher import WebResearcher


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="pithos-research",
        description=(
            "Run a subagent-driven web research pass against a configurable "
            "whitelist of trusted domains and print a Markdown report."
        ),
    )
    p.add_argument("inquiry", nargs="+", help="The research question / topic.")
    p.add_argument(
        "--domains",
        action="append",
        default=None,
        help="Override whitelisted domains (repeatable). Defaults to config.",
    )
    p.add_argument(
        "--seed-url",
        action="append",
        default=None,
        dest="seed_urls",
        help="Additional starting URL (repeatable). Must be on a whitelisted domain.",
    )
    p.add_argument(
        "--json",
        action="store_true",
        help="Emit a JSON document instead of Markdown.",
    )
    p.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress non-error logging.",
    )
    p.add_argument(
        "--config-dir",
        default=None,
        help="Override the configs/ directory (defaults to project configs/).",
    )
    return p


def main(argv: Optional[list[str]] = None) -> int:
    args = _build_parser().parse_args(argv)

    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    try:
        from . import WEB_RESEARCH_AVAILABLE
    except Exception as exc:
        print(f"web-research package failed to import: {exc}", file=sys.stderr)
        return 2
    if not WEB_RESEARCH_AVAILABLE:
        print(
            "Optional web dependencies are not installed. " "Run: pip install .[web]",
            file=sys.stderr,
        )
        return 2

    cm = ConfigManager(args.config_dir) if args.config_dir else ConfigManager()
    researcher = WebResearcher(cm)

    inquiry = " ".join(args.inquiry).strip()
    if not inquiry:
        print("inquiry must not be empty", file=sys.stderr)
        return 2

    from .models import WebResearchRequest

    request = WebResearchRequest(
        inquiry=inquiry,
        domains_override=args.domains,
        extra_seed_urls=args.seed_urls or [],
    )

    try:
        report = researcher.research(request)
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"research failed: {exc}", file=sys.stderr)
        return 1

    if args.json:
        payload = {
            "inquiry": report.inquiry,
            "summary": report.summary,
            "sources": report.sources,
            "excerpt_count": len(report.excerpts),
            "stats": report.stats,
            "errors": report.errors,
        }
        print(json.dumps(payload, indent=2))
    else:
        print(report.to_markdown())
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
