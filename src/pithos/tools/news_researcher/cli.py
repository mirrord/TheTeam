"""Command-line entry point for the NewsResearcher tool.

Examples::

    pithos-research-news "recent advances in cache quantization"
    pithos-research-news --json "new transformer architectures"
    pithos-research-news --recency-days 7 --domains arxiv.org "diffusion models"
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from typing import Optional

from ...config_manager import ConfigManager


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="pithos-research-news",
        description=(
            "Collect and summarise recent news articles relevant to an inquiry "
            "from a configurable whitelist of domains and feeds."
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
        "--feed",
        action="append",
        default=None,
        dest="feeds",
        help="Override RSS/Atom feed URLs (repeatable). Defaults to config.",
    )
    p.add_argument(
        "--recency-days",
        type=int,
        default=None,
        help="Only include articles newer than this many days. Defaults to config.",
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
        from . import NEWS_RESEARCH_AVAILABLE
    except Exception as exc:
        print(f"news-research package failed to import: {exc}", file=sys.stderr)
        return 2
    if not NEWS_RESEARCH_AVAILABLE:
        print(
            "Optional web dependencies are not installed. Run: pip install .[web]",
            file=sys.stderr,
        )
        return 2

    from .models import NewsResearchRequest
    from .researcher import NewsResearcher

    cm = ConfigManager(args.config_dir) if args.config_dir else ConfigManager()
    researcher = NewsResearcher(cm)

    inquiry = " ".join(args.inquiry).strip()
    if not inquiry:
        print("inquiry must not be empty", file=sys.stderr)
        return 2

    request = NewsResearchRequest(
        inquiry=inquiry,
        domains_override=args.domains,
        feeds_override=args.feeds,
        recency_days_override=args.recency_days,
    )

    try:
        report = researcher.research(request)
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"news research failed: {exc}", file=sys.stderr)
        return 1

    if args.json:
        payload = {
            "inquiry": report.inquiry,
            "terms": report.terms,
            "document_path": report.document_path,
            "relevant": [
                {
                    "url": a.url,
                    "title": a.title,
                    "published": a.published_iso,
                    "summary": a.summary,
                    "reason": a.reason,
                }
                for a in report.relevant
            ],
            "articles_reviewed": len(report.assessments),
            "stats": report.stats,
            "errors": report.errors,
        }
        print(json.dumps(payload, indent=2))
    else:
        print(report.to_markdown())
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
