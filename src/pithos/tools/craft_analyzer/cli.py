"""Command-line entry point for the CraftAnalyzer tool.

Examples::

    pithos-craft-notes story.txt
    pithos-craft-notes --json story.txt
    pithos-craft-notes --roots ./data/research/stories --dimension dialogue
    pithos-craft-notes --text "Once upon a time..."
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
        prog="pithos-craft-notes",
        description=(
            "Analyze a story's creative-writing craft (characterization, scene "
            "construction, themes, prose style, dialogue, plot structure) and "
            "produce prescriptive how-to notes for writing similar stories."
        ),
    )
    p.add_argument(
        "source",
        nargs="?",
        help="Path to a single story file, or a directory to scan for text files.",
    )
    p.add_argument(
        "--text",
        default=None,
        help="Raw story text (alternative to a positional file/directory path).",
    )
    p.add_argument(
        "--roots",
        action="append",
        default=None,
        help="Directory root to scan for story files (repeatable).",
    )
    p.add_argument(
        "--title",
        default=None,
        help="Title used for the report and stored notes. Defaults to the file/dir name.",
    )
    p.add_argument(
        "--dimension",
        action="append",
        default=None,
        dest="dimensions",
        help="Limit analysis to specific craft dimensions (repeatable). Defaults to config.",
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

    from .analyzer import CraftAnalyzer
    from .models import CraftAnalysisRequest

    provided = [
        name
        for name, value in (
            ("source", args.source),
            ("--text", args.text),
            ("--roots", args.roots),
        )
        if value
    ]
    if len(provided) == 0:
        print("one of source, --text, or --roots must be provided", file=sys.stderr)
        return 2
    if len(provided) > 1:
        print(
            f"only one of source, --text, or --roots may be given (got: {provided})",
            file=sys.stderr,
        )
        return 2

    import os

    if args.text:
        request = CraftAnalysisRequest(text=args.text, title=args.title)
    elif args.roots:
        request = CraftAnalysisRequest(roots=args.roots, title=args.title)
    elif os.path.isdir(args.source):
        request = CraftAnalysisRequest(roots=[args.source], title=args.title)
    else:
        request = CraftAnalysisRequest(file_path=args.source, title=args.title)

    if args.dimensions:
        request.dimensions_override = args.dimensions

    cm = ConfigManager(args.config_dir) if args.config_dir else ConfigManager()
    analyzer = CraftAnalyzer(cm)

    try:
        report = analyzer.analyze(request)
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"craft analysis failed: {exc}", file=sys.stderr)
        return 1

    if args.json:
        payload = {
            "title": report.title,
            "document_path": report.document_path,
            "notes": [
                {
                    "dimension": n.dimension,
                    "note": n.note,
                    "evidence": n.evidence,
                }
                for n in report.notes
            ],
            "stats": report.stats,
            "errors": report.errors,
        }
        print(json.dumps(payload, indent=2))
    else:
        print(report.to_markdown())
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
