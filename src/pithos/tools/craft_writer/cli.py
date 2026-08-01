"""Command-line entry point for the CraftWriter tool.

Examples::

    pithos-craft-write "a heist gone wrong, melancholy tone"
    pithos-craft-write --title "The Last Job" --genre thriller --tone tense "a heist gone wrong"
    pithos-craft-write --source-title "Some Analyzed Story" --json "a quiet reunion"
    pithos-craft-write --no-revise --words 800 "a quiet reunion"
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
        prog="pithos-craft-write",
        description=(
            "Write a short story guided by previously analyzed craft notes "
            "(see pithos-craft-notes)."
        ),
    )
    p.add_argument(
        "direction",
        help="Freeform direction describing what to write.",
    )
    p.add_argument(
        "--title",
        default=None,
        help="Story title (otherwise proposed by the outline stage).",
    )
    p.add_argument(
        "--genre", default=None, help="Genre guidance for the outline stage."
    )
    p.add_argument("--tone", default=None, help="Tone guidance for the outline stage.")
    p.add_argument(
        "--words",
        type=int,
        default=None,
        dest="target_word_count",
        help="Approximate target word count for the story.",
    )
    p.add_argument(
        "--sections",
        type=int,
        default=None,
        dest="num_sections",
        help="Number of sections to plan (defaults to derived from --words or config).",
    )
    p.add_argument(
        "--source-title",
        default=None,
        help="Restrict retrieved craft notes to those derived from a specific analyzed story.",
    )
    p.add_argument(
        "--dimension",
        action="append",
        default=None,
        dest="dimensions",
        help="Limit note retrieval to specific craft dimensions (repeatable). Defaults to config.",
    )
    p.add_argument(
        "--no-revise",
        action="store_true",
        help="Skip the final whole-draft revision pass.",
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

    from .models import CraftWriteRequest
    from .writer import CraftWriter

    request = CraftWriteRequest(
        direction=args.direction,
        title=args.title,
        genre=args.genre,
        tone=args.tone,
        target_word_count=args.target_word_count,
        num_sections=args.num_sections,
        source_title=args.source_title,
        dimensions_override=args.dimensions,
        revise=False if args.no_revise else None,
    )

    cm = ConfigManager(args.config_dir) if args.config_dir else ConfigManager()
    writer = CraftWriter(cm)

    try:
        story = writer.write(request)
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"craft writing failed: {exc}", file=sys.stderr)
        return 1

    if args.json:
        payload = {
            "title": story.title,
            "premise": story.premise,
            "document_path": story.document_path,
            "sections": [
                {"heading": s.heading, "summary": s.summary, "text": s.text}
                for s in story.sections
            ],
            "full_text": story.full_text,
            "stats": story.stats,
            "errors": story.errors,
        }
        print(json.dumps(payload, indent=2))
    else:
        print(story.to_markdown())
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
