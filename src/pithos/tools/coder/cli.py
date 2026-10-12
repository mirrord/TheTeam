"""Command-line entry point for the coding harness (``code`` tool).

Examples::

    pithos-code "fix the failing test in tests/test_foo.py"
    pithos-code --root ./myrepo "add a --verbose flag to the CLI"
    pithos-code --json --steps 20 "refactor utils.py into smaller modules"
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
        prog="pithos-code",
        description=(
            "Run a coding task through the agentic harness: plan, edit files, "
            "run tests, and self-correct in a workspace-jailed loop."
        ),
    )
    p.add_argument("task", help="Natural-language description of the coding task.")
    p.add_argument(
        "--root",
        default=None,
        help="Workspace root the agent is confined to (defaults to config/cwd).",
    )
    p.add_argument(
        "--test",
        default=None,
        dest="test_command",
        help="Optional test command hint shown to the agent.",
    )
    p.add_argument(
        "--steps",
        type=int,
        default=None,
        dest="step_limit",
        help="Override the maximum number of agent steps.",
    )
    p.add_argument(
        "--json",
        action="store_true",
        help="Emit a JSON result instead of Markdown.",
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

    from .coder import Coder
    from .models import CodeHarnessConfig, CodeTaskRequest

    cm = ConfigManager(args.config_dir) if args.config_dir else ConfigManager()

    config = None
    if args.step_limit is not None:
        config = _load_config_with_overrides(cm, step_limit=args.step_limit)

    coder = Coder(cm, config=config)
    request = CodeTaskRequest(
        task=args.task, root=args.root, test_command=args.test_command
    )

    try:
        result = coder.code(request)
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"coding harness failed: {exc}", file=sys.stderr)
        return 1

    if args.json:
        payload = {
            "task": result.task,
            "exit_status": result.exit_status.value,
            "summary": result.summary,
            "diff": result.diff,
            "steps": result.steps,
            "trajectory_path": result.trajectory_path,
            "errors": result.errors,
        }
        print(json.dumps(payload, indent=2))
    else:
        print(result.to_markdown())
    return 0 if result.success else 1


def _load_config_with_overrides(cm, **overrides) -> "object":
    """Load the harness config and apply CLI overrides (small helper)."""
    from .harness import CodeHarness

    config = CodeHarness._load_config(cm)
    for key, value in overrides.items():
        if value is not None and hasattr(config, key):
            setattr(config, key, value)
    return config


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
