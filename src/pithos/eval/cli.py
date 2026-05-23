"""``pithos-eval`` command-line interface.

Subcommands:

* ``run`` — load an :class:`EvalConfig` YAML, execute the runner, and
  write per-case JSONL plus an aggregated report.
* ``report`` — re-aggregate an existing run directory (no execution).
* ``list-configs`` — list YAML configs under ``configs/eval/``.
* ``list-suites`` — list registered capability suites (placeholder until
  Phase 6 lands).
* ``analyze`` — placeholder for re-running analyzers on stored traces;
  currently equivalent to ``report``.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from typing import Optional, Sequence

import yaml

from .config import EvalConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_config(path: str) -> EvalConfig:
    with open(path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    return EvalConfig.from_dict(data)


def _load_price_map(path: Optional[str]) -> dict:
    if not path:
        return {}
    if not os.path.exists(path):
        logger.warning("Price map %s not found; ignoring", path)
        return {}
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def _find_configs(root: str = "configs/eval") -> list[str]:
    if not os.path.isdir(root):
        return []
    return sorted(
        os.path.join(root, name)
        for name in os.listdir(root)
        if name.endswith((".yaml", ".yml"))
    )


# ---------------------------------------------------------------------------
# Subcommands
# ---------------------------------------------------------------------------


def cmd_run(args: argparse.Namespace) -> int:
    from .reporter import Reporter
    from .runner import EvalRunner

    config = _load_config(args.config)
    if args.rounds:
        config.execution.rounds = int(args.rounds)
    if args.output_dir:
        config.output.base_dir = args.output_dir

    price_map = _load_price_map(args.price_map)

    runner = EvalRunner(
        config,
        resume=not args.no_resume,
        write_outputs=not args.dry_run,
        max_cases_per_task=args.max_cases,
    )
    records = runner.run()

    reporter = Reporter(
        config_name=config.name,
        rounds=config.execution.rounds,
        price_map=price_map,
    )
    report = reporter.build_report(records)

    if not args.dry_run:
        paths = reporter.write(report, config.run_dir)
        print(f"Wrote {paths['report_json']}")
        print(f"Wrote {paths['class_report_csv']}")

    print(json.dumps(report.class_report, indent=2, default=str))
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    from .reporter import Reporter, load_records_from_run_dir

    records = load_records_from_run_dir(args.run_dir)
    if not records:
        print(f"No case records found under {args.run_dir}", file=sys.stderr)
        return 1
    price_map = _load_price_map(args.price_map)
    reporter = Reporter(
        config_name=os.path.basename(args.run_dir),
        rounds=max((r.round_num for r in records), default=1),
        price_map=price_map,
    )
    report = reporter.build_report(records)
    paths = reporter.write(report, args.run_dir)
    print(f"Wrote {paths['report_json']}")
    print(f"Wrote {paths['class_report_csv']}")
    print(json.dumps(report.class_report, indent=2, default=str))
    return 0


def cmd_list_configs(args: argparse.Namespace) -> int:
    paths = _find_configs(args.dir)
    if not paths:
        print(f"No eval configs found under {args.dir}")
        return 0
    for p in paths:
        print(p)
    return 0


def cmd_list_suites(args: argparse.Namespace) -> int:
    builtin = [
        ("multiple_choice", "Multi-choice accuracy task"),
        ("free_form", "Free-form output with exact/regex/LLM judge"),
        ("tool_use", "Tool-call expectations + ToolTraceGrader"),
        ("memory_recall", "Two-turn memory recall + MemoryRecallGrader"),
        ("self_reflection", "Planted-error correction + RegexMatchGrader"),
    ]
    for name, desc in builtin:
        print(f"{name:20s}  {desc}")
    return 0


def cmd_analyze(args: argparse.Namespace) -> int:
    # In v1 ``analyze`` is an alias for ``report``; once traces are
    # rehydrated from disk (Phase 6+) this will re-run analyzers.
    return cmd_report(args)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="pithos-eval", description=__doc__)
    sub = parser.add_subparsers(dest="command")

    p_run = sub.add_parser("run", help="Run an evaluation config")
    p_run.add_argument("--config", required=True, help="Path to EvalConfig YAML")
    p_run.add_argument("--rounds", type=int, default=None)
    p_run.add_argument("--output-dir", default=None)
    p_run.add_argument("--price-map", default=None)
    p_run.add_argument("--max-cases", type=int, default=None)
    p_run.add_argument("--no-resume", action="store_true")
    p_run.add_argument(
        "--dry-run", action="store_true", help="Do not write any output files"
    )
    p_run.set_defaults(func=cmd_run)

    p_report = sub.add_parser("report", help="Re-aggregate an existing run directory")
    p_report.add_argument("--run-dir", required=True)
    p_report.add_argument("--price-map", default=None)
    p_report.set_defaults(func=cmd_report)

    p_list = sub.add_parser("list-configs", help="List eval YAML configs")
    p_list.add_argument("--dir", default="configs/eval")
    p_list.set_defaults(func=cmd_list_configs)

    p_suites = sub.add_parser("list-suites", help="List registered capability suites")
    p_suites.set_defaults(func=cmd_list_suites)

    p_analyze = sub.add_parser("analyze", help="Re-run analyzers on a stored run")
    p_analyze.add_argument("--run-dir", required=True)
    p_analyze.add_argument("--price-map", default=None)
    p_analyze.set_defaults(func=cmd_analyze)

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    logging.basicConfig(
        level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s"
    )
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "command", None):
        parser.print_help()
        return 1
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
