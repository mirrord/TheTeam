"""Talos CLI entry point.

Provides ``talos shell``, ``talos voice``, and ``talos telegram`` subcommands.
The first invocation triggers an interactive wizard to create
``~/.talos/config.yaml``.
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

from .config import DEFAULT_CONFIG_PATH, ensure_config, build_agent, load_config


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="talos",
        description="Talos — local-first AI assistant with shell/voice/telegram interfaces.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to Talos config file (default: ~/.talos/config.yaml).",
    )
    parser.add_argument(
        "--reconfigure",
        action="store_true",
        help="Force re-run of the setup wizard even if a config already exists.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable DEBUG-level logging.",
    )
    parser.add_argument(
        "--trace-flowcharts",
        action="store_true",
        help=(
            "Stream every flowchart's per-node activity "
            "(timestamp, node, input/output) to a trace file. "
            "Defaults to <config_dir>/traces/flowchart-trace-<timestamp>.log; "
            "override with --trace-flowcharts-path."
        ),
    )
    parser.add_argument(
        "--trace-flowcharts-path",
        type=Path,
        default=None,
        metavar="PATH",
        help="Override the trace file path used by --trace-flowcharts.",
    )

    sub = parser.add_subparsers(dest="interface", required=False)
    sub.add_parser("shell", help="Interactive stdin/stdout chat.")
    sub.add_parser("voice", help="Wake-word voice interface (speech-to-speech).")
    telegram_p = sub.add_parser("telegram", help="Telegram bot interface.")
    telegram_p.add_argument(
        "--show",
        action="store_true",
        help="Mirror the conversation to stdout and stream the agent's response.",
    )
    sub.add_parser("config", help="Run the setup wizard and exit.")

    # ---- tools subcommand --------------------------------------------------
    tools_parser = sub.add_parser(
        "tools",
        help="Manage tools available to the Talos agent.",
    )
    tools_sub = tools_parser.add_subparsers(
        dest="tools_action", metavar="ACTION", required=True
    )

    enable_p = tools_sub.add_parser(
        "enable", help="Allow a tool (adds to local allow list)."
    )
    enable_p.add_argument("tool_name", metavar="TOOL", help="Tool name to enable.")

    disable_p = tools_sub.add_parser(
        "disable", help="Block a tool (adds to local deny list)."
    )
    disable_p.add_argument("tool_name", metavar="TOOL", help="Tool name to disable.")

    tools_sub.add_parser(
        "list", help="List tools currently available to the Talos agent."
    )
    tools_sub.add_parser("ls", help="Alias for list.")
    tools_sub.add_parser(
        "list-all",
        help="List all configured tools with their availability status.",
    )
    # ------------------------------------------------------------------------

    mic = sub.add_parser(
        "mic-test",
        help="List microphone devices and record a test clip to verify input.",
    )
    mic.add_argument(
        "--device",
        type=int,
        default=None,
        metavar="N",
        help="Device index to test (default: system default).",
    )
    mic.add_argument(
        "--duration",
        type=float,
        default=5.0,
        metavar="SECS",
        help="Recording length in seconds (default: 5).",
    )
    mic.add_argument(
        "--no-transcribe",
        action="store_true",
        help="Skip whisper transcription (faster; no model download needed).",
    )
    mic.add_argument(
        "--whisper-device",
        default="cpu",
        metavar="DEV",
        help="Device for whisper: 'cpu' or 'cuda' (default: cpu).",
    )

    return parser


def _resolve_trace_path(explicit_path: Optional[Path], config_path: Path) -> Path:
    """Resolve the trace file path for ``--trace-flowcharts``.

    Returns *explicit_path* if given, otherwise a timestamped default file
    under ``<config_dir>/traces/``.
    """
    if explicit_path:
        return Path(explicit_path)
    config_dir = config_path.parent
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    return config_dir / "traces" / f"flowchart-trace-{ts}.log"


def main(argv: Optional[list[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )

    if args.trace_flowcharts:
        from pithos.flowchart import enable_global_trace

        config_path = Path(args.config) if args.config else DEFAULT_CONFIG_PATH
        trace_path = _resolve_trace_path(args.trace_flowcharts_path, config_path)
        enable_global_trace(trace_path)
        print(f"Streaming flowchart trace to {trace_path}")

    if args.interface is None:
        parser.print_help()
        return 1

    # tools subcommand does not require a full agent build — handle early.
    if args.interface == "tools":
        from .tools_cmd import run as run_tools
        from .config import DEFAULT_CONFIG_PATH

        config_path = Path(args.config) if args.config else DEFAULT_CONFIG_PATH
        return run_tools(args, config_path)

    config, path = ensure_config(args.config, force_wizard=args.reconfigure)

    if args.interface == "config":
        print(f"Config ready at {path}")
        return 0

    if args.interface == "mic-test":
        try:
            from .interfaces.voice import test_microphone
        except ImportError as exc:
            print(f"Microphone test unavailable: {exc}", file=sys.stderr)
            return 2
        test_microphone(
            device=args.device,
            duration=args.duration,
            transcribe=not args.no_transcribe,
            whisper_device=args.whisper_device,
        )
        return 0

    agent = build_agent(config)

    if args.interface == "shell":
        from .interfaces.shell import ShellInterface

        ShellInterface(agent).run()
        return 0

    if args.interface == "voice":
        try:
            from .interfaces.voice import VoiceInterface
        except ImportError as exc:
            print(f"Voice interface unavailable: {exc}", file=sys.stderr)
            return 2
        VoiceInterface(agent, config.voice).run()
        return 0

    if args.interface == "telegram":
        try:
            from .interfaces.telegram import TelegramInterface
        except ImportError as exc:
            print(f"Telegram interface unavailable: {exc}", file=sys.stderr)
            return 2
        TelegramInterface(agent, config.telegram, show=args.show).run()
        return 0

    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
