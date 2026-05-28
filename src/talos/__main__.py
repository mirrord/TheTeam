"""Talos CLI entry point.

Provides ``talos shell``, ``talos voice``, and ``talos telegram`` subcommands.
The first invocation triggers an interactive wizard to create
``~/.talos/config.yaml``.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

from .config import ensure_config, build_agent


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

    sub = parser.add_subparsers(dest="interface", required=False)
    sub.add_parser("shell", help="Interactive stdin/stdout chat.")
    sub.add_parser("voice", help="Wake-word voice interface (speech-to-speech).")
    sub.add_parser("telegram", help="Telegram bot interface.")
    sub.add_parser("config", help="Run the setup wizard and exit.")

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


def main(argv: Optional[list[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )

    if args.interface is None:
        parser.print_help()
        return 1

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
        TelegramInterface(agent, config.telegram).run()
        return 0

    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
