"""pithos CLI entrypoint."""

import sys
import argparse


def main():
    """Main entry point for pithos CLI."""
    parser = argparse.ArgumentParser(
        description="pithos - LLM Agent Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # Agent command (default behavior)
    agent_parser = subparsers.add_parser(
        "agent", help="Run agent interactions", add_help=False
    )

    # Diagnostic command
    subparsers.add_parser("diagnostic", help="Run environment and connectivity checks")

    # Tool command - execute tools with agent-formatted output
    tool_parser = subparsers.add_parser(
        "tool", help="Execute a tool and see agent-formatted output"
    )
    tool_parser.add_argument(
        "--format",
        choices=["agent", "json", "simple"],
        default="agent",
        help="Output format (default: agent)",
    )
    tool_parser.add_argument(
        "tool_command",
        nargs=argparse.REMAINDER,
        help="Tool command to execute (e.g., 'python --version')",
    )

    # Parse args
    args, unknown = parser.parse_known_args()

    # If no command or agent command, delegate to agent CLI
    if not args.command or args.command == "agent":
        from .agent import main as agent_main

        # Put unknown args back for agent to parse
        sys.argv = [sys.argv[0]] + unknown
        agent_main()
    elif args.command == "diagnostic":
        from .diagnostic import run_diagnostics

        verbose = "--verbose" in unknown or "-v" in unknown
        ok = run_diagnostics(verbose=verbose)
        sys.exit(0 if ok else 1)
    elif args.command == "tool":
        from .tools import tool_cli_main

        tool_cli_main(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
