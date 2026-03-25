"""CLI entry points for the pithos tool management system."""

import argparse
import json
import sys

from ..config_manager import ConfigManager
from .executor import ToolExecutor, format_tool_result_for_agent
from .registry import ToolRegistry


def tool_cli_main(args) -> None:
    """Execute a tool command and display agent-formatted output.

    Args:
        args: Parsed command-line arguments with 'tool_command' and 'format' fields.
    """
    # Build command string from args
    command = " ".join(args.tool_command)

    # Initialize tool system
    config_manager = ConfigManager()
    tool_registry = ToolRegistry(config_manager)
    executor = ToolExecutor(
        timeout=tool_registry.config.get("timeout", 30),
        max_output_size=tool_registry.config.get("max_output_size", 10000),
    )

    # Execute the tool
    result = executor.run(command, tool_registry)

    # Format output based on requested format
    if args.format == "agent":
        # Agent-formatted output (default)
        print(format_tool_result_for_agent(result))
    elif args.format == "json":
        # JSON output
        output = {
            "command": result.command,
            "success": result.success,
            "exit_code": result.exit_code,
            "execution_time": result.execution_time,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }
        if result.error_hint:
            output["error_hint"] = result.error_hint
        print(json.dumps(output, indent=2))
    elif args.format == "simple":
        # Simple output - just stdout/stderr
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print(result.stderr, file=sys.stderr)
        # Exit with tool's exit code
        sys.exit(result.exit_code)

    # Exit with non-zero if tool failed (except for simple format which already exited)
    if not result.success and args.format != "simple":
        sys.exit(1)


def main() -> None:
    """CLI entry point for tool management."""
    parser = argparse.ArgumentParser(description="pithos Tool Management")
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # List command
    subparsers.add_parser("list", help="List available tools")

    # Show command
    show_parser = subparsers.add_parser("show", help="Show tool details")
    show_parser.add_argument("tool_name", help="Name of tool to show")

    # Refresh command
    subparsers.add_parser("refresh", help="Refresh tool cache")

    # Test command
    test_parser = subparsers.add_parser("test", help="Test tool execution")
    test_parser.add_argument("tool_name", help="Name of tool to test")
    test_parser.add_argument("args", nargs="*", help="Arguments to pass to tool")

    args = parser.parse_args()

    # Initialize config manager and registry
    config_manager = ConfigManager()
    tool_registry = ToolRegistry(config_manager)

    if args.command == "list":
        print("Available tools:")
        for tool_name in tool_registry.list_tools():
            tool = tool_registry.get_tool(tool_name)
            if tool:
                print(f"  {tool_name}: {tool.description}")

    elif args.command == "show":
        tool = tool_registry.get_tool(args.tool_name)
        if tool:
            print(f"Name: {tool.name}")
            print(f"Path: {tool.path}")
            print(f"Description: {tool.description}")
            print(f"Platform: {tool.platform}")
            print(f"Source: {tool.source}")
        else:
            print(f"Tool '{args.tool_name}' not found or not allowed")

    elif args.command == "refresh":
        print("Refreshing tool registry...")
        tool_registry.refresh()
        print(f"Found {len(tool_registry.tools)} tools")

    elif args.command == "test":
        command = f"{args.tool_name} {' '.join(args.args)}"
        print(f"Executing: {command}")
        executor = ToolExecutor()
        result = executor.run(command, tool_registry)
        print(f"\nSuccess: {result.success}")
        print(f"Exit code: {result.exit_code}")
        print(f"Execution time: {result.execution_time:.3f}s")
        if result.stdout:
            print(f"\nStdout:\n{result.stdout}")
        if result.stderr:
            print(f"\nStderr:\n{result.stderr}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
