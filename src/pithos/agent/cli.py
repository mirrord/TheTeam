"""CLI utilities for pithos agents."""

from pathlib import Path
import argparse
import sys
from typing import List

from ..config_manager import ConfigManager
from .agent import Agent
from .ollama_agent import OllamaAgent

# Sequences that open a tool-call span in agent output.  Used to detect the
# start of markup so it can be held back from the display buffer.
_TOOL_OPEN_SEQS: List[str] = [
    "[RUN]",
    "<RUN>",
    "[EXEC]",
    "<EXEC>",
    "RUN: ",
    "EXEC: ",
    "TOOL: ",
    "run(",
    "tool(",
    "execute(",
    "runcommand(",
]


def _safe_prefix_len(buf: str) -> int:
    """Return the number of leading characters in *buf* that are safe to print.

    Holds back any text from the earliest position where a tool-call opener
    might be starting (including partial matches at the end of the buffer).
    """
    earliest = len(buf)
    for seq in _TOOL_OPEN_SEQS:
        idx = buf.find(seq)
        if idx >= 0:
            earliest = min(earliest, idx)
            continue
        # Partial prefix at end of buffer — hold it back until we know more.
        for length in range(1, len(seq)):
            if buf.endswith(seq[:length]):
                earliest = min(earliest, len(buf) - length)
                break
    return earliest


def _format_tool_block(command: str, result: str) -> str:
    """Return a formatted string showing a command and its output.

    The block uses box-drawing characters to visually separate the tool
    execution from surrounding prose.  Only used for display; the agent's
    context is not affected.
    """
    cmd_label = f"$ {command}"
    bar_width = max(len(cmd_label) + 6, 50)
    top = f"  ┌─ {cmd_label} {'─' * max(0, bar_width - len(cmd_label) - 4)}┐"
    bottom = f"  └{'─' * (bar_width + 2)}┘"
    body_lines: List[str] = []
    output = result.rstrip() if result else ""
    if output:
        for line in output.splitlines():
            body_lines.append(f"  │ {line}")
    else:
        body_lines.append(f"  │ (no output)")
    return "\n" + top + "\n" + "\n".join(body_lines) + "\n" + bottom + "\n"


def interactive_chat(agent: Agent, verbose: bool = False) -> None:
    """Interactive streaming chat interface for an agent.

    Tokens are printed as they arrive so the user sees output immediately.
    Tool executions are shown as formatted command blocks; the raw markup
    tags (e.g. ``[RUN]...[/RUN]``) are suppressed from the display.
    The agent's context is not affected — only the text shown to the user
    changes.
    """
    print("Starting interactive chat. Press Ctrl+C to end the chat.")

    from pithos.tools import ToolCallExtractor

    _extractor = ToolCallExtractor()

    try:
        while True:
            user_input = input("You: ")
            if not user_input.strip():
                continue

            # Per-turn mutable state shared between the loop and the callback.
            _state: dict = {"buf": "", "pending_cmds": []}

            def _status_cb(status: str, detail=None) -> None:
                if status == "tool_call" and detail:
                    _state["pending_cmds"].append(detail)
                elif status == "tool_result":
                    # Strip any raw tool-call markup that's still in the buffer.
                    for call in _extractor.extract(_state["buf"]):
                        _state["buf"] = _state["buf"].replace(call.raw_text, "", 1)
                    # Flush buffered prose before the tool block.
                    if _state["buf"]:
                        sys.stdout.write(_state["buf"])
                        sys.stdout.flush()
                        _state["buf"] = ""
                    # Print the formatted command + result block.
                    cmd = (
                        "; ".join(_state["pending_cmds"])
                        if _state["pending_cmds"]
                        else ""
                    )
                    _state["pending_cmds"].clear()
                    sys.stdout.write(_format_tool_block(cmd, detail or ""))
                    sys.stdout.flush()

            print("Agent: ", end="", flush=True)
            for token in agent.stream(
                user_input, verbose=verbose, status_callback=_status_cb
            ):
                _state["buf"] += token
                # Flush the safe prefix of the buffer (everything before a
                # potential tool-call opener).
                safe_len = _safe_prefix_len(_state["buf"])
                if safe_len > 0:
                    sys.stdout.write(_state["buf"][:safe_len])
                    sys.stdout.flush()
                    _state["buf"] = _state["buf"][safe_len:]

            # Stream ended — flush any remaining buffered text.
            if _state["buf"]:
                # Remove residual markup (edge case: stream ended mid-call).
                for call in _extractor.extract(_state["buf"]):
                    _state["buf"] = _state["buf"].replace(call.raw_text, "", 1)
                sys.stdout.write(_state["buf"])
                sys.stdout.flush()
            print()

    except KeyboardInterrupt:
        print("\nEnding chat.")


def main() -> None:
    """CLI entrypoint for agent management."""
    parser = argparse.ArgumentParser(description="pithos Agent CLI")
    subparsers = parser.add_subparsers(dest="command")

    chat_parser = subparsers.add_parser("chat", help="Chat with an agent")
    chat_parser.add_argument(
        "agent_config",
        type=str,
        help="Path to agent config file, registered agent name, or model name",
    )
    chat_parser.add_argument(
        "--verbose", action="store_true", help="Enable verbose output"
    )

    reg_parser = subparsers.add_parser("register", help="Register agent config")
    reg_parser.add_argument(
        "agent_config", type=str, help="Path to the agent config file"
    )
    reg_parser.add_argument("--name", type=str, help="Name to register the agent as")

    args = parser.parse_args()
    config_manager = ConfigManager()

    if args.command == "chat":
        agent_path = Path(args.agent_config)
        if agent_path.exists():
            agent = OllamaAgent.from_yaml(str(agent_path), config_manager)
            print(f"Using agent config: {args.agent_config}")
        elif args.agent_config in config_manager.get_registered_agent_names():
            agent = OllamaAgent.from_config(args.agent_config, config_manager)
            print(f"Using registered agent: {args.agent_config}")
        else:
            agent = OllamaAgent(default_model=args.agent_config)
            print(f"Using base model: {args.agent_config}")

        interactive_chat(agent, args.verbose)

    elif args.command == "register":
        agent = OllamaAgent.from_yaml(args.agent_config, config_manager)
        agent.register(config_manager, args.name)
        print(f"Agent registered as '{agent.agent_name}'")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
