"""CLI utilities for pithos agents."""

from pathlib import Path
import argparse

from ..config_manager import ConfigManager
from .agent import Agent
from .ollama_agent import OllamaAgent


def interactive_chat(agent: Agent, verbose: bool = False) -> None:
    """Interactive streaming chat interface for an agent.

    Tokens are printed as they arrive so the user sees output immediately.
    Tool executions (mid-stream interruptions) are transparent — the stream
    resumes after each tool result without any visible pause in output.
    """
    print("Starting interactive chat. Press Ctrl+C to end the chat.")
    try:
        while True:
            user_input = input("You: ")
            if not user_input.strip():
                continue
            print("Agent: ", end="", flush=True)
            for token in agent.stream(user_input, verbose=verbose):
                print(token, end="", flush=True)
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
