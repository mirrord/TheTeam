"""Demo script for pithos framework.

This provides a simple interactive demonstration of pithos capabilities.
"""

import sys
from .agent import OllamaAgent, interactive_chat
from .config_manager import ConfigManager


def main() -> None:
    """Run interactive demo with a sample agent."""
    print("=" * 60)
    print("pithos Interactive Demo")
    print("=" * 60)
    print("\nThis demo lets you chat with an LLM agent.")
    print("You can optionally use a predefined agent config or flowchart.")
    print()

    config_manager = ConfigManager()

    # Show available configs
    agents = list(config_manager.get_registered_agent_names())

    if agents:
        print(f"Available agents: {', '.join(agents)}")

    print("\nOptions:")
    print("  1. Chat with default model (llama3.2:3b)")
    if agents:
        print("  2. Chat with a registered agent")
    print("  q. Quit")
    print()

    choice = input("Select option (1/2/q): ").strip()

    if choice == "q":
        print("Goodbye!")
        sys.exit(0)

    agent = None

    if choice == "1":
        print("\nUsing default model: llama3.2:3b")
        agent = OllamaAgent(default_model="llama3.2:3b")

    elif choice == "2" and agents:
        print("\nAvailable agents:")
        for i, name in enumerate(agents, 1):
            print(f"  {i}. {name}")
        agent_choice = input("Select agent number: ").strip()
        try:
            agent_idx = int(agent_choice) - 1
            agent_name = agents[agent_idx]
            print(f"\nLoading agent: {agent_name}")
            agent = OllamaAgent.from_config(agent_name, config_manager)
        except (ValueError, IndexError):
            print("Invalid choice. Using default model.")
            agent = OllamaAgent(default_model="llama3.2:3b")

    else:
        print("\nUsing default model: llama3.2:3b")
        agent = OllamaAgent(default_model="llama3.2:3b")

    print("\n" + "=" * 60)
    print("Starting interactive chat...")
    print("Type 'quit' or 'exit' to end the conversation.")
    print("=" * 60 + "\n")

    interactive_chat(agent, verbose=False)


if __name__ == "__main__":
    main()
