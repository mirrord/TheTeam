"""Shell interface — interactive stdin/stdout chat backed by pithos."""

from pithos.agent.cli import interactive_chat
from pithos.agent import Agent

from .system_commands import handle_system_command


class ShellInterface:
    """Thin wrapper that delegates to pithos's built-in interactive chat."""

    def __init__(self, agent: Agent, verbose: bool = False) -> None:
        self.agent = agent
        self.verbose = verbose

    def run(self) -> None:
        """Block on interactive chat until the user exits with Ctrl+C."""
        interactive_chat(
            self.agent,
            verbose=self.verbose,
            command_handler=lambda text: handle_system_command(self.agent, text),
        )


__all__ = ["ShellInterface"]
