"""Shell interface — interactive stdin/stdout chat backed by pithos."""

from pithos.agent.cli import interactive_chat
from pithos.agent import Agent


class ShellInterface:
    """Thin wrapper that delegates to pithos's built-in interactive chat."""

    def __init__(self, agent: Agent, verbose: bool = False) -> None:
        self.agent = agent
        self.verbose = verbose

    def run(self) -> None:
        """Block on interactive chat until the user exits with Ctrl+C."""
        interactive_chat(self.agent, verbose=self.verbose)


__all__ = ["ShellInterface"]
