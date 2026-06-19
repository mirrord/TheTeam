"""Abstract base class for tool providers in the pithos tool calling system.

A :class:`ToolProvider` encapsulates everything needed to surface a group of
related tools to the :class:`~pithos.tools.registry.ToolRegistry`:

- **discovery** — what tools are available and what are their metadata
- **routing**   — does this provider own a given tool name
- **execution** — how to run the tool given a raw command string

Concrete implementations live in:
- :mod:`pithos.tools.cli_provider`    — system PATH executables
- :mod:`pithos.tools.flowchart_tool`  — registered pithos flowcharts
- :mod:`pithos.tools.web_researcher.researcher` — web-research subagent
- :mod:`pithos.tools.memory_provider` — agent memory operations
"""

from abc import ABC, abstractmethod
from typing import Any, Optional

from .models import ToolMetadata, ToolResult


class ToolProvider(ABC):
    """Interface that all tool providers must implement.

    The :class:`~pithos.tools.registry.ToolRegistry` calls :meth:`discover`
    once on startup and then routes each execution request to whichever
    registered provider first returns ``True`` from :meth:`can_execute`.

    ``context`` passed to :meth:`execute` is an optional dict of runtime
    dependencies (e.g. ``{"agent": agent_instance}``) that providers may
    need but that are not known at construction time.
    """

    @abstractmethod
    def discover(self) -> dict[str, ToolMetadata]:
        """Return all tools this provider makes available.

        Returns:
            Mapping of tool name → :class:`~pithos.tools.models.ToolMetadata`.
            The registry merges all provider results; first-registered wins on
            name collision.
        """

    @abstractmethod
    def can_execute(self, tool_name: str) -> bool:
        """Return True if this provider can handle *tool_name*.

        Args:
            tool_name: The leading token of the command string (e.g.
                ``"python"``, ``"flowchart"``, ``"memory"``).
        """

    @abstractmethod
    def execute(
        self,
        command: str,
        context: Optional[dict[str, Any]] = None,
    ) -> ToolResult:
        """Execute *command* and return a :class:`~pithos.tools.models.ToolResult`.

        Args:
            command: Full command string as written by the agent (e.g.
                ``"python --version"`` or ``"flowchart my-flow input text"``).
            context: Optional runtime dependencies.  Common keys:
                ``"agent"`` — the calling :class:`~pithos.agent.agent.Agent`
                instance (needed by memory and flowchart providers).
        """
