"""Tool registry - aggregates tool metadata from registered ToolProviders.� aggregates tool metadata from registered ToolProviders.

The registry is a passive coordinator: it holds no discovery logic of its own.
All discovery, filtering, and execution is delegated to
:class:`~pithos.tools.provider.ToolProvider` implementations.

Typical usage::

    cli = CLIToolProvider(config)
    fc  = FlowchartToolExecutor(config_manager)
    registry = ToolRegistry(config_manager, providers=[cli, fc])
    registry.list_tools()         # -> sorted list of all tool names
    registry.get_provider("python")  # -> CLIToolProvider instance
"""

from typing import Any, Optional

from ..config_manager import ConfigManager
from .models import ToolMetadata
from .provider import ToolProvider


class ToolRegistry:
    """Registry that aggregates ToolProvider results.

    The registry:
    - calls ToolProvider.discover on each registered provider at init
    - merges the results into a single flat {name: ToolMetadata} dict
    - exposes lookup, filtering, and confirmation helpers for use by
      ToolExecutor

    Parameters
    ----------
    config_manager:
        Used to load the tool configuration block (``tool_config.yaml``).
    providers:
        List of ToolProvider instances to register immediately.  More can be
        added later via register_provider.  Pass an empty list (or omit) to
        start with no tools.
    """

    def __init__(
        self,
        config_manager: ConfigManager,
        providers: Optional[list[ToolProvider]] = None,
    ) -> None:
        if config_manager is None:
            raise ValueError("config_manager cannot be None")

        self.config_manager = config_manager
        self.config: dict[str, Any] = self._load_config()
        self.tools: dict[str, ToolMetadata] = {}
        self._providers: list[ToolProvider] = []

        for provider in providers or []:
            self.register_provider(provider)

    # ------------------------------------------------------------------
    # Provider management
    # ------------------------------------------------------------------

    def register_provider(self, provider: ToolProvider) -> None:
        """Register a provider and merge its tools into the registry.

        Discovery is run immediately; first-registered wins on name collision.

        Args:
            provider: ToolProvider instance to add.
        """
        self._providers.append(provider)
        discovered = provider.discover()
        for name, meta in discovered.items():
            if name not in self.tools:
                self.tools[name] = meta

    def get_provider(self, tool_name: str) -> Optional[ToolProvider]:
        """Return the first registered provider that can execute tool_name.

        Args:
            tool_name: Leading command token (e.g. "python").

        Returns:
            ToolProvider instance, or None if no provider claims this tool.
        """
        for provider in self._providers:
            if provider.can_execute(tool_name):
                return provider
        return None

    def refresh(self) -> None:
        """Re-run discovery on all providers and reload configuration.

        Useful when the tool configuration changes at runtime (e.g. new
        flowcharts registered or PATH updated).
        """
        self.config = self._load_config()
        self.tools.clear()
        for provider in self._providers:
            discovered = provider.discover()
            for name, meta in discovered.items():
                if name not in self.tools:
                    self.tools[name] = meta

    # ------------------------------------------------------------------
    # Tool lookup helpers
    # ------------------------------------------------------------------

    def get_tool(self, name: str) -> Optional[ToolMetadata]:
        """Return metadata for name, or None if not found."""
        return self.tools.get(name)

    def list_tools(self) -> list[str]:
        """Return a sorted list of all available tool names."""
        return sorted(self.tools.keys())

    def is_allowed(self, tool_name: str) -> bool:
        """Return True when tool_name is present in the registry.

        A tool is allowed if any registered provider discovered it (providers
        enforce their own filtering in ToolProvider.discover).

        Raises:
            ValueError: If tool_name is empty.
        """
        if not tool_name or not tool_name.strip():
            raise ValueError("tool_name cannot be empty")
        return tool_name in self.tools

    def requires_confirmation(self, tool_name: str) -> bool:
        """Return True when the tool config marks tool_name for confirmation.

        Only active when ``mode`` is ``"confirm"`` and the tool appears in the
        ``confirm`` list.
        """
        if self.config.get("mode") != "confirm":
            return False
        return tool_name in self.config.get("confirm", [])

    def get_tool_list_text(self) -> str:
        """Return a formatted tool summary for agent system prompts."""
        if not self.tools:
            return "No tools available."

        cli_lines: list[str] = []
        flowchart_lines: list[str] = []
        web_research_lines: list[str] = []
        memory_lines: list[str] = []

        for tool_name in sorted(self.tools.keys()):
            tool = self.tools[tool_name]
            if tool.tool_type == "flowchart":
                if not tool_name.startswith("flowchart:"):
                    flowchart_lines.append(f"  - {tool_name}: {tool.description}")
                else:
                    short = tool_name.removeprefix("flowchart:")
                    flowchart_lines.append(f"      {short}")
            elif tool.tool_type == "web_research":
                web_research_lines.append(f"  - {tool_name}: {tool.description}")
            elif tool.tool_type == "memory":
                memory_lines.append(f"  - {tool_name}: {tool.description}")
            else:
                cli_lines.append(f"  - {tool_name}: {tool.description}")

        sections: list[str] = []
        if cli_lines:
            sections.append("CLI tools:\n" + "\n".join(cli_lines))
        if flowchart_lines:
            sections.append(
                "Flowchart tools (use: flowchart <name> [input]):\n"
                + "\n".join(flowchart_lines)
            )
        if web_research_lines:
            sections.append("Web research:\n" + "\n".join(web_research_lines))
        if memory_lines:
            sections.append("Memory operations:\n" + "\n".join(memory_lines))

        return "\n\n".join(sections)

    # ------------------------------------------------------------------
    # Config loading
    # ------------------------------------------------------------------

    def _load_config(self) -> dict[str, Any]:
        config = self.config_manager.get_config("tool_config", "tools")
        if not config:
            return {
                "enabled": True,
                "timeout": 30,
                "max_output_size": 10000,
                "mode": "include",
                "include": [
                    "python",
                    "pip",
                    "git",
                    "curl",
                    "wget",
                    "node",
                    "npm",
                    "echo",
                    "cat",
                    "ls",
                    "dir",
                    "pwd",
                    "cd",
                ],
                "exclude": ["rm", "del", "format", "shutdown", "reboot", "kill"],
                "descriptions": {},
            }
        return config
