"""Flowchart tool — lets agents invoke registered flowcharts as tools."""

import logging
import time
from typing import Any, Optional

from ..config_manager import ConfigManager
from .models import ToolMetadata, ToolResult
from .provider import ToolProvider

logger = logging.getLogger(__name__)


class FlowchartToolExecutor(ToolProvider):
    """Discovers and executes flowcharts on behalf of an agent."""

    def __init__(
        self,
        config_manager: ConfigManager,
        max_steps: int = 100,
    ):
        self.config_manager = config_manager
        self.max_steps = max_steps

    # ------------------------------------------------------------------
    # ToolProvider interface
    # ------------------------------------------------------------------

    def discover(self, platform: str = "cross-platform") -> dict[str, ToolMetadata]:
        """Return ToolMetadata for the flowchart dispatcher and each registered flowchart.

        Registers:
        - ``flowchart`` — dispatcher tool agents use to invoke any flowchart
        - ``flowchart:<name>`` — one entry per *enabled* registered flowchart

        Per-flowchart enable/disable is controlled by ``flowcharts.items`` in
        ``tool_config.yaml``.  Flowcharts not listed there default to enabled.
        """
        tools: dict[str, ToolMetadata] = {
            "flowchart": ToolMetadata(
                name="flowchart",
                path="",
                description="Run a registered pithos flowchart. Usage: flowchart <name> [input text]",
                platform=platform,
                source="virtual",
                tool_type="flowchart",
            )
        }
        for name in self.config_manager.get_registered_flowchart_names():
            if not self._is_flowchart_enabled(name):
                continue
            tools[f"flowchart:{name}"] = ToolMetadata(
                name=f"flowchart:{name}",
                path="",
                description=f"Run the '{name}' flowchart workflow",
                platform=platform,
                source="flowchart",
                tool_type="flowchart",
            )
        return tools

    def can_execute(self, tool_name: str) -> bool:
        """Return True for the ``flowchart`` dispatcher and ``flowchart:*`` names."""
        return tool_name == "flowchart" or tool_name.startswith("flowchart:")

    def execute(
        self,
        command: str,
        context: Optional[dict[str, Any]] = None,
    ) -> ToolResult:
        """Execute a flowchart tool call.

        Parses ``flowchart <name> [input]`` from *command*, builds the agents
        dict from ``context["agent"]``, and delegates to :meth:`run`.

        Args:
            command: Full command string, e.g. ``"flowchart my-flow some input"``.
            context: Must contain ``"agent"`` key with the calling agent instance
                for flowcharts that contain prompt nodes.
        """
        parts = command.strip().split(None, 2)
        # parts[0] == "flowchart"
        if len(parts) < 2:
            available = self.list_flowcharts()
            return ToolResult(
                success=False,
                stdout="",
                stderr="No flowchart name provided.",
                exit_code=-1,
                execution_time=0.0,
                command=command,
                error_hint=f"Usage: flowchart <name> [input]\nAvailable: {', '.join(available)}",
            )

        fc_name = parts[1]
        fc_input = parts[2] if len(parts) > 2 else ""

        agents_dict = self._build_agents_dict(fc_name, context)
        return self.run(fc_name, fc_input, agents_dict)

    # ------------------------------------------------------------------
    # Backward-compat alias
    # ------------------------------------------------------------------

    def discover_flowcharts(
        self, platform: str = "cross-platform"
    ) -> dict[str, ToolMetadata]:
        """Alias for :meth:`discover` kept for backward compatibility."""
        return self.discover(platform=platform)

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def run(
        self,
        flowchart_name: str,
        initial_input: str,
        agents: dict[str, Any],
    ) -> ToolResult:
        """Load and execute a registered flowchart, returning a ToolResult.

        Args:
            flowchart_name: Registered config name of the flowchart.
            initial_input: Text input to feed into the flowchart.
            agents: ``{name: agent_instance}`` dict required by the flowchart.

        Returns:
            ToolResult with the flowchart's final output.
        """
        from ..flowchart import Flowchart

        start = time.time()
        command = f"flowchart {flowchart_name} {initial_input}"

        try:
            fc = Flowchart.from_registered(flowchart_name, self.config_manager)
        except ValueError as exc:
            return ToolResult(
                success=False,
                stdout="",
                stderr=str(exc),
                exit_code=-1,
                execution_time=time.time() - start,
                command=command,
                error_hint=f"Flowchart '{flowchart_name}' not found. "
                f"Available flowcharts: {', '.join(self.list_flowcharts())}",
            )
        except Exception as exc:
            return ToolResult(
                success=False,
                stdout="",
                stderr=f"Failed to load flowchart: {exc}",
                exit_code=-1,
                execution_time=time.time() - start,
                command=command,
                error_hint="Check the flowchart YAML for syntax errors.",
            )

        try:
            output = fc.run(
                agents=agents,
                initial_input=initial_input,
                max_steps=self.max_steps,
            )
            return ToolResult(
                success=True,
                stdout=output,
                stderr="",
                exit_code=0,
                execution_time=time.time() - start,
                command=command,
            )
        except Exception as exc:
            return ToolResult(
                success=False,
                stdout="",
                stderr=str(exc),
                exit_code=1,
                execution_time=time.time() - start,
                command=command,
                error_hint="The flowchart execution failed. "
                "Check agent availability and flowchart configuration.",
            )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def list_flowcharts(self) -> list[str]:
        """Return names of all *enabled* registered flowcharts."""
        return [
            name
            for name in self.config_manager.get_registered_flowchart_names()
            if self._is_flowchart_enabled(name)
        ]

    # ------------------------------------------------------------------
    # Per-flowchart enable/disable
    # ------------------------------------------------------------------

    def _get_items_config(self) -> dict[str, Any]:
        """Return the ``flowcharts.items`` dict from tool_config, or ``{}``."""
        try:
            cfg = self.config_manager.get_config("tool_config", "tools")
            if not isinstance(cfg, dict):
                return {}
            items = cfg.get("flowcharts", {}).get("items")
            return items if isinstance(items, dict) else {}
        except Exception:
            return {}

    def _is_flowchart_enabled(self, name: str) -> bool:
        """Return True when *name* is enabled per the items config.

        Flowcharts absent from ``flowcharts.items`` default to ``True``
        (opt-out model — existing flowcharts are unaffected by the new config).
        """
        items = self._get_items_config()
        if name not in items:
            return True
        entry = items[name]
        if isinstance(entry, dict):
            return bool(entry.get("enabled", True))
        return True

    def _build_agents_dict(
        self,
        flowchart_name: str,
        context: Optional[dict[str, Any]],
    ) -> dict[str, Any]:
        """Build the agents dict required by the flowchart.

        Maps every agent name referenced in the flowchart to the calling
        agent from *context*, allowing single-agent flowcharts to just work.
        Falls back to an empty dict if the flowchart cannot be loaded or no
        agent is available.
        """
        agent = (context or {}).get("agent")
        if agent is None:
            return {}

        from ..flowchart import Flowchart
        from ..flownode import AgentPromptNode, GetHistoryNode, SetHistoryNode

        try:
            fc = Flowchart.from_registered(flowchart_name, self.config_manager)
            required: set[str] = set()
            for nid in fc.graph.nodes:
                nobj = fc.graph.nodes[nid]["nodeobj"]
                if isinstance(nobj, (AgentPromptNode, GetHistoryNode, SetHistoryNode)):
                    required.add(nobj.agent)
            return {name: agent for name in required}
        except Exception:
            return {}
