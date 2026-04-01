"""Flowchart tool — lets agents invoke registered flowcharts as tools."""

import logging
import time
from typing import Any

from ..config_manager import ConfigManager
from .models import ToolMetadata, ToolResult

logger = logging.getLogger(__name__)


class FlowchartToolExecutor:
    """Discovers and executes flowcharts on behalf of an agent."""

    def __init__(
        self,
        config_manager: ConfigManager,
        timeout: int = 120,
        max_steps: int = 100,
    ):
        self.config_manager = config_manager
        self.timeout = timeout
        self.max_steps = max_steps

    # ------------------------------------------------------------------
    # Discovery
    # ------------------------------------------------------------------

    def discover_flowcharts(
        self, platform: str = "cross-platform"
    ) -> dict[str, ToolMetadata]:
        """Return ToolMetadata entries for every registered flowchart.

        Each flowchart is exposed as a virtual tool named
        ``flowchart:<config_name>`` so that agents can call
        ``RUN: flowchart <name> <input>``.
        """
        tools: dict[str, ToolMetadata] = {}

        flowchart_files = list(self.config_manager.get_registered_flowchart_names())
        for name in flowchart_files:
            tools[f"flowchart:{name}"] = ToolMetadata(
                name=f"flowchart:{name}",
                path="",
                description=f"Run the '{name}' flowchart workflow",
                platform=platform,
                source="flowchart",
                tool_type="flowchart",
            )

        return tools

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
        """Return names of all registered flowcharts."""
        return list(self.config_manager.get_registered_flowchart_names())
