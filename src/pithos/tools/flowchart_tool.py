"""Flowchart tool — lets agents invoke registered flowcharts as tools."""

import logging
import threading
import time
from typing import Any, Optional

from ..config_manager import ConfigManager
from .models import ToolMetadata, ToolResult
from .provider import ToolProvider, parse_no_timeout

logger = logging.getLogger(__name__)


class FlowchartToolExecutor(ToolProvider):
    """Discovers and executes flowcharts on behalf of an agent."""

    def __init__(
        self,
        config_manager: ConfigManager,
        max_steps: int = 100,
        timeout: Optional[float] = 120.0,
    ):
        self.config_manager = config_manager
        self.max_steps = max_steps
        # Overall wall-clock cap on a single flowchart run (seconds). ``None``
        # disables the cap entirely (blocks until the flowchart finishes).
        # Enforced via a daemon thread since a hung flowchart step (e.g. a
        # CustomNode or GitNode with its own internal timeout) cannot be
        # forcibly interrupted from another thread in Python.
        self.timeout = timeout

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

        result_holder: list[str] = []
        exc_holder: list[BaseException] = []

        def _run() -> None:
            try:
                result_holder.append(
                    fc.run(
                        agents=agents,
                        initial_input=initial_input,
                        max_steps=self.max_steps,
                    )
                )
            except BaseException as exc:  # noqa: BLE001 - re-raised in caller thread
                exc_holder.append(exc)

        # Run in a daemon thread with an overall wall-clock cap. A hung node
        # (e.g. a CustomNode or GitNode already blocked past its own internal
        # timeout) cannot be forcibly interrupted, so a timed-out run is
        # abandoned rather than joined indefinitely -- the thread finishes on
        # its own eventually since it is a daemon thread.
        timeout = self.get_timeout(flowchart_name)
        thread = threading.Thread(target=_run, daemon=True)
        thread.start()
        thread.join(timeout=timeout)

        if thread.is_alive():
            return ToolResult(
                success=False,
                stdout="",
                stderr=f"Flowchart '{flowchart_name}' timed out after "
                f"{timeout} seconds",
                exit_code=-1,
                execution_time=time.time() - start,
                command=command,
                error_hint="This flowchart run timed out. Try a smaller or "
                "simpler input, break the task into smaller steps, or "
                "increase 'flowcharts.timeout' in tool_config.yaml.",
            )

        if exc_holder:
            exc = exc_holder[0]
            if isinstance(exc, TimeoutError):
                # A node inside the flowchart (e.g. CustomNode, GitNode) hit
                # its own internal timeout -- give a timeout-specific hint
                # instead of the generic failure hint below.
                return ToolResult(
                    success=False,
                    stdout="",
                    stderr=str(exc),
                    exit_code=1,
                    execution_time=time.time() - start,
                    command=command,
                    error_hint="A step inside this flowchart timed out. Try "
                    "a smaller or simpler input, or increase that step's "
                    "timeout in the flowchart's YAML config.",
                )
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

        output = result_holder[0] if result_holder else ""
        return ToolResult(
            success=True,
            stdout=output,
            stderr="",
            exit_code=0,
            execution_time=time.time() - start,
            command=command,
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
    # Per-flowchart config (enable/disable, timeout override)
    # ------------------------------------------------------------------

    def _get_tool_config(self) -> dict[str, Any]:
        """Return the full tool_config dict, or ``{}`` if unavailable."""
        try:
            cfg = self.config_manager.get_config("tool_config", "tools")
            return cfg if isinstance(cfg, dict) else {}
        except Exception:
            return {}

    def get_timeout(self, flowchart_name: str) -> Optional[float]:
        """Return the effective wall-clock timeout for running *flowchart_name*.

        The timeout is disabled (``None``) when the top-level ``no_timeout``
        list in ``tool_config.yaml`` contains either ``flowchart`` (all
        flowcharts) or ``flowchart:<flowchart_name>`` (this flowchart only).

        Args:
            flowchart_name: Registered config name of the flowchart.

        Returns:
            ``None`` for no limit, otherwise :attr:`timeout`.
        """
        no_timeout = parse_no_timeout(self._get_tool_config())
        if "flowchart" in no_timeout or f"flowchart:{flowchart_name}" in no_timeout:
            return None
        return self.timeout

    def _get_items_config(self) -> dict[str, Any]:
        """Return the ``flowcharts.items`` dict from tool_config, or ``{}``."""
        try:
            items = self._get_tool_config().get("flowcharts", {}).get("items")
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
