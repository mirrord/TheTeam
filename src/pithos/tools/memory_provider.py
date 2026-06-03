"""Memory tool provider — routes memory tool calls to the agent's memory system.

When the ``memory`` virtual tool is invoked via the tool pipeline (e.g.
``RUN: memory STORE[notes]: some content``), :class:`MemoryToolProvider`
parses the memory operation and delegates execution to the calling agent's
:meth:`~pithos.agent.agent.Agent._execute_memory_ops` method, which owns
the :class:`~pithos.tools.memory_tool.MemoryStore` and metrics integration.

This provider is registered in the :class:`~pithos.tools.registry.ToolRegistry`
when memory is enabled, via
:meth:`~pithos.tools.registry.ToolRegistry.register_provider`.
"""

import time
from typing import Any, Optional

from .memory_ops import MemoryOpExtractor
from .models import ToolMetadata, ToolResult
from .provider import ToolProvider

_MEMORY_TOOL_NAMES = frozenset(
    {"memory", "memory:search", "memory:add", "memory:delete"}
)


class MemoryToolProvider(ToolProvider):
    """Exposes the agent's vector memory as a virtual tool.

    Execution requires ``context["agent"]`` to be the calling agent instance.
    If no agent is available the operation returns an error result rather than
    raising.

    Args:
        config: Tool configuration dict (used for manual description overrides).
    """

    def __init__(self, config: Optional[dict[str, Any]] = None) -> None:
        self._config = config or {}
        self._extractor = MemoryOpExtractor()

    # ------------------------------------------------------------------
    # ToolProvider interface
    # ------------------------------------------------------------------

    def discover(self) -> dict[str, ToolMetadata]:
        """Return metadata for the memory dispatcher and operation sub-tools."""
        descriptions = self._config.get("descriptions", {})

        tools: dict[str, ToolMetadata] = {
            "memory": ToolMetadata(
                name="memory",
                path="",
                description=descriptions.get(
                    "memory",
                    "Access agent long-term memory. "
                    "Usage: memory STORE[category]: content  |  memory RETRIEVE[category]: query",
                ),
                platform="cross-platform",
                source="virtual",
                tool_type="memory",
            ),
        }
        for operation in ("search", "add", "delete"):
            key = f"memory:{operation}"
            tools[key] = ToolMetadata(
                name=key,
                path="",
                description=descriptions.get(
                    key,
                    f"Memory operation '{operation}'. Usage: memory {operation} <query>",
                ),
                platform="cross-platform",
                source="virtual",
                tool_type="memory",
            )
        return tools

    def can_execute(self, tool_name: str) -> bool:
        """Return True for ``memory`` and ``memory:*`` tool names."""
        return tool_name in _MEMORY_TOOL_NAMES or tool_name.startswith("memory:")

    def execute(
        self,
        command: str,
        context: Optional[dict[str, Any]] = None,
    ) -> ToolResult:
        """Parse memory operations from *command* and execute them.

        Delegates to ``context["agent"]._execute_memory_ops()`` so that all
        memory store/retrieve logic, metrics recording, and error formatting
        remain in one place.

        Args:
            command: Full command string, e.g.
                ``"memory STORE[notes]: some text"`` or
                ``"memory RETRIEVE[notes]: query"``.
            context: Must contain ``"agent"`` with the calling agent instance.
        """
        start = time.time()
        agent = (context or {}).get("agent")

        if agent is None:
            return ToolResult(
                success=False,
                stdout="",
                stderr="Memory tool requires an agent context.",
                exit_code=-1,
                execution_time=0.0,
                command=command,
                error_hint="Memory operations are only available when running inside an agent.",
            )

        mem_ops = self._extractor.extract(command)
        if not mem_ops:
            return ToolResult(
                success=False,
                stdout="",
                stderr="No valid memory operation found in command.",
                exit_code=-1,
                execution_time=time.time() - start,
                command=command,
                error_hint=(
                    "Memory tool expects commands like:\n"
                    "  memory STORE[category]: <content>\n"
                    "  memory RETRIEVE[category]: <query>"
                ),
            )

        result_msg = agent._execute_memory_ops(mem_ops)
        return ToolResult(
            success=True,
            stdout=result_msg,
            stderr="",
            exit_code=0,
            execution_time=time.time() - start,
            command=command,
        )
