"""Memory tool provider — owns memory operation execution for the agent system.

When the ``memory`` virtual tool is invoked via the tool pipeline (e.g.
``RUN: memory STORE[notes]: some content``), :class:`MemoryToolProvider`
parses the memory operation and executes it directly against the agent's
:class:`~pithos.tools.memory_tool.MemoryStore`.

:class:`MemoryToolProvider` also provides :meth:`extract_and_execute` for
the agent's post-stream pass that scans the full response text for inline
memory operations (STORE / RETRIEVE) and executes them automatically.

This provider is registered in the :class:`~pithos.tools.registry.ToolRegistry`
when memory is enabled, via
:meth:`~pithos.tools.registry.ToolRegistry.register_provider`.
"""

import time
from typing import Any, Optional

from .memory_ops import MemoryOpExtractor, MemoryOpRequest
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

        result_msg = self._execute_ops(mem_ops, agent.memory_store, agent.metrics)
        return ToolResult(
            success=True,
            stdout=result_msg,
            stderr="",
            exit_code=0,
            execution_time=time.time() - start,
            command=command,
        )

    # ------------------------------------------------------------------
    # Core execution (used by both the tool-call path and the post-stream pass)
    # ------------------------------------------------------------------

    def _execute_ops(
        self,
        operations: list[MemoryOpRequest],
        memory_store: Any,
        metrics: Optional[Any] = None,
    ) -> str:
        """Execute a list of memory operations and return a formatted result string.

        This is the single authoritative implementation for store/retrieve
        logic, metrics recording, error formatting, and tag-suggestion
        reporting.  Both the tool-call path (:meth:`execute`) and the
        post-stream inline-op path (:meth:`extract_and_execute`) delegate
        here.

        Args:
            operations: Parsed memory operation requests.
            memory_store: The agent's :class:`~pithos.tools.memory_tool.MemoryStore`
                instance.  May be ``None`` when memory is not enabled.
            metrics: Optional :class:`~pithos.metrics.MetricsCollector` for
                recording store / retrieve events.

        Returns:
            A human-readable result string suitable for injection into the
            agent context as a system message.
        """
        if not memory_store:
            return "Memory system is not available."

        results: list[str] = []
        for op in operations:
            try:
                if op.operation == "store":
                    if not op.content:
                        results.append(
                            f"\u2717 Store operation failed: No content provided\n"
                            f"\U0001f4a1 Hint: Use format like STORE[{op.category}]: your content here"
                        )
                        continue

                    entry_id = memory_store.store(op.category, op.content)
                    if metrics is not None:
                        try:
                            metrics.record_memory_store()
                        except Exception:
                            pass
                    store_msg = (
                        f"\u2713 Stored in {op.category}: "
                        f"{op.content[:50]}... (ID: {entry_id})"
                    )
                    if memory_store.tag_suggestions_enabled:
                        try:
                            entry_meta = memory_store.get_all_entries(op.category)
                            tags: list[str] = []
                            for e in entry_meta:
                                if e.get("id") == entry_id:
                                    tags = e.get("metadata", {}).get(
                                        "suggested_tags", []
                                    )
                                    break
                            if tags:
                                store_msg += (
                                    f"\n  \U0001f3f7 Suggested tags: {', '.join(tags)}"
                                )
                        except Exception:
                            pass
                    results.append(store_msg)

                elif op.operation == "retrieve":
                    if not op.query:
                        results.append(
                            f"\u2717 Retrieve operation failed: No query provided\n"
                            f"\U0001f4a1 Hint: Use format like RETRIEVE[{op.category}]: your search query"
                        )
                        continue

                    search_results = memory_store.retrieve(op.category, op.query)
                    if metrics is not None:
                        try:
                            metrics.record_memory_retrieve(
                                result_count=len(search_results)
                            )
                        except Exception:
                            pass
                    if search_results:
                        results.append(
                            f"\u2713 Retrieved {len(search_results)} results from "
                            f"{op.category} for query: {op.query}"
                        )
                        for i, result in enumerate(search_results[:3], 1):
                            results.append(
                                f"  {i}. [Score: {result.relevance_score:.2f}] {result.content}"
                            )
                        if len(search_results) > 3:
                            results.append(f"  ... and {len(search_results) - 3} more")
                    else:
                        results.append(
                            f"\u2717 No relevant results found in {op.category} for: {op.query}"
                        )

            except Exception as exc:
                results.append(
                    f"\u2717 Error in {op.operation} operation: {exc}\n"
                    "\U0001f4a1 Hint: Check that the category name is valid and "
                    "content/query is properly formatted"
                )

        return "\n".join(results)

    def extract_and_execute(
        self,
        response_text: str,
        agent: Any,
    ) -> Optional[str]:
        """Scan *response_text* for inline memory operations and execute them.

        Called by the agent's post-stream pass to process STORE / RETRIEVE
        syntax that the model emitted as part of its response text.

        Args:
            response_text: The full assistant response to scan.
            agent: The agent instance — must expose ``memory_store`` and
                ``metrics`` attributes.

        Returns:
            A formatted result string when at least one operation was found
            and executed, or ``None`` when no operations were detected.
        """
        ops = self._extractor.extract(response_text)
        if not ops:
            return None
        return self._execute_ops(ops, agent.memory_store, agent.metrics)
