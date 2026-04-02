"""pithos Agent - Abstract base class for LLM agents."""

from abc import ABC, abstractmethod
from typing import Optional, Any, Type, TypeVar, Iterator
import logging
import uuid
import yaml

from ..config_manager import ConfigManager
from ..tools import ToolRegistry, ToolExecutor, MemoryOpRequest, MemoryOpExtractor
from ..tools.flowchart_tool import FlowchartToolExecutor
from ..context import Msg, UserMsg, AgentMsg, AgentContext
from .history import ConversationStore, HistorySearchResult
from .compaction import CompactionConfig, MemoryCompactor
from .recall import RecallConfig, AutoRecall
from ..metrics import MetricsCollector

try:
    from ..tools.memory_tool import MemoryStore

    MEMORY_AVAILABLE = True
except ImportError:
    MEMORY_AVAILABLE = False
    MemoryStore = None

_AgentT = TypeVar("_AgentT", bound="Agent")

logger = logging.getLogger(__name__)


class Agent(ABC):
    """
    Abstract base class for LLM agents. Manages multiple contexts.
    Subclasses must implement `send()` to provide the backend-specific LLM call.
    """

    def __init__(
        self,
        default_model: str,
        agent_name: Optional[str] = None,
        system_prompt: str = "",
        temperature: Optional[float] = None,
    ):
        self.default_model = default_model
        self.agent_name = agent_name or default_model
        self.default_system_prompt = system_prompt
        self.temperature = temperature if temperature is not None else 0.7
        self.max_tokens = -1
        self.contexts: dict[str, AgentContext] = {}
        self.current_context: Optional[str] = None
        # Tool calling support
        self.tools_enabled = False
        self.tool_registry: Optional[ToolRegistry] = None
        self.tool_executor: Optional[ToolExecutor] = None
        self.flowchart_executor: Optional[FlowchartToolExecutor] = None
        self.tool_auto_loop = False
        self.tool_max_iterations = 5
        # Memory tool support
        self.memory_enabled = False
        self.memory_store: Optional[Any] = None  # MemoryStore instance
        # Conversation history support
        self.history_store: Optional[ConversationStore] = None
        self.session_id: Optional[str] = None
        self._last_history_message_id: Optional[str] = None
        # Metrics collection (optional, attached via attach_metrics())
        self.metrics: Optional[MetricsCollector] = None
        # Automatic context compaction (optional, enabled via enable_compaction())
        self.compaction_enabled = False
        self._compactor: Optional[MemoryCompactor] = None
        # Automatic memory recall (optional, enabled via enable_recall())
        self.recall_enabled = False
        self._auto_recall: Optional[AutoRecall] = None
        # Chain-of-thought inference flowchart (optional)
        self.inference_flowchart: Optional[Any] = None
        self._inference_config: Optional[Any] = None
        self._running_inference: bool = False
        # Create default context
        self.create_context("default", system_prompt)

    @classmethod
    def from_dict(
        cls: Type[_AgentT], config: dict[str, Any], config_manager: ConfigManager
    ) -> _AgentT:
        """Create agent from configuration dictionary."""
        model = config.get("model")
        if not model:
            raise ValueError("Agent config must specify 'model'")
        agent = cls(
            model,
            config.get("name"),
            config.get("system_prompt", ""),
            config.get("temperature"),
        )

        # Load contexts
        contexts = config.get("contexts", {})
        for ctx_name, ctx_data in contexts.items():
            if ctx_name != "default":
                agent.contexts[ctx_name] = AgentContext.from_dict(
                    ctx_data, ctx_name, config_manager
                )

        # Switch to specified context
        current_ctx = config.get("current_context", "default")
        if current_ctx in agent.contexts:
            agent.current_context = current_ctx

        # Load compaction config if present
        compaction_cfg = config.get("compaction")
        if compaction_cfg and compaction_cfg.get("enabled", False):
            cfg = CompactionConfig(
                threshold=compaction_cfg.get("threshold", 20),
                keep_last=compaction_cfg.get("keep_last", 6),
                summary_model=compaction_cfg.get("summary_model"),
                memory_category=compaction_cfg.get(
                    "memory_category", "context_summaries"
                ),
            )
            agent.enable_compaction(cfg)

        # Load recall config if present
        recall_cfg = config.get("recall")
        if recall_cfg and recall_cfg.get("enabled", False):
            cfg_r = RecallConfig(
                sources=recall_cfg.get("sources", ["memory", "history"]),
                n_results=recall_cfg.get("n_results", 5),
                recall_model=recall_cfg.get("recall_model"),
                categories=recall_cfg.get("categories", []),
                min_relevance=recall_cfg.get("min_relevance", 0.5),
            )
            agent.enable_recall(cfg_r)

        # Load inference flowchart if present
        inference_cfg = config.get("inference")
        if inference_cfg is not None:
            agent.set_inference_flowchart(inference_cfg, config_manager)

        return agent

    @classmethod
    def from_yaml(
        cls: Type[_AgentT], config_file: str, config_manager: ConfigManager
    ) -> _AgentT:
        """Load agent from YAML configuration file."""
        with open(config_file, "r") as file:
            config = yaml.safe_load(file)
        return cls.from_dict(config, config_manager)

    @classmethod
    def from_config(
        cls: Type[_AgentT], name: str, config_manager: ConfigManager
    ) -> _AgentT:
        """Load agent from registered configuration."""
        config = config_manager.get_config(name, "agents")
        if not config:
            return cls(default_model=name)
        return cls.from_dict(config, config_manager)

    def to_dict(self) -> dict[str, Any]:
        """Serialize agent configuration to dictionary."""
        d: dict[str, Any] = {
            "name": self.agent_name,
            "model": self.default_model,
            "system_prompt": self.default_system_prompt,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "current_context": self.current_context,
        }

        # Serialize non-default contexts
        contexts = {}
        for ctx_name, ctx in self.contexts.items():
            if ctx_name != "default":
                contexts[ctx_name] = ctx.to_dict(with_history=True)
        if contexts:
            d["contexts"] = contexts

        # Serialize inference flowchart config
        if self._inference_config is not None:
            d["inference"] = self._inference_config
        elif self.inference_flowchart is not None:
            d["inference"] = self.inference_flowchart.to_dict()

        return d

    def register(
        self, config_manager: ConfigManager, registered_name: Optional[str] = None
    ) -> None:
        """Register this agent configuration."""
        registered_name = registered_name or self.agent_name
        config_manager.register_config(self.to_dict(), registered_name, "agents")
        self.agent_name = registered_name

    def create_context(
        self,
        context_name: str,
        system_prompt: Optional[str] = None,
    ) -> None:
        """Create a new context and switch to it."""
        prompt = (
            system_prompt if system_prompt is not None else self.default_system_prompt
        )
        self.contexts[context_name] = AgentContext(context_name, prompt)
        self.current_context = context_name

    def switch_context(self, context_name: str) -> None:
        """Switch to an existing context."""
        if context_name not in self.contexts:
            raise ValueError(f"Context '{context_name}' does not exist.")
        self.current_context = context_name

    def copy_context(
        self,
        source_context: str,
        new_context_name: str,
        new_system_prompt: Optional[str] = None,
    ) -> None:
        """
        Create an independent copy of a context.
        Changes to the new context will not affect the source.
        """
        if source_context not in self.contexts:
            raise ValueError(f"Context '{source_context}' does not exist.")

        new_ctx = self.contexts[source_context].copy(new_context_name)
        if new_system_prompt is not None:
            new_ctx.set_system_prompt(new_system_prompt)
        self.contexts[new_context_name] = new_ctx
        self.current_context = new_context_name

    def share_context(self, context_name: str) -> AgentContext:
        """
        Get a reference to a context that can be shared with another agent.
        Both agents will modify the same history.
        """
        if context_name not in self.contexts:
            raise ValueError(f"Context '{context_name}' does not exist.")
        return self.contexts[context_name]

    def use_shared_context(self, context_name: str, context: AgentContext) -> None:
        """Use a context shared from another agent."""
        self.contexts[context_name] = context
        self.current_context = context_name

    def list_contexts(self) -> list[str]:
        """List all available context names."""
        return list(self.contexts.keys())

    def get_current_context_name(self) -> Optional[str]:
        """Get the name of the current context."""
        return self.current_context

    def set_system_prompt(self, system_prompt: str) -> None:
        """Set the system prompt for the current context."""
        if not self.current_context:
            raise ValueError("No context selected.")
        self.contexts[self.current_context].set_system_prompt(system_prompt)

    def clear_context(self, context_name: Optional[str] = None) -> None:
        """Clear message history in a context."""
        ctx = context_name or self.current_context
        if not ctx:
            raise ValueError("No context selected.")
        if ctx not in self.contexts:
            raise ValueError(f"Context '{ctx}' does not exist.")
        self.contexts[ctx].clear()

    def delete_context(self, context_name: str) -> None:
        """Delete a context entirely."""
        if context_name not in self.contexts:
            raise ValueError(f"Context '{context_name}' does not exist.")
        if context_name == self.current_context:
            self.current_context = "default" if "default" in self.contexts else None
        del self.contexts[context_name]

    def attach_metrics(self, collector: MetricsCollector) -> None:
        """Attach a :class:`~pithos.metrics.MetricsCollector` to this agent.

        Once attached, every LLM call, tool execution, and memory operation
        will automatically record metrics into *collector*.

        Args:
            collector: The collector instance to receive metrics.
        """
        self.metrics = collector

    def enable_compaction(self, config: Optional[CompactionConfig] = None) -> None:
        """Enable automatic context compaction.

        When enabled, the oldest messages in the active context are
        summarised and replaced with a compact summary whenever the message
        count reaches ``config.threshold``.

        Args:
            config: Compaction settings.  Defaults to
                :class:`~pithos.agent.compaction.CompactionConfig` with its
                default values when not supplied.
        """
        self.compaction_enabled = True
        self._compactor = MemoryCompactor(config or CompactionConfig())

    def disable_compaction(self) -> None:
        """Disable automatic context compaction."""
        self.compaction_enabled = False
        self._compactor = None

    def enable_recall(self, config: Optional[RecallConfig] = None) -> None:
        """Enable automatic memory recall.

        When enabled, relevant memories are retrieved via RAG before each
        user turn and prepended to the context as a ``[RECALLED CONTEXT]``
        system message.  The injection is not subject to compaction and
        replaces any previous recall injection.

        Memory and/or history must be enabled separately via
        :meth:`enable_memory` / :meth:`enable_history` for the respective
        recall sources to work.  If neither is available the recall pass
        simply produces no snippets.

        Args:
            config: Recall settings.  Defaults to
                :class:`~pithos.agent.recall.RecallConfig` with its default
                values when not supplied.
        """
        self.recall_enabled = True
        self._auto_recall = AutoRecall(config or RecallConfig())

    def disable_recall(self) -> None:
        """Disable automatic memory recall."""
        self.recall_enabled = False
        self._auto_recall = None

    def set_inference_flowchart(
        self,
        config: Any,
        config_manager: Optional["ConfigManager"] = None,
    ) -> None:
        """Set an optional chain-of-thought flowchart for inference.

        When set, each call to :meth:`send` runs the flowchart instead of a
        single LLM round-trip.  The flowchart receives the user message as
        ``initial_input`` and its final output becomes the assistant response.
        PromptNodes inside the flowchart invoke the agent's underlying LLM
        call automatically.

        Args:
            config: One of:

                * A :class:`~pithos.flowchart.Flowchart` instance.
                * A ``str`` naming a registered flowchart configuration.
                * A ``dict`` with ``nodes``, ``edges``, ``start_node`` keys
                  (inline flowchart definition).

            config_manager: Required when *config* is a ``str`` or ``dict``.
                Can be ``None`` when passing a pre-built ``Flowchart``.

        Raises:
            TypeError: If *config* is not a supported type.
            ValueError: If a registered name cannot be resolved.
        """
        from ..flowchart import Flowchart

        if isinstance(config, Flowchart):
            self.inference_flowchart = config
            self._inference_config = None
        elif isinstance(config, str):
            if config_manager is None:
                raise ValueError(
                    "config_manager is required to load a registered flowchart."
                )
            self.inference_flowchart = Flowchart.from_registered(config, config_manager)
            self._inference_config = config
        elif isinstance(config, dict):
            if config_manager is None:
                raise ValueError(
                    "config_manager is required to build an inline flowchart."
                )
            self.inference_flowchart = Flowchart.from_dict(config, config_manager)
            self._inference_config = config
        else:
            raise TypeError(
                f"Unsupported inference flowchart config type: {type(config).__name__}"
            )

    def clear_inference_flowchart(self) -> None:
        """Remove the chain-of-thought inference flowchart."""
        self.inference_flowchart = None
        self._inference_config = None

    def close(self) -> None:
        """Close all open database connections held by this agent.

        Releases file handles for the SQLite and ChromaDB connections used
        by :attr:`history_store` and :attr:`memory_store`.  Should be called
        when the agent is no longer needed, especially before the persistence
        directory is deleted (required on Windows to avoid
        ``PermissionError: [WinError 32]``).

        It is safe to call this method multiple times, or when no stores are
        open.
        """
        if self.history_store is not None:
            try:
                self.history_store.close()
            except Exception:
                pass
        if self.memory_store is not None:
            try:
                self.memory_store.close()
            except Exception:
                pass

    @abstractmethod
    def stream(
        self,
        content: str,
        context_name: Optional[str] = None,
        workspace: Optional[str] = None,
        verbose: bool = False,
        model: Optional[str] = None,
    ) -> Iterator[str]:
        """Stream response tokens one chunk at a time.

        This is the primary method that subclasses must implement.  Yields
        each token/chunk as it is produced by the backend.  The full
        response is committed to context history only after the iterator is
        exhausted, so callers MUST consume it completely for side-effects
        (context update, tool/memory processing, compaction) to take place.

        Tool calls encountered during streaming are executed mid-stream and
        the results injected before the model continues generating.

        Args:
            content: The message to send.
            context_name: Context to use (uses current if None).
            workspace: Optional workspace context to prepend.
            verbose: Print conversation details.
            model: Model to use (uses default_model if None).

        Yields:
            Text chunks of the response.
        """

    def send(
        self,
        content: str,
        context_name: Optional[str] = None,
        workspace: Optional[str] = None,
        verbose: bool = False,
        model: Optional[str] = None,
    ) -> str:
        """Send a message and return the complete response as a string.

        Convenience wrapper around :meth:`stream` that collects all chunks
        into a single string.  Prefer :meth:`stream` when incremental output
        is needed; this method is provided for backward compatibility and
        simple use-cases.

        Args:
            content: The message to send.
            context_name: Context to use (uses current if None).
            workspace: Optional workspace context to prepend.
            verbose: Print conversation details.
            model: Model to use (uses default_model if None).

        Returns:
            The agent's complete response.
        """
        return "".join(self.stream(content, context_name, workspace, verbose, model))

    def _extract_tool_calls(self, content: str) -> list:
        """Extract tool calls from agent response using multiple formats.

        Args:
            content: Agent response text.

        Returns:
            List of ToolCallRequest objects.
        """
        from ..tools import ToolCallExtractor

        if not hasattr(self, "_tool_extractor"):
            self._tool_extractor = ToolCallExtractor()

        return self._tool_extractor.extract(content)

    def _execute_tools(self, requests: list) -> str:
        """Execute tool commands and format results with clear error feedback.

        Args:
            requests: List of ToolCallRequest objects to execute.

        Returns:
            Formatted string with tool execution results and error guidance.
        """
        if not self.tool_executor or not self.tool_registry:
            return "Tool execution is not available."

        results = []
        for req in requests:
            parts = req.command.split(None, 1) if req.command else []
            tool_name = parts[0] if parts else ""

            if tool_name == "flowchart":
                result = self._execute_flowchart_tool(
                    parts[1] if len(parts) > 1 else ""
                )
            else:
                result = self.tool_executor.run(req.command, self.tool_registry)

            # Record tool call metrics
            if self.metrics is not None:
                try:
                    self.metrics.record_tool_call(
                        tool_name=tool_name or "unknown",
                        success=result.success,
                        execution_time_ms=result.execution_time * 1000.0,
                    )
                except Exception:
                    pass
            results.append(self._format_tool_result(result))

        return "\n\n".join(results)

    def _execute_flowchart_tool(self, args_str: str) -> Any:
        """Run a flowchart invoked via the ``flowchart`` virtual tool.

        Expected format: ``<flowchart_name> [input text ...]``

        The calling agent is injected into the flowchart's agent dict under
        every required agent name so single-agent flowcharts work
        out-of-the-box.  For multi-agent flowcharts the caller is used as a
        fallback for any unresolved agent name.
        """
        from ..tools.models import ToolResult

        if not self.flowchart_executor:
            return ToolResult(
                success=False,
                stdout="",
                stderr="Flowchart tools are not enabled.",
                exit_code=-1,
                execution_time=0.0,
                command=f"flowchart {args_str}",
                error_hint="Enable flowcharts in tool_config.yaml under 'flowcharts.enabled: true'.",
            )

        parts = args_str.strip().split(None, 1)
        if not parts:
            available = self.flowchart_executor.list_flowcharts()
            return ToolResult(
                success=False,
                stdout="",
                stderr="No flowchart name provided.",
                exit_code=-1,
                execution_time=0.0,
                command="flowchart",
                error_hint=f"Usage: flowchart <name> [input]\nAvailable: {', '.join(available)}",
            )

        fc_name = parts[0]
        fc_input = parts[1] if len(parts) > 1 else ""

        # Build an agents dict: map every required agent name to *self* so
        # that single-agent flowcharts (prompt nodes) just work.
        from ..flowchart import Flowchart
        from ..flownode import AgentPromptNode, GetHistoryNode, SetHistoryNode

        try:
            fc = Flowchart.from_registered(
                fc_name, self.flowchart_executor.config_manager
            )
            required: set[str] = set()
            for nid in fc.graph.nodes:
                nobj = fc.graph.nodes[nid]["nodeobj"]
                if isinstance(nobj, (AgentPromptNode, GetHistoryNode, SetHistoryNode)):
                    required.add(nobj.agent)
            agents_dict = {name: self for name in required}
        except Exception:
            agents_dict = {}

        return self.flowchart_executor.run(fc_name, fc_input, agents_dict)

    def _format_tool_result(self, result) -> str:
        """Format a tool result with clear error feedback for the agent.

        Args:
            result: ToolResult from execution.

        Returns:
            Formatted string describing the result with actionable feedback.
        """
        lines = [f"Tool execution: {result.command}"]
        lines.append(f"Status: {'✓ Success' if result.success else '✗ Failed'}")
        lines.append(f"Exit code: {result.exit_code}")

        if result.stdout:
            lines.append(f"\nOutput:\n{result.stdout}")

        if result.stderr:
            lines.append(f"\nStderr:\n{result.stderr}")

        # Add error hints if present
        if result.error_hint:
            lines.append(f"\n💡 Hint: {result.error_hint}")

        return "\n".join(lines)

    def enable_tools(
        self,
        config_manager: ConfigManager,
        auto_loop: bool = False,
        max_iterations: int = 5,
    ) -> None:
        """Enable tool calling for this agent.

        Args:
            config_manager: ConfigManager for loading tool configurations.
            auto_loop: Whether to automatically continue conversation after tool execution.
            max_iterations: Maximum number of tool calling iterations to prevent loops.
        """
        self.tools_enabled = True
        self.tool_registry = ToolRegistry(config_manager)

        # Get timeout and max_output_size from config
        tool_config = self.tool_registry.config
        timeout = tool_config.get("timeout", 30)
        max_output_size = tool_config.get("max_output_size", 10000)

        self.tool_executor = ToolExecutor(timeout, max_output_size)
        self.tool_auto_loop = auto_loop
        self.tool_max_iterations = max_iterations

        # Set up flowchart tool executor if flowcharts are enabled
        fc_config = tool_config.get("flowcharts", {})
        if fc_config.get("enabled", False):
            self.flowchart_executor = FlowchartToolExecutor(
                config_manager=config_manager,
                timeout=fc_config.get("timeout", 120),
                max_steps=fc_config.get("max_steps", 100),
            )

        # Enhance system prompt with tool usage instructions
        self._add_tool_prompt_to_contexts()

    def _add_tool_prompt_to_contexts(self) -> None:
        """Add tool usage instructions to all context system prompts."""
        if not self.tool_registry:
            return

        tool_prompt = self._get_tool_usage_prompt()

        for ctx_name, context in self.contexts.items():
            current_prompt = context.get_system_prompt()
            if "You have access to command-line tools" not in current_prompt:
                new_prompt = (
                    current_prompt + "\n\n" + tool_prompt
                    if current_prompt
                    else tool_prompt
                )
                context.set_system_prompt(new_prompt)

    def _get_tool_usage_prompt(self) -> str:
        """Generate tool usage instructions for system prompt.

        Returns:
            Formatted prompt with tool usage instructions and available tools.
        """
        if not self.tool_registry:
            return ""

        from ..tools import ToolCallExtractor

        if not hasattr(self, "_tool_extractor"):
            self._tool_extractor = ToolCallExtractor()

        tool_list = self.tool_registry.get_tool_list_text()
        format_examples = self._tool_extractor.get_usage_examples()

        return f"""You have access to command-line tools via multiple formats for reliability.

{format_examples}

Available tools:
{tool_list}

The tool output will be provided to you automatically, and you can continue reasoning.
Only use tools when necessary. If a tool fails, you will receive clear error feedback."""

    def enable_memory(
        self,
        config_manager: ConfigManager,
        persist_directory: Optional[str] = None,
    ) -> None:
        """Enable memory storage/retrieval for this agent.

        Args:
            config_manager: ConfigManager for loading memory configurations.
            persist_directory: Optional directory for persistent storage.

        Raises:
            RuntimeError: If memory tool is not available (ChromaDB not installed).
        """
        if not MEMORY_AVAILABLE or MemoryStore is None:
            raise RuntimeError(
                "Memory tool is not available. Install with: pip install chromadb"
            )

        self.memory_enabled = True
        self.memory_store = MemoryStore(config_manager, persist_directory)

        # Enhance system prompt with memory usage instructions
        self._add_memory_prompt_to_contexts()

    def enable_tag_suggestions(
        self,
        model: str,
        max_suggestions: int = 3,
        temperature: float = 0.2,
        timeout: int = 30,
    ) -> None:
        """Enable automatic LLM category tag suggestions for memory storage.

        Each time the agent stores a memory entry, the LLM will be asked to
        suggest up to *max_suggestions* category tags for the content.  The
        suggestions are saved in the entry's metadata (``suggested_tags`` and
        ``suggested_tags_confidence``) and reported back to the agent as part
        of the store result message.

        :meth:`enable_memory` must be called before this method.

        Args:
            model: Ollama model name to use for generating suggestions.
            max_suggestions: Maximum tags to suggest per entry (1–10).
            temperature: LLM sampling temperature (lower = more deterministic).
            timeout: HTTP timeout in seconds for the LLM request.

        Raises:
            RuntimeError: If memory has not been enabled on this agent.
        """
        if not self.memory_enabled or self.memory_store is None:
            raise RuntimeError(
                "Memory must be enabled before enabling tag suggestions. "
                "Call enable_memory() first."
            )
        self.memory_store.enable_tag_suggestions(
            model=model,
            max_suggestions=max_suggestions,
            temperature=temperature,
            timeout=timeout,
        )

    def _add_memory_prompt_to_contexts(self) -> None:
        """Add memory usage instructions to all context system prompts."""
        if not self.memory_store:
            return

        memory_prompt = self._get_memory_usage_prompt()

        for ctx_name, context in self.contexts.items():
            current_prompt = context.get_system_prompt()
            if "You have access to a knowledge memory system" not in current_prompt:
                new_prompt = (
                    current_prompt + "\n\n" + memory_prompt
                    if current_prompt
                    else memory_prompt
                )
                context.set_system_prompt(new_prompt)

    def _get_memory_usage_prompt(self) -> str:
        """Generate memory usage instructions for system prompt.

        Returns:
            Formatted prompt with memory usage instructions.
        """
        if not hasattr(self, "_memory_extractor"):
            self._memory_extractor = MemoryOpExtractor()

        categories = []
        if self.memory_store:
            try:
                categories = self.memory_store.list_categories()
            except Exception:
                pass

        categories_text = ", ".join(categories) if categories else "No categories yet"
        format_examples = self._memory_extractor.get_usage_examples()

        return f"""You have access to a knowledge memory system organized by categories.

{format_examples}

Current categories: {categories_text}

Use memory to:
1. Store important facts, insights, or learnings
2. Retrieve relevant context from previous interactions
3. Build up domain knowledge over time

Results will be provided to you automatically. If an operation fails, you will receive clear error feedback."""

    def _extract_memory_ops(self, content: str) -> list[MemoryOpRequest]:
        """Extract memory operations from agent response using multiple formats.

        Args:
            content: Agent response text.

        Returns:
            List of MemoryOpRequest objects.
        """
        if not hasattr(self, "_memory_extractor"):
            self._memory_extractor = MemoryOpExtractor()

        return self._memory_extractor.extract(content)

    def _execute_memory_ops(self, operations: list[MemoryOpRequest]) -> str:
        """Execute memory operations and format results with clear error feedback.

        Args:
            operations: List of MemoryOpRequest objects.

        Returns:
            Formatted string with operation results.
        """
        if not self.memory_store:
            return "Memory system is not available."

        results = []
        for op in operations:
            try:
                if op.operation == "store":
                    if not op.content:
                        results.append(
                            f"✗ Store operation failed: No content provided\n"
                            f"💡 Hint: Use format like STORE[{op.category}]: your content here"
                        )
                        continue

                    entry_id = self.memory_store.store(op.category, op.content)
                    # Record store metric
                    if self.metrics is not None:
                        try:
                            self.metrics.record_memory_store()
                        except Exception:
                            pass
                    store_msg = f"✓ Stored in {op.category}: {op.content[:50]}... (ID: {entry_id})"
                    # Append suggested tags to the result message when available.
                    if self.memory_store.tag_suggestions_enabled:
                        try:
                            entry_meta = self.memory_store.get_all_entries(op.category)
                            tags: list[str] = []
                            for e in entry_meta:
                                if e.get("id") == entry_id:
                                    tags = e.get("metadata", {}).get(
                                        "suggested_tags", []
                                    )
                                    break
                            if tags:
                                store_msg += f"\n  🏷 Suggested tags: {', '.join(tags)}"
                        except Exception:
                            pass
                    results.append(store_msg)

                elif op.operation == "retrieve":
                    if not op.query:
                        results.append(
                            f"✗ Retrieve operation failed: No query provided\n"
                            f"💡 Hint: Use format like RETRIEVE[{op.category}]: your search query"
                        )
                        continue

                    search_results = self.memory_store.retrieve(op.category, op.query)
                    # Record retrieve metric (hit if ≥1 result returned)
                    if self.metrics is not None:
                        try:
                            self.metrics.record_memory_retrieve(
                                result_count=len(search_results)
                            )
                        except Exception:
                            pass
                    if search_results:
                        results.append(
                            f"✓ Retrieved {len(search_results)} results from {op.category} for query: {op.query}"
                        )
                        for i, result in enumerate(search_results[:3], 1):
                            results.append(
                                f"  {i}. [Score: {result.relevance_score:.2f}] {result.content}"
                            )
                        if len(search_results) > 3:
                            results.append(f"  ... and {len(search_results) - 3} more")
                    else:
                        results.append(
                            f"✗ No relevant results found in {op.category} for: {op.query}"
                        )

            except Exception as e:
                results.append(
                    f"✗ Error in {op.operation} operation: {str(e)}\n"
                    f"💡 Hint: Check that the category name is valid and content/query is properly formatted"
                )

        return "\n".join(results)

    # ------------------------------------------------------------------
    # Conversation history
    # ------------------------------------------------------------------

    def enable_history(
        self,
        persist_directory: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> None:
        """Enable persistent conversation history for this agent.

        All subsequent messages sent and received are stored in a SQLite
        database and (when ChromaDB is installed) a vector index, enabling
        later retrieval via :meth:`search_history`.

        Can be called multiple times to switch to a different session; the
        same underlying store is reused when ``persist_directory`` does not
        change.

        Args:
            persist_directory: Directory for the history database.  Defaults
                to ``./data/conversations``.
            session_id: Identifier for the current conversation session.
                A new random UUID is generated when not provided.
        """
        directory = persist_directory or "./data/conversations"
        # Reuse existing store if the directory is unchanged
        if self.history_store is None:
            self.history_store = ConversationStore(directory)
        self.session_id = session_id or str(uuid.uuid4())
        self._last_history_message_id = None

    def tag_current_message(self, tags: list[str]) -> None:
        """Attach metadata tags to the most recent agent response.

        Tags can be used to filter :meth:`search_history` results.  Call this
        method immediately after :meth:`send` or after consuming the
        :meth:`stream` iterator.

        Args:
            tags: List of tag strings (e.g. ``["important", "bug-fix"]``).

        Raises:
            RuntimeError: If history is not enabled or no message has been
                stored yet in this session.

        Example::

            response = agent.send("Fix the authentication error")
            agent.tag_current_message(["important", "bug-fix"])
        """
        if self.history_store is None or self.session_id is None:
            raise RuntimeError("History is not enabled. Call enable_history() first.")
        if self._last_history_message_id is None:
            raise RuntimeError(
                "No message has been stored yet. Send a message before tagging."
            )
        self.history_store.add_tags(self._last_history_message_id, tags)

    def search_history(
        self,
        query: str,
        n_results: int = 10,
        tags: Optional[list[str]] = None,
        role: Optional[str] = None,
        semantic: bool = True,
        all_sessions: bool = False,
    ) -> list[HistorySearchResult]:
        """Search stored conversation history.

        Uses vector (semantic) search when ChromaDB is available and
        ``semantic=True``; otherwise falls back to SQLite full-text search.

        By default only the current session is searched.  Set
        ``all_sessions=True`` to search across every session stored for this
        agent.

        Args:
            query: Search phrase or natural-language question.
            n_results: Maximum number of results to return.
            tags: If given, restrict results to messages tagged with *at
                least one* of the listed tags.
            role: If given, restrict results to ``'user'`` or
                ``'assistant'`` messages only.
            semantic: Prefer semantic (vector) search when available.
            all_sessions: If ``True``, search all sessions for this agent.
                Default is to search only the current session.

        Returns:
            List of :class:`~pithos.agent.history.HistorySearchResult`
            objects ordered by relevance.

        Raises:
            RuntimeError: If history is not enabled.

        Example::

            results = agent.search_history("authentication error")
            for r in results:
                print(r.message.role, r.message.content)
        """
        if self.history_store is None or self.session_id is None:
            raise RuntimeError("History is not enabled. Call enable_history() first.")
        session_filter = None if all_sessions else self.session_id
        return self.history_store.search(
            query=query,
            n_results=n_results,
            agent_name=self.agent_name,
            session_id=session_filter,
            tags=tags,
            role=role,
            semantic=semantic,
        )

    def _history_persist(
        self,
        context_name: str,
        role: str,
        content: str,
        set_as_last: bool = False,
    ) -> Optional[str]:
        """Persist a message to history if history is enabled.

        Failures are silently swallowed so history issues never break the
        normal send/stream flow.

        Args:
            context_name: Name of the active context.
            role: Message role.
            content: Message text.
            set_as_last: When ``True``, store the returned ID as
                ``_last_history_message_id`` for use by
                :meth:`tag_current_message`.

        Returns:
            The message ID, or ``None`` if history is disabled or storage
            failed.
        """
        if self.history_store is None or self.session_id is None:
            return None
        try:
            msg_id = self.history_store.store_message(
                session_id=self.session_id,
                agent_name=self.agent_name,
                context_name=context_name,
                role=role,
                content=content,
            )
            if set_as_last:
                self._last_history_message_id = msg_id
            return msg_id
        except Exception:
            return None
