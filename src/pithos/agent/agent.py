"""pithos Agent - LLM agent with context management."""

from abc import ABC, abstractmethod
from typing import Optional, Any, Type, TypeVar, Iterator
from pathlib import Path
import argparse
import logging
import uuid
import yaml

from ollama import chat
from ollama import ChatResponse
from ollama._types import ResponseError as OllamaResponseError
import ollama

from ..config_manager import ConfigManager
from ..tools import ToolRegistry, ToolExecutor, MemoryOpRequest, MemoryOpExtractor
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
        max_tokens: Optional[int] = None,
    ):
        self.default_model = default_model
        self.agent_name = agent_name or default_model
        self.default_system_prompt = system_prompt
        self.temperature = temperature if temperature is not None else 0.7
        self.max_tokens = max_tokens if max_tokens is not None else -1
        self.contexts: dict[str, AgentContext] = {}
        self.current_context: Optional[str] = None
        # Tool calling support
        self.tools_enabled = False
        self.tool_registry: Optional[ToolRegistry] = None
        self.tool_executor: Optional[ToolExecutor] = None
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
        # Create default context
        self.create_context("default", system_prompt)

    @classmethod
    def from_dict(
        cls: Type[_AgentT], config: dict[str, Any], config_manager: ConfigManager
    ) -> _AgentT:
        """Create agent from configuration dictionary."""
        # Support both 'default_model' and legacy 'model' key
        model = config.get("default_model") or config.get("model")
        if not model:
            raise ValueError("Agent config must specify 'default_model' or 'model'")
        agent = cls(
            model,
            config.get("name"),
            config.get("system_prompt", ""),
            config.get("temperature"),
            config.get("max_tokens"),
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
                memory_category=compaction_cfg.get("memory_category", "context_summaries"),
                summary_max_tokens=compaction_cfg.get("summary_max_tokens", 512),
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
            "default_model": self.default_model,
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
    def send(
        self,
        content: str,
        context_name: Optional[str] = None,
        workspace: Optional[str] = None,
        verbose: bool = False,
        model: Optional[str] = None,
    ) -> str:
        """
        Send a message and get a response.

        Args:
            content: The message to send
            context_name: Context to use (uses current if None)
            workspace: Optional workspace context to prepend
            verbose: Print conversation details
            model: Model to use (uses default_model if None)

        Returns:
            The agent's response
        """

    def stream(
        self,
        content: str,
        context_name: Optional[str] = None,
        workspace: Optional[str] = None,
        verbose: bool = False,
        model: Optional[str] = None,
    ) -> Iterator[str]:
        """
        Stream response tokens one chunk at a time.

        Yields each token/chunk as it is produced by the backend.  The full
        response is committed to the context history only after the iterator
        is exhausted, so callers MUST consume the iterator to completion for
        side-effects (context update, tool/memory processing) to take place.

        The default implementation falls back to ``send()`` and yields the
        whole response as a single chunk, so subclasses that don't override
        this still work correctly.

        Args:
            content: The message to send.
            context_name: Context to use (uses current if None).
            workspace: Optional workspace context to prepend.
            verbose: Print conversation details.
            model: Model to use (uses default_model if None).

        Yields:
            Text chunks of the response.
        """
        yield self.send(content, context_name, workspace, verbose, model)

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
            result = self.tool_executor.run(req.command, self.tool_registry)
            # Record tool call metrics
            if self.metrics is not None:
                try:
                    tool_name = req.command.split()[0] if req.command else "unknown"
                    self.metrics.record_tool_call(
                        tool_name=tool_name,
                        success=result.success,
                        execution_time_ms=result.execution_time * 1000.0,
                    )
                except Exception:
                    pass
            results.append(self._format_tool_result(result))

        return "\n\n".join(results)

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
                    store_msg = (
                        f"✓ Stored in {op.category}: {op.content[:50]}... (ID: {entry_id})"
                    )
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


class OllamaAgent(Agent):
    """LLM agent backed by Ollama."""

    def send(
        self,
        content: str,
        context_name: Optional[str] = None,
        workspace: Optional[str] = None,
        verbose: bool = False,
        model: Optional[str] = None,
    ) -> str:
        """Send a message via Ollama and get a response."""
        import time as _time

        ctx = context_name or self.current_context
        if not ctx:
            raise ValueError("No context selected.")
        if ctx not in self.contexts:
            self.create_context(ctx)

        context = self.contexts[ctx]

        # Auto-recall: inject relevant memories before the user message is appended
        if self.recall_enabled and self._auto_recall:
            try:
                self._auto_recall.inject_recall(agent=self, context=context, content=content, model=model)
            except Exception as exc:
                logger.warning("Auto-recall failed (non-fatal): %s", exc)

        context.add_message(UserMsg(content))

        try:
            messages = context.get_messages(workspace)
            if verbose:
                logger.debug(">>> SEND: %s", content)

            # Build options dict
            options = {"temperature": self.temperature}
            # Only include num_predict if max_tokens is not -1 (unlimited)
            if self.max_tokens != -1:
                options["num_predict"] = self.max_tokens

            model_to_use = model or self.default_model

            _t0 = _time.monotonic()
            response: ChatResponse = chat(
                model=model_to_use,
                messages=messages,
                options=options,
            )
            _response_ms = (_time.monotonic() - _t0) * 1000.0

            if verbose:
                logger.debug("<<< RECV: %s", response.message.content)
                logger.debug("-" * 40)

            # Record token usage and response time metrics
            if self.metrics is not None:
                try:
                    usage = response.usage
                    prompt_tok = getattr(usage, "prompt_tokens", 0) or 0
                    completion_tok = getattr(usage, "completion_tokens", 0) or 0
                    self.metrics.record_token_usage(
                        model=model_to_use,
                        prompt_tokens=prompt_tok,
                        completion_tokens=completion_tok,
                        response_time_ms=_response_ms,
                    )
                except Exception:
                    pass

        except OllamaResponseError as e:
            context.remove_last_message()
            if "not found" in str(e):
                model_to_use = model or self.default_model
                print(f"Model '{model_to_use}' not found. Download it? (y/n)")
                user_input = input("y/[n]: ")
                if user_input.lower() == "y":
                    print("Downloading model...")
                    ollama.pull(model_to_use)
                    print("Model downloaded.")
                    return self.send(content, context_name, workspace, verbose, model)
                else:
                    raise RuntimeError("Model not available.") from e
            raise e
        except Exception as exc:
            context.remove_last_message()
            raise RuntimeError(
                f"Failed to communicate with Ollama: {exc}. "
                "Ensure Ollama is running. "
                "If using localhost, try setting "
                "OLLAMA_HOST=http://127.0.0.1:11434 to avoid IPv6 resolution issues."
            ) from exc

        context.add_message(AgentMsg(response.message.content or ""))

        # Persist to conversation history if enabled (non-empty user turns only)
        if content:
            self._history_persist(ctx, "user", content)
        self._history_persist(
            ctx, "assistant", response.message.content or "", set_as_last=True
        )

        # Check for tool calls if tools are enabled
        if self.tools_enabled and self.tool_registry and self.tool_executor:
            tool_requests = self._extract_tool_calls(response.message.content or "")
            if tool_requests:
                result_message = self._execute_tools(tool_requests)
                context.add_message(Msg("system", result_message))

                if self.tool_auto_loop:
                    return self.send(
                        "", context_name=ctx, workspace=workspace, verbose=verbose
                    )

        # Check for memory operations if memory is enabled
        if self.memory_enabled and self.memory_store:
            mem_ops = self._extract_memory_ops(response.message.content or "")
            if mem_ops:
                result_message = self._execute_memory_ops(mem_ops)
                context.add_message(Msg("system", result_message))

                if self.tool_auto_loop:
                    return self.send(
                        "", context_name=ctx, workspace=workspace, verbose=verbose
                    )

        # Auto-compaction: summarise old messages when the context is too large
        if self.compaction_enabled and self._compactor:
            try:
                self._compactor.compact(agent=self, context=context, context_name=ctx)
            except Exception as exc:
                logger.warning("Auto-compaction failed (non-fatal): %s", exc)

        return response.message.content or ""

    def stream(
        self,
        content: str,
        context_name: Optional[str] = None,
        workspace: Optional[str] = None,
        verbose: bool = False,
        model: Optional[str] = None,
    ) -> Iterator[str]:
        """Stream response tokens from Ollama one chunk at a time.

        The full assembled response is committed to context history and
        tool/memory post-processing runs only after the iterator is exhausted.
        Callers MUST consume the iterator completely.

        Yields:
            Text chunks as produced by the model.
        """
        import time as _time

        ctx = context_name or self.current_context
        if not ctx:
            raise ValueError("No context selected.")
        if ctx not in self.contexts:
            self.create_context(ctx)

        context = self.contexts[ctx]

        # Auto-recall: inject relevant memories before the user message is appended
        if self.recall_enabled and self._auto_recall:
            try:
                self._auto_recall.inject_recall(agent=self, context=context, content=content, model=model)
            except Exception as exc:
                logger.warning("Auto-recall failed (non-fatal): %s", exc)

        context.add_message(UserMsg(content))

        try:
            messages = context.get_messages(workspace)
            if verbose:
                logger.debug(">>> STREAM: %s", content)

            options: dict[str, Any] = {"temperature": self.temperature}
            if self.max_tokens != -1:
                options["num_predict"] = self.max_tokens

            model_to_use = model or self.default_model

            stream_iter = chat(
                model=model_to_use,
                messages=messages,
                options=options,
                stream=True,
            )

            full_response = ""
            _t0 = _time.monotonic()
            _last_chunk = None
            for chunk in stream_iter:
                token = chunk.message.content or ""
                full_response += token
                if verbose:
                    logger.debug("%s", token)
                _last_chunk = chunk
                yield token
            _response_ms = (_time.monotonic() - _t0) * 1000.0

            if verbose:
                logger.debug("-" * 40)

            # Record token usage and response time metrics (final chunk has usage)
            if self.metrics is not None:
                try:
                    usage = getattr(_last_chunk, "usage", None) if _last_chunk else None
                    prompt_tok = getattr(usage, "prompt_tokens", 0) or 0
                    completion_tok = getattr(usage, "completion_tokens", 0) or 0
                    self.metrics.record_token_usage(
                        model=model_to_use,
                        prompt_tokens=prompt_tok,
                        completion_tokens=completion_tok,
                        response_time_ms=_response_ms,
                    )
                except Exception:
                    pass

        except OllamaResponseError as e:
            context.remove_last_message()
            raise e
        except Exception as exc:
            context.remove_last_message()
            raise RuntimeError(
                f"Failed to communicate with Ollama: {exc}. "
                "Ensure Ollama is running. "
                "If using localhost, try setting "
                "OLLAMA_HOST=http://127.0.0.1:11434 to avoid IPv6 resolution issues."
            ) from exc

        # Commit full response to context after streaming is done
        context.add_message(AgentMsg(full_response))

        # Persist to conversation history if enabled (non-empty user turns only)
        if content:
            self._history_persist(ctx, "user", content)
        self._history_persist(ctx, "assistant", full_response, set_as_last=True)

        # Tool calls post-processing
        if self.tools_enabled and self.tool_registry and self.tool_executor:
            tool_requests = self._extract_tool_calls(full_response)
            if tool_requests:
                result_message = self._execute_tools(tool_requests)
                context.add_message(Msg("system", result_message))

        # Memory operations post-processing
        if self.memory_enabled and self.memory_store:
            mem_ops = self._extract_memory_ops(full_response)
            if mem_ops:
                result_message = self._execute_memory_ops(mem_ops)
                context.add_message(Msg("system", result_message))

        # Auto-compaction: summarise old messages when the context is too large
        if self.compaction_enabled and self._compactor:
            try:
                self._compactor.compact(agent=self, context=context, context_name=ctx)
            except Exception as exc:
                logger.warning("Auto-compaction failed (non-fatal): %s", exc)


class EXLAgent(Agent):
    """LLM agent backed by ExLlamaV2. (Stub — not yet implemented.)"""

    def send(
        self,
        content: str,
        context_name: Optional[str] = None,
        workspace: Optional[str] = None,
        verbose: bool = False,
        model: Optional[str] = None,
    ) -> str:
        raise NotImplementedError(
            "EXLAgent.send() is not yet implemented. "
            "ExLlamaV2 backend support is planned for a future release."
        )


class LlamacppAgent(Agent):
    """LLM agent backed by llama.cpp. (Stub — not yet implemented.)"""

    def send(
        self,
        content: str,
        context_name: Optional[str] = None,
        workspace: Optional[str] = None,
        verbose: bool = False,
        model: Optional[str] = None,
    ) -> str:
        raise NotImplementedError(
            "LlamacppAgent.send() is not yet implemented. "
            "llama.cpp backend support is planned for a future release."
        )


def interactive_chat(agent: Agent, verbose: bool = False) -> None:
    """Interactive chat interface for an agent."""
    print("Starting interactive chat. Press Ctrl+C to end the chat.")
    try:
        while True:
            user_input = input("You: ")
            if not user_input.strip():
                continue
            response = agent.send(user_input, verbose=verbose)
            print(f"Agent: {response}")
    except KeyboardInterrupt:
        print("\nEnding chat.")


def main() -> None:
    """CLI entrypoint for agent management."""
    parser = argparse.ArgumentParser(description="pithos Agent CLI")
    subparsers = parser.add_subparsers(dest="command")

    # Chat command
    chat_parser = subparsers.add_parser("chat", help="Chat with an agent")
    chat_parser.add_argument(
        "agent_config",
        type=str,
        help="Path to agent config file, registered agent name, or model name",
    )
    chat_parser.add_argument(
        "--verbose", action="store_true", help="Enable verbose output"
    )

    # Register command
    reg_parser = subparsers.add_parser("register", help="Register agent config")
    reg_parser.add_argument(
        "agent_config", type=str, help="Path to the agent config file"
    )
    reg_parser.add_argument("--name", type=str, help="Name to register the agent as")

    args = parser.parse_args()
    config_manager = ConfigManager()

    if args.command == "chat":
        agent = None
        agent_path = Path(args.agent_config)
        if agent_path.exists():
            agent = OllamaAgent.from_yaml(str(agent_path), config_manager)
            print(f"Using agent config: {args.agent_config}")
        elif args.agent_config in config_manager.get_registered_agent_names():
            agent = OllamaAgent.from_config(args.agent_config, config_manager)
            print(f"Using registered agent: {args.agent_config}")
        else:
            agent = OllamaAgent(default_model=args.agent_config)
            print(f"Using base model: {args.agent_config}")

        interactive_chat(agent, args.verbose)

    elif args.command == "register":
        agent = OllamaAgent.from_yaml(args.agent_config, config_manager)
        agent.register(config_manager, args.name)
        print(f"Agent registered as '{agent.agent_name}'")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
