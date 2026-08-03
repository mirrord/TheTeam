"""pithos Agent - Abstract base class for LLM agents."""

from abc import ABC, abstractmethod
from typing import Optional, Any, Type, TypeVar, Iterator, Callable
import logging
import uuid
import yaml

from ..config_manager import ConfigManager
from ..tools import ToolRegistry, ToolExecutor
from ..tools.provider import ToolProvider
from ..context import AgentContext
from .history import ConversationStore, HistorySearchResult
from .compaction import CompactionConfig
from .recall import RecallConfig
from .memory import MemoryModule
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
        # Optional wall-clock cap (seconds) on a single generation. When set,
        # ``stream()`` stops consuming the model stream once this many seconds
        # have elapsed, committing whatever was produced so far. This bounds
        # worst-case runtime so a runaway/looping model cannot hang forever
        # (Ollama streaming has no timeout of its own). ``None`` = unbounded.
        self.generation_timeout: Optional[float] = None
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
        # Memory module — manages recall, compaction, and context prompt injection.
        # Created lazily by enable_memory() / enable_recall() / enable_compaction().
        self._memory_module: Optional[MemoryModule] = None
        # Memory provider — set by enable_memory(); used for post-stream op extraction.
        self._memory_provider: Optional[Any] = None
        # Conversation history support
        self.history_store: Optional[ConversationStore] = None
        self.session_id: Optional[str] = None
        self._last_history_message_id: Optional[str] = None
        # Metrics collection (optional, attached via attach_metrics())
        self.metrics: Optional[MetricsCollector] = None
        # Chain-of-thought inference flowchart (optional)
        self.inference_flowchart: Optional[Any] = None
        self._inference_config: Optional[Any] = None
        self._running_inference: bool = False
        # Artifact paths collected from the most recent send() call.
        self._pending_image_paths: list[str] = []
        self._pending_report_paths: list[str] = []
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

    def prefix_system_prompt(self, prefix: str) -> None:
        """Prefix the current system prompt with additional text."""
        if not self.current_context:
            raise ValueError("No context selected.")
        ctx = self.contexts[self.current_context]
        new_prompt = prefix + "\n\n" + ctx.system_prompt.content
        ctx.set_system_prompt(new_prompt)

    def suffix_system_prompt(self, suffix: str) -> None:
        """Suffix the current system prompt with additional text."""
        if not self.current_context:
            raise ValueError("No context selected.")
        ctx = self.contexts[self.current_context]
        new_prompt = ctx.system_prompt.content + "\n\n" + suffix
        ctx.set_system_prompt(new_prompt)

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

    # ------------------------------------------------------------------
    # Memory module helpers
    # ------------------------------------------------------------------

    def _get_or_create_memory_module(self) -> MemoryModule:
        """Return the existing :class:`~pithos.agent.memory.MemoryModule` or create one."""
        if self._memory_module is None:
            self._memory_module = MemoryModule()
        return self._memory_module

    @property
    def recall_enabled(self) -> bool:
        """Return ``True`` if automatic memory recall is active."""
        return self._memory_module is not None and self._memory_module.recall_enabled

    @property
    def compaction_enabled(self) -> bool:
        """Return ``True`` if automatic context compaction is active."""
        return (
            self._memory_module is not None and self._memory_module.compaction_enabled
        )

    @property
    def _auto_recall(self) -> Optional[Any]:
        """Return the :class:`~pithos.agent.recall.AutoRecall` instance, or ``None``."""
        return (
            self._memory_module._auto_recall
            if self._memory_module is not None
            else None
        )

    @property
    def _compactor(self) -> Optional[Any]:
        """Return the :class:`~pithos.agent.compaction.MemoryCompactor` instance, or ``None``."""
        return (
            self._memory_module._compactor if self._memory_module is not None else None
        )

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
        self._get_or_create_memory_module().enable_compaction(config)

    def disable_compaction(self) -> None:
        """Disable automatic context compaction."""
        if self._memory_module is not None:
            self._memory_module.disable_compaction()

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
        self._get_or_create_memory_module().enable_recall(config)

    def disable_recall(self) -> None:
        """Disable automatic memory recall."""
        if self._memory_module is not None:
            self._memory_module.disable_recall()

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

    # ------------------------------------------------------------------
    # Backend hooks (subclasses implement)
    # ------------------------------------------------------------------

    @abstractmethod
    def _raw_stream(
        self,
        messages: list[dict[str, Any]],
        model: str,
        options: dict[str, Any],
    ) -> Iterator[tuple[str, Any]]:
        """Yield ``(token, chunk_metadata)`` pairs from the backend.

        Subclasses implement this to perform a single backend streaming
        call with the supplied chat messages, model identifier and
        ``options`` dict (containing ``temperature`` and optionally
        ``num_predict`` / ``max_tokens``).

        The base :meth:`stream` method handles all surrounding
        orchestration (recall, tool detection, history, compaction,
        metrics) — backends should focus solely on producing tokens.

        For each emitted chunk yield a ``(text, raw_chunk)`` tuple.  The
        ``raw_chunk`` of the *last* yielded item is passed to
        :meth:`_extract_token_usage` for metric reporting; intermediate
        ``raw_chunk`` values may be ``None``.
        """

    def _extract_token_usage(self, last_chunk: Any) -> tuple[int, int]:
        """Return ``(prompt_tokens, completion_tokens)`` for the final chunk.

        Default implementation reads ``usage.prompt_tokens`` /
        ``usage.completion_tokens`` from a chunk-like object (Ollama
        format).  Backends with a different shape should override.
        """
        try:
            usage = getattr(last_chunk, "usage", None) if last_chunk else None
            prompt_tok = getattr(usage, "prompt_tokens", 0) or 0
            completion_tok = getattr(usage, "completion_tokens", 0) or 0
            return int(prompt_tok), int(completion_tok)
        except Exception:
            return 0, 0

    def _wrap_backend_error(self, exc: BaseException) -> BaseException:
        """Translate a backend exception into a user-facing error.

        Return the exception to ``raise from exc``.  Default implementation
        wraps in a generic :class:`RuntimeError`.  Backends that have
        domain-specific errors (e.g. Ollama's ``ResponseError``) should
        override and re-raise those untouched while wrapping anything else.
        """
        return RuntimeError(f"Backend communication failed: {exc}")

    # ------------------------------------------------------------------
    # Public streaming + send (backend-agnostic orchestration)
    # ------------------------------------------------------------------

    def stream(
        self,
        content: str,
        context_name: Optional[str] = None,
        workspace: Optional[str] = None,
        verbose: bool = False,
        model: Optional[str] = None,
        status_callback: Optional[Callable[[str, Optional[str]], None]] = None,
    ) -> Iterator[str]:
        """Stream response tokens with mid-stream tool execution.

        Yields chunks as they arrive.  When a complete tool call is detected
        in the accumulated output the stream is interrupted: the partial
        response is committed to context, the tool is executed and its
        result injected as a system message, then a new continuation
        stream is started transparently.

        Memory operations (STORE / RETRIEVE) and auto-compaction are
        performed after the *final* continuation exits.  Callers MUST
        consume the iterator to completion for all side-effects to take
        place.

        Args:
            content: The message to send.
            context_name: Context to use (uses current if None).
            workspace: Optional workspace context to prepend.
            verbose: Print conversation details.
            model: Model to use (uses default_model if None).
            status_callback: Optional callback invoked with progress status
                updates of the form ``(status, detail)``. Known status
                values: ``"thinking"`` (before first token), ``"tool_call"``
                (with tool name in detail, before execution),
                ``"tool_result"`` (with tool name in detail, after
                execution), ``"generating"`` (when resuming generation
                after a tool result).

        Yields:
            Text chunks produced by the model.
        """
        import time as _time

        ctx = context_name or self.current_context
        if not ctx:
            raise ValueError("No context selected.")
        if ctx not in self.contexts:
            self.create_context(ctx)

        context = self.contexts[ctx]

        # Auto-recall: inject relevant memories before the user message.
        if self._memory_module is not None:
            self._memory_module.before_send(
                agent=self, context=context, content=content, model=model
            )

        # Inference flowchart short-circuit (single-chunk yield).
        if self.inference_flowchart and not self._running_inference:
            from ..context import UserMsg as _UserMsg  # local import to avoid cycles

            _ = _UserMsg  # silence unused
            result = self._inference_send(
                content, ctx, context, workspace, verbose, model
            )
            yield result
            return

        from ..context import UserMsg, AgentMsg, Msg

        context.add_message(UserMsg(content))

        try:
            messages = context.get_messages(workspace)
            if verbose:
                logger.debug(">>> STREAM: %s", content)

            options: dict[str, Any] = {"temperature": self.temperature}
            if self.max_tokens != -1:
                options["num_predict"] = self.max_tokens
                options["max_tokens"] = self.max_tokens

            model_to_use = model or self.default_model

            accumulated = ""
            _t0 = _time.monotonic()
            _last_chunk: Any = None
            # Hashes of raw_text for tool calls already executed this turn.
            _seen_raw: set[str] = set()

            # Notify caller that the model is being invoked (pre-first-token).
            if status_callback is not None:
                try:
                    status_callback("thinking", None)
                except Exception:
                    pass
            _first_token_seen = False

            for token, raw_chunk in self._raw_stream(messages, model_to_use, options):
                token = token or ""
                accumulated += token
                if verbose:
                    logger.debug("%s", token)
                if raw_chunk is not None:
                    _last_chunk = raw_chunk
                if not _first_token_seen and token:
                    _first_token_seen = True
                    if status_callback is not None:
                        try:
                            status_callback("generating", None)
                        except Exception:
                            pass
                yield token

                # Wall-clock generation cap: stop consuming a runaway/looping
                # stream so a single call cannot hang indefinitely. Whatever was
                # produced so far falls through and is committed as the response.
                if (
                    self.generation_timeout is not None
                    and (_time.monotonic() - _t0) > self.generation_timeout
                ):
                    logger.warning(
                        "generation_timeout (%.1fs) exceeded for model '%s'; "
                        "stopping stream after %d chars.",
                        self.generation_timeout,
                        model_to_use,
                        len(accumulated),
                    )
                    break

                # Mid-stream tool detection: only execute newly-seen complete calls.
                if self.tools_enabled and self.tool_registry and self.tool_executor:
                    all_calls = self._extract_tool_calls(accumulated)
                    new_calls = [c for c in all_calls if c.raw_text not in _seen_raw]
                    if new_calls:
                        for c in new_calls:
                            _seen_raw.add(c.raw_text)
                        context.add_message(AgentMsg(accumulated))
                        if status_callback is not None:
                            for c in new_calls:
                                try:
                                    status_callback(
                                        "tool_call",
                                        c.command or "",
                                    )
                                except Exception:
                                    pass
                        raw_results = self._run_tool_requests(new_calls)
                        for _r in raw_results:
                            if _r.image_paths:
                                self._pending_image_paths.extend(_r.image_paths)
                            if _r.report_paths:
                                self._pending_report_paths.extend(_r.report_paths)
                        result_msg = (
                            "\n\n".join(
                                self._format_tool_result(r) for r in raw_results
                            )
                            if raw_results
                            else "Tool execution is not available."
                        )
                        context.add_message(Msg("system", result_msg))
                        if status_callback is not None:
                            try:
                                status_callback(
                                    "tool_result",
                                    self._build_tool_display(raw_results),
                                )
                            except Exception:
                                pass
                        if content:
                            self._history_persist(ctx, "user", content)
                        yield from self.stream(
                            "",
                            context_name=ctx,
                            workspace=workspace,
                            verbose=verbose,
                            model=model,
                            status_callback=status_callback,
                        )
                        return

            _response_ms = (_time.monotonic() - _t0) * 1000.0

            if verbose:
                logger.debug("-" * 40)

            # Record token usage from the final chunk.
            if self.metrics is not None:
                try:
                    prompt_tok, completion_tok = self._extract_token_usage(_last_chunk)
                    self.metrics.record_token_usage(
                        model=model_to_use,
                        prompt_tokens=prompt_tok,
                        completion_tokens=completion_tok,
                        response_time_ms=_response_ms,
                    )
                except Exception:
                    pass

        except Exception as exc:
            context.remove_last_message()
            wrapped = self._wrap_backend_error(exc)
            if wrapped is exc:
                raise
            raise wrapped from exc

        # No tool interruption occurred — commit the full response.
        context.add_message(AgentMsg(accumulated))

        # Persist to history (skip empty content = continuation turns).
        if content:
            self._history_persist(ctx, "user", content)
        self._history_persist(ctx, "assistant", accumulated, set_as_last=True)

        # Memory operations post-stream.
        if self.memory_enabled and self._memory_provider is not None:
            mem_result = self._memory_provider.extract_and_execute(accumulated, self)
            if mem_result:
                context.add_message(Msg("system", mem_result))

        # Auto-compaction.
        if self._memory_module is not None:
            self._memory_module.after_send(
                agent=self, context=context, response=accumulated, context_name=ctx
            )

    # ------------------------------------------------------------------
    # Inference flowchart path (non-streaming; yields a single chunk)
    # ------------------------------------------------------------------

    def _inference_send(
        self,
        content: str,
        ctx: str,
        context: AgentContext,
        workspace: Optional[str],
        verbose: bool,
        model: Optional[str],
    ) -> str:
        """Run the chain-of-thought inference flowchart for a user message.

        Executes the inference flowchart with *content* as initial input.
        PromptNodes inside the flowchart invoke the agent's underlying LLM
        call (the ``_running_inference`` guard prevents infinite recursion).

        The user message and final response are recorded in the main
        conversation context (*context*); intermediate flowchart reasoning
        happens in a temporary context that is discarded afterwards.
        """
        from ..context import UserMsg, AgentMsg, Msg

        context.add_message(UserMsg(content))

        tmp_ctx_name = f"_cot_{uuid.uuid4().hex[:8]}"
        self.create_context(tmp_ctx_name, self.default_system_prompt)
        saved_current = ctx

        self._running_inference = True
        try:
            fc = self.inference_flowchart
            assert fc is not None
            fc.reset()
            fc._initialize_message_routing()

            fc.message_router.shared_context["agent"] = self
            fc.message_router.shared_context["context_name"] = tmp_ctx_name
            fc.message_router.shared_context["model"] = model or self.default_model
            fc.message_router.shared_context["verbose"] = verbose

            result = fc.run_message_based(initial_data=content)
            response = ""
            if result.get("messages"):
                response = str(result["messages"][-1].data)
        except Exception as exc:
            logger.error("Inference flowchart failed: %s", exc)
            context.remove_last_message()
            raise RuntimeError(f"Inference flowchart execution failed: {exc}") from exc
        finally:
            self._running_inference = False
            if tmp_ctx_name in self.contexts:
                del self.contexts[tmp_ctx_name]
            self.current_context = saved_current

        context.add_message(AgentMsg(response))

        if content:
            self._history_persist(ctx, "user", content)
        self._history_persist(ctx, "assistant", response, set_as_last=True)

        # Tool calls post-processing on the final response.
        if self.tools_enabled and self.tool_registry and self.tool_executor:
            tool_requests = self._extract_tool_calls(response)
            if tool_requests:
                result_msg = self._execute_tools(tool_requests)
                context.add_message(Msg("system", result_msg))
                if self.tool_auto_loop:
                    return self.send(
                        "", context_name=ctx, workspace=workspace, verbose=verbose
                    )

        # Memory operations post-processing.
        if self.memory_enabled and self._memory_provider is not None:
            mem_result = self._memory_provider.extract_and_execute(response, self)
            if mem_result:
                context.add_message(Msg("system", mem_result))
                if self.tool_auto_loop:
                    return self.send(
                        "", context_name=ctx, workspace=workspace, verbose=verbose
                    )

        # Auto-compaction.
        if self._memory_module is not None:
            self._memory_module.after_send(
                agent=self, context=context, response=response, context_name=ctx
            )

        return response

    @property
    def last_image_paths(self) -> list[str]:
        """Return image paths collected from the most recent :meth:`send` call."""
        return list(self._pending_image_paths)

    @property
    def last_report_paths(self) -> list[str]:
        """Return report file paths collected from the most recent :meth:`send` call."""
        return list(self._pending_report_paths)

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
        self._pending_image_paths = []
        self._pending_report_paths = []
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

    def _run_tool_requests(self, requests: list) -> list:
        """Execute tool requests, record metrics, and return raw ToolResult list.

        Args:
            requests: List of ToolCallRequest objects to execute.

        Returns:
            List of ToolResult objects (empty list if tools are not available).
        """
        if not self.tool_executor or not self.tool_registry:
            return []

        results = []
        for req in requests:
            parts = req.command.split(None, 1) if req.command else []
            tool_name = parts[0] if parts else ""

            result = self.tool_executor.run(
                req.command,
                self.tool_registry,
                context={"agent": self},
            )

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
            results.append(result)

        return results

    def _execute_tools(self, requests: list) -> str:
        """Execute tool commands and format results with clear error feedback.

        Args:
            requests: List of ToolCallRequest objects to execute.

        Returns:
            Formatted string with tool execution results and error guidance.
        """
        if not self.tool_executor or not self.tool_registry:
            return "Tool execution is not available."

        raw_results = self._run_tool_requests(requests)
        return "\n\n".join(self._format_tool_result(r) for r in raw_results)

    def _build_tool_display(self, results: list) -> str:
        """Build a user-facing display string from tool results.

        Returns only stdout/stderr — no agent-facing metadata.  Used by the
        shell interface to show the command output in a formatted block.

        Args:
            results: List of ToolResult objects.

        Returns:
            Plain-text representation of the combined output.
        """
        parts = []
        for result in results:
            lines = []
            if result.stdout:
                lines.append(result.stdout.rstrip())
            if result.stderr:
                lines.append(result.stderr.rstrip())
            if not lines:
                if result.success:
                    lines.append("(no output)")
                else:
                    lines.append(f"(failed, exit code {result.exit_code})")
            parts.append("\n".join(lines))
        return "\n\n".join(parts)

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

        Builds a list of ToolProviders from the tool configuration, creates a
        ToolRegistry populated with those providers, then creates a ToolExecutor
        that routes all tool calls through the registry.

        Args:
            config_manager: ConfigManager for loading tool configurations.
            auto_loop: Whether to automatically continue conversation after tool execution.
            max_iterations: Maximum number of tool calling iterations to prevent loops.
        """
        from ..tools.cli_provider import CLIToolProvider
        from ..tools.flowchart_tool import FlowchartToolExecutor
        from ..tools.safety import CommandSafetyChecker

        # Load config via a temporary registry (reads tool_config.yaml).
        _tmp = ToolRegistry(config_manager, providers=[])
        tool_config = _tmp.config

        timeout = tool_config.get("timeout", 30)
        max_output_size = tool_config.get("max_output_size", 10000)

        # Always register a CLI provider.
        safety = CommandSafetyChecker(tool_config.get("safety", {}))
        cli = CLIToolProvider(
            config=tool_config,
            timeout=timeout,
            max_output_size=max_output_size,
            safety_checker=safety,
        )
        providers: list[ToolProvider] = [cli]

        # Optionally add a flowchart provider.
        fc_config = tool_config.get("flowcharts", {})
        if fc_config.get("enabled", False):
            providers.append(
                FlowchartToolExecutor(
                    config_manager=config_manager,
                    max_steps=fc_config.get("max_steps", 100),
                )
            )

        # Optionally add a web-research provider.
        wr_config = tool_config.get("web_research", {})
        if wr_config.get("enabled", False):
            try:
                from ..tools.web_researcher import (
                    WEB_RESEARCH_AVAILABLE,
                    WebResearcherToolExecutor,
                )

                if WEB_RESEARCH_AVAILABLE:
                    providers.append(
                        WebResearcherToolExecutor(config_manager=config_manager)
                    )
            except Exception:
                pass  # optional feature — silently skip if deps are missing

        # Optionally add a news-research provider.
        news_config = tool_config.get("news_research", {})
        if news_config.get("enabled", False):
            try:
                from ..tools.news_researcher import (
                    NEWS_RESEARCH_AVAILABLE,
                    NewsResearcherToolExecutor,
                )

                if NEWS_RESEARCH_AVAILABLE:
                    providers.append(
                        NewsResearcherToolExecutor(config_manager=config_manager)
                    )
            except Exception:
                pass  # optional feature — silently skip if deps are missing

        # Optionally add a craft-analysis provider.
        craft_config = tool_config.get("craft_analysis", {})
        if craft_config.get("enabled", False):
            try:
                from ..tools.craft_analyzer import (
                    CRAFT_ANALYSIS_AVAILABLE,
                    CraftAnalyzerToolExecutor,
                )

                if CRAFT_ANALYSIS_AVAILABLE:
                    providers.append(
                        CraftAnalyzerToolExecutor(config_manager=config_manager)
                    )
            except Exception:
                pass  # optional feature — silently skip if deps are missing

        # Optionally add a craft-writing provider.
        craft_write_config = tool_config.get("craft_writing", {})
        if craft_write_config.get("enabled", False):
            try:
                from ..tools.craft_writer import (
                    CRAFT_WRITING_AVAILABLE,
                    CraftWriterToolExecutor,
                )

                if CRAFT_WRITING_AVAILABLE:
                    providers.append(
                        CraftWriterToolExecutor(config_manager=config_manager)
                    )
            except Exception:
                pass  # optional feature — silently skip if deps are missing

        # Optionally add a prompt2image provider.
        t2i_config = tool_config.get("prompt2image", {})
        if t2i_config.get("enabled", False):
            try:
                from ..tools.prompt2image import (
                    TEXT2IMAGE_AVAILABLE,
                    Prompt2ImageToolProvider,
                )

                if TEXT2IMAGE_AVAILABLE:
                    providers.append(
                        Prompt2ImageToolProvider(config_manager=config_manager)
                    )
            except Exception:
                pass  # optional feature — silently skip if deps are missing

        self.tool_registry = ToolRegistry(config_manager, providers=providers)
        self.tool_executor = ToolExecutor()
        self.tools_enabled = True
        self.tool_auto_loop = auto_loop
        self.tool_max_iterations = max_iterations

        # Enhance system prompt with tool usage instructions
        self._add_tool_prompt_to_contexts()

    def _add_tool_prompt_to_contexts(self) -> None:
        """Add tool usage instructions to all context system prompts."""
        if not self.tool_registry:
            return

        tool_prompt = self._get_tool_usage_prompt()
        # print("Tool usage prompt:\n", tool_prompt)  # Debug output

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

        return f"""You have access to command-line tools invoked with the bracket format.

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

        from ..tools.memory_provider import MemoryToolProvider

        self.memory_enabled = True
        self.memory_store = MemoryStore(config_manager, persist_directory)

        # Create the provider and register it with the tool registry (if tools enabled).
        provider_config = (
            self.tool_registry.config if self.tool_registry is not None else {}
        )
        self._memory_provider = MemoryToolProvider(provider_config)
        if self.tool_registry is not None:
            self.tool_registry.register_provider(self._memory_provider)

        # Create module if not already present and inject memory usage instructions.
        module = self._get_or_create_memory_module()
        module.inject_memory_prompt(self)

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
