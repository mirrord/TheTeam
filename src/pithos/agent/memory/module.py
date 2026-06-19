"""MemoryModule — context management callbacks for agent memory operations.

:class:`MemoryModule` owns the recall and compaction lifecycle, providing
:meth:`before_send` and :meth:`after_send` hooks that the agent calls around
each model invocation.  Subclass and override those methods to customise
memory behaviour without touching the agent itself.

The module also provides :meth:`inject_memory_prompt`, which appends memory
usage instructions to all of an agent's context system prompts when memory
is first enabled.
"""

import logging
from typing import TYPE_CHECKING, Optional

from ..compaction import CompactionConfig, MemoryCompactor
from ..recall import RecallConfig, AutoRecall

if TYPE_CHECKING:  # pragma: no cover
    from ..agent import Agent
    from ...context import AgentContext

logger = logging.getLogger(__name__)


class MemoryModule:
    """Manages automatic memory operations around each model invocation.

    The module acts as an extensible lifecycle manager for recall injection
    and context compaction.  Override :meth:`before_send` or
    :meth:`after_send` in a subclass to add or replace the default behaviour.

    Args:
        recall_config: Initial recall configuration.  ``None`` means recall
            is disabled until :meth:`enable_recall` is called.
        compaction_config: Initial compaction configuration.  ``None`` means
            compaction is disabled until :meth:`enable_compaction` is called.
    """

    def __init__(
        self,
        recall_config: Optional[RecallConfig] = None,
        compaction_config: Optional[CompactionConfig] = None,
    ) -> None:
        self._auto_recall: Optional[AutoRecall] = None
        self._compactor: Optional[MemoryCompactor] = None

        if recall_config is not None:
            self.enable_recall(recall_config)
        if compaction_config is not None:
            self.enable_compaction(compaction_config)

    # ------------------------------------------------------------------
    # State accessors
    # ------------------------------------------------------------------

    @property
    def recall_enabled(self) -> bool:
        """Return ``True`` if automatic recall is active."""
        return self._auto_recall is not None

    @property
    def compaction_enabled(self) -> bool:
        """Return ``True`` if automatic compaction is active."""
        return self._compactor is not None

    # ------------------------------------------------------------------
    # Enable / disable
    # ------------------------------------------------------------------

    def enable_recall(self, config: Optional[RecallConfig] = None) -> None:
        """Enable automatic memory recall with *config*.

        Args:
            config: Recall settings.  Defaults to :class:`~pithos.agent.recall.RecallConfig`
                with its default values when not supplied.
        """
        self._auto_recall = AutoRecall(config or RecallConfig())

    def disable_recall(self) -> None:
        """Disable automatic memory recall."""
        self._auto_recall = None

    def enable_compaction(self, config: Optional[CompactionConfig] = None) -> None:
        """Enable automatic context compaction with *config*.

        Args:
            config: Compaction settings.  Defaults to
                :class:`~pithos.agent.compaction.CompactionConfig` with its
                default values when not supplied.
        """
        self._compactor = MemoryCompactor(config or CompactionConfig())

    def disable_compaction(self) -> None:
        """Disable automatic context compaction."""
        self._compactor = None

    # ------------------------------------------------------------------
    # System prompt injection
    # ------------------------------------------------------------------

    def inject_memory_prompt(self, agent: "Agent") -> None:
        """Append memory usage instructions to all context system prompts.

        Reads available categories from ``agent.memory_store`` (if present)
        and formats usage examples via
        :class:`~pithos.tools.memory_ops.MemoryOpExtractor`.  The
        instructions are injected once per context — if the sentinel phrase
        is already present in a context's system prompt it is not added again.

        Args:
            agent: The agent whose contexts will be updated.
        """
        if not agent.memory_store:
            return

        from ...tools.memory_ops import MemoryOpExtractor

        extractor = MemoryOpExtractor()
        categories: list[str] = []
        try:
            categories = agent.memory_store.list_categories()
        except Exception:
            pass

        categories_text = ", ".join(categories) if categories else "No categories yet"
        format_examples = extractor.get_usage_examples()

        memory_prompt = (
            "You have access to a knowledge memory system organized by categories.\n\n"
            f"{format_examples}\n\n"
            f"Current categories: {categories_text}\n\n"
            "Use memory to:\n"
            "1. Store important facts, insights, or learnings\n"
            "2. Retrieve relevant context from previous interactions\n"
            "3. Build up domain knowledge over time\n\n"
            "Results will be provided to you automatically. "
            "If an operation fails, you will receive clear error feedback."
        )

        for _ctx_name, context in agent.contexts.items():
            current_prompt = context.get_system_prompt()
            if "You have access to a knowledge memory system" not in current_prompt:
                new_prompt = (
                    current_prompt + "\n\n" + memory_prompt
                    if current_prompt
                    else memory_prompt
                )
                context.set_system_prompt(new_prompt)

    # ------------------------------------------------------------------
    # Lifecycle callbacks
    # ------------------------------------------------------------------

    def before_send(
        self,
        agent: "Agent",
        context: "AgentContext",
        content: str,
        model: Optional[str],
    ) -> None:
        """Called before each user message is forwarded to the model.

        Default behaviour: inject recalled memories into *context* when
        recall is enabled.  Override in a subclass to add or replace this
        behaviour.

        Args:
            agent: The calling agent instance.
            context: The active :class:`~pithos.context.AgentContext`.
            content: The user message about to be sent.
            model: The model identifier that will be used (may be ``None``).
        """
        if self._auto_recall is not None:
            try:
                self._auto_recall.inject_recall(
                    agent=agent, context=context, content=content, model=model
                )
            except Exception as exc:
                logger.warning("Auto-recall failed (non-fatal): %s", exc)

    def after_send(
        self,
        agent: "Agent",
        context: "AgentContext",
        response: str,
        context_name: str,
    ) -> None:
        """Called after the model response has been committed to *context*.

        Default behaviour: compact the context when compaction is enabled.
        Override in a subclass to add or replace this behaviour.

        Args:
            agent: The calling agent instance.
            context: The active :class:`~pithos.context.AgentContext`.
            response: The full assistant response text (available for
                subclass inspection, e.g. custom post-processing).
            context_name: Name of the active context, forwarded to
                :meth:`~pithos.agent.compaction.MemoryCompactor.compact`.
        """
        if self._compactor is not None:
            try:
                self._compactor.compact(
                    agent=agent, context=context, context_name=context_name
                )
            except Exception as exc:
                logger.warning("Auto-compaction failed (non-fatal): %s", exc)
