"""Automatic memory recall for agent conversation histories.

Before each LLM call, :class:`AutoRecall` performs an ephemeral query to
discover relevant search criteria, retrieves matching entries from the
configured sources (vector memory store and/or conversation history), and
prepends the results as a single ``[RECALLED CONTEXT]`` system message in
the agent's active context.

Key properties
--------------
* The injected recall message is tagged ``_pithos_no_compact: True`` so the
  compaction pass never includes it in its candidates.
* Any *previous* auto-recall injection (tagged ``_pithos_auto_recall: True``)
  is removed before inserting the new one, ensuring at most one recall
  message is present at any time.
* The ephemeral "what should I search for?" LLM call is kept out of context
  history; it never appears in the persistent conversation record.
"""

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

from ..context import AgentContext

if TYPE_CHECKING:  # pragma: no cover
    from .agent import Agent

logger = logging.getLogger(__name__)

_QUERY_PROMPT = """\
You are helping retrieve relevant background knowledge for a conversation. \
Based on the conversation history and the new message below, generate 1-3 \
concise search queries that would retrieve the most relevant prior context \
or background knowledge.

Conversation so far:
{history}

New message: {new_message}

Respond with only the search queries, one per line. No numbering. No explanation."""

_RECALL_HEADER = (
    "[RECALLED CONTEXT]\n"
    "The following memories were automatically retrieved as relevant context:"
)


@dataclass
class RecallConfig:
    """Configuration for automatic memory recall.

    Attributes:
        sources: Which data sources to search.  Valid values are ``"memory"``
            (the vector memory store) and ``"history"`` (the persistent
            conversation history store).
        n_results: Maximum total number of snippets to inject.
        recall_model: Ollama model to use for generating search queries.
            Falls back to the agent's ``default_model`` when ``None``.
        categories: Vector memory categories to search.  An empty list means
            all available categories are searched.
        min_relevance: Minimum relevance score (0–1) for memory results to be
            included in the injection.
    """

    sources: list[str] = field(default_factory=lambda: ["memory", "history"])
    n_results: int = 5
    recall_model: Optional[str] = None
    categories: list[str] = field(default_factory=list)
    min_relevance: float = 0.5


class AutoRecall:
    """Manages automatic memory recall before each agent response.

    Instantiate with a :class:`RecallConfig` and call :meth:`inject_recall`
    at the start of each ``send()`` / ``stream()`` call (before the user
    message is appended to the context).
    """

    def __init__(self, config: RecallConfig) -> None:
        self.config = config

    def inject_recall(
        self,
        agent: "Agent",
        context: AgentContext,
        content: str,
        model: Optional[str],
    ) -> None:
        """Retrieve relevant memories and prepend them to *context*.

        The method is a no-op when *content* is empty (e.g. tool-loop
        continuation calls) so that intermediate recursive calls are not
        charged with an extra LLM round-trip.

        Steps:
        1. Remove any existing auto-recall message from the context.
        2. Use an ephemeral LLM prompt to derive 1–3 search queries.
        3. Search the configured sources with those queries.
        4. If any snippets were found, prepend a ``[RECALLED CONTEXT]``
           system message tagged ``_pithos_no_compact`` and
           ``_pithos_auto_recall``.

        Args:
            agent: The owning agent instance.
            context: The active conversation context.
            content: The user message about to be sent.  Recall is skipped
                when this is empty.
            model: Override model (or ``None`` to use agent default).
        """
        # Skip recall for internal continuation calls (tool/memory auto-loop)
        if not content or not content.strip():
            return

        # Always clean up stale recall message from previous turn
        self._remove_previous_recall(context)

        # Generate search queries via a lightweight ephemeral LLM call
        queries = self._generate_queries(agent, context, content, model)
        if not queries:
            return

        # Retrieve snippets from all configured sources
        snippets = self._retrieve(agent, queries)
        if not snippets:
            return

        # Build and inject the recall context message
        recall_lines = [_RECALL_HEADER]
        for i, (source, text) in enumerate(snippets, 1):
            recall_lines.append(f"\n{i}. [{source}] {text}")
        recall_content = "\n".join(recall_lines)

        recall_msg: dict = {
            "role": "system",
            "content": recall_content,
            "_pithos_no_compact": True,
            "_pithos_auto_recall": True,
        }
        context.message_history.insert(0, recall_msg)

        logger.debug(
            "Injected %d recalled snippets into context '%s'.",
            len(snippets),
            context.name,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _remove_previous_recall(context: AgentContext) -> None:
        """Remove an existing auto-recall injection from the context."""
        context.message_history = [
            m for m in context.message_history if not m.get("_pithos_auto_recall")
        ]

    def _generate_queries(
        self,
        agent: "Agent",
        context: AgentContext,
        content: str,
        model: Optional[str],
    ) -> list[str]:
        """Ask the LLM what to search for, based on recent history + new message.

        Uses a brief sliding window of the last few turns (non-recall, non-
        compaction-summary messages) to avoid sending the full context.

        Returns:
            Up to 3 search query strings.
        """
        from ollama import chat as ollama_chat

        # Build a concise history snippet (last 6 non-internal messages)
        visible = [
            m
            for m in context.message_history[-6:]
            if not m.get("_pithos_auto_recall") and not m.get("_pithos_no_compact")
        ]
        history_text = (
            "\n".join(f"{m['role'].upper()}: {m['content'][:300]}" for m in visible)
            if visible
            else "(no prior conversation)"
        )

        prompt = _QUERY_PROMPT.format(history=history_text, new_message=content)
        model_to_use = self.config.recall_model or model or agent.default_model
        options: dict = {"temperature": 0.2, "num_predict": 200}

        try:
            response = ollama_chat(
                model=model_to_use,
                messages=[{"role": "user", "content": prompt}],
                options=options,
            )
            raw = (response.message.content or "").strip()
        except Exception as exc:
            logger.warning("Recall query generation failed: %s", exc)
            return []

        queries = [q.strip() for q in raw.splitlines() if q.strip()]
        return queries[:3]

    def _retrieve(self, agent: "Agent", queries: list[str]) -> list[tuple[str, str]]:
        """Search all configured sources and return deduplicated snippets.

        Returns:
            List of ``(source_label, text)`` tuples, capped at
            ``config.n_results``.
        """
        snippets: list[tuple[str, str]] = []
        seen: set[str] = set()

        for query in queries:
            if "memory" in self.config.sources:
                for text, _score in self._search_memory(agent, query):
                    if text not in seen:
                        seen.add(text)
                        snippets.append(("memory", text))

            if "history" in self.config.sources:
                for text in self._search_history(agent, query):
                    if text not in seen:
                        seen.add(text)
                        snippets.append(("history", text))

            if len(snippets) >= self.config.n_results:
                break

        return snippets[: self.config.n_results]

    def _search_memory(self, agent: "Agent", query: str) -> list[tuple[str, float]]:
        """Search the vector memory store.

        Returns:
            Sorted list of ``(content, relevance_score)`` pairs.
        """
        if not (agent.memory_enabled and agent.memory_store):
            return []

        results: list[tuple[str, float]] = []
        try:
            categories = list(self.config.categories)
            if not categories:
                categories = agent.memory_store.list_categories()
            for cat in categories:
                for r in agent.memory_store.retrieve(
                    cat,
                    query,
                    n_results=self.config.n_results,
                    min_relevance=self.config.min_relevance,
                ):
                    results.append((r.content, r.relevance_score))
        except Exception as exc:
            logger.warning("Memory search failed during recall: %s", exc)

        results.sort(key=lambda x: x[1], reverse=True)
        return results[: self.config.n_results]

    def _search_history(self, agent: "Agent", query: str) -> list[str]:
        """Search the persistent conversation history store.

        Returns:
            List of formatted snippet strings.
        """
        if agent.history_store is None:
            return []
        try:
            hits = agent.history_store.search(
                query=query, n_results=self.config.n_results, semantic=True
            )
            return [f"[{h.message.role}] {h.message.content[:300]}" for h in hits]
        except Exception as exc:
            logger.warning("History search failed during recall: %s", exc)
            return []
