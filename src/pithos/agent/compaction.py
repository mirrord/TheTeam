"""Automatic context compaction for agent conversation histories.

When the number of messages in a context exceeds a configured threshold,
:class:`MemoryCompactor` summarises the oldest messages via an LLM call,
optionally archives the raw messages to the vector memory store, and replaces
them with a single system-level summary so the active context stays within
budget.

Messages tagged with ``_pithos_no_compact: True`` (auto-recall injections,
previous summaries) are never included in the compaction candidates and are
preserved in place.
"""

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

from ..context import AgentContext

if TYPE_CHECKING:  # pragma: no cover
    from .agent import Agent

logger = logging.getLogger(__name__)

_SUMMARY_PROMPT = """\
The following is a portion of a conversation history that needs to be summarised \
to free up context space. Summarise it concisely, capturing the key topics, \
decisions, and outcomes. Then list important named entities (people, tools, files, \
concepts, technical terms) that appear in this conversation but may not be \
explicitly named in the summary.

Conversation history:
{history}

Respond in exactly this format (do not add anything else):
Summary: <your summary here>
Entities: <comma-separated entity list, or "none" if there are none>"""


@dataclass
class CompactionConfig:
    """Configuration for automatic context compaction.

    Attributes:
        threshold: Total message count that triggers compaction.
        keep_last: Number of most-recent (non-protected) messages to leave
            untouched so they remain available for immediate context.
        summary_model: Ollama model to use for summarisation.  Falls back to
            the agent's ``default_model`` when ``None``.
        memory_category: Category name used when archiving summaries to the
            vector memory store.
        summary_max_tokens: Maximum tokens allowed for the summarisation
            response.
    """

    threshold: int = 20
    keep_last: int = 6
    summary_model: Optional[str] = None
    memory_category: str = "context_summaries"
    summary_max_tokens: int = -1


class MemoryCompactor:
    """Manages automatic context compaction for an agent.

    Instantiate with a :class:`CompactionConfig` and call :meth:`compact`
    after each assistant turn.  Compaction only runs when
    :meth:`should_compact` returns ``True``.
    """

    def __init__(self, config: CompactionConfig) -> None:
        self.config = config

    def should_compact(self, context: AgentContext) -> bool:
        """Return ``True`` if the context is large enough to warrant compaction.

        Compaction is warranted when the *total* message count meets or exceeds
        ``config.threshold`` AND there are more non-protected messages than
        ``config.keep_last`` (i.e. there is something to actually compact).
        """
        if len(context.message_history) < self.config.threshold:
            return False
        compactable = sum(
            1 for m in context.message_history if not m.get("_pithos_no_compact")
        )
        return compactable > self.config.keep_last

    def compact(self, agent: "Agent", context: AgentContext, context_name: str) -> None:
        """Compact the context if the threshold is met.

        1. Identifies non-protected messages and picks the oldest ones to
           compact (all except the last ``keep_last`` non-protected messages).
        2. Summarises them via the configured model.
        3. Optionally archives the summary to the memory store.
        4. Removes the compacted messages and inserts a summary entry in their
           place, tagged ``_pithos_no_compact: True`` so it is never itself
           compacted.

        Args:
            agent: The owning :class:`~pithos.agent.Agent` instance (needed
                for the LLM call and optional memory store access).
            context: The :class:`~pithos.context.AgentContext` to compact.
            context_name: Human-readable context name used in archive metadata.
        """
        if not self.should_compact(context):
            return

        # Determine which message indices are compactable
        compactable_indices = [
            i
            for i, m in enumerate(context.message_history)
            if not m.get("_pithos_no_compact")
        ]

        # Keep the last `keep_last` compactable messages; compact the rest
        keep = self.config.keep_last
        to_compact_indices = (
            compactable_indices[:-keep] if keep > 0 else compactable_indices
        )

        if not to_compact_indices:
            return

        msgs_to_compact = [context.message_history[i] for i in to_compact_indices]

        logger.debug(
            "Compacting %d messages in context '%s'",
            len(msgs_to_compact),
            context_name,
        )

        # Generate summary via LLM
        summary, entities = self._generate_summary(agent, msgs_to_compact)

        # Archive to memory store if available (stores the summary, not raw messages,
        # to avoid blowing up the vector DB with potentially redundant content)
        if agent.memory_enabled and agent.memory_store:
            self._archive_to_memory(agent.memory_store, context_name, summary, entities)

        # Remove compacted messages (iterate in reverse to preserve indices)
        for i in sorted(to_compact_indices, reverse=True):
            context.message_history.pop(i)

        # Insert the summary at the position of the first removed message,
        # clamped to the (now shorter) list length
        insert_pos = min(to_compact_indices[0], len(context.message_history))

        summary_content = f"[CONTEXT SUMMARY]\n{summary}"
        if entities and entities.strip().lower() not in ("none", ""):
            summary_content += f"\n\nKey entities: {entities}"

        summary_msg: dict = {
            "role": "system",
            "content": summary_content,
            "_pithos_no_compact": True,
        }
        context.message_history.insert(insert_pos, summary_msg)

        logger.debug(
            "Compaction complete. Summary inserted at position %d.", insert_pos
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _generate_summary(
        self, agent: "Agent", messages: list[dict]
    ) -> tuple[str, str]:
        """Summarise *messages* via the LLM.

        Returns:
            A ``(summary, entities)`` tuple where *entities* is a
            comma-separated string (may be ``"none"``).
        """
        from ollama import chat as ollama_chat

        history_text = "\n".join(
            f"{m['role'].upper()}: {m['content']}" for m in messages
        )
        prompt = _SUMMARY_PROMPT.format(history=history_text)

        print("Compaction max tokens config:", self.config.summary_max_tokens)
        model = self.config.summary_model or agent.default_model
        options: dict = {
            "temperature": 0.3,
            "num_predict": self.config.summary_max_tokens,
        }

        raw = ""
        for attempt in range(3):  # Try twice: if parsing fails, retry once more
            try:
                print("Compaction LLM prompt:\n", prompt)
                response = ollama_chat(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    options=options,
                )
                print("Compaction LLM raw response:\n", response.message.content)
                raw = (response.message.content or "").strip()
            except Exception as exc:
                logger.warning("Compaction LLM call failed: %s", exc)
                return "[Summary unavailable due to LLM error]", "none"
            if raw:
                break  # Exit loop if we got a non-empty response

        summary, entities = self._parse_summary_response(raw)
        # If parsing produced an empty summary, fall back to the raw response so
        # the context block is never blank and archiving never fails with an empty
        # content error.
        if not summary.strip():
            summary = raw if raw else "[Summary unavailable]"
        return summary, entities

    @staticmethod
    def _parse_summary_response(raw: str) -> tuple[str, str]:
        """Parse the structured summary/entities response from the LLM."""
        summary = raw
        entities = "none"

        if "Entities:" in raw:
            parts = raw.split("Entities:", 1)
            entities = parts[1].strip()
            summary_part = parts[0]
            if summary_part.strip().lower().startswith("summary:"):
                summary_part = summary_part.strip()[len("summary:") :].strip()
            summary = summary_part.strip()
        elif raw.lower().startswith("summary:"):
            summary = raw[len("summary:") :].strip()

        return summary, entities

    def _archive_to_memory(
        self,
        memory_store,
        context_name: str,
        summary: str,
        entities: str,
    ) -> None:
        """Store the compaction summary in the vector memory for future recall."""
        if not summary or not summary.strip():
            logger.debug("Skipping compaction archive: summary is empty.")
            return
        try:
            metadata: dict = {
                "type": "compaction_summary",
                "context_name": context_name,
            }
            if entities and entities.strip().lower() not in ("none", ""):
                metadata["entities"] = entities
            memory_store.store(self.config.memory_category, summary, metadata=metadata)
        except Exception as exc:
            logger.warning("Failed to archive compaction summary to memory: %s", exc)
