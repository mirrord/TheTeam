"""Unit tests for automatic context compaction (MemoryCompactor) and
automatic memory recall (AutoRecall)."""

import pytest
from unittest.mock import MagicMock, patch, call
from pithos.context import AgentContext, Msg, UserMsg, AgentMsg
from pithos.agent.compaction import CompactionConfig, MemoryCompactor
from pithos.agent.recall import RecallConfig, AutoRecall


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_context(n_messages: int) -> AgentContext:
    """Return a context with *n_messages* alternating user/assistant messages."""
    ctx = AgentContext("test", "sys")
    for i in range(n_messages):
        role = "user" if i % 2 == 0 else "assistant"
        ctx.message_history.append({"role": role, "content": f"msg {i}"})
    return ctx


def _make_agent(model: str = "glm-4.7-flash") -> MagicMock:
    """Return a mock agent with the attributes MemoryCompactor/AutoRecall expect."""
    agent = MagicMock()
    agent.default_model = model
    agent.memory_enabled = False
    agent.memory_store = None
    agent.history_store = None
    return agent


# ===========================================================================
# CompactionConfig
# ===========================================================================


class TestCompactionConfig:
    def test_defaults(self):
        cfg = CompactionConfig()
        assert cfg.threshold == 20
        assert cfg.keep_last == 6
        assert cfg.summary_model is None
        assert cfg.memory_category == "context_summaries"
        assert cfg.summary_max_tokens == 512

    def test_custom(self):
        cfg = CompactionConfig(threshold=10, keep_last=2, summary_model="phi3")
        assert cfg.threshold == 10
        assert cfg.keep_last == 2
        assert cfg.summary_model == "phi3"


# ===========================================================================
# MemoryCompactor.should_compact
# ===========================================================================


class TestShouldCompact:
    def test_below_threshold_returns_false(self):
        compactor = MemoryCompactor(CompactionConfig(threshold=20, keep_last=2))
        ctx = _make_context(10)
        assert compactor.should_compact(ctx) is False

    def test_at_threshold_with_enough_compactable_returns_true(self):
        compactor = MemoryCompactor(CompactionConfig(threshold=10, keep_last=2))
        ctx = _make_context(
            10
        )  # 10 messages, all compactable, keep_last=2 → 8 to compact
        assert compactor.should_compact(ctx) is True

    def test_no_compactable_due_to_keep_last_returns_false(self):
        compactor = MemoryCompactor(CompactionConfig(threshold=5, keep_last=10))
        ctx = _make_context(5)
        # 5 total compactable, keep_last=10 → nothing to compact
        assert compactor.should_compact(ctx) is False

    def test_protected_messages_excluded_from_count(self):
        compactor = MemoryCompactor(CompactionConfig(threshold=6, keep_last=2))
        ctx = _make_context(0)
        # 4 normal messages + 4 protected
        for i in range(4):
            ctx.message_history.append({"role": "user", "content": f"real {i}"})
        for i in range(4):
            ctx.message_history.append(
                {"role": "system", "content": "summary", "_pithos_no_compact": True}
            )
        # total=8 >= threshold=6, but only 4 compactable, keep_last=2 → 2 to compact → True
        assert compactor.should_compact(ctx) is True

    def test_all_protected_returns_false(self):
        compactor = MemoryCompactor(CompactionConfig(threshold=3, keep_last=1))
        ctx = _make_context(0)
        for i in range(5):
            ctx.message_history.append(
                {"role": "system", "content": "summary", "_pithos_no_compact": True}
            )
        # 0 compactable → False even though total >= threshold
        assert compactor.should_compact(ctx) is False


# ===========================================================================
# MemoryCompactor.compact
# ===========================================================================


class TestCompact:
    def _make_compactor(self, threshold=10, keep_last=2):
        return MemoryCompactor(
            CompactionConfig(threshold=threshold, keep_last=keep_last)
        )

    def test_compact_reduces_message_count(self):
        compactor = self._make_compactor(threshold=6, keep_last=2)
        ctx = _make_context(8)
        agent = _make_agent()

        with patch.object(
            compactor, "_generate_summary", return_value=("Summary text", "none")
        ):
            compactor.compact(agent, ctx, "test")

        # Should have: (8-6 compacted msgs replaced by 1 summary) + 2 kept = 3
        assert len(ctx.message_history) == 3

    def test_compact_inserts_summary_message(self):
        compactor = self._make_compactor(threshold=6, keep_last=2)
        ctx = _make_context(8)
        agent = _make_agent()

        with patch.object(
            compactor, "_generate_summary", return_value=("Great summary", "Alice, Bob")
        ):
            compactor.compact(agent, ctx, "test")

        summary_msgs = [
            m for m in ctx.message_history if "[CONTEXT SUMMARY]" in m["content"]
        ]
        assert len(summary_msgs) == 1
        assert "Great summary" in summary_msgs[0]["content"]
        assert "Alice, Bob" in summary_msgs[0]["content"]

    def test_summary_message_is_protected(self):
        compactor = self._make_compactor(threshold=6, keep_last=2)
        ctx = _make_context(8)
        agent = _make_agent()

        with patch.object(
            compactor, "_generate_summary", return_value=("Summary", "none")
        ):
            compactor.compact(agent, ctx, "test")

        summary_msgs = [
            m for m in ctx.message_history if "[CONTEXT SUMMARY]" in m["content"]
        ]
        assert summary_msgs[0].get("_pithos_no_compact") is True

    def test_compact_skips_when_threshold_not_met(self):
        compactor = self._make_compactor(threshold=20, keep_last=2)
        ctx = _make_context(5)
        agent = _make_agent()
        original_history = list(ctx.message_history)

        with patch.object(compactor, "_generate_summary") as mock_summarise:
            compactor.compact(agent, ctx, "test")
            mock_summarise.assert_not_called()

        assert ctx.message_history == original_history

    def test_compact_archives_to_memory_when_enabled(self):
        compactor = self._make_compactor(threshold=6, keep_last=2)
        ctx = _make_context(8)
        agent = _make_agent()
        agent.memory_enabled = True
        mock_store = MagicMock()
        agent.memory_store = mock_store

        with patch.object(
            compactor, "_generate_summary", return_value=("Summary", "entity1")
        ):
            compactor.compact(agent, ctx, "test")

        mock_store.store.assert_called_once()
        args, kwargs = mock_store.store.call_args
        assert args[1] == "Summary"  # category, content, metadata

    def test_compact_does_not_archive_when_memory_disabled(self):
        compactor = self._make_compactor(threshold=6, keep_last=2)
        ctx = _make_context(8)
        agent = _make_agent()
        agent.memory_enabled = False

        with patch.object(
            compactor, "_generate_summary", return_value=("Summary", "none")
        ):
            with patch.object(compactor, "_archive_to_memory") as mock_archive:
                compactor.compact(agent, ctx, "test")
                mock_archive.assert_not_called()

    def test_protected_messages_not_compacted(self):
        compactor = self._make_compactor(threshold=5, keep_last=1)
        ctx = _make_context(0)
        protected = {
            "role": "system",
            "content": "recall context",
            "_pithos_no_compact": True,
        }
        ctx.message_history.append(protected)
        for i in range(6):
            ctx.message_history.append({"role": "user", "content": f"msg {i}"})

        agent = _make_agent()

        with patch.object(
            compactor, "_generate_summary", return_value=("Summary", "none")
        ):
            compactor.compact(agent, ctx, "test")

        # Protected message must still be present
        assert any(
            m.get("_pithos_auto_recall") is True or "_pithos_no_compact" in m
            for m in ctx.message_history
            if m["content"] == "recall context"
        )

    def test_entities_not_shown_when_none(self):
        compactor = self._make_compactor(threshold=6, keep_last=2)
        ctx = _make_context(8)
        agent = _make_agent()

        with patch.object(
            compactor, "_generate_summary", return_value=("Summary only", "none")
        ):
            compactor.compact(agent, ctx, "test")

        summary_msg = next(
            m
            for m in ctx.message_history
            if "[CONTEXT SUMMARY]" in m.get("content", "")
        )
        assert "Key entities" not in summary_msg["content"]


# ===========================================================================
# MemoryCompactor._parse_summary_response
# ===========================================================================


class TestParseSummaryResponse:
    def test_parses_structured_response(self):
        raw = "Summary: This is the summary.\nEntities: Alice, Bob, pytest"
        summary, entities = MemoryCompactor._parse_summary_response(raw)
        assert summary == "This is the summary."
        assert entities == "Alice, Bob, pytest"

    def test_parses_summary_only(self):
        raw = "Summary: Just a summary."
        summary, entities = MemoryCompactor._parse_summary_response(raw)
        assert summary == "Just a summary."
        assert entities == "none"

    def test_parses_plain_text(self):
        raw = "Plain response without structure."
        summary, entities = MemoryCompactor._parse_summary_response(raw)
        assert summary == raw
        assert entities == "none"


# ===========================================================================
# RecallConfig
# ===========================================================================


class TestRecallConfig:
    def test_defaults(self):
        cfg = RecallConfig()
        assert "memory" in cfg.sources
        assert "history" in cfg.sources
        assert cfg.n_results == 5
        assert cfg.recall_model is None
        assert cfg.categories == []
        assert cfg.min_relevance == 0.5

    def test_custom(self):
        cfg = RecallConfig(sources=["memory"], n_results=3, min_relevance=0.7)
        assert cfg.sources == ["memory"]
        assert cfg.n_results == 3
        assert cfg.min_relevance == 0.7


# ===========================================================================
# AutoRecall._remove_previous_recall
# ===========================================================================


class TestRemovePreviousRecall:
    def test_removes_recall_message(self):
        ctx = _make_context(3)
        ctx.message_history.append(
            {"role": "system", "content": "recall", "_pithos_auto_recall": True}
        )
        AutoRecall._remove_previous_recall(ctx)
        assert not any(m.get("_pithos_auto_recall") for m in ctx.message_history)

    def test_does_not_remove_non_recall_messages(self):
        ctx = _make_context(3)
        before = len(ctx.message_history)
        AutoRecall._remove_previous_recall(ctx)
        assert len(ctx.message_history) == before


# ===========================================================================
# AutoRecall.inject_recall
# ===========================================================================


class TestInjectRecall:
    def test_skips_empty_content(self):
        recall = AutoRecall(RecallConfig())
        ctx = _make_context(2)
        agent = _make_agent()

        with patch.object(recall, "_generate_queries") as mock_gen:
            recall.inject_recall(agent, ctx, content="", model=None)
            mock_gen.assert_not_called()

    def test_skips_whitespace_content(self):
        recall = AutoRecall(RecallConfig())
        ctx = _make_context(2)
        agent = _make_agent()

        with patch.object(recall, "_generate_queries") as mock_gen:
            recall.inject_recall(agent, ctx, content="   ", model=None)
            mock_gen.assert_not_called()

    def test_no_snippets_means_no_injection(self):
        recall = AutoRecall(RecallConfig())
        ctx = _make_context(2)
        agent = _make_agent()

        with patch.object(recall, "_generate_queries", return_value=["query"]):
            with patch.object(recall, "_retrieve", return_value=[]):
                before = len(ctx.message_history)
                recall.inject_recall(agent, ctx, content="hello", model=None)
                assert len(ctx.message_history) == before

    def test_snippets_injected_at_position_zero(self):
        recall = AutoRecall(RecallConfig())
        ctx = _make_context(4)
        agent = _make_agent()
        fake_snippets = [("memory", "Relevant fact A"), ("history", "Prior answer B")]

        with patch.object(recall, "_generate_queries", return_value=["q"]):
            with patch.object(recall, "_retrieve", return_value=fake_snippets):
                recall.inject_recall(agent, ctx, content="question", model=None)

        assert ctx.message_history[0].get("_pithos_auto_recall") is True
        assert ctx.message_history[0].get("_pithos_no_compact") is True
        assert "RECALLED CONTEXT" in ctx.message_history[0]["content"]
        assert "Relevant fact A" in ctx.message_history[0]["content"]

    def test_previous_recall_replaced(self):
        recall = AutoRecall(RecallConfig())
        ctx = _make_context(2)
        # Pre-insert an old recall message
        ctx.message_history.insert(
            0,
            {
                "role": "system",
                "content": "OLD RECALL",
                "_pithos_auto_recall": True,
                "_pithos_no_compact": True,
            },
        )
        agent = _make_agent()
        fake_snippets = [("memory", "New fact")]

        with patch.object(recall, "_generate_queries", return_value=["q"]):
            with patch.object(recall, "_retrieve", return_value=fake_snippets):
                recall.inject_recall(agent, ctx, content="question", model=None)

        recall_msgs = [m for m in ctx.message_history if m.get("_pithos_auto_recall")]
        assert len(recall_msgs) == 1
        assert "OLD RECALL" not in recall_msgs[0]["content"]
        assert "New fact" in recall_msgs[0]["content"]

    def test_queries_generated_with_no_queries_skips_retrieve(self):
        recall = AutoRecall(RecallConfig())
        ctx = _make_context(2)
        agent = _make_agent()

        with patch.object(recall, "_generate_queries", return_value=[]):
            with patch.object(recall, "_retrieve") as mock_retrieve:
                recall.inject_recall(agent, ctx, content="hello", model=None)
                mock_retrieve.assert_not_called()


# ===========================================================================
# AutoRecall._retrieve deduplication
# ===========================================================================


class TestRetrieve:
    def test_deduplicates_snippets(self):
        recall = AutoRecall(RecallConfig(n_results=10))
        agent = _make_agent()
        agent.memory_enabled = True
        mock_result = MagicMock()
        mock_result.content = "Same fact"
        mock_result.relevance_score = 0.9
        agent.memory_store = MagicMock()
        agent.memory_store.list_categories.return_value = ["cat"]
        agent.memory_store.retrieve.return_value = [mock_result, mock_result]

        snippets = recall._retrieve(agent, ["query1", "query2"])
        texts = [s[1] for s in snippets]
        assert texts.count("Same fact") == 1

    def test_caps_at_n_results(self):
        recall = AutoRecall(RecallConfig(n_results=2))
        agent = _make_agent()
        agent.memory_enabled = True
        results = []
        for i in range(5):
            r = MagicMock()
            r.content = f"fact {i}"
            r.relevance_score = 0.9
            results.append(r)
        agent.memory_store = MagicMock()
        agent.memory_store.list_categories.return_value = ["cat"]
        agent.memory_store.retrieve.return_value = results

        snippets = recall._retrieve(agent, ["query"])
        assert len(snippets) <= 2

    def test_filters_by_min_relevance(self):
        # Filtering by min_relevance is now delegated to retrieve(); _search_memory
        # passes min_relevance as a kwarg and trusts the store to filter.
        # Simulate a well-behaved store mock that already filters on min_relevance.
        recall = AutoRecall(
            RecallConfig(sources=["memory"], n_results=10, min_relevance=0.8)
        )
        agent = _make_agent()
        agent.memory_enabled = True

        high = MagicMock()
        high.content = "high"
        high.relevance_score = 0.9
        # The store only returns results that pass min_relevance (as retrieve() now does)
        agent.memory_store = MagicMock()
        agent.memory_store.list_categories.return_value = ["cat"]
        agent.memory_store.retrieve.return_value = [high]

        snippets = recall._retrieve(agent, ["query"])
        texts = [s[1] for s in snippets]
        assert "high" in texts
        # Confirm retrieve was called with the correct min_relevance
        _args, kwargs = agent.memory_store.retrieve.call_args
        assert kwargs.get("min_relevance") == 0.8


# ===========================================================================
# AgentContext.get_messages strips _pithos_* metadata
# ===========================================================================


class TestGetMessagesStripsMetadata:
    def test_strips_pithos_keys(self):
        ctx = AgentContext("test", "sys")
        ctx.message_history.append(
            {
                "role": "system",
                "content": "recalled context",
                "_pithos_no_compact": True,
                "_pithos_auto_recall": True,
            }
        )
        messages = ctx.get_messages()
        for msg in messages:
            for key in msg:
                assert not key.startswith("_pithos_")

    def test_preserves_role_and_content(self):
        ctx = AgentContext("test", "sys")
        ctx.message_history.append(
            {
                "role": "user",
                "content": "hello",
                "_pithos_no_compact": True,
            }
        )
        messages = ctx.get_messages()
        user_msgs = [m for m in messages if m.get("role") == "user"]
        assert user_msgs[0]["content"] == "hello"


# ===========================================================================
# Integration: Agent.enable_compaction / enable_recall flags
# ===========================================================================


class TestAgentEnableMethods:
    def _make_real_agent(self):
        from pithos.agent import OllamaAgent

        agent = OllamaAgent.__new__(OllamaAgent)
        # Manually init without calling __init__ to avoid LLM setup
        from pithos.agent.agent import Agent

        Agent.__init__(agent, "glm-4.7-flash", "test-agent")
        return agent

    def test_enable_compaction_sets_flags(self):
        agent = self._make_real_agent()
        assert agent.compaction_enabled is False
        agent.enable_compaction(CompactionConfig(threshold=10))
        assert agent.compaction_enabled is True
        assert agent._compactor is not None

    def test_disable_compaction_clears_flags(self):
        agent = self._make_real_agent()
        agent.enable_compaction()
        agent.disable_compaction()
        assert agent.compaction_enabled is False
        assert agent._compactor is None

    def test_enable_recall_sets_flags(self):
        agent = self._make_real_agent()
        assert agent.recall_enabled is False
        agent.enable_recall(RecallConfig(n_results=3))
        assert agent.recall_enabled is True
        assert agent._auto_recall is not None

    def test_disable_recall_clears_flags(self):
        agent = self._make_real_agent()
        agent.enable_recall()
        agent.disable_recall()
        assert agent.recall_enabled is False
        assert agent._auto_recall is None

    def test_enable_compaction_default_config(self):
        agent = self._make_real_agent()
        agent.enable_compaction()
        assert agent._compactor.config.threshold == 20

    def test_enable_recall_default_config(self):
        agent = self._make_real_agent()
        agent.enable_recall()
        assert agent._auto_recall.config.n_results == 5


# ===========================================================================
# Bug-regression: empty summary fallback in _generate_summary
# ===========================================================================


class TestGenerateSummaryEmptyFallback:
    """_generate_summary must never return an empty summary string."""

    def _make_compactor(self):
        return MemoryCompactor(CompactionConfig())

    def _make_agent(self):
        return _make_agent()

    def test_empty_raw_response_uses_fallback(self):
        """When the LLM returns an empty string, summary falls back to
        '[Summary unavailable]'."""
        compactor = self._make_compactor()
        agent = self._make_agent()

        # ollama_chat is imported inside _generate_summary, so patch at ollama level
        with patch("ollama.chat") as mock_chat:
            mock_chat.return_value.message.content = ""
            summary, entities = compactor._generate_summary(
                agent, [{"role": "user", "content": "hi"}]
            )

        assert summary.strip() != ""

    def test_summary_only_prefix_no_body_uses_raw(self):
        """'Summary: \\nEntities: none' parsed to empty body triggers fallback."""
        compactor = self._make_compactor()
        agent = self._make_agent()

        raw_response = "Summary: \nEntities: none"
        with patch("ollama.chat") as mock_chat:
            mock_chat.return_value.message.content = raw_response
            summary, entities = compactor._generate_summary(
                agent, [{"role": "user", "content": "hi"}]
            )

        # Should fall back to raw instead of returning empty string
        assert summary.strip() != ""

    def test_normal_summary_unchanged(self):
        """A properly formatted response is parsed and returned as-is."""
        compactor = self._make_compactor()
        agent = self._make_agent()

        raw_response = (
            "Summary: The conversation covered astronomy.\nEntities: Sun, Jupiter"
        )
        with patch("ollama.chat") as mock_chat:
            mock_chat.return_value.message.content = raw_response
            summary, entities = compactor._generate_summary(
                agent, [{"role": "user", "content": "hi"}]
            )

        assert summary == "The conversation covered astronomy."
        assert entities == "Sun, Jupiter"


# ===========================================================================
# Bug-regression: _archive_to_memory skips empty summaries gracefully
# ===========================================================================


class TestArchiveToMemoryEmptySummary:
    """_archive_to_memory must not invoke the store with empty content."""

    def test_empty_summary_does_not_call_store(self):
        compactor = MemoryCompactor(CompactionConfig())
        mock_store = MagicMock()
        compactor._archive_to_memory(mock_store, "ctx", "", "none")
        mock_store.store.assert_not_called()

    def test_whitespace_summary_does_not_call_store(self):
        compactor = MemoryCompactor(CompactionConfig())
        mock_store = MagicMock()
        compactor._archive_to_memory(mock_store, "ctx", "   ", "none")
        mock_store.store.assert_not_called()

    def test_nonempty_summary_calls_store(self):
        compactor = MemoryCompactor(CompactionConfig())
        mock_store = MagicMock()
        compactor._archive_to_memory(mock_store, "ctx", "A real summary.", "none")
        mock_store.store.assert_called_once()


# ===========================================================================
# Bug-regression: MemoryStore.retrieve respects caller-supplied min_relevance
# ===========================================================================


class TestRetrieveMinRelevanceOverride:
    """retrieve() must use the caller-supplied min_relevance when provided,
    overriding the config similarity_threshold."""

    def _make_mock_collection(self, distances):
        """Return a MagicMock collection that yields *distances*."""
        n = len(distances)
        collection = MagicMock()
        collection.query.return_value = {
            "ids": [[f"id{i}" for i in range(n)]],
            "distances": [distances],
            "documents": [[f"doc {i}" for i in range(n)]],
            "metadatas": [[{} for _ in range(n)]],
        }
        return collection

    def test_min_relevance_overrides_config_threshold(self):
        """Results that would be filtered by config threshold (0.5) are returned
        when min_relevance=0.0 is supplied."""
        from pithos.tools.memory_tool import MemoryStore

        store = MagicMock(spec=MemoryStore)
        store.config = {"similarity_threshold": 0.5}
        store._get_collection = MagicMock(
            return_value=self._make_mock_collection(
                [1.1]
            )  # exp(-1.1) ≈ 0.33, below 0.5
        )

        # Call the real retrieve implementation directly via unbound call
        results = MemoryStore.retrieve(
            store, "cat", "query", n_results=5, min_relevance=0.0
        )
        # With min_relevance=0.0 every result should pass
        assert len(results) == 1
        assert (
            results[0].relevance_score < 0.5
        )  # confirms it was below config threshold

    def test_default_threshold_used_when_no_min_relevance(self):
        """Without min_relevance, config similarity_threshold still applies."""
        from pithos.tools.memory_tool import MemoryStore

        store = MagicMock(spec=MemoryStore)
        store.config = {"similarity_threshold": 0.5}
        store._get_collection = MagicMock(
            return_value=self._make_mock_collection([1.1])  # score ≈ 0.33
        )

        results = MemoryStore.retrieve(store, "cat", "query", n_results=5)
        # Without override, result is below config threshold → filtered out
        assert len(results) == 0


# ===========================================================================
# Bug-regression: format-specifier safe prompt construction
# ===========================================================================


class TestFormatSpecifierSafePrompts:
    """Prompts built from LLM-generated content must not crash when that
    content contains Python format specifiers like {0}, {name}, or bare {}."""

    def test_compaction_summary_with_format_specifiers_in_history(self):
        """_generate_summary must not crash when messages contain {0} or {key}."""
        compactor = MemoryCompactor(CompactionConfig())
        agent = _make_agent()

        messages = [
            {"role": "user", "content": "Explain the Hubble constant"},
            {
                "role": "assistant",
                "content": "The Hubble constant H{0} is ~70 km/s/Mpc. "
                "See {docs} for details and {} for more.",
            },
        ]

        with patch("ollama.chat") as mock_chat:
            mock_chat.return_value.message.content = (
                "Summary: Discussion of Hubble constant.\nEntities: Hubble, H0"
            )
            summary, entities = compactor._generate_summary(agent, messages)

        assert "Hubble" in summary
        assert entities != "none"

    def test_recall_query_gen_with_format_specifiers_in_history(self):
        """_generate_queries must not crash when context has format specifiers."""
        recall = AutoRecall(RecallConfig())
        agent = _make_agent()
        ctx = AgentContext("test", "sys")
        ctx.message_history.append(
            {
                "role": "assistant",
                "content": "Use config[{key}] and item {0} for lookups.",
            }
        )

        with patch("ollama.chat") as mock_chat:
            mock_chat.return_value.message.content = "config lookup syntax"
            queries = recall._generate_queries(
                agent, ctx, "How do I look things up?", None
            )

        assert len(queries) >= 1

    def test_recall_query_gen_with_format_specifiers_in_user_message(self):
        """_generate_queries must not crash when the user message has specifiers."""
        recall = AutoRecall(RecallConfig())
        agent = _make_agent()
        ctx = AgentContext("test", "sys")

        with patch("ollama.chat") as mock_chat:
            mock_chat.return_value.message.content = "JSON parsing"
            queries = recall._generate_queries(
                agent, ctx, 'Parse {"key": "value"} in Python', None
            )

        assert len(queries) >= 1


# ===========================================================================
# Bug-regression: relevance score formula for L2 distance
# ===========================================================================


class TestRelevanceScoreFormula:
    """The distance-to-relevance conversion must produce meaningful scores
    for ChromaDB's default L2 distance metric."""

    def _make_mock_collection(self, distances):
        n = len(distances)
        collection = MagicMock()
        collection.query.return_value = {
            "ids": [[f"id{i}" for i in range(n)]],
            "distances": [distances],
            "documents": [[f"doc {i}" for i in range(n)]],
            "metadatas": [[{} for _ in range(n)]],
        }
        return collection

    def test_typical_l2_distances_produce_usable_scores(self):
        """L2 distances in the typical range [0.5, 2.0] must map to scores
        that allow relevant results through a 0.4 threshold."""
        from pithos.tools.memory_tool import MemoryStore

        store = MagicMock(spec=MemoryStore)
        store.config = {"similarity_threshold": 0.0}
        store._get_collection = MagicMock(
            return_value=self._make_mock_collection([0.5, 1.0, 1.25, 2.0])
        )

        results = MemoryStore.retrieve(
            store, "cat", "query", n_results=10, min_relevance=0.0
        )
        scores = [r.relevance_score for r in results]

        # All scores must be in (0, 1]
        assert all(0 < s <= 1 for s in scores)
        # Closer distance must yield higher score
        assert scores[0] > scores[1] > scores[2] > scores[3]
        # Typical L2 distance of 1.25 (semantically related text) must
        # produce a score above 0.4 so recall's default threshold works
        assert (
            scores[2] > 0.4
        ), f"Score {scores[2]} for distance 1.25 is too low for recall"

    def test_zero_distance_gives_max_score(self):
        """Identical vectors (distance 0) must produce relevance 1.0."""
        from pithos.tools.memory_tool import MemoryStore

        store = MagicMock(spec=MemoryStore)
        store.config = {"similarity_threshold": 0.0}
        store._get_collection = MagicMock(
            return_value=self._make_mock_collection([0.0])
        )

        results = MemoryStore.retrieve(
            store, "cat", "query", n_results=5, min_relevance=0.0
        )
        assert results[0].relevance_score == 1.0


# ===========================================================================
# Bug-regression: AutoRecall._search_memory passes min_relevance to retrieve
# ===========================================================================


class TestSearchMemoryPassesMinRelevance:
    """_search_memory must forward RecallConfig.min_relevance to retrieve()."""

    def test_min_relevance_forwarded_to_retrieve(self):
        recall = AutoRecall(
            RecallConfig(sources=["memory"], n_results=5, min_relevance=0.3)
        )
        agent = _make_agent()
        agent.memory_enabled = True
        mock_store = MagicMock()
        mock_store.list_categories.return_value = ["facts"]
        mock_result = MagicMock()
        mock_result.content = "A fact"
        mock_result.relevance_score = (
            0.35  # above 0.3 but below typical config threshold
        )
        mock_store.retrieve.return_value = [mock_result]
        agent.memory_store = mock_store

        snippets = recall._search_memory(agent, "query")

        # retrieve must have been called with min_relevance=0.3
        _args, kwargs = mock_store.retrieve.call_args
        assert kwargs.get("min_relevance") == 0.3

    def test_snippet_returned_when_above_min_relevance(self):
        recall = AutoRecall(
            RecallConfig(sources=["memory"], n_results=5, min_relevance=0.3)
        )
        agent = _make_agent()
        agent.memory_enabled = True
        mock_store = MagicMock()
        mock_store.list_categories.return_value = ["facts"]
        mock_result = MagicMock()
        mock_result.content = "A fact"
        mock_result.relevance_score = 0.35
        mock_store.retrieve.return_value = [mock_result]
        agent.memory_store = mock_store

        snippets = recall._search_memory(agent, "query")
        assert len(snippets) == 1
        assert snippets[0][0] == "A fact"
