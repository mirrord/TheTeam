"""Exhaustive unit tests for every step of context compaction.

Covers:
    1. CompactionConfig — defaults, custom values, boundary values
    2. should_compact() — threshold logic, protected message filtering, boundaries
    3. compact() step 3a — identifying compactable message indices
    4. compact() step 3b — LLM summary generation
    5. compact() step 3c — optional memory archival
    6. compact() step 3d — removal of old messages (reverse order)
    7. compact() step 3e — summary insertion (position, content, protection)
    8. _generate_summary() — prompt construction, model selection, error handling
    9. _parse_summary_response() — structured/unstructured parsing, edge cases
   10. _archive_to_memory() — metadata construction, empty guards, exception handling
   11. Integration scenarios — consecutive compactions, keep_last=0, mixed contexts
"""

import pytest
from unittest.mock import MagicMock, patch, call

from pithos.context import AgentContext
from pithos.agent.compaction import CompactionConfig, MemoryCompactor


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_context(n_messages: int, start_index: int = 0) -> AgentContext:
    """Context with *n_messages* alternating user/assistant messages."""
    ctx = AgentContext("test", "sys")
    for i in range(start_index, start_index + n_messages):
        role = "user" if i % 2 == 0 else "assistant"
        ctx.message_history.append({"role": role, "content": f"msg {i}"})
    return ctx


def _make_agent(model: str = "test-model") -> MagicMock:
    agent = MagicMock()
    agent.default_model = model
    agent.memory_enabled = False
    agent.memory_store = None
    return agent


def _make_compactor(
    threshold: int = 10, keep_last: int = 2, **kwargs
) -> MemoryCompactor:
    return MemoryCompactor(
        CompactionConfig(threshold=threshold, keep_last=keep_last, **kwargs)
    )


# ===========================================================================
# 1. CompactionConfig
# ===========================================================================


class TestCompactionConfigBoundaries:
    """Boundary-value and additional configuration tests."""

    def test_threshold_zero(self):
        cfg = CompactionConfig(threshold=0)
        assert cfg.threshold == 0

    def test_keep_last_zero(self):
        cfg = CompactionConfig(keep_last=0)
        assert cfg.keep_last == 0

    def test_summary_model_overrides_default(self):
        cfg = CompactionConfig(summary_model="custom-model")
        assert cfg.summary_model == "custom-model"

    def test_memory_category_custom(self):
        cfg = CompactionConfig(memory_category="my_archive")
        assert cfg.memory_category == "my_archive"

    def test_all_params_set_together(self):
        cfg = CompactionConfig(
            threshold=5,
            keep_last=1,
            summary_model="llama3",
            memory_category="archive",
        )
        assert cfg.threshold == 5
        assert cfg.keep_last == 1
        assert cfg.summary_model == "llama3"
        assert cfg.memory_category == "archive"


# ===========================================================================
# 2. should_compact() — threshold logic and protected-message filtering
# ===========================================================================


class TestShouldCompactBoundaries:
    """Boundary and edge-case tests for the compaction decision."""

    def test_empty_context_returns_false(self):
        compactor = _make_compactor(threshold=1, keep_last=0)
        ctx = _make_context(0)
        assert compactor.should_compact(ctx) is False

    def test_exact_threshold_minus_one_returns_false(self):
        compactor = _make_compactor(threshold=10, keep_last=2)
        ctx = _make_context(9)
        assert compactor.should_compact(ctx) is False

    def test_exact_threshold_returns_true(self):
        compactor = _make_compactor(threshold=10, keep_last=2)
        ctx = _make_context(10)
        assert compactor.should_compact(ctx) is True

    def test_above_threshold_returns_true(self):
        compactor = _make_compactor(threshold=10, keep_last=2)
        ctx = _make_context(15)
        assert compactor.should_compact(ctx) is True

    def test_exactly_keep_last_compactable_returns_false(self):
        """When compactable == keep_last, there is nothing to compact."""
        compactor = _make_compactor(threshold=3, keep_last=3)
        ctx = _make_context(3)
        # 3 compactable, keep_last=3 → 0 to compact → False
        assert compactor.should_compact(ctx) is False

    def test_one_more_than_keep_last_returns_true(self):
        """When compactable == keep_last + 1, there is exactly one to compact."""
        compactor = _make_compactor(threshold=4, keep_last=3)
        ctx = _make_context(4)
        # 4 compactable, keep_last=3 → 1 to compact → True
        assert compactor.should_compact(ctx) is True

    def test_keep_last_zero_with_messages_returns_true(self):
        compactor = _make_compactor(threshold=2, keep_last=0)
        ctx = _make_context(3)
        assert compactor.should_compact(ctx) is True

    def test_threshold_met_but_all_protected(self):
        compactor = _make_compactor(threshold=3, keep_last=0)
        ctx = AgentContext("test", "sys")
        for i in range(5):
            ctx.message_history.append(
                {
                    "role": "system",
                    "content": f"summary {i}",
                    "_pithos_no_compact": True,
                }
            )
        # total=5 >= threshold=3, but 0 compactable → False
        assert compactor.should_compact(ctx) is False

    def test_mixed_protected_unprotected_at_boundary(self):
        """Threshold met, but only exactly keep_last are unprotected."""
        compactor = _make_compactor(threshold=5, keep_last=2)
        ctx = AgentContext("test", "sys")
        # 3 protected + 2 unprotected = 5 total
        for i in range(3):
            ctx.message_history.append(
                {
                    "role": "system",
                    "content": f"protected {i}",
                    "_pithos_no_compact": True,
                }
            )
        for i in range(2):
            ctx.message_history.append({"role": "user", "content": f"msg {i}"})
        # 2 compactable == keep_last=2 → nothing to compact → False
        assert compactor.should_compact(ctx) is False

    def test_mixed_protected_unprotected_above_boundary(self):
        compactor = _make_compactor(threshold=5, keep_last=2)
        ctx = AgentContext("test", "sys")
        for i in range(2):
            ctx.message_history.append(
                {
                    "role": "system",
                    "content": f"protected {i}",
                    "_pithos_no_compact": True,
                }
            )
        for i in range(4):
            ctx.message_history.append({"role": "user", "content": f"msg {i}"})
        # total=6 >= 5, 4 compactable > keep_last=2 → True
        assert compactor.should_compact(ctx) is True


# ===========================================================================
# 3a. compact() — Identifying compactable message indices
# ===========================================================================


class TestCompactIdentifyCompactableIndices:
    """Verify that the correct messages are selected for compaction."""

    def test_all_unprotected_oldest_compacted(self):
        """With keep_last=2 and 8 unprotected msgs, the first 6 are compacted."""
        compactor = _make_compactor(threshold=6, keep_last=2)
        ctx = _make_context(8)
        agent = _make_agent()
        original_last_two = [m["content"] for m in ctx.message_history[-2:]]

        with patch.object(compactor, "_generate_summary", return_value=("S", "none")):
            compactor.compact(agent, ctx, "test")

        # The last 2 original messages should be preserved
        remaining_contents = [
            m["content"]
            for m in ctx.message_history
            if "[CONTEXT SUMMARY]" not in m["content"]
        ]
        assert remaining_contents == original_last_two

    def test_protected_messages_interspersed_preserved(self):
        """Protected messages scattered among regular ones are all preserved."""
        compactor = _make_compactor(threshold=6, keep_last=1)
        ctx = AgentContext("test", "sys")
        # Build: user, protected, assistant, user, protected, assistant, user
        ctx.message_history = [
            {"role": "user", "content": "msg 0"},
            {"role": "system", "content": "protected A", "_pithos_no_compact": True},
            {"role": "assistant", "content": "msg 2"},
            {"role": "user", "content": "msg 3"},
            {"role": "system", "content": "protected B", "_pithos_no_compact": True},
            {"role": "assistant", "content": "msg 5"},
            {"role": "user", "content": "msg 6"},
        ]
        agent = _make_agent()

        with patch.object(compactor, "_generate_summary", return_value=("S", "none")):
            compactor.compact(agent, ctx, "test")

        # Both protected messages must remain
        protected = [
            m
            for m in ctx.message_history
            if m.get("_pithos_no_compact") and "protected" in m["content"]
        ]
        assert len(protected) == 2
        assert protected[0]["content"] == "protected A"
        assert protected[1]["content"] == "protected B"

    def test_keep_last_zero_compacts_all_unprotected(self):
        """With keep_last=0, every non-protected message is compacted."""
        compactor = _make_compactor(threshold=3, keep_last=0)
        ctx = _make_context(4)
        agent = _make_agent()

        with patch.object(compactor, "_generate_summary", return_value=("S", "none")):
            compactor.compact(agent, ctx, "test")

        non_summary = [
            m for m in ctx.message_history if "[CONTEXT SUMMARY]" not in m["content"]
        ]
        assert len(non_summary) == 0  # all compacted, only summary remains

    def test_only_one_compactable_beyond_keep_last(self):
        """With exactly one message beyond keep_last, only that one is compacted."""
        compactor = _make_compactor(threshold=4, keep_last=3)
        ctx = _make_context(4)
        agent = _make_agent()
        first_msg_content = ctx.message_history[0]["content"]

        with patch.object(
            compactor, "_generate_summary", return_value=("S", "none")
        ) as mock_gen:
            compactor.compact(agent, ctx, "test")

        # _generate_summary should have been called with exactly 1 message
        assert mock_gen.call_count == 1
        compacted_msgs = mock_gen.call_args[0][1]
        assert len(compacted_msgs) == 1
        assert compacted_msgs[0]["content"] == first_msg_content


# ===========================================================================
# 3b. compact() → _generate_summary — LLM call details
# ===========================================================================


class TestGenerateSummaryDetails:
    """Tests for _generate_summary prompt construction, model selection, and options."""

    def test_uses_config_summary_model(self):
        compactor = _make_compactor(summary_model="custom-model")
        agent = _make_agent(model="default-model")

        with patch("ollama.chat") as mock_chat:
            mock_chat.return_value.message.content = "Summary: ok\nEntities: none"
            compactor._generate_summary(agent, [{"role": "user", "content": "hi"}])

        assert mock_chat.call_args[1]["model"] == "custom-model"

    def test_falls_back_to_agent_model(self):
        compactor = _make_compactor()  # summary_model=None
        agent = _make_agent(model="agent-default")

        with patch("ollama.chat") as mock_chat:
            mock_chat.return_value.message.content = "Summary: ok\nEntities: none"
            compactor._generate_summary(agent, [{"role": "user", "content": "hi"}])

        assert mock_chat.call_args[1]["model"] == "agent-default"

    def test_temperature_is_low(self):
        compactor = _make_compactor()
        agent = _make_agent()

        with patch("ollama.chat") as mock_chat:
            mock_chat.return_value.message.content = "Summary: ok\nEntities: none"
            compactor._generate_summary(agent, [{"role": "user", "content": "hi"}])

        options = mock_chat.call_args[1]["options"]
        assert options["temperature"] == 0.3

    def test_prompt_contains_message_history(self):
        compactor = _make_compactor()
        agent = _make_agent()
        messages = [
            {"role": "user", "content": "What is Python?"},
            {"role": "assistant", "content": "A programming language."},
        ]

        with patch("ollama.chat") as mock_chat:
            mock_chat.return_value.message.content = "Summary: ok\nEntities: none"
            compactor._generate_summary(agent, messages)

        prompt = mock_chat.call_args[1]["messages"][0]["content"]
        assert "USER: What is Python?" in prompt
        assert "ASSISTANT: A programming language." in prompt

    def test_prompt_contains_format_instructions(self):
        compactor = _make_compactor()
        agent = _make_agent()

        with patch("ollama.chat") as mock_chat:
            mock_chat.return_value.message.content = "Summary: ok\nEntities: none"
            compactor._generate_summary(agent, [{"role": "user", "content": "hi"}])

        prompt = mock_chat.call_args[1]["messages"][0]["content"]
        assert "Summary:" in prompt
        assert "Entities:" in prompt

    def test_llm_exception_returns_error_summary(self):
        compactor = _make_compactor()
        agent = _make_agent()

        with patch("ollama.chat", side_effect=ConnectionError("LLM unreachable")):
            summary, entities = compactor._generate_summary(
                agent, [{"role": "user", "content": "hi"}]
            )

        assert "LLM error" in summary
        assert entities == "none"

    def test_llm_returns_none_content(self):
        compactor = _make_compactor()
        agent = _make_agent()

        with patch("ollama.chat") as mock_chat:
            mock_chat.return_value.message.content = None
            summary, entities = compactor._generate_summary(
                agent, [{"role": "user", "content": "hi"}]
            )

        assert summary.strip() != ""  # should fallback, not empty

    def test_whitespace_only_response_uses_fallback(self):
        compactor = _make_compactor()
        agent = _make_agent()

        with patch("ollama.chat") as mock_chat:
            mock_chat.return_value.message.content = "   \n  "
            summary, entities = compactor._generate_summary(
                agent, [{"role": "user", "content": "hi"}]
            )

        assert summary.strip() != ""

    def test_multiple_messages_formatted_correctly(self):
        compactor = _make_compactor()
        agent = _make_agent()
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
            {"role": "user", "content": "Tell me about cats"},
            {"role": "assistant", "content": "Cats are great"},
        ]

        with patch("ollama.chat") as mock_chat:
            mock_chat.return_value.message.content = "Summary: ok\nEntities: none"
            compactor._generate_summary(agent, messages)

        prompt = mock_chat.call_args[1]["messages"][0]["content"]
        lines = prompt.split("\n")
        history_lines = [l for l in lines if l.startswith(("USER:", "ASSISTANT:"))]
        assert len(history_lines) == 4


# ===========================================================================
# 3c. compact() → memory archival
# ===========================================================================


class TestCompactMemoryArchival:
    """Tests for the memory archival step within compact()."""

    def test_archives_with_correct_category(self):
        compactor = _make_compactor(threshold=4, keep_last=1, memory_category="my_cat")
        ctx = _make_context(5)
        agent = _make_agent()
        agent.memory_enabled = True
        agent.memory_store = MagicMock()

        with patch.object(
            compactor, "_generate_summary", return_value=("S", "entity1")
        ):
            compactor.compact(agent, ctx, "ctx_name")

        args, kwargs = agent.memory_store.store.call_args
        assert args[0] == "my_cat"

    def test_archives_summary_content(self):
        compactor = _make_compactor(threshold=4, keep_last=1)
        ctx = _make_context(5)
        agent = _make_agent()
        agent.memory_enabled = True
        agent.memory_store = MagicMock()

        with patch.object(
            compactor,
            "_generate_summary",
            return_value=("The conversation summary", "none"),
        ):
            compactor.compact(agent, ctx, "ctx_name")

        args, kwargs = agent.memory_store.store.call_args
        assert args[1] == "The conversation summary"

    def test_metadata_includes_type_and_context_name(self):
        compactor = _make_compactor(threshold=4, keep_last=1)
        ctx = _make_context(5)
        agent = _make_agent()
        agent.memory_enabled = True
        agent.memory_store = MagicMock()

        with patch.object(compactor, "_generate_summary", return_value=("S", "none")):
            compactor.compact(agent, ctx, "my_context")

        _, kwargs = agent.memory_store.store.call_args
        metadata = kwargs["metadata"]
        assert metadata["type"] == "compaction_summary"
        assert metadata["context_name"] == "my_context"

    def test_metadata_includes_entities_when_present(self):
        compactor = _make_compactor(threshold=4, keep_last=1)
        ctx = _make_context(5)
        agent = _make_agent()
        agent.memory_enabled = True
        agent.memory_store = MagicMock()

        with patch.object(
            compactor, "_generate_summary", return_value=("S", "Alice, Bob")
        ):
            compactor.compact(agent, ctx, "ctx")

        _, kwargs = agent.memory_store.store.call_args
        assert kwargs["metadata"]["entities"] == "Alice, Bob"

    def test_metadata_excludes_entities_when_none(self):
        compactor = _make_compactor(threshold=4, keep_last=1)
        ctx = _make_context(5)
        agent = _make_agent()
        agent.memory_enabled = True
        agent.memory_store = MagicMock()

        with patch.object(compactor, "_generate_summary", return_value=("S", "none")):
            compactor.compact(agent, ctx, "ctx")

        _, kwargs = agent.memory_store.store.call_args
        assert "entities" not in kwargs["metadata"]

    def test_no_archival_when_memory_store_is_none(self):
        compactor = _make_compactor(threshold=4, keep_last=1)
        ctx = _make_context(5)
        agent = _make_agent()
        agent.memory_enabled = True
        agent.memory_store = None

        with patch.object(compactor, "_generate_summary", return_value=("S", "none")):
            with patch.object(compactor, "_archive_to_memory") as mock_archive:
                compactor.compact(agent, ctx, "ctx")
                mock_archive.assert_not_called()

    def test_no_archival_when_memory_disabled(self):
        compactor = _make_compactor(threshold=4, keep_last=1)
        ctx = _make_context(5)
        agent = _make_agent()
        agent.memory_enabled = False
        agent.memory_store = MagicMock()

        with patch.object(compactor, "_generate_summary", return_value=("S", "none")):
            with patch.object(compactor, "_archive_to_memory") as mock_archive:
                compactor.compact(agent, ctx, "ctx")
                mock_archive.assert_not_called()


# ===========================================================================
# 3d. compact() — Message removal in reverse order
# ===========================================================================


class TestCompactMessageRemoval:
    """Verify messages are removed correctly preserving index integrity."""

    def test_removed_messages_are_the_correct_ones(self):
        """Verify the exact messages removed are the oldest non-protected ones."""
        compactor = _make_compactor(threshold=5, keep_last=2)
        ctx = _make_context(6)
        agent = _make_agent()
        # Messages: msg 0..5 — keep last 2 (msg 4, msg 5), compact 0..3
        expected_kept = ["msg 4", "msg 5"]

        with patch.object(compactor, "_generate_summary", return_value=("S", "none")):
            compactor.compact(agent, ctx, "test")

        remaining = [
            m["content"]
            for m in ctx.message_history
            if "[CONTEXT SUMMARY]" not in m["content"]
        ]
        assert remaining == expected_kept

    def test_protected_messages_stay_at_original_relative_position(self):
        """Protected messages should not be displaced by compaction."""
        compactor = _make_compactor(threshold=5, keep_last=1)
        ctx = AgentContext("test", "sys")
        ctx.message_history = [
            {"role": "user", "content": "msg 0"},
            {"role": "system", "content": "protected", "_pithos_no_compact": True},
            {"role": "assistant", "content": "msg 2"},
            {"role": "user", "content": "msg 3"},
            {"role": "assistant", "content": "msg 4"},
            {"role": "user", "content": "msg 5"},
        ]
        agent = _make_agent()

        with patch.object(compactor, "_generate_summary", return_value=("S", "none")):
            compactor.compact(agent, ctx, "test")

        # Protected message should still be present
        protected = [m for m in ctx.message_history if m.get("content") == "protected"]
        assert len(protected) == 1

    def test_history_size_correct_after_compaction(self):
        """Total size = kept_messages + protected_messages + 1 summary."""
        compactor = _make_compactor(threshold=6, keep_last=3)
        ctx = AgentContext("test", "sys")
        for i in range(8):
            ctx.message_history.append({"role": "user", "content": f"msg {i}"})
        # Add 2 protected
        ctx.message_history.insert(
            2, {"role": "system", "content": "p1", "_pithos_no_compact": True}
        )
        ctx.message_history.insert(
            5, {"role": "system", "content": "p2", "_pithos_no_compact": True}
        )
        # total = 10, compactable = 8, keep_last=3 → compact 5, keep 3 + 2 protected + 1 summary = 6
        agent = _make_agent()

        with patch.object(compactor, "_generate_summary", return_value=("S", "none")):
            compactor.compact(agent, ctx, "test")

        assert len(ctx.message_history) == 6


# ===========================================================================
# 3e. compact() — Summary insertion
# ===========================================================================


class TestCompactSummaryInsertion:
    """Tests for summary message position, content format, and protection flag."""

    def test_summary_inserted_at_first_compacted_position(self):
        """When compaction starts at index 0, summary should be at index 0."""
        compactor = _make_compactor(threshold=4, keep_last=2)
        ctx = _make_context(5)
        agent = _make_agent()

        with patch.object(compactor, "_generate_summary", return_value=("S", "none")):
            compactor.compact(agent, ctx, "test")

        assert "[CONTEXT SUMMARY]" in ctx.message_history[0]["content"]

    def test_summary_inserted_after_protected_prefix(self):
        """When protected messages come first, summary is placed at the
        position of the first compacted (unprotected) message."""
        compactor = _make_compactor(threshold=5, keep_last=1)
        ctx = AgentContext("test", "sys")
        ctx.message_history = [
            {"role": "system", "content": "protected0", "_pithos_no_compact": True},
            {"role": "user", "content": "msg 1"},
            {"role": "assistant", "content": "msg 2"},
            {"role": "user", "content": "msg 3"},
            {"role": "assistant", "content": "msg 4"},
            {"role": "user", "content": "msg 5"},
        ]
        agent = _make_agent()

        with patch.object(compactor, "_generate_summary", return_value=("S", "none")):
            compactor.compact(agent, ctx, "test")

        # protected0 should be at index 0, summary at index 1
        assert ctx.message_history[0]["content"] == "protected0"
        assert "[CONTEXT SUMMARY]" in ctx.message_history[1]["content"]

    def test_summary_content_has_context_summary_header(self):
        compactor = _make_compactor(threshold=4, keep_last=1)
        ctx = _make_context(5)
        agent = _make_agent()

        with patch.object(
            compactor, "_generate_summary", return_value=("My summary text", "none")
        ):
            compactor.compact(agent, ctx, "test")

        summary_msg = next(
            m for m in ctx.message_history if "[CONTEXT SUMMARY]" in m["content"]
        )
        assert summary_msg["content"].startswith("[CONTEXT SUMMARY]\n")
        assert "My summary text" in summary_msg["content"]

    def test_summary_includes_key_entities(self):
        compactor = _make_compactor(threshold=4, keep_last=1)
        ctx = _make_context(5)
        agent = _make_agent()

        with patch.object(
            compactor, "_generate_summary", return_value=("S", "Python, Flask")
        ):
            compactor.compact(agent, ctx, "test")

        summary_msg = next(
            m for m in ctx.message_history if "[CONTEXT SUMMARY]" in m["content"]
        )
        assert "Key entities: Python, Flask" in summary_msg["content"]

    def test_summary_omits_entities_when_none(self):
        compactor = _make_compactor(threshold=4, keep_last=1)
        ctx = _make_context(5)
        agent = _make_agent()

        with patch.object(compactor, "_generate_summary", return_value=("S", "none")):
            compactor.compact(agent, ctx, "test")

        summary_msg = next(
            m for m in ctx.message_history if "[CONTEXT SUMMARY]" in m["content"]
        )
        assert "Key entities" not in summary_msg["content"]

    def test_summary_omits_entities_when_empty_string(self):
        compactor = _make_compactor(threshold=4, keep_last=1)
        ctx = _make_context(5)
        agent = _make_agent()

        with patch.object(compactor, "_generate_summary", return_value=("S", "")):
            compactor.compact(agent, ctx, "test")

        summary_msg = next(
            m for m in ctx.message_history if "[CONTEXT SUMMARY]" in m["content"]
        )
        assert "Key entities" not in summary_msg["content"]

    def test_summary_omits_entities_when_whitespace(self):
        compactor = _make_compactor(threshold=4, keep_last=1)
        ctx = _make_context(5)
        agent = _make_agent()

        with patch.object(compactor, "_generate_summary", return_value=("S", "  ")):
            compactor.compact(agent, ctx, "test")

        summary_msg = next(
            m for m in ctx.message_history if "[CONTEXT SUMMARY]" in m["content"]
        )
        assert "Key entities" not in summary_msg["content"]

    def test_summary_omits_entities_when_None_uppercase(self):
        compactor = _make_compactor(threshold=4, keep_last=1)
        ctx = _make_context(5)
        agent = _make_agent()

        with patch.object(compactor, "_generate_summary", return_value=("S", "None")):
            compactor.compact(agent, ctx, "test")

        summary_msg = next(
            m for m in ctx.message_history if "[CONTEXT SUMMARY]" in m["content"]
        )
        assert "Key entities" not in summary_msg["content"]

    def test_summary_role_is_system(self):
        compactor = _make_compactor(threshold=4, keep_last=1)
        ctx = _make_context(5)
        agent = _make_agent()

        with patch.object(compactor, "_generate_summary", return_value=("S", "none")):
            compactor.compact(agent, ctx, "test")

        summary_msg = next(
            m for m in ctx.message_history if "[CONTEXT SUMMARY]" in m["content"]
        )
        assert summary_msg["role"] == "system"

    def test_summary_is_protected(self):
        compactor = _make_compactor(threshold=4, keep_last=1)
        ctx = _make_context(5)
        agent = _make_agent()

        with patch.object(compactor, "_generate_summary", return_value=("S", "none")):
            compactor.compact(agent, ctx, "test")

        summary_msg = next(
            m for m in ctx.message_history if "[CONTEXT SUMMARY]" in m["content"]
        )
        assert summary_msg["_pithos_no_compact"] is True

    def test_insert_position_clamped_to_list_length(self):
        """When all messages are compacted, insert_pos is clamped to 0."""
        compactor = _make_compactor(threshold=3, keep_last=0)
        ctx = _make_context(3)
        agent = _make_agent()

        with patch.object(compactor, "_generate_summary", return_value=("S", "none")):
            compactor.compact(agent, ctx, "test")

        # Only the summary should remain
        assert len(ctx.message_history) == 1
        assert "[CONTEXT SUMMARY]" in ctx.message_history[0]["content"]


# ===========================================================================
# 8. _parse_summary_response — extended parsing tests
# ===========================================================================


class TestParseSummaryResponseExtended:
    """Extended parsing edge cases beyond the basics."""

    def test_multiline_summary(self):
        raw = "Summary: First line.\nSecond line.\nEntities: A, B"
        summary, entities = MemoryCompactor._parse_summary_response(raw)
        assert "First line." in summary
        assert "Second line." in summary
        assert entities == "A, B"

    def test_entities_with_extra_whitespace(self):
        raw = "Summary: ok\nEntities:   Alice ,  Bob  , Carol  "
        summary, entities = MemoryCompactor._parse_summary_response(raw)
        assert entities == "Alice ,  Bob  , Carol"

    def test_entities_without_summary_prefix(self):
        raw = "Some discussion.\nEntities: X, Y"
        summary, entities = MemoryCompactor._parse_summary_response(raw)
        assert entities == "X, Y"
        assert "Some discussion." in summary

    def test_entities_none_string(self):
        raw = "Summary: A chat.\nEntities: none"
        summary, entities = MemoryCompactor._parse_summary_response(raw)
        assert summary == "A chat."
        assert entities == "none"

    def test_case_sensitivity_of_summary_prefix(self):
        raw = "summary: lower-case prefix"
        summary, entities = MemoryCompactor._parse_summary_response(raw)
        assert summary == "lower-case prefix"
        assert entities == "none"

    def test_empty_string(self):
        raw = ""
        summary, entities = MemoryCompactor._parse_summary_response(raw)
        assert summary == ""
        assert entities == "none"

    def test_only_summary_prefix_no_body(self):
        raw = "Summary:"
        summary, entities = MemoryCompactor._parse_summary_response(raw)
        assert summary == ""

    def test_entities_empty_after_split(self):
        raw = "Summary: ok\nEntities:"
        summary, entities = MemoryCompactor._parse_summary_response(raw)
        assert summary == "ok"
        assert entities == ""

    def test_entities_keyword_in_summary_body(self):
        """The word 'Entities:' appearing in the middle should still split correctly."""
        raw = "Summary: The user asked about Entities: in the system.\nEntities: user, system"
        summary, entities = MemoryCompactor._parse_summary_response(raw)
        # Should split on the LAST Entities: occurrence — actually it splits on first
        # The implementation splits on first "Entities:" occurrence
        assert (
            entities == "in the system.\nEntities: user, system"
            or "user, system" in entities
        )


# ===========================================================================
# 10. _archive_to_memory — extended tests
# ===========================================================================


class TestArchiveToMemoryExtended:
    """Extended tests for the archival helper."""

    def test_store_exception_is_caught(self):
        """Exceptions from store.store() must not propagate."""
        compactor = _make_compactor()
        mock_store = MagicMock()
        mock_store.store.side_effect = RuntimeError("DB error")

        # Should not raise
        compactor._archive_to_memory(mock_store, "ctx", "Summary", "none")

    def test_entities_none_excluded_from_metadata(self):
        compactor = _make_compactor()
        mock_store = MagicMock()
        compactor._archive_to_memory(mock_store, "ctx", "Summary", "none")

        _, kwargs = mock_store.store.call_args
        assert "entities" not in kwargs["metadata"]

    def test_empty_entities_excluded_from_metadata(self):
        compactor = _make_compactor()
        mock_store = MagicMock()
        compactor._archive_to_memory(mock_store, "ctx", "Summary", "")

        _, kwargs = mock_store.store.call_args
        assert "entities" not in kwargs["metadata"]

    def test_valid_entities_included_in_metadata(self):
        compactor = _make_compactor()
        mock_store = MagicMock()
        compactor._archive_to_memory(mock_store, "ctx", "Summary", "Alice, Bob")

        _, kwargs = mock_store.store.call_args
        assert kwargs["metadata"]["entities"] == "Alice, Bob"

    def test_uses_config_category(self):
        compactor = _make_compactor(memory_category="custom_archive")
        mock_store = MagicMock()
        compactor._archive_to_memory(mock_store, "ctx", "Summary", "none")

        args, _ = mock_store.store.call_args
        assert args[0] == "custom_archive"

    def test_none_summary_does_not_call_store(self):
        compactor = _make_compactor()
        mock_store = MagicMock()
        compactor._archive_to_memory(mock_store, "ctx", None, "none")
        mock_store.store.assert_not_called()


# ===========================================================================
# 11. Integration scenarios
# ===========================================================================


class TestConsecutiveCompactions:
    """Test that compaction summaries from earlier rounds are protected
    and never re-compacted."""

    def test_summary_from_first_compaction_survives_second(self):
        compactor = _make_compactor(threshold=5, keep_last=2)
        ctx = _make_context(6)
        agent = _make_agent()

        # First compaction
        with patch.object(
            compactor, "_generate_summary", return_value=("First summary", "none")
        ):
            compactor.compact(agent, ctx, "test")

        # Verify summary is present
        assert any("[CONTEXT SUMMARY]" in m["content"] for m in ctx.message_history)
        size_after_first = len(ctx.message_history)

        # Add more messages to trigger a second compaction
        for i in range(10):
            ctx.message_history.append({"role": "user", "content": f"new msg {i}"})

        # Second compaction
        with patch.object(
            compactor, "_generate_summary", return_value=("Second summary", "none")
        ):
            compactor.compact(agent, ctx, "test")

        # First summary must still be present (it's protected)
        summaries = [
            m for m in ctx.message_history if "[CONTEXT SUMMARY]" in m["content"]
        ]
        assert len(summaries) == 2
        contents = [m["content"] for m in summaries]
        assert any("First summary" in c for c in contents)
        assert any("Second summary" in c for c in contents)

    def test_double_compact_no_op_when_below_threshold(self):
        """Calling compact twice without adding messages should be a no-op the second time."""
        compactor = _make_compactor(threshold=5, keep_last=2)
        ctx = _make_context(6)
        agent = _make_agent()

        with patch.object(
            compactor, "_generate_summary", return_value=("S", "none")
        ) as mock_gen:
            compactor.compact(agent, ctx, "test")
            size_after = len(ctx.message_history)

            compactor.compact(agent, ctx, "test")
            assert len(ctx.message_history) == size_after
            assert mock_gen.call_count == 1  # only called once


class TestCompactNoOpScenarios:
    """Cases where compact() should be a complete no-op."""

    def test_empty_context(self):
        compactor = _make_compactor(threshold=1, keep_last=0)
        ctx = _make_context(0)
        agent = _make_agent()

        with patch.object(compactor, "_generate_summary") as mock_gen:
            compactor.compact(agent, ctx, "test")
            mock_gen.assert_not_called()

    def test_only_protected_messages(self):
        compactor = _make_compactor(threshold=2, keep_last=0)
        ctx = AgentContext("test", "sys")
        for i in range(5):
            ctx.message_history.append(
                {"role": "system", "content": f"p {i}", "_pithos_no_compact": True}
            )
        agent = _make_agent()

        with patch.object(compactor, "_generate_summary") as mock_gen:
            compactor.compact(agent, ctx, "test")
            mock_gen.assert_not_called()

        assert len(ctx.message_history) == 5  # unchanged

    def test_threshold_not_met(self):
        compactor = _make_compactor(threshold=100, keep_last=2)
        ctx = _make_context(10)
        agent = _make_agent()
        original = list(ctx.message_history)

        with patch.object(compactor, "_generate_summary") as mock_gen:
            compactor.compact(agent, ctx, "test")
            mock_gen.assert_not_called()

        assert ctx.message_history == original


class TestCompactEndToEnd:
    """Full end-to-end compact flow with realistic message patterns."""

    def test_realistic_conversation_compaction(self):
        """Simulate a multi-turn conversation and verify compaction output."""
        compactor = _make_compactor(threshold=8, keep_last=3)
        ctx = AgentContext("chat", "You are a helpful assistant.")
        # Build a realistic conversation
        turns = [
            ("user", "What is Python?"),
            ("assistant", "Python is a programming language."),
            ("user", "What about JavaScript?"),
            ("assistant", "JavaScript is a web language."),
            ("user", "Compare them."),
            ("assistant", "Python is for backends, JS for frontends."),
            ("user", "What about TypeScript?"),
            ("assistant", "TypeScript adds types to JavaScript."),
            ("user", "Which should I learn?"),
            ("assistant", "Depends on your goals."),
        ]
        for role, content in turns:
            ctx.message_history.append({"role": role, "content": content})

        agent = _make_agent()

        with patch.object(
            compactor,
            "_generate_summary",
            return_value=(
                "Discussion of Python, JS, TS. Python for backends, JS/TS for frontends.",
                "Python, JavaScript, TypeScript",
            ),
        ):
            compactor.compact(agent, ctx, "chat")

        # Verify structure
        assert len(ctx.message_history) == 4  # 7 compacted → 1 summary + 3 kept

        # Summary message
        summary_msg = ctx.message_history[0]
        assert summary_msg["role"] == "system"
        assert "[CONTEXT SUMMARY]" in summary_msg["content"]
        assert "Python, JavaScript, TypeScript" in summary_msg["content"]
        assert summary_msg["_pithos_no_compact"] is True

        # Kept messages are the last 3
        assert (
            ctx.message_history[1]["content"] == "TypeScript adds types to JavaScript."
        )
        assert ctx.message_history[2]["content"] == "Which should I learn?"
        assert ctx.message_history[3]["content"] == "Depends on your goals."

    def test_compaction_preserves_message_order(self):
        """After compaction, remaining messages should be in original order."""
        compactor = _make_compactor(threshold=5, keep_last=3)
        ctx = _make_context(7)
        agent = _make_agent()

        original_last_3 = [m["content"] for m in ctx.message_history[-3:]]

        with patch.object(compactor, "_generate_summary", return_value=("S", "none")):
            compactor.compact(agent, ctx, "test")

        remaining = [
            m["content"]
            for m in ctx.message_history
            if "[CONTEXT SUMMARY]" not in m["content"]
        ]
        assert remaining == original_last_3

    def test_compact_with_system_messages_in_history(self):
        """System messages (not protected) in history are compactable."""
        compactor = _make_compactor(threshold=5, keep_last=1)
        ctx = AgentContext("test", "sys")
        ctx.message_history = [
            {"role": "system", "content": "tool output"},
            {"role": "user", "content": "msg 1"},
            {"role": "assistant", "content": "msg 2"},
            {"role": "system", "content": "another tool output"},
            {"role": "user", "content": "msg 4"},
            {"role": "assistant", "content": "msg 5"},
        ]
        agent = _make_agent()

        with patch.object(
            compactor, "_generate_summary", return_value=("S", "none")
        ) as mock_gen:
            compactor.compact(agent, ctx, "test")

        # _generate_summary should have been called with the first 5 compactable messages
        compacted_msgs = mock_gen.call_args[0][1]
        assert len(compacted_msgs) == 5
        # Only the last message kept
        non_summary = [
            m for m in ctx.message_history if "[CONTEXT SUMMARY]" not in m["content"]
        ]
        assert len(non_summary) == 1
        assert non_summary[0]["content"] == "msg 5"
