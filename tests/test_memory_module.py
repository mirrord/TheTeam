"""Unit tests for MemoryModule — the context management callback system."""

import pytest
from unittest.mock import MagicMock, call, patch

from pithos.agent.memory import MemoryModule
from pithos.agent.compaction import CompactionConfig, MemoryCompactor
from pithos.agent.recall import RecallConfig, AutoRecall

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_agent(memory_store=None):
    """Return a minimal mock agent with the attributes MemoryModule needs."""
    agent = MagicMock()
    agent.memory_store = memory_store
    agent.metrics = None
    agent.contexts = {}
    return agent


def _make_mock_context():
    ctx = MagicMock()
    ctx.get_system_prompt.return_value = "You are helpful."
    return ctx


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestMemoryModuleConstruction:
    def test_default_construction(self):
        mod = MemoryModule()
        assert mod.recall_enabled is False
        assert mod.compaction_enabled is False
        assert mod._auto_recall is None
        assert mod._compactor is None

    def test_with_recall_config(self):
        cfg = RecallConfig(n_results=3)
        mod = MemoryModule(recall_config=cfg)
        assert mod.recall_enabled is True
        assert isinstance(mod._auto_recall, AutoRecall)
        assert mod._auto_recall.config.n_results == 3

    def test_with_compaction_config(self):
        cfg = CompactionConfig(threshold=5)
        mod = MemoryModule(compaction_config=cfg)
        assert mod.compaction_enabled is True
        assert isinstance(mod._compactor, MemoryCompactor)
        assert mod._compactor.config.threshold == 5

    def test_with_both_configs(self):
        mod = MemoryModule(
            recall_config=RecallConfig(),
            compaction_config=CompactionConfig(),
        )
        assert mod.recall_enabled is True
        assert mod.compaction_enabled is True


# ---------------------------------------------------------------------------
# Enable / disable
# ---------------------------------------------------------------------------


class TestMemoryModuleEnableDisable:
    def test_enable_recall_default_config(self):
        mod = MemoryModule()
        mod.enable_recall()
        assert mod.recall_enabled is True
        assert isinstance(mod._auto_recall, AutoRecall)

    def test_enable_recall_custom_config(self):
        mod = MemoryModule()
        mod.enable_recall(RecallConfig(n_results=7))
        assert mod._auto_recall.config.n_results == 7

    def test_disable_recall(self):
        mod = MemoryModule(recall_config=RecallConfig())
        mod.disable_recall()
        assert mod.recall_enabled is False
        assert mod._auto_recall is None

    def test_enable_compaction_default_config(self):
        mod = MemoryModule()
        mod.enable_compaction()
        assert mod.compaction_enabled is True
        assert isinstance(mod._compactor, MemoryCompactor)

    def test_enable_compaction_custom_config(self):
        mod = MemoryModule()
        mod.enable_compaction(CompactionConfig(threshold=10, keep_last=2))
        assert mod._compactor.config.threshold == 10

    def test_disable_compaction(self):
        mod = MemoryModule(compaction_config=CompactionConfig())
        mod.disable_compaction()
        assert mod.compaction_enabled is False
        assert mod._compactor is None


# ---------------------------------------------------------------------------
# inject_memory_prompt
# ---------------------------------------------------------------------------


class TestInjectMemoryPrompt:
    def test_no_op_when_memory_store_absent(self):
        mod = MemoryModule()
        agent = _make_mock_agent(memory_store=None)
        # inject_memory_prompt returns early when memory_store is None;
        # no context system prompts should be touched.
        agent.contexts = {}  # plain empty dict, not a Mock attribute
        mod.inject_memory_prompt(agent)
        # No contexts exist, so no set_system_prompt calls can have been made.

    def test_appends_prompt_to_contexts(self):
        mod = MemoryModule()

        mock_store = MagicMock()
        mock_store.list_categories.return_value = ["facts", "notes"]

        agent = _make_mock_agent(memory_store=mock_store)

        ctx = MagicMock()
        ctx.get_system_prompt.return_value = "Base prompt."
        agent.contexts = {"default": ctx}

        mod.inject_memory_prompt(agent)

        set_calls = ctx.set_system_prompt.call_args_list
        assert len(set_calls) == 1
        injected = set_calls[0][0][0]
        assert "knowledge memory system" in injected
        assert "Base prompt." in injected

    def test_does_not_duplicate_if_already_present(self):
        mod = MemoryModule()

        mock_store = MagicMock()
        mock_store.list_categories.return_value = []

        agent = _make_mock_agent(memory_store=mock_store)

        ctx = MagicMock()
        ctx.get_system_prompt.return_value = (
            "You have access to a knowledge memory system already injected."
        )
        agent.contexts = {"default": ctx}

        mod.inject_memory_prompt(agent)

        ctx.set_system_prompt.assert_not_called()

    def test_handles_list_categories_exception(self):
        mod = MemoryModule()

        mock_store = MagicMock()
        mock_store.list_categories.side_effect = RuntimeError("db error")

        agent = _make_mock_agent(memory_store=mock_store)
        ctx = MagicMock()
        ctx.get_system_prompt.return_value = ""
        agent.contexts = {"default": ctx}

        # Should not raise.
        mod.inject_memory_prompt(agent)
        # Prompt is still injected.
        assert ctx.set_system_prompt.called


# ---------------------------------------------------------------------------
# before_send
# ---------------------------------------------------------------------------


class TestBeforeSend:
    def test_noop_when_recall_disabled(self):
        mod = MemoryModule()
        agent = _make_mock_agent()
        ctx = _make_mock_context()
        # Should not raise and recall is never called.
        mod.before_send(agent, ctx, "hello", None)

    def test_calls_inject_recall_when_enabled(self):
        mod = MemoryModule(recall_config=RecallConfig())
        mock_recall = MagicMock(spec=AutoRecall)
        mod._auto_recall = mock_recall

        agent = _make_mock_agent()
        ctx = _make_mock_context()
        mod.before_send(agent, ctx, "test message", "llama3")

        mock_recall.inject_recall.assert_called_once_with(
            agent=agent, context=ctx, content="test message", model="llama3"
        )

    def test_swallows_recall_exception(self):
        mod = MemoryModule(recall_config=RecallConfig())
        mock_recall = MagicMock(spec=AutoRecall)
        mock_recall.inject_recall.side_effect = RuntimeError("network error")
        mod._auto_recall = mock_recall

        agent = _make_mock_agent()
        ctx = _make_mock_context()
        # Must not propagate.
        mod.before_send(agent, ctx, "hello", None)


# ---------------------------------------------------------------------------
# after_send
# ---------------------------------------------------------------------------


class TestAfterSend:
    def test_noop_when_compaction_disabled(self):
        mod = MemoryModule()
        agent = _make_mock_agent()
        ctx = _make_mock_context()
        # Should not raise.
        mod.after_send(agent, ctx, "response", "default")

    def test_calls_compact_when_enabled(self):
        mod = MemoryModule(compaction_config=CompactionConfig())
        mock_compactor = MagicMock(spec=MemoryCompactor)
        mod._compactor = mock_compactor

        agent = _make_mock_agent()
        ctx = _make_mock_context()
        mod.after_send(agent, ctx, "the response", "default")

        mock_compactor.compact.assert_called_once_with(
            agent=agent, context=ctx, context_name="default"
        )

    def test_swallows_compaction_exception(self):
        mod = MemoryModule(compaction_config=CompactionConfig())
        mock_compactor = MagicMock(spec=MemoryCompactor)
        mock_compactor.compact.side_effect = RuntimeError("db full")
        mod._compactor = mock_compactor

        agent = _make_mock_agent()
        ctx = _make_mock_context()
        # Must not propagate.
        mod.after_send(agent, ctx, "response", "default")

    def test_response_text_available_to_subclass(self):
        """after_send passes `response` so subclasses can inspect it."""
        captured = {}

        class MyModule(MemoryModule):
            def after_send(self, agent, context, response, context_name):
                captured["response"] = response
                super().after_send(agent, context, response, context_name)

        mod = MyModule()
        agent = _make_mock_agent()
        ctx = _make_mock_context()
        mod.after_send(agent, ctx, "hello world", "default")
        assert captured["response"] == "hello world"


# ---------------------------------------------------------------------------
# Subclassing / extensibility
# ---------------------------------------------------------------------------


class TestMemoryModuleSubclassing:
    def test_override_before_send(self):
        before_calls = []

        class CustomModule(MemoryModule):
            def before_send(self, agent, context, content, model):
                before_calls.append(content)
                # Do NOT call super — completely replace behaviour.

        mod = CustomModule(recall_config=RecallConfig())
        mock_recall = MagicMock(spec=AutoRecall)
        mod._auto_recall = mock_recall

        agent = _make_mock_agent()
        ctx = _make_mock_context()
        mod.before_send(agent, ctx, "custom msg", None)

        assert before_calls == ["custom msg"]
        mock_recall.inject_recall.assert_not_called()

    def test_override_after_send(self):
        after_calls = []

        class CustomModule(MemoryModule):
            def after_send(self, agent, context, response, context_name):
                after_calls.append(context_name)
                # Do NOT call super.

        mod = CustomModule(compaction_config=CompactionConfig())
        mock_compactor = MagicMock(spec=MemoryCompactor)
        mod._compactor = mock_compactor

        agent = _make_mock_agent()
        ctx = _make_mock_context()
        mod.after_send(agent, ctx, "resp", "myctx")

        assert after_calls == ["myctx"]
        mock_compactor.compact.assert_not_called()

    def test_extend_before_send_with_super(self):
        extra_calls = []

        class ExtendedModule(MemoryModule):
            def before_send(self, agent, context, content, model):
                extra_calls.append("pre")
                super().before_send(agent, context, content, model)
                extra_calls.append("post")

        mock_recall = MagicMock(spec=AutoRecall)
        mod = ExtendedModule()
        mod._auto_recall = mock_recall

        agent = _make_mock_agent()
        ctx = _make_mock_context()
        mod.before_send(agent, ctx, "x", None)

        assert extra_calls == ["pre", "post"]
        mock_recall.inject_recall.assert_called_once()
