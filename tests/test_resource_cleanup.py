"""Tests for resource-cleanup bugs:

Bug 1 — Connection errors in OllamaAgent.send():
    * Non-OllamaResponseError exceptions from chat() are surfaced with a
      helpful message and the pending user message is cleaned from context.
    * inject_recall() / compact() failures are non-fatal (no propagation).

Bug 2 — Windows PermissionError on temp-dir cleanup (WinError 32):
    * ConversationStore.close() closes the ChromaDB PersistentClient so
      that chroma.sqlite3 is no longer locked.
    * MemoryStore.close() releases its ChromaDB client.
    * Agent.close() delegates to both stores.
"""

import shutil
import tempfile
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

from pithos.agent.agent import OllamaAgent
from pithos.agent.compaction import CompactionConfig, MemoryCompactor
from pithos.agent.recall import RecallConfig, AutoRecall
from pithos.context import AgentContext


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_agent(model: str = "test-model") -> OllamaAgent:
    return OllamaAgent(default_model=model)


def _patched_chat(response_text: str = "pong"):
    """Return a context-manager patch for ollama.chat that succeeds."""
    mock_response = MagicMock()
    mock_response.message.content = response_text
    mock_response.usage = None
    return patch("pithos.agent.agent.chat", return_value=mock_response)


# ===========================================================================
# Bug 1 — Connection / non-OllamaResponseError handling in send()
# ===========================================================================


class TestSendConnectionErrorHandling:
    """chat() can raise non-OllamaResponseError exceptions (e.g. httpx.ConnectError,
    ConnectionRefusedError).  send() must:
      1. Remove the pending user message that was added before the try-block.
      2. Re-raise as RuntimeError with a helpful message.
    """

    def test_connection_error_removes_user_message(self):
        agent = _make_agent()
        ctx = agent.contexts["default"]
        original_len = len(ctx.message_history)

        with patch(
            "pithos.agent.agent.chat", side_effect=ConnectionRefusedError("refused")
        ):
            with pytest.raises((RuntimeError, ConnectionRefusedError)):
                agent.send("hello")

        # The user message must not remain in history after the failed call
        assert len(ctx.message_history) == original_len

    def test_connection_error_raises_runtime_error_with_hint(self):
        agent = _make_agent()

        with patch(
            "pithos.agent.agent.chat", side_effect=OSError("connection timed out")
        ):
            with pytest.raises(RuntimeError) as exc_info:
                agent.send("hello")

        # Error message must mention Ollama and provide actionable guidance
        msg = str(exc_info.value)
        assert "Ollama" in msg or "ollama" in msg

    def test_os_error_during_chat_is_wrapped(self):
        """Generic OSError (covers socket errors) must also be caught."""
        agent = _make_agent()

        with patch(
            "pithos.agent.agent.chat", side_effect=OSError("Network unreachable")
        ):
            with pytest.raises(RuntimeError):
                agent.send("ping")

    def test_successful_send_still_works(self):
        agent = _make_agent()

        with _patched_chat("hello back"):
            result = agent.send("hello")

        assert result == "hello back"
        ctx = agent.contexts["default"]
        roles = [m["role"] for m in ctx.message_history]
        assert "user" in roles
        assert "assistant" in roles


class TestSendRecallFailureIsNonFatal:
    """inject_recall() must never prevent a successful LLM call."""

    def test_inject_recall_exception_does_not_propagate(self):
        agent = _make_agent()
        agent.enable_recall(RecallConfig())

        # Patch inject_recall to raise an arbitrary exception
        with patch.object(
            agent._auto_recall,
            "inject_recall",
            side_effect=RuntimeError("recall DB error"),
        ):
            with _patched_chat("response despite recall failure"):
                result = agent.send("ping")

        assert result == "response despite recall failure"

    def test_inject_recall_exception_leaves_context_clean(self):
        """After a recall failure the context must still end up with user+assistant messages."""
        agent = _make_agent()
        agent.enable_recall(RecallConfig())
        ctx = agent.contexts["default"]
        initial_len = len(ctx.message_history)

        with patch.object(
            agent._auto_recall,
            "inject_recall",
            side_effect=ValueError("bad chroma query"),
        ):
            with _patched_chat("ok"):
                agent.send("test message")

        final_msgs = ctx.message_history
        assert len(final_msgs) == initial_len + 2  # user + assistant


class TestSendCompactionFailureIsNonFatal:
    """compact() must never prevent send() from returning the LLM response."""

    def test_compact_exception_does_not_propagate(self):
        agent = _make_agent()
        agent.enable_compaction(CompactionConfig(threshold=1, keep_last=0))

        with patch.object(
            agent._compactor, "compact", side_effect=RuntimeError("compaction DB error")
        ):
            with _patched_chat("response despite compaction failure"):
                result = agent.send("ping")

        assert result == "response despite compaction failure"

    def test_compact_exception_response_still_added_to_context(self):
        agent = _make_agent()
        agent.enable_compaction(CompactionConfig(threshold=1, keep_last=0))
        ctx = agent.contexts["default"]

        with patch.object(
            agent._compactor, "compact", side_effect=ValueError("disk full")
        ):
            with _patched_chat("assistant answer"):
                agent.send("user question")

        contents = [m["content"] for m in ctx.message_history]
        assert "assistant answer" in contents


# ===========================================================================
# Bug 1 — stream() mirrors send() behaviour
# ===========================================================================


class TestStreamConnectionErrorHandling:
    def test_stream_connection_error_removes_user_message(self):
        agent = _make_agent()
        ctx = agent.contexts["default"]
        original_len = len(ctx.message_history)

        with patch(
            "pithos.agent.agent.chat", side_effect=ConnectionRefusedError("refused")
        ):
            with pytest.raises((RuntimeError, ConnectionRefusedError)):
                list(agent.stream("hello"))

        assert len(ctx.message_history) == original_len

    def test_stream_recall_failure_is_nonfatal(self):
        agent = _make_agent()
        agent.enable_recall(RecallConfig())

        mock_chunk = MagicMock()
        mock_chunk.message.content = "streamed"
        mock_chunk.usage = None

        with patch.object(
            agent._auto_recall,
            "inject_recall",
            side_effect=RuntimeError("stream recall error"),
        ):
            with patch("pithos.agent.agent.chat", return_value=iter([mock_chunk])):
                chunks = list(agent.stream("ping"))

        assert "".join(chunks) == "streamed"


# ===========================================================================
# Bug 2 — ConversationStore.close() releases ChromaDB file handles
# ===========================================================================


class TestConversationStoreClose:
    def test_close_allows_temp_dir_deletion(self):
        """After close(), the temp directory containing chroma.sqlite3 must be
        deletable (no PermissionError on Windows from lingering file handles)."""
        pytest.importorskip("chromadb")

        tmpdir = tempfile.mkdtemp(prefix="pithos_test_history_")
        try:
            from pithos.agent.history import ConversationStore

            store = ConversationStore(persist_directory=tmpdir)
            # Write something so ChromaDB actually opens chroma.sqlite3
            store.store_message(
                session_id="s1",
                agent_name="test",
                context_name="default",
                role="user",
                content="hello",
            )
            store.close()
            # Must not raise PermissionError on Windows
            shutil.rmtree(tmpdir)
            tmpdir = None  # prevent double-removal in finally
        finally:
            if tmpdir is not None:
                shutil.rmtree(tmpdir, ignore_errors=True)

    def test_close_sets_chroma_refs_to_none(self):
        """After close(), internal ChromaDB references must be cleared."""
        pytest.importorskip("chromadb")

        tmpdir = tempfile.mkdtemp(prefix="pithos_test_hist_")
        try:
            from pithos.agent.history import ConversationStore

            store = ConversationStore(persist_directory=tmpdir)
            store.close()

            assert store._chroma_client is None
            assert store._chroma_collection is None
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_close_without_chromadb_is_safe(self):
        """close() must work even when ChromaDB was never initialised."""
        tmpdir = tempfile.mkdtemp(prefix="pithos_test_hist_nc_")
        try:
            from pithos.agent.history import ConversationStore

            store = ConversationStore(persist_directory=tmpdir)
            # Force no ChromaDB
            store._chroma_client = None
            store._chroma_collection = None
            store.close()  # should not raise
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_close_is_idempotent(self):
        """Calling close() twice must not raise."""
        tmpdir = tempfile.mkdtemp(prefix="pithos_test_hist_idem_")
        try:
            from pithos.agent.history import ConversationStore

            store = ConversationStore(persist_directory=tmpdir)
            store.close()
            store.close()  # second call must be safe
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)


# ===========================================================================
# Bug 2 — MemoryStore.close() releases ChromaDB file handles
# ===========================================================================


class TestMemoryStoreClose:
    def test_close_allows_temp_dir_deletion(self):
        """After close(), the temp directory must be deletable."""
        pytest.importorskip("chromadb")

        tmpdir = tempfile.mkdtemp(prefix="pithos_test_memory_")
        try:
            from pithos.tools.memory_tool import MemoryStore

            store = MemoryStore(persist_directory=tmpdir)
            store.store("test_cat", "some fact")
            store.close()
            shutil.rmtree(tmpdir)
            tmpdir = None
        finally:
            if tmpdir is not None:
                shutil.rmtree(tmpdir, ignore_errors=True)

    def test_close_clears_collection_cache(self):
        """After close(), the in-memory collection cache must be empty."""
        pytest.importorskip("chromadb")

        tmpdir = tempfile.mkdtemp(prefix="pithos_test_mem_cache_")
        try:
            from pithos.tools.memory_tool import MemoryStore

            store = MemoryStore(persist_directory=tmpdir)
            store.store("cat_a", "entry")  # populates cache
            assert len(store._collections) > 0
            store.close()
            assert len(store._collections) == 0
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_close_sets_client_to_none(self):
        pytest.importorskip("chromadb")

        tmpdir = tempfile.mkdtemp(prefix="pithos_test_mem_null_")
        try:
            from pithos.tools.memory_tool import MemoryStore

            store = MemoryStore(persist_directory=tmpdir)
            store.close()
            assert store.client is None
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)


# ===========================================================================
# Bug 2 — Agent.close() propagates to stores
# ===========================================================================


class TestAgentClose:
    def test_close_calls_history_store_close(self):
        agent = _make_agent()
        mock_store = MagicMock()
        agent.history_store = mock_store
        agent.close()
        mock_store.close.assert_called_once()

    def test_close_calls_memory_store_close(self):
        agent = _make_agent()
        mock_store = MagicMock()
        agent.memory_store = mock_store
        agent.close()
        mock_store.close.assert_called_once()

    def test_close_with_no_stores_is_safe(self):
        agent = _make_agent()
        assert agent.history_store is None
        assert agent.memory_store is None
        agent.close()  # must not raise

    def test_close_continues_if_store_close_raises(self):
        """If one store.close() raises, the other is still attempted."""
        agent = _make_agent()
        failing_history = MagicMock()
        failing_history.close.side_effect = RuntimeError("cannot close")
        ok_memory = MagicMock()

        agent.history_store = failing_history
        agent.memory_store = ok_memory

        agent.close()  # must not propagate
        ok_memory.close.assert_called_once()

    def test_close_with_real_stores_in_tempdir(self):
        """Integration: close() + shutil.rmtree() must not raise PermissionError."""
        pytest.importorskip("chromadb")

        tmpdir = tempfile.mkdtemp(prefix="pithos_test_agent_close_")
        try:
            from pithos.config_manager import ConfigManager

            agent = OllamaAgent(default_model="test-model")
            agent.enable_memory(ConfigManager(), persist_directory=f"{tmpdir}/memory")
            agent.enable_history(
                persist_directory=f"{tmpdir}/history",
                session_id="close-test",
            )
            agent.close()
            shutil.rmtree(tmpdir)
            tmpdir = None
        finally:
            if tmpdir is not None:
                shutil.rmtree(tmpdir, ignore_errors=True)
