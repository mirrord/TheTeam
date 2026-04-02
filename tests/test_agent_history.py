"""Tests for conversation history storage and retrieval (ConversationStore + Agent API)."""

import shutil
import tempfile
import uuid
from unittest.mock import MagicMock, patch

import pytest

from pithos.agent import OllamaAgent
from pithos.agent.history import (
    CHROMADB_AVAILABLE,
    ConversationStore,
    HistorySearchResult,
    MessageRecord,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_dir():
    """Temporary directory removed after each test."""
    d = tempfile.mkdtemp()
    yield d
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def store(tmp_dir):
    """Fresh ConversationStore backed by a temp directory."""
    return ConversationStore(tmp_dir)


@pytest.fixture
def agent():
    """OllamaAgent instance (no real LLM calls)."""
    return OllamaAgent(default_model="test-model", agent_name="test-agent")


# ---------------------------------------------------------------------------
# ConversationStore — store_message
# ---------------------------------------------------------------------------


class TestStoreMessage:
    def test_returns_string_id(self, store):
        msg_id = store.store_message("sess1", "agent", "default", "user", "Hello")
        assert isinstance(msg_id, str)
        assert msg_id.startswith("msg_")

    def test_stored_message_is_retrievable(self, store):
        session = str(uuid.uuid4())
        store.store_message(session, "agent", "default", "user", "Test content")
        msgs = store.get_session_messages(session)
        assert len(msgs) == 1
        assert msgs[0].content == "Test content"
        assert msgs[0].role == "user"

    def test_multiple_messages_ordered_by_timestamp(self, store):
        session = str(uuid.uuid4())
        store.store_message(session, "agent", "default", "user", "First")
        store.store_message(session, "agent", "default", "assistant", "Second")
        msgs = store.get_session_messages(session)
        assert len(msgs) == 2
        assert msgs[0].content == "First"
        assert msgs[1].content == "Second"

    def test_separate_sessions_are_independent(self, store):
        s1, s2 = str(uuid.uuid4()), str(uuid.uuid4())
        store.store_message(s1, "agent", "default", "user", "Session 1 message")
        store.store_message(s2, "agent", "default", "user", "Session 2 message")
        assert len(store.get_session_messages(s1)) == 1
        assert len(store.get_session_messages(s2)) == 1

    def test_duplicate_content_gets_unique_ids(self, store):
        session = str(uuid.uuid4())
        id1 = store.store_message(session, "agent", "default", "user", "Same text")
        id2 = store.store_message(session, "agent", "default", "user", "Same text")
        assert id1 != id2


# ---------------------------------------------------------------------------
# ConversationStore — add_tags
# ---------------------------------------------------------------------------


class TestAddTags:
    def test_add_tags_to_message(self, store):
        session = str(uuid.uuid4())
        msg_id = store.store_message(session, "agent", "default", "user", "Tagged msg")
        store.add_tags(msg_id, ["important", "python"])
        msgs = store.get_session_messages(session)
        assert set(msgs[0].tags) == {"important", "python"}

    def test_add_tags_idempotent(self, store):
        """Adding same tag twice should not duplicate it."""
        session = str(uuid.uuid4())
        msg_id = store.store_message(session, "agent", "default", "user", "Msg")
        store.add_tags(msg_id, ["tag1"])
        store.add_tags(msg_id, ["tag1"])
        msgs = store.get_session_messages(session)
        assert msgs[0].tags.count("tag1") == 1

    def test_message_without_tags_has_empty_list(self, store):
        session = str(uuid.uuid4())
        store.store_message(session, "agent", "default", "user", "No tags")
        msgs = store.get_session_messages(session)
        assert msgs[0].tags == []

    def test_add_tags_strips_whitespace(self, store):
        session = str(uuid.uuid4())
        msg_id = store.store_message(session, "agent", "default", "user", "Spaced tags")
        store.add_tags(msg_id, ["  important  ", "  debug  "])
        msgs = store.get_session_messages(session)
        assert "important" in msgs[0].tags
        assert "debug" in msgs[0].tags

    def test_empty_tags_list_noop(self, store):
        """add_tags with empty list should not raise."""
        session = str(uuid.uuid4())
        msg_id = store.store_message(session, "agent", "default", "user", "Msg")
        store.add_tags(msg_id, [])  # Should not raise
        msgs = store.get_session_messages(session)
        assert msgs[0].tags == []


# ---------------------------------------------------------------------------
# ConversationStore — get_session_messages filtering
# ---------------------------------------------------------------------------


class TestGetSessionMessages:
    def test_filter_by_role(self, store):
        session = str(uuid.uuid4())
        store.store_message(session, "agent", "default", "user", "User msg")
        store.store_message(session, "agent", "default", "assistant", "Agent msg")
        user_msgs = store.get_session_messages(session, role="user")
        assert all(m.role == "user" for m in user_msgs)
        assert len(user_msgs) == 1

    def test_filter_by_context(self, store):
        session = str(uuid.uuid4())
        store.store_message(session, "agent", "ctx_a", "user", "Context A msg")
        store.store_message(session, "agent", "ctx_b", "user", "Context B msg")
        ctx_a_msgs = store.get_session_messages(session, context_name="ctx_a")
        assert len(ctx_a_msgs) == 1
        assert ctx_a_msgs[0].content == "Context A msg"

    def test_empty_session_returns_empty_list(self, store):
        assert store.get_session_messages("nonexistent-session") == []


# ---------------------------------------------------------------------------
# ConversationStore — list_sessions
# ---------------------------------------------------------------------------


class TestListSessions:
    def test_list_all_sessions(self, store):
        s1, s2 = str(uuid.uuid4()), str(uuid.uuid4())
        store.store_message(s1, "agent_a", "default", "user", "Hello")
        store.store_message(s2, "agent_b", "default", "user", "World")
        sessions = store.list_sessions()
        session_ids = [s["session_id"] for s in sessions]
        assert s1 in session_ids
        assert s2 in session_ids

    def test_list_sessions_filtered_by_agent(self, store):
        s1, s2 = str(uuid.uuid4()), str(uuid.uuid4())
        store.store_message(s1, "agent_a", "default", "user", "Hello")
        store.store_message(s2, "agent_b", "default", "user", "World")
        sessions = store.list_sessions(agent_name="agent_a")
        assert all(s["agent_name"] == "agent_a" for s in sessions)
        assert len(sessions) == 1

    def test_session_message_count(self, store):
        session = str(uuid.uuid4())
        store.store_message(session, "agent", "default", "user", "Msg 1")
        store.store_message(session, "agent", "default", "assistant", "Msg 2")
        sessions = store.list_sessions()
        match = next(s for s in sessions if s["session_id"] == session)
        assert match["message_count"] == 2


# ---------------------------------------------------------------------------
# ConversationStore — text search
# ---------------------------------------------------------------------------


class TestTextSearch:
    def test_search_finds_matching_content(self, store):
        session = str(uuid.uuid4())
        store.store_message(
            session, "agent", "default", "user", "authentication error fix"
        )
        store.store_message(
            session, "agent", "default", "user", "unrelated content here"
        )
        results = store.search_text("authentication error")
        contents = [r.message.content for r in results]
        assert any("authentication" in c for c in contents)

    def test_search_returns_history_search_results(self, store):
        session = str(uuid.uuid4())
        store.store_message(session, "agent", "default", "user", "test content")
        results = store.search_text("test content")
        assert all(isinstance(r, HistorySearchResult) for r in results)

    def test_search_match_type_is_text(self, store):
        session = str(uuid.uuid4())
        store.store_message(session, "agent", "default", "user", "something unique")
        results = store.search_text("something unique")
        assert all(r.match_type == "text" for r in results)

    def test_search_filters_by_role(self, store):
        session = str(uuid.uuid4())
        store.store_message(session, "agent", "default", "user", "error in the system")
        store.store_message(
            session, "agent", "default", "assistant", "error in the response"
        )
        results = store.search_text("error", role="user")
        assert all(r.message.role == "user" for r in results)

    def test_search_filters_by_tags(self, store):
        session = str(uuid.uuid4())
        id1 = store.store_message(
            session, "agent", "default", "user", "important message here"
        )
        id2 = store.store_message(
            session, "agent", "default", "user", "ordinary message here"
        )
        store.add_tags(id1, ["critical"])
        results = store.search_text("message here", tags=["critical"])
        assert all("critical" in r.message.tags for r in results)

    def test_search_filters_by_session(self, store):
        s1, s2 = str(uuid.uuid4()), str(uuid.uuid4())
        store.store_message(s1, "agent", "default", "user", "target message content")
        store.store_message(s2, "agent", "default", "user", "target message content")
        results = store.search_text("target message content", session_id=s1)
        assert all(r.message.session_id == s1 for r in results)

    def test_search_filters_by_agent_name(self, store):
        session = str(uuid.uuid4())
        store.store_message(
            session, "agent_alpha", "default", "user", "unique phrase alpha"
        )
        store.store_message(
            session, "agent_beta", "default", "user", "unique phrase alpha"
        )
        results = store.search_text("unique phrase alpha", agent_name="agent_alpha")
        assert all(r.message.agent_name == "agent_alpha" for r in results)

    def test_search_no_match_returns_empty(self, store):
        session = str(uuid.uuid4())
        store.store_message(session, "agent", "default", "user", "completely unrelated")
        results = store.search_text("xyzzy_nonexistent_token_123")
        assert len(results) == 0

    def test_search_respects_n_results_limit(self, store):
        session = str(uuid.uuid4())
        for i in range(10):
            store.store_message(session, "agent", "default", "user", f"result item {i}")
        results = store.search_text("result item", n_results=3)
        assert len(results) <= 3

    def test_search_on_empty_store_returns_empty(self, store):
        results = store.search_text("anything")
        assert results == []

    def test_search_relevance_score_is_float(self, store):
        session = str(uuid.uuid4())
        store.store_message(session, "agent", "default", "user", "searchable content")
        results = store.search_text("searchable")
        assert all(isinstance(r.relevance_score, float) for r in results)


# ---------------------------------------------------------------------------
# ConversationStore — search() dispatch
# ---------------------------------------------------------------------------


class TestSearchDispatch:
    def test_search_returns_results(self, store):
        session = str(uuid.uuid4())
        store.store_message(
            session, "agent", "default", "user", "dispatch test content"
        )
        results = store.search("dispatch test content", semantic=False)
        assert len(results) > 0

    def test_search_semantic_false_uses_text(self, store):
        """With semantic=False, search_text is used regardless of ChromaDB."""
        session = str(uuid.uuid4())
        store.store_message(
            session, "agent", "default", "user", "test dispatch semantic false"
        )
        results = store.search("test dispatch semantic false", semantic=False)
        assert all(r.match_type == "text" for r in results)

    def test_semantic_search_available_property(self, store):
        """Property should reflect ChromaDB availability."""
        assert isinstance(store.semantic_search_available, bool)


# ---------------------------------------------------------------------------
# ConversationStore — persistence (reopen)
# ---------------------------------------------------------------------------


class TestPersistence:
    def test_data_survives_reopen(self, tmp_dir):
        session = str(uuid.uuid4())
        store1 = ConversationStore(tmp_dir)
        mid = store1.store_message(
            session, "agent", "default", "user", "persisted content"
        )
        store1.add_tags(mid, ["persisted"])
        store1.close()

        store2 = ConversationStore(tmp_dir)
        msgs = store2.get_session_messages(session)
        assert len(msgs) == 1
        assert msgs[0].content == "persisted content"
        assert "persisted" in msgs[0].tags
        store2.close()

    def test_search_works_after_reopen(self, tmp_dir):
        session = str(uuid.uuid4())
        store1 = ConversationStore(tmp_dir)
        store1.store_message(
            session, "agent", "default", "user", "reopened search test"
        )
        store1.close()

        store2 = ConversationStore(tmp_dir)
        results = store2.search_text("reopened search test")
        assert len(results) > 0
        store2.close()


# ---------------------------------------------------------------------------
# Agent.enable_history
# ---------------------------------------------------------------------------


class TestAgentEnableHistory:
    def test_enable_history_sets_store(self, agent, tmp_dir):
        agent.enable_history(tmp_dir)
        assert agent.history_store is not None
        assert isinstance(agent.history_store, ConversationStore)

    def test_enable_history_sets_session_id(self, agent, tmp_dir):
        agent.enable_history(tmp_dir)
        assert agent.session_id is not None
        assert isinstance(agent.session_id, str)

    def test_enable_history_with_explicit_session_id(self, agent, tmp_dir):
        sid = "my-custom-session"
        agent.enable_history(tmp_dir, session_id=sid)
        assert agent.session_id == sid

    def test_enable_history_generates_unique_sessions(self, agent, tmp_dir):
        agent.enable_history(tmp_dir)
        sid1 = agent.session_id
        agent.enable_history(tmp_dir)
        sid2 = agent.session_id
        assert sid1 != sid2

    def test_enable_history_default_directory_created(self, tmp_dir, monkeypatch):
        """Default persist_directory is auto-created."""
        import os

        default_path = os.path.join(tmp_dir, "data", "conversations")
        monkeypatch.chdir(tmp_dir)
        agent2 = OllamaAgent("test-model")
        agent2.enable_history()
        assert agent2.history_store is not None


# ---------------------------------------------------------------------------
# Agent.tag_current_message
# ---------------------------------------------------------------------------


class TestAgentTagCurrentMessage:
    def test_tag_requires_history_enabled(self, agent):
        with pytest.raises(RuntimeError, match="History is not enabled"):
            agent.tag_current_message(["tag"])

    def test_tag_requires_prior_message(self, agent, tmp_dir):
        agent.enable_history(tmp_dir)
        with pytest.raises(RuntimeError, match="No message has been stored yet"):
            agent.tag_current_message(["tag"])

    def test_tag_after_manual_persist(self, agent, tmp_dir):
        agent.enable_history(tmp_dir)
        agent._history_persist(
            "default", "assistant", "Response content", set_as_last=True
        )
        agent.tag_current_message(["important", "debug"])

        # Verify tag was stored
        msgs = agent.history_store.get_session_messages(agent.session_id)
        assert any("important" in m.tags for m in msgs)

    def test_tag_multiple_calls_accumulate(self, agent, tmp_dir):
        agent.enable_history(tmp_dir)
        agent._history_persist("default", "assistant", "Response", set_as_last=True)
        agent.tag_current_message(["tag1"])
        agent.tag_current_message(["tag2"])
        msgs = agent.history_store.get_session_messages(agent.session_id)
        tagged = next(m for m in msgs if m.id == agent._last_history_message_id)
        assert "tag1" in tagged.tags
        assert "tag2" in tagged.tags


# ---------------------------------------------------------------------------
# Agent.search_history
# ---------------------------------------------------------------------------


class TestAgentSearchHistory:
    def test_search_requires_history_enabled(self, agent):
        with pytest.raises(RuntimeError, match="History is not enabled"):
            agent.search_history("query")

    def test_search_finds_stored_message(self, agent, tmp_dir):
        agent.enable_history(tmp_dir)
        agent._history_persist("default", "user", "authentication error occurred")
        results = agent.search_history("authentication error", semantic=False)
        assert len(results) > 0

    def test_search_restricted_to_current_session_by_default(self, agent, tmp_dir):
        agent.enable_history(tmp_dir, session_id="session-a")
        agent._history_persist("default", "user", "session_a unique term zyx")

        # Switch session — content from session_a should NOT appear
        agent.enable_history(tmp_dir, session_id="session-b")
        results = agent.search_history("session_a unique term zyx", semantic=False)
        assert len(results) == 0

    def test_search_all_sessions(self, agent, tmp_dir):
        agent.enable_history(tmp_dir, session_id="session-a")
        agent._history_persist("default", "user", "all_sessions unique phrase qrs")

        agent.enable_history(tmp_dir, session_id="session-b")
        results = agent.search_history(
            "all_sessions unique phrase qrs", semantic=False, all_sessions=True
        )
        assert len(results) > 0

    def test_search_with_tag_filter(self, agent, tmp_dir):
        agent.enable_history(tmp_dir)
        agent._history_persist(
            "default", "assistant", "tagged response content", set_as_last=True
        )
        agent.tag_current_message(["relevant"])
        agent._history_persist("default", "assistant", "untagged response content")

        results = agent.search_history(
            "response content", tags=["relevant"], semantic=False
        )
        assert all("relevant" in r.message.tags for r in results)

    def test_search_with_role_filter(self, agent, tmp_dir):
        agent.enable_history(tmp_dir)
        agent._history_persist("default", "user", "user side message text")
        agent._history_persist("default", "assistant", "assistant side message text")
        results = agent.search_history("side message text", role="user", semantic=False)
        assert all(r.message.role == "user" for r in results)

    def test_search_returns_history_search_results(self, agent, tmp_dir):
        agent.enable_history(tmp_dir)
        agent._history_persist("default", "user", "result type check content")
        results = agent.search_history("result type check content", semantic=False)
        assert all(isinstance(r, HistorySearchResult) for r in results)


# ---------------------------------------------------------------------------
# Agent.send() integration — history is persisted automatically
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_ollama_response():
    """Fake streaming response returned by ollama.chat (list of chunks)."""
    chunk = MagicMock()
    chunk.message.content = "Mocked agent response"
    # Return a list so it can be iterated safely for multiple send() calls.
    return [chunk]


class TestAgentSendHistoryIntegration:
    @patch("pithos.agent.ollama_agent.chat")
    def test_send_persists_user_and_agent_messages(
        self, mock_chat, agent, tmp_dir, mock_ollama_response
    ):
        mock_chat.return_value = mock_ollama_response
        agent.enable_history(tmp_dir)
        agent.send("Hello world")

        msgs = agent.history_store.get_session_messages(agent.session_id)
        roles = {m.role for m in msgs}
        assert "user" in roles
        assert "assistant" in roles

    @patch("pithos.agent.ollama_agent.chat")
    def test_send_sets_last_history_message_id(
        self, mock_chat, agent, tmp_dir, mock_ollama_response
    ):
        mock_chat.return_value = mock_ollama_response
        agent.enable_history(tmp_dir)
        agent.send("Hello")
        assert agent._last_history_message_id is not None

    @patch("pithos.agent.ollama_agent.chat")
    def test_tag_current_message_after_send(
        self, mock_chat, agent, tmp_dir, mock_ollama_response
    ):
        mock_chat.return_value = mock_ollama_response
        agent.enable_history(tmp_dir)
        agent.send("Fix authentication bug")
        agent.tag_current_message(["important", "bug-fix"])

        # The tagged message should be the assistant response
        msgs = agent.history_store.get_session_messages(agent.session_id)
        tagged = next((m for m in msgs if m.id == agent._last_history_message_id), None)
        assert tagged is not None
        assert "important" in tagged.tags
        assert "bug-fix" in tagged.tags

    @patch("pithos.agent.ollama_agent.chat")
    def test_search_history_after_send(
        self, mock_chat, agent, tmp_dir, mock_ollama_response
    ):
        mock_chat.return_value = mock_ollama_response
        agent.enable_history(tmp_dir)
        agent.send("How do I fix the authentication error?")

        results = agent.search_history("authentication error", semantic=False)
        assert len(results) > 0

    @patch("pithos.agent.ollama_agent.chat")
    def test_send_without_history_works_normally(
        self, mock_chat, agent, mock_ollama_response
    ):
        """History not enabled: send() must behave identically to before."""
        mock_chat.return_value = mock_ollama_response
        response = agent.send("Hello")
        assert response == "Mocked agent response"

    @patch("pithos.agent.ollama_agent.chat")
    def test_history_failure_does_not_break_send(
        self, mock_chat, agent, tmp_dir, mock_ollama_response
    ):
        """If history persistence raises unexpectedly, send() should still succeed."""
        mock_chat.return_value = mock_ollama_response
        agent.enable_history(tmp_dir)
        # Sabotage the store
        agent.history_store._conn.close()

        # Should not raise; response should still come through
        response = agent.send("Hello despite broken history")
        assert response == "Mocked agent response"


# ---------------------------------------------------------------------------
# Agent.stream() integration
# ---------------------------------------------------------------------------


class TestAgentStreamHistoryIntegration:
    @patch("pithos.agent.ollama_agent.chat")
    def test_stream_persists_after_exhaustion(self, mock_chat, agent, tmp_dir):
        """History is written only after the iterator is fully consumed."""

        def _make_chunk(text):
            c = MagicMock()
            c.message.content = text
            return c

        mock_chat.return_value = iter([_make_chunk("Hello"), _make_chunk(" world")])
        agent.enable_history(tmp_dir)

        # Consume generator fully
        list(agent.stream("Stream test message"))

        msgs = agent.history_store.get_session_messages(agent.session_id)
        roles = {m.role for m in msgs}
        assert "user" in roles
        assert "assistant" in roles

    @patch("pithos.agent.ollama_agent.chat")
    def test_stream_sets_last_history_message_id(self, mock_chat, agent, tmp_dir):
        def _chunk(text):
            c = MagicMock()
            c.message.content = text
            return c

        mock_chat.return_value = iter([_chunk("Response")])
        agent.enable_history(tmp_dir)
        list(agent.stream("Stream tagging test"))
        assert agent._last_history_message_id is not None

    @patch("pithos.agent.ollama_agent.chat")
    def test_tag_after_stream(self, mock_chat, agent, tmp_dir):
        def _chunk(text):
            c = MagicMock()
            c.message.content = text
            return c

        mock_chat.return_value = iter([_chunk("Streamed response")])
        agent.enable_history(tmp_dir)
        list(agent.stream("Stream with tag"))
        agent.tag_current_message(["streamed", "tagged"])

        msgs = agent.history_store.get_session_messages(agent.session_id)
        tagged = next(m for m in msgs if m.id == agent._last_history_message_id)
        assert "streamed" in tagged.tags


# ---------------------------------------------------------------------------
# MessageRecord dataclass
# ---------------------------------------------------------------------------


class TestMessageRecord:
    def test_fields(self):
        record = MessageRecord(
            id="msg_abc",
            session_id="sess",
            agent_name="agent",
            context_name="default",
            role="user",
            content="Hello",
            timestamp="2026-01-01T00:00:00+00:00",
        )
        assert record.id == "msg_abc"
        assert record.tags == []

    def test_default_tags_are_independent(self):
        r1 = MessageRecord("1", "s", "a", "c", "user", "c", "t")
        r2 = MessageRecord("2", "s", "a", "c", "user", "c", "t")
        r1.tags.append("x")
        assert "x" not in r2.tags
