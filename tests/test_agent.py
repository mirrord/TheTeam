"""Unit tests for agent module."""

import pytest
from unittest.mock import Mock, patch
from pithos.agent import (
    Agent,
    Msg,
    UserMsg,
    AgentMsg,
    AgentContext,
    OllamaAgent,
)


class TestMsg:
    """Test Msg data container."""

    def test_msg_creation(self):
        msg = Msg(role="user", content="Hello")
        assert msg.role == "user"
        assert msg.content == "Hello"

    def test_msg_getitem(self):
        msg = Msg(role="user", content="Hello")
        assert msg["role"] == "user"
        assert msg["content"] == "Hello"

    def test_msg_setitem(self):
        msg = Msg(role="user", content="Hello")
        msg["role"] = "assistant"
        assert msg.role == "assistant"

    def test_msg_raw(self):
        msg = Msg(role="user", content="Hello")
        raw = msg.raw()
        assert raw == {"role": "user", "content": "Hello"}


class TestUserMsg:
    """Test UserMsg helper."""

    def test_user_msg_creation(self):
        msg = UserMsg("Hello")
        assert msg.role == "user"
        assert msg.content == "Hello"


class TestAgentMsg:
    """Test AgentMsg helper."""

    def test_agent_msg_creation(self):
        msg = AgentMsg("Response")
        assert msg.role == "assistant"
        assert msg.content == "Response"


class TestAgentContext:
    """Test AgentContext for conversation management."""

    def test_context_creation(self):
        ctx = AgentContext("test_context", "System prompt")
        assert ctx.name == "test_context"
        assert ctx.get_system_prompt() == "System prompt"
        assert ctx.message_history == []
        assert ctx.completed is False

    def test_context_default_values(self):
        ctx = AgentContext("test")
        assert ctx.get_system_prompt() == ""

    def test_add_message(self):
        ctx = AgentContext("test")
        msg = UserMsg("Hello")
        ctx.add_message(msg)
        assert len(ctx.message_history) == 1
        assert ctx.message_history[0]["role"] == "user"
        assert ctx.message_history[0]["content"] == "Hello"

    def test_get_last_output(self):
        ctx = AgentContext("test")
        ctx.add_message(UserMsg("Question"))
        ctx.add_message(AgentMsg("Answer"))
        assert ctx.get_last_output() == "Answer"

    def test_get_last_output_empty(self):
        ctx = AgentContext("test")
        assert ctx.get_last_output() == ""

    def test_get_last_input(self):
        ctx = AgentContext("test")
        ctx.add_message(UserMsg("Question"))
        ctx.add_message(AgentMsg("Answer"))
        assert ctx.get_last_input() == "Question"

    def test_get_last_input_empty(self):
        ctx = AgentContext("test")
        assert ctx.get_last_input() == ""

    def test_clear(self):
        ctx = AgentContext("test")
        ctx.add_message(UserMsg("Hello"))
        ctx.add_message(AgentMsg("Hi"))
        ctx.clear()
        assert ctx.message_history == []

    def test_remove_last_message(self):
        ctx = AgentContext("test")
        ctx.add_message(UserMsg("Hello"))
        ctx.add_message(AgentMsg("Hi"))
        ctx.remove_last_message()
        assert len(ctx.message_history) == 1
        assert ctx.message_history[0]["role"] == "user"

    def test_set_system_prompt(self):
        ctx = AgentContext("test", "Original")
        ctx.set_system_prompt("Updated")
        assert ctx.get_system_prompt() == "Updated"

    def test_copy_context(self):
        """Test that context copy creates independent history."""
        ctx = AgentContext("original", "System prompt")
        ctx.add_message(UserMsg("Hello"))

        copy_ctx = ctx.copy("copy")
        assert copy_ctx.name == "copy"
        assert copy_ctx.get_system_prompt() == "System prompt"
        assert len(copy_ctx.message_history) == 1

        # Modify original - copy should be unaffected
        ctx.add_message(AgentMsg("Response"))
        assert len(ctx.message_history) == 2
        assert len(copy_ctx.message_history) == 1

    def test_copy_context_default_name(self):
        ctx = AgentContext("original")
        copy_ctx = ctx.copy()
        assert copy_ctx.name == "original_copy"

    def test_get_messages(self):
        ctx = AgentContext("test", "System")
        ctx.add_message(UserMsg("Hello"))
        messages = ctx.get_messages()

        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == "System"
        assert messages[1]["role"] == "user"

    def test_get_messages_with_workspace(self):
        ctx = AgentContext("test", "System")
        ctx.add_message(UserMsg("Hello"))
        messages = ctx.get_messages(workspace="Workspace context")

        assert len(messages) == 3
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        assert messages[1]["content"] == "Workspace context"
        assert messages[2]["role"] == "user"
        assert messages[2]["content"] == "Hello"

    def test_to_dict(self):
        ctx = AgentContext("test", "System prompt")
        d = ctx.to_dict()
        assert d["system_prompt"] == "System prompt"
        assert "message_history" not in d

    def test_to_dict_with_history(self):
        ctx = AgentContext("test", "System")
        ctx.add_message(UserMsg("Hello"))
        d = ctx.to_dict(with_history=True)
        assert d["system_prompt"] == "System"
        assert len(d["message_history"]) == 1

    def test_from_dict(self):
        data = {
            "system_prompt": "Test system",
            "message_history": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi"},
            ],
        }
        ctx = AgentContext.from_dict(data, "test_context")
        assert ctx.name == "test_context"
        assert ctx.get_system_prompt() == "Test system"
        assert len(ctx.message_history) == 2


class TestOllamaAgent:
    """Test OllamaAgent class."""

    def test_agent_creation(self):
        agent = OllamaAgent("glm-4.7-flash")
        assert agent.default_model == "glm-4.7-flash"
        assert agent.agent_name == "glm-4.7-flash"
        assert agent.default_system_prompt == ""
        assert "default" in agent.contexts
        assert agent.current_context == "default"

    def test_agent_creation_with_name(self):
        agent = OllamaAgent("glm-4.7-flash", agent_name="my_agent")
        assert agent.agent_name == "my_agent"

    def test_agent_creation_with_system_prompt(self):
        agent = OllamaAgent("glm-4.7-flash", system_prompt="You are helpful")
        assert agent.default_system_prompt == "You are helpful"
        assert agent.contexts["default"].get_system_prompt() == "You are helpful"

    def test_agent_creation_default_temperature(self):
        """Test that agent has default temperature of 0.7."""
        agent = OllamaAgent("glm-4.7-flash")
        assert agent.temperature == 0.7

    def test_agent_creation_with_custom_temperature(self):
        """Test creating agent with custom temperature."""
        agent = OllamaAgent("glm-4.7-flash", temperature=0.3)
        assert agent.temperature == 0.3

    def test_agent_creation_with_zero_temperature(self):
        """Test creating agent with temperature 0 (deterministic)."""
        agent = OllamaAgent("glm-4.7-flash", temperature=0)
        assert agent.temperature == 0

    def test_create_context(self):
        agent = OllamaAgent("glm-4.7-flash")
        agent.create_context("test_ctx", "Test prompt")
        assert "test_ctx" in agent.contexts
        assert agent.current_context == "test_ctx"
        assert agent.contexts["test_ctx"].get_system_prompt() == "Test prompt"

    def test_create_context_uses_default_prompt(self):
        agent = OllamaAgent("glm-4.7-flash", system_prompt="Default")
        agent.create_context("test")
        assert agent.contexts["test"].get_system_prompt() == "Default"

    def test_switch_context(self):
        agent = OllamaAgent("glm-4.7-flash")
        agent.create_context("ctx1")
        agent.create_context("ctx2")
        agent.switch_context("ctx1")
        assert agent.current_context == "ctx1"

    def test_switch_context_nonexistent_raises(self):
        agent = OllamaAgent("glm-4.7-flash")
        with pytest.raises(ValueError, match="does not exist"):
            agent.switch_context("nonexistent")

    def test_copy_context(self):
        """Test context copying creates independent history."""
        agent = OllamaAgent("glm-4.7-flash")
        agent.contexts["default"].add_message(UserMsg("Original message"))

        agent.copy_context("default", "copied")
        assert "copied" in agent.contexts
        assert agent.current_context == "copied"
        assert len(agent.contexts["copied"].message_history) == 1

        # Modify original
        agent.contexts["default"].add_message(AgentMsg("Response"))
        assert len(agent.contexts["default"].message_history) == 2
        assert len(agent.contexts["copied"].message_history) == 1

    def test_copy_context_with_new_prompt(self):
        agent = OllamaAgent("glm-4.7-flash")
        agent.copy_context("default", "copied", new_system_prompt="New prompt")
        assert agent.contexts["copied"].get_system_prompt() == "New prompt"

    def test_copy_context_nonexistent_raises(self):
        agent = OllamaAgent("glm-4.7-flash")
        with pytest.raises(ValueError, match="does not exist"):
            agent.copy_context("nonexistent", "new")

    def test_share_context(self):
        """Test that shared context returns reference."""
        agent1 = OllamaAgent("glm-4.7-flash")
        agent1.contexts["default"].add_message(UserMsg("Shared message"))

        shared_ctx = agent1.share_context("default")
        assert shared_ctx is agent1.contexts["default"]

    def test_use_shared_context(self):
        """Test that agents can share the same context."""
        agent1 = OllamaAgent("glm-4.7-flash")
        agent2 = OllamaAgent("glm-4.7-flash")

        agent1.contexts["default"].add_message(UserMsg("From agent1"))
        shared_ctx = agent1.share_context("default")

        agent2.use_shared_context("shared", shared_ctx)
        assert agent2.contexts["shared"] is agent1.contexts["default"]
        assert len(agent2.contexts["shared"].message_history) == 1

        # Both agents modify same history
        agent2.contexts["shared"].add_message(AgentMsg("From agent2"))
        assert len(agent1.contexts["default"].message_history) == 2

    def test_list_contexts(self):
        agent = OllamaAgent("glm-4.7-flash")
        agent.create_context("ctx1")
        agent.create_context("ctx2")
        contexts = agent.list_contexts()
        assert "default" in contexts
        assert "ctx1" in contexts
        assert "ctx2" in contexts

    def test_get_current_context_name(self):
        agent = OllamaAgent("glm-4.7-flash")
        assert agent.get_current_context_name() == "default"
        agent.create_context("test")
        assert agent.get_current_context_name() == "test"

    def test_set_system_prompt(self):
        agent = OllamaAgent("glm-4.7-flash")
        agent.set_system_prompt("New prompt")
        assert agent.contexts["default"].get_system_prompt() == "New prompt"

    def test_set_system_prompt_no_context_raises(self):
        agent = OllamaAgent("glm-4.7-flash")
        agent.current_context = None
        with pytest.raises(ValueError, match="No context selected"):
            agent.set_system_prompt("Test")

    def test_clear_context(self):
        agent = OllamaAgent("glm-4.7-flash")
        agent.contexts["default"].add_message(UserMsg("Test"))
        agent.clear_context()
        assert len(agent.contexts["default"].message_history) == 0

    def test_clear_specific_context(self):
        agent = OllamaAgent("glm-4.7-flash")
        agent.create_context("test")
        agent.contexts["test"].add_message(UserMsg("Test"))
        agent.clear_context("test")
        assert len(agent.contexts["test"].message_history) == 0

    def test_delete_context(self):
        agent = OllamaAgent("glm-4.7-flash")
        agent.create_context("test")
        agent.delete_context("test")
        assert "test" not in agent.contexts

    def test_delete_current_context_switches_to_default(self):
        agent = OllamaAgent("glm-4.7-flash")
        agent.create_context("test")
        assert agent.current_context == "test"
        agent.delete_context("test")
        assert agent.current_context == "default"

    def test_delete_context_nonexistent_raises(self):
        agent = OllamaAgent("glm-4.7-flash")
        with pytest.raises(ValueError, match="does not exist"):
            agent.delete_context("nonexistent")

    @patch("pithos.agent.ollama_agent.chat")
    def test_send_message(self, mock_chat):
        """Test sending a message with mocked LLM."""
        mock_chunk = Mock()
        mock_chunk.message.content = "Response from LLM"
        mock_chat.return_value = iter([mock_chunk])

        agent = OllamaAgent("glm-4.7-flash")
        response = agent.send("Hello")

        assert response == "Response from LLM"
        assert len(agent.contexts["default"].message_history) == 2
        assert agent.contexts["default"].message_history[0]["role"] == "user"
        assert agent.contexts["default"].message_history[1]["role"] == "assistant"

    @patch("pithos.agent.ollama_agent.chat")
    def test_send_passes_temperature(self, mock_chat):
        """Test that send passes temperature to chat function."""
        mock_chunk = Mock()
        mock_chunk.message.content = "Response"
        mock_chat.return_value = iter([mock_chunk])

        agent = OllamaAgent("glm-4.7-flash", temperature=0.5)
        agent.send("Hello")

        # Verify chat was called with correct temperature in options
        mock_chat.assert_called_once()
        call_kwargs = mock_chat.call_args[1]
        assert "options" in call_kwargs
        assert call_kwargs["options"]["temperature"] == 0.5

    @patch("pithos.agent.ollama_agent.chat")
    def test_send_passes_default_temperature(self, mock_chat):
        """Test that send passes default temperature when not specified."""
        mock_chunk = Mock()
        mock_chunk.message.content = "Response"
        mock_chat.return_value = iter([mock_chunk])

        agent = OllamaAgent("glm-4.7-flash")  # Should default to 0.7
        agent.send("Hello")

        call_kwargs = mock_chat.call_args[1]
        assert call_kwargs["options"]["temperature"] == 0.7

    @patch("pithos.agent.ollama_agent.chat")
    def test_send_does_not_pass_num_predict(self, mock_chat):
        """Test that send does not pass num_predict since max_tokens is always -1."""
        mock_chunk = Mock()
        mock_chunk.message.content = "Response"
        mock_chat.return_value = iter([mock_chunk])

        agent = OllamaAgent("glm-4.7-flash")
        agent.send("Hello")

        call_kwargs = mock_chat.call_args[1]
        assert "num_predict" not in call_kwargs["options"]
        assert call_kwargs["options"]["temperature"] == 0.7

    @patch("pithos.agent.ollama_agent.chat")
    def test_send_message_to_specific_context(self, mock_chat):
        """Test sending to a specific context."""
        mock_chunk = Mock()
        mock_chunk.message.content = "Response"
        mock_chat.return_value = iter([mock_chunk])

        agent = OllamaAgent("glm-4.7-flash")
        agent.create_context("test")
        agent.switch_context("default")
        agent.send("Hello", context_name="test")

        assert len(agent.contexts["test"].message_history) == 2
        assert len(agent.contexts["default"].message_history) == 0

    @patch("pithos.agent.ollama_agent.chat")
    def test_send_creates_context_if_missing(self, mock_chat):
        """Test that send creates context if it doesn't exist."""
        mock_chunk = Mock()
        mock_chunk.message.content = "Response"
        mock_chat.return_value = iter([mock_chunk])

        agent = OllamaAgent("glm-4.7-flash")
        agent.send("Hello", context_name="new_context")

        assert "new_context" in agent.contexts

    def test_to_dict(self):
        agent = OllamaAgent(
            "glm-4.7-flash", agent_name="test_agent", system_prompt="Prompt"
        )
        d = agent.to_dict()

        assert d["name"] == "test_agent"
        assert d["model"] == "glm-4.7-flash"
        assert d["system_prompt"] == "Prompt"
        assert d["temperature"] == 0.7  # Default temperature
        assert d["max_tokens"] == -1  # Default max_tokens (unlimited)
        assert d["current_context"] == "default"

    def test_to_dict_with_custom_temperature(self):
        """Test serialization includes custom temperature."""
        agent = OllamaAgent("glm-4.7-flash", temperature=0.3)
        d = agent.to_dict()
        assert d["temperature"] == 0.3

    def test_to_dict_with_multiple_contexts(self):
        agent = OllamaAgent("glm-4.7-flash")
        agent.create_context("ctx1", "Prompt1")
        agent.contexts["ctx1"].add_message(UserMsg("Test"))

        d = agent.to_dict()
        assert "contexts" in d
        assert "ctx1" in d["contexts"]

    @patch("pithos.agent.agent.ConfigManager")
    def test_from_dict(self, mock_config_manager):
        """Test creating agent from dictionary."""
        config = {
            "model": "glm-4.7-flash",
            "name": "test_agent",
            "system_prompt": "Test prompt",
        }
        agent = OllamaAgent.from_dict(config, mock_config_manager)

        assert agent.default_model == "glm-4.7-flash"
        assert agent.agent_name == "test_agent"
        assert agent.default_system_prompt == "Test prompt"
        assert agent.temperature == 0.7  # Default when not specified

    @patch("pithos.agent.agent.ConfigManager")
    def test_from_dict_with_temperature(self, mock_config_manager):
        """Test creating agent from dictionary with temperature."""
        config = {
            "model": "glm-4.7-flash",
            "name": "test_agent",
            "system_prompt": "Test prompt",
            "temperature": 0.2,
        }
        agent = OllamaAgent.from_dict(config, mock_config_manager)

        assert agent.temperature == 0.2

    @patch("pithos.agent.agent.ConfigManager")
    def test_from_dict_with_zero_temperature(self, mock_config_manager):
        """Test creating agent with temperature 0 from config."""
        config = {
            "model": "glm-4.7-flash",
            "name": "test_agent",
            "temperature": 0,
        }
        agent = OllamaAgent.from_dict(config, mock_config_manager)

        assert agent.temperature == 0

    @patch("pithos.agent.agent.ConfigManager")
    def test_from_dict_defaults(self, mock_config_manager):
        """Test that from_dict uses defaults when params not specified."""
        config = {
            "model": "glm-4.7-flash",
        }
        agent = OllamaAgent.from_dict(config, mock_config_manager)

        assert agent.temperature == 0.7  # Default temperature
        assert agent.max_tokens == -1  # Default max_tokens (unlimited)


class TestOllamaAgentStreaming:
    """Tests for OllamaAgent.stream() method."""

    @patch("pithos.agent.ollama_agent.chat")
    def test_stream_yields_chunks(self, mock_chat):
        """stream() should yield each token as it arrives."""
        chunk1, chunk2, chunk3 = Mock(), Mock(), Mock()
        chunk1.message.content = "Hello"
        chunk2.message.content = " world"
        chunk3.message.content = "!"
        mock_chat.return_value = iter([chunk1, chunk2, chunk3])

        agent = OllamaAgent("glm-4.7-flash")
        chunks = list(agent.stream("Hi"))

        assert chunks == ["Hello", " world", "!"]

    @patch("pithos.agent.ollama_agent.chat")
    def test_stream_calls_ollama_with_stream_true(self, mock_chat):
        """stream() must call ollama with stream=True."""
        mock_chat.return_value = iter([])

        agent = OllamaAgent("glm-4.7-flash")
        list(agent.stream("test"))

        call_kwargs = mock_chat.call_args[1]
        assert call_kwargs.get("stream") is True

    @patch("pithos.agent.ollama_agent.chat")
    def test_stream_updates_context_after_exhaustion(self, mock_chat):
        """Context history is updated only after all chunks are consumed (generator is lazy)."""
        chunk = Mock()
        chunk.message.content = "response"
        mock_chat.return_value = iter([chunk])

        agent = OllamaAgent("glm-4.7-flash")
        gen = agent.stream("Hello")

        # Generator has not been advanced at all — no code has run yet
        assert len(agent.contexts["default"].message_history) == 0

        list(gen)  # exhaust all chunks

        # After exhaustion: user + assistant messages are committed
        history = agent.contexts["default"].message_history
        assert len(history) == 2
        assert history[0]["role"] == "user"
        assert history[1]["role"] == "assistant"
        assert history[1]["content"] == "response"

    @patch("pithos.agent.ollama_agent.chat")
    def test_stream_assembles_full_response(self, mock_chat):
        """The assembled text in context should equal the concatenation of all chunks."""
        chunks = [Mock(), Mock(), Mock()]
        for tok, c in zip(["foo", " ", "bar"], chunks):
            c.message.content = tok
        mock_chat.return_value = iter(chunks)

        agent = OllamaAgent("glm-4.7-flash")
        list(agent.stream("question"))

        content = agent.contexts["default"].message_history[-1]["content"]
        assert content == "foo bar"

    @patch("pithos.agent.ollama_agent.chat")
    def test_stream_passes_temperature(self, mock_chat):
        """stream() should forward temperature in options."""
        mock_chat.return_value = iter([])

        agent = OllamaAgent("glm-4.7-flash", temperature=0.2)
        list(agent.stream("Hi"))

        call_kwargs = mock_chat.call_args[1]
        assert call_kwargs["options"]["temperature"] == 0.2

    @patch("pithos.agent.ollama_agent.chat")
    def test_stream_omits_num_predict(self, mock_chat):
        """stream() must not include num_predict since max_tokens is always -1."""
        mock_chat.return_value = iter([])

        agent = OllamaAgent("glm-4.7-flash")
        list(agent.stream("Hi"))

        call_kwargs = mock_chat.call_args[1]
        assert "num_predict" not in call_kwargs["options"]

    @patch("pithos.agent.ollama_agent.chat")
    def test_stream_handles_empty_chunk_content(self, mock_chat):
        """Chunks with None or empty content should not corrupt the output."""
        c1, c2, c3 = Mock(), Mock(), Mock()
        c1.message.content = "A"
        c2.message.content = None
        c3.message.content = "B"
        mock_chat.return_value = iter([c1, c2, c3])

        agent = OllamaAgent("glm-4.7-flash")
        result = list(agent.stream("Hi"))

        assert result == ["A", "", "B"]
        assembled = agent.contexts["default"].message_history[-1]["content"]
        assert assembled == "AB"

    @patch("pithos.agent.ollama_agent.chat")
    def test_stream_removes_message_on_error(self, mock_chat):
        """If the Ollama call raises, the queued user message must be rolled back."""
        from ollama._types import ResponseError

        mock_chat.side_effect = ResponseError("boom")

        agent = OllamaAgent("glm-4.7-flash")
        with pytest.raises(ResponseError):
            list(agent.stream("Hi"))

        assert len(agent.contexts["default"].message_history) == 0

    def test_send_collects_stream_into_string(self):
        """send() must join all stream() chunks into a single string."""

        class _StubAgent(Agent):
            """Minimal concrete agent that implements stream()."""

            def stream(
                self,
                content,
                context_name=None,
                workspace=None,
                verbose=False,
                model=None,
            ):
                yield "chunk1"
                yield " chunk2"

        agent = _StubAgent("stub-model")
        result = agent.send("Hello")

        assert result == "chunk1 chunk2"


class TestAgentInferenceFlowchart:
    """Tests for optional chain-of-thought inference flowchart."""

    def test_inference_flowchart_not_set_by_default(self):
        agent = OllamaAgent("glm-4.7-flash")
        assert agent.inference_flowchart is None
        assert agent._inference_config is None
        assert agent._running_inference is False

    def test_set_inference_flowchart_with_instance(self):
        """Setting a Flowchart instance directly."""
        from pithos.flowchart import Flowchart
        from pithos.config_manager import ConfigManager
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            cm = ConfigManager(config_dir=tmpdir)
            fc = Flowchart(cm)
            fc.add_node("n1", type="prompt", prompt="{current_input}", extraction={})
            fc.set_start_node("n1")

            agent = OllamaAgent("glm-4.7-flash")
            agent.set_inference_flowchart(fc)

            assert agent.inference_flowchart is fc
            assert agent._inference_config is None

    def test_set_inference_flowchart_with_dict(self):
        """Setting an inline flowchart via dict config."""
        from pithos.config_manager import ConfigManager
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            cm = ConfigManager(config_dir=tmpdir)
            fc_dict = {
                "nodes": {
                    "Generate": {
                        "type": "prompt",
                        "prompt": "{current_input}",
                        "extraction": {},
                    },
                },
                "edges": [],
                "start_node": "Generate",
            }
            agent = OllamaAgent("glm-4.7-flash")
            agent.set_inference_flowchart(fc_dict, cm)

            assert agent.inference_flowchart is not None
            assert agent._inference_config is fc_dict

    def test_set_inference_flowchart_with_registered_name(self):
        """Setting a flowchart by registered name."""
        from pithos.config_manager import ConfigManager
        from pithos.flowchart import Flowchart
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            cm = ConfigManager(config_dir=tmpdir)
            # Register a flowchart
            fc_data = {
                "nodes": {
                    "N1": {
                        "type": "prompt",
                        "prompt": "{current_input}",
                        "extraction": {},
                    },
                },
                "edges": [],
                "start_node": "N1",
            }
            cm.register_config(fc_data, "test_fc", "flowcharts")

            agent = OllamaAgent("glm-4.7-flash")
            agent.set_inference_flowchart("test_fc", cm)

            assert agent.inference_flowchart is not None
            assert agent._inference_config == "test_fc"

    def test_set_inference_flowchart_no_config_manager_for_string(self):
        agent = OllamaAgent("glm-4.7-flash")
        with pytest.raises(ValueError, match="config_manager is required"):
            agent.set_inference_flowchart("some_flowchart")

    def test_set_inference_flowchart_no_config_manager_for_dict(self):
        agent = OllamaAgent("glm-4.7-flash")
        with pytest.raises(ValueError, match="config_manager is required"):
            agent.set_inference_flowchart({"nodes": {}, "edges": []})

    def test_set_inference_flowchart_invalid_type(self):
        agent = OllamaAgent("glm-4.7-flash")
        with pytest.raises(TypeError, match="Unsupported"):
            agent.set_inference_flowchart(42)

    def test_clear_inference_flowchart(self):
        from pithos.flowchart import Flowchart
        from pithos.config_manager import ConfigManager
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            cm = ConfigManager(config_dir=tmpdir)
            fc = Flowchart(cm)
            fc.add_node("n1", type="prompt", prompt="{current_input}", extraction={})
            fc.set_start_node("n1")

            agent = OllamaAgent("glm-4.7-flash")
            agent.set_inference_flowchart(fc)
            assert agent.inference_flowchart is not None

            agent.clear_inference_flowchart()
            assert agent.inference_flowchart is None
            assert agent._inference_config is None

    def test_to_dict_includes_inference_config_string(self):
        from pithos.config_manager import ConfigManager
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            cm = ConfigManager(config_dir=tmpdir)
            fc_data = {
                "nodes": {
                    "N1": {
                        "type": "prompt",
                        "prompt": "{current_input}",
                        "extraction": {},
                    },
                },
                "edges": [],
                "start_node": "N1",
            }
            cm.register_config(fc_data, "my_fc", "flowcharts")

            agent = OllamaAgent("glm-4.7-flash")
            agent.set_inference_flowchart("my_fc", cm)

            d = agent.to_dict()
            assert d["inference"] == "my_fc"

    def test_to_dict_includes_inference_config_dict(self):
        from pithos.config_manager import ConfigManager
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            cm = ConfigManager(config_dir=tmpdir)
            fc_dict = {
                "nodes": {
                    "N": {
                        "type": "prompt",
                        "prompt": "{current_input}",
                        "extraction": {},
                    },
                },
                "edges": [],
                "start_node": "N",
            }
            agent = OllamaAgent("glm-4.7-flash")
            agent.set_inference_flowchart(fc_dict, cm)

            d = agent.to_dict()
            assert d["inference"] is fc_dict

    def test_to_dict_no_inference_when_not_set(self):
        agent = OllamaAgent("glm-4.7-flash")
        d = agent.to_dict()
        assert "inference" not in d

    @patch("pithos.agent.agent.ConfigManager")
    def test_from_dict_loads_inline_inference(self, MockCM):
        """from_dict should build an inference flowchart from inline config."""
        from pithos.config_manager import ConfigManager
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            cm = ConfigManager(config_dir=tmpdir)
            config = {
                "model": "glm-4.7-flash",
                "name": "cot_agent",
                "inference": {
                    "nodes": {
                        "Generate": {
                            "type": "prompt",
                            "prompt": "{current_input}",
                            "extraction": {},
                        },
                    },
                    "edges": [],
                    "start_node": "Generate",
                },
            }

            agent = OllamaAgent.from_dict(config, cm)
            assert agent.inference_flowchart is not None
            assert agent._inference_config == config["inference"]

    @patch("pithos.agent.ollama_agent.chat")
    def test_send_uses_inference_flowchart(self, mock_chat):
        """When inference flowchart is set, send() should route through it."""
        from pithos.config_manager import ConfigManager
        from pithos.flowchart import Flowchart
        import tempfile

        # The flowchart has a single PromptNode that calls the agent.
        # The agent.send() (with _running_inference=True) will call the
        # LLM directly. We mock to return a known response.
        mock_chunk = Mock()
        mock_chunk.message.content = "Reflected answer"
        mock_chat.return_value = iter([mock_chunk])

        with tempfile.TemporaryDirectory() as tmpdir:
            cm = ConfigManager(config_dir=tmpdir)
            fc_dict = {
                "nodes": {
                    "Generate": {
                        "type": "prompt",
                        "prompt": "{current_input}",
                        "extraction": {},
                    },
                },
                "edges": [],
                "start_node": "Generate",
            }
            fc = Flowchart.from_dict(fc_dict, cm)

            agent = OllamaAgent("glm-4.7-flash")
            agent.set_inference_flowchart(fc)

            result = agent.send("Hello world")

            # The flowchart PromptNode should have invoked chat()
            assert mock_chat.called
            # The result should be the LLM response routed through the flowchart
            assert result == "Reflected answer"
            # Main context records user + assistant messages
            history = agent.contexts["default"].message_history
            assert len(history) == 2
            assert history[0]["role"] == "user"
            assert history[0]["content"] == "Hello world"
            assert history[1]["role"] == "assistant"
            assert history[1]["content"] == "Reflected answer"

    @patch("pithos.agent.ollama_agent.chat")
    def test_send_without_inference_flowchart_unchanged(self, mock_chat):
        """Without inference flowchart, send() behaves exactly as before."""
        mock_chunk = Mock()
        mock_chunk.message.content = "Direct response"
        mock_chat.return_value = iter([mock_chunk])

        agent = OllamaAgent("glm-4.7-flash")
        result = agent.send("Hello")

        assert result == "Direct response"
        assert agent.inference_flowchart is None

    @patch("pithos.agent.ollama_agent.chat")
    def test_inference_cleans_up_temp_context(self, mock_chat):
        """Temporary inference context should not leak."""
        mock_chunk = Mock()
        mock_chunk.message.content = "response"
        mock_chat.return_value = iter([mock_chunk])

        from pithos.config_manager import ConfigManager
        from pithos.flowchart import Flowchart
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            cm = ConfigManager(config_dir=tmpdir)

            fc_dict = {
                "nodes": {
                    "N": {
                        "type": "prompt",
                        "prompt": "{current_input}",
                        "extraction": {},
                    },
                },
                "edges": [],
                "start_node": "N",
            }
            fc = Flowchart.from_dict(fc_dict, cm)

            agent = OllamaAgent("glm-4.7-flash")
            agent.set_inference_flowchart(fc)

            contexts_before = set(agent.list_contexts())
            agent.send("test")
            contexts_after = set(agent.list_contexts())

            # No temp contexts should remain
            assert contexts_before == contexts_after
            # Current context should still be default
            assert agent.current_context == "default"

    @patch("pithos.agent.ollama_agent.chat")
    def test_inference_resets_running_flag_on_error(self, mock_chat):
        """_running_inference flag resets even if flowchart fails."""
        mock_chat.side_effect = Exception("LLM down")

        from pithos.config_manager import ConfigManager
        from pithos.flowchart import Flowchart
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            cm = ConfigManager(config_dir=tmpdir)
            fc_dict = {
                "nodes": {
                    "N": {
                        "type": "prompt",
                        "prompt": "{current_input}",
                        "extraction": {},
                    },
                },
                "edges": [],
                "start_node": "N",
            }
            fc = Flowchart.from_dict(fc_dict, cm)

            agent = OllamaAgent("glm-4.7-flash")
            agent.set_inference_flowchart(fc)

            with pytest.raises(RuntimeError, match="Inference flowchart"):
                agent.send("test")

            assert agent._running_inference is False
            assert agent.current_context == "default"

    @patch("pithos.agent.ollama_agent.chat")
    def test_stream_with_inference_flowchart(self, mock_chat):
        """stream() should yield flowchart result as single chunk."""
        mock_chunk = Mock()
        mock_chunk.message.content = "streamed via cot"
        mock_chat.return_value = iter([mock_chunk])

        from pithos.config_manager import ConfigManager
        from pithos.flowchart import Flowchart
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            cm = ConfigManager(config_dir=tmpdir)
            fc_dict = {
                "nodes": {
                    "N": {
                        "type": "prompt",
                        "prompt": "{current_input}",
                        "extraction": {},
                    },
                },
                "edges": [],
                "start_node": "N",
            }
            fc = Flowchart.from_dict(fc_dict, cm)

            agent = OllamaAgent("glm-4.7-flash")
            agent.set_inference_flowchart(fc)

            chunks = list(agent.stream("Hello"))
            assert chunks == ["streamed via cot"]

    @patch("pithos.agent.ollama_agent.chat")
    def test_multi_step_inference_flowchart(self, mock_chat):
        """A multi-node inference flowchart should call LLM multiple times."""
        call_count = [0]

        def side_effect(**kwargs):
            call_count[0] += 1
            chunk = Mock()
            chunk.message.content = f"Step {call_count[0]} output"
            return iter([chunk])

        mock_chat.side_effect = side_effect

        from pithos.config_manager import ConfigManager
        from pithos.flowchart import Flowchart
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            cm = ConfigManager(config_dir=tmpdir)
            fc_dict = {
                "nodes": {
                    "Step1": {
                        "type": "prompt",
                        "prompt": "{current_input}",
                        "extraction": {},
                    },
                    "Step2": {
                        "type": "prompt",
                        "prompt": "Reflect: {current_input}",
                        "extraction": {},
                    },
                },
                "edges": [
                    {
                        "from": "Step1",
                        "to": "Step2",
                        "condition": {"type": "AlwaysCondition"},
                    }
                ],
                "start_node": "Step1",
            }
            fc = Flowchart.from_dict(fc_dict, cm)

            agent = OllamaAgent("glm-4.7-flash")
            agent.set_inference_flowchart(fc)

            result = agent.send("Question")

            # LLM should have been called at least twice (once per PromptNode)
            assert call_count[0] >= 2
            # Main context should only have user + final assistant
            history = agent.contexts["default"].message_history
            assert history[0]["role"] == "user"
            assert history[0]["content"] == "Question"
            assert history[-1]["role"] == "assistant"
