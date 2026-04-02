"""Integration tests for agent tool calling."""

import pytest
from unittest.mock import Mock, patch
from pithos.agent import OllamaAgent
from pithos.config_manager import ConfigManager
from pithos.tools import ToolRegistry, ToolExecutor, ToolResult


class TestAgentToolCalling:
    """Tests for agent tool calling integration."""

    @pytest.fixture
    def config_manager(self, tmp_path):
        """Create a ConfigManager with test configuration."""
        config_dir = tmp_path / "configs" / "tools"
        config_dir.mkdir(parents=True)

        config_file = config_dir / "tool_config.yaml"
        config_content = """
enabled: true
timeout: 5
max_output_size: 1000
mode: include
include:
  - echo
  - python
descriptions:
  echo: "Display a line of text"
exclude: []
"""
        config_file.write_text(config_content)

        return ConfigManager(str(tmp_path / "configs"))

    @pytest.fixture
    def agent(self):
        """Create a test agent."""
        return OllamaAgent(
            default_model="test-model", system_prompt="You are a helpful assistant."
        )

    def test_agent_enable_tools(self, agent, config_manager):
        """Test enabling tools for an agent."""
        agent.enable_tools(config_manager)

        assert agent.tools_enabled is True
        assert agent.tool_registry is not None
        assert agent.tool_executor is not None
        assert isinstance(agent.tool_registry, ToolRegistry)
        assert isinstance(agent.tool_executor, ToolExecutor)

    def test_agent_tool_prompt_enhancement(self, agent, config_manager):
        """Test that tool prompt is added to system prompt."""
        initial_prompt = agent.contexts["default"].get_system_prompt()
        agent.enable_tools(config_manager)

        enhanced_prompt = agent.contexts["default"].get_system_prompt()
        assert len(enhanced_prompt) > len(initial_prompt)
        # Check for new multi-format support
        assert any(
            keyword in enhanced_prompt
            for keyword in ["RUN:", "run(", "Tool Call Formats"]
        )
        assert "Available tools" in enhanced_prompt

    def test_extract_tool_calls(self, agent):
        """Test extracting tool calls from agent response with multiple formats."""
        # Legacy format - double quotes
        content1 = 'Let me check: runcommand("python --version")'
        calls1 = agent._extract_tool_calls(content1)
        assert len(calls1) == 1
        assert calls1[0].command == "python --version"
        assert calls1[0].format == "legacy"

        # CLI-style format
        content2 = "Let me check:\nRUN: python --version"
        calls2 = agent._extract_tool_calls(content2)
        assert len(calls2) == 1
        assert calls2[0].command == "python --version"
        assert calls2[0].format == "cli"

        # Function-style format
        content3 = "Let me check: run(python --version)"
        calls3 = agent._extract_tool_calls(content3)
        assert len(calls3) == 1
        assert calls3[0].command == "python --version"
        assert calls3[0].format == "function"

        # Multiple calls with mixed formats
        content4 = 'First runcommand("python --version") then\nRUN: echo done'
        calls4 = agent._extract_tool_calls(content4)
        assert len(calls4) == 2
        commands = [call.command for call in calls4]
        assert "python --version" in commands
        assert "echo done" in commands

        # No calls
        content5 = "Just a regular response"
        calls5 = agent._extract_tool_calls(content5)
        assert calls5 == []

    def test_format_tool_result(self, agent):
        """Test formatting tool results."""
        result = ToolResult(
            success=True,
            stdout="Python 3.10.0",
            stderr="",
            exit_code=0,
            execution_time=0.1,
            command="python --version",
            error_hint=None,
        )

        formatted = agent._format_tool_result(result)
        assert "python --version" in formatted
        assert "Success" in formatted or "✓" in formatted
        assert "Python 3.10.0" in formatted

    def test_format_tool_result_with_error(self, agent):
        """Test formatting tool results with errors."""
        result = ToolResult(
            success=False,
            stdout="",
            stderr="Command not found",
            exit_code=127,
            execution_time=0.01,
            command="badcommand",
            error_hint="Check that the command exists",
        )

        formatted = agent._format_tool_result(result)
        assert "badcommand" in formatted
        assert "Failed" in formatted or "✗" in formatted
        assert "Command not found" in formatted
        assert "Hint" in formatted or "💡" in formatted

    def test_execute_tools(self, agent, config_manager):
        """Test executing tools with new request format."""
        from pithos.tools import ToolCallRequest

        agent.enable_tools(config_manager)

        # Mock the executor
        mock_result = ToolResult(
            success=True,
            stdout="Test output",
            stderr="",
            exit_code=0,
            execution_time=0.1,
            command="echo test",
            error_hint=None,
        )
        agent.tool_executor.run = Mock(return_value=mock_result)

        # Create tool request
        request = ToolCallRequest(
            command="echo test", format="cli", raw_text="RUN: echo test"
        )

        # Execute tools
        result_text = agent._execute_tools([request])

        assert "echo test" in result_text
        assert "Test output" in result_text
        assert "Success" in result_text or "✓" in result_text

    @patch("pithos.agent.ollama_agent.chat")
    def test_agent_send_with_tool_calls(self, mock_chat, agent, config_manager):
        """Test agent send with tool calling."""
        agent.enable_tools(config_manager)

        # First stream: LLM emits a tool call mid-response.
        mock_tool_chunk = Mock()
        mock_tool_chunk.message.content = (
            'Let me check: runcommand("python --version")\n'
        )
        # Continuation stream after tool result injected.
        mock_cont_chunk = Mock()
        mock_cont_chunk.message.content = "The Python version is 3.10.0."
        mock_chat.side_effect = [iter([mock_tool_chunk]), iter([mock_cont_chunk])]

        # Mock tool execution
        mock_result = ToolResult(
            success=True,
            stdout="Python 3.10.0",
            stderr="",
            exit_code=0,
            execution_time=0.1,
            command="python --version",
        )
        agent.tool_executor.run = Mock(return_value=mock_result)

        # Send message
        response = agent.send("What Python version?", verbose=False)

        # Check that tool was executed
        agent.tool_executor.run.assert_called_once()

        # Check that tool result was added to context
        messages = agent.contexts["default"].message_history
        assert (
            len(messages) >= 2
        )  # User message + agent message (+ potentially system message)

    def test_agent_without_tools_enabled(self, agent):
        """Test that agent works normally without tools enabled."""
        # Agent should not have tools enabled by default
        assert agent.tools_enabled is False
        assert agent.tool_registry is None
        assert agent.tool_executor is None

        # Extract should return empty list
        calls = agent._extract_tool_calls('runcommand("test")')
        assert isinstance(calls, list)

    @patch("pithos.agent.ollama_agent.chat")
    def test_agent_send_no_tool_calls(self, mock_chat, agent, config_manager):
        """Test agent send without tool calls."""
        agent.enable_tools(config_manager)

        # Mock the LLM response without tool calls
        mock_chunk = Mock()
        mock_chunk.message.content = "Just a regular response"
        mock_chat.return_value = iter([mock_chunk])

        # Send message
        response = agent.send("Tell me something", verbose=False)

        assert response == "Just a regular response"
        assert len(agent.contexts["default"].message_history) == 2  # User + assistant

    def test_tool_auto_loop_disabled(self, agent, config_manager):
        """Test that auto loop is disabled by default."""
        agent.enable_tools(config_manager, auto_loop=False)
        assert agent.tool_auto_loop is False

    def test_tool_auto_loop_enabled(self, agent, config_manager):
        """Test enabling auto loop."""
        agent.enable_tools(config_manager, auto_loop=True, max_iterations=3)
        assert agent.tool_auto_loop is True
        assert agent.tool_max_iterations == 3


class TestToolCallNode:
    """Tests for ToolCallNode in flowcharts."""

    @pytest.fixture
    def config_manager(self, tmp_path):
        """Create a ConfigManager."""
        config_dir = tmp_path / "configs"
        config_dir.mkdir(parents=True)
        return ConfigManager(str(config_dir))

    def test_toolcall_node_creation(self):
        """Test creating a ToolCallNode."""
        from pithos.flownode import ToolCallNode

        node = ToolCallNode(
            command="python --version",
            save_to="python_version",
            error_handling="continue",
        )

        assert node.command == "python --version"
        assert node.save_to == "python_version"
        assert node.error_handling == "continue"

    def test_toolcall_node_execution(self):
        """Test ToolCallNode execution."""
        from pithos.flownode import ToolCallNode
        from pithos.tools import ToolResult
        from pithos.message import NodeInputState, Message, MessageRouter

        node = ToolCallNode(
            command="python --version",
            save_to="result",
            error_handling="continue",
            inputs=["default"],
            outputs=["default"],
        )

        # Mock tool executor and registry
        mock_executor = Mock()
        mock_registry = Mock()
        mock_executor.run.return_value = ToolResult(
            success=True,
            stdout="Python 3.10.0",
            stderr="",
            exit_code=0,
            execution_time=0.1,
            command="python --version",
        )

        # Create message router with tools in shared_context
        router = MessageRouter()
        router.shared_context["tool_executor"] = mock_executor
        router.shared_context["tool_registry"] = mock_registry

        # Create input state and send message
        input_state = NodeInputState(node_id="test", required_inputs=["default"])
        msg = Message(data="input", input_key="default")
        input_state.receive_message(msg)

        # Execute node
        outputs = node.execute_with_messages(input_state, message_router=router)

        # Verify
        assert len(outputs) > 0
        assert outputs[0].data["success"] is True
        assert outputs[0].data["stdout"] == "Python 3.10.0"
        mock_executor.run.assert_called_once()

    def test_toolcall_node_with_state_variables(self):
        """Test ToolCallNode with state variable substitution."""
        from pithos.flownode import ToolCallNode
        from pithos.tools import ToolResult
        from pithos.message import NodeInputState, Message, MessageRouter

        node = ToolCallNode(
            command="echo {message}",
            save_to="result",
            inputs=["default"],
            outputs=["default"],
        )

        mock_executor = Mock()
        mock_registry = Mock()
        mock_executor.run.return_value = ToolResult(
            success=True,
            stdout="Hello World",
            stderr="",
            exit_code=0,
            execution_time=0.1,
            command="echo Hello World",
        )

        # Create message router with tools and variables
        router = MessageRouter()
        router.shared_context["tool_executor"] = mock_executor
        router.shared_context["tool_registry"] = mock_registry
        router.shared_context["message"] = "Hello World"

        # Create input state
        input_state = NodeInputState(node_id="test", required_inputs=["default"])
        msg = Message(data="input", input_key="default")
        input_state.receive_message(msg)

        # Execute node
        outputs = node.execute_with_messages(input_state, message_router=router)

        # Verify command was formatted with state variable
        call_args = mock_executor.run.call_args[0][0]
        assert "Hello World" in call_args

    def test_toolcall_node_error_handling_stop(self):
        """Test ToolCallNode with stop error handling."""
        from pithos.flownode import ToolCallNode
        from pithos.tools import ToolResult
        from pithos.message import NodeInputState, Message, MessageRouter

        node = ToolCallNode(
            command="badcommand",
            save_to="result",
            error_handling="stop",
            inputs=["default"],
            outputs=["default"],
        )

        mock_executor = Mock()
        mock_registry = Mock()
        mock_executor.run.return_value = ToolResult(
            success=False,
            stdout="",
            stderr="Command not found",
            exit_code=127,
            execution_time=0.01,
            command="badcommand",
        )

        # Create message router with tools
        router = MessageRouter()
        router.shared_context["tool_executor"] = mock_executor
        router.shared_context["tool_registry"] = mock_registry

        # Create input state
        input_state = NodeInputState(node_id="test", required_inputs=["default"])
        msg = Message(data="input", input_key="default")
        input_state.receive_message(msg)

        # Should raise RuntimeError
        with pytest.raises(RuntimeError, match="Tool execution failed"):
            node.execute_with_messages(input_state, message_router=router)

    def test_toolcall_node_no_tools_available(self):
        """Test ToolCallNode when tools are not available."""
        from pithos.flownode import ToolCallNode
        from pithos.message import NodeInputState, Message, MessageRouter

        node = ToolCallNode(
            command="python --version",
            save_to="result",
            error_handling="continue",
            inputs=["default"],
            outputs=["default"],
        )

        # Create message router without tools
        router = MessageRouter()

        # Create input state
        input_state = NodeInputState(node_id="test", required_inputs=["default"])
        msg = Message(data="input", input_key="default")
        input_state.receive_message(msg)

        # Execute node
        outputs = node.execute_with_messages(input_state, message_router=router)

        assert len(outputs) > 0
        assert outputs[0].data["success"] is False
        assert "not available" in outputs[0].data["stderr"]

    def test_create_node_toolcall(self):
        """Test creating ToolCallNode via factory."""
        from pithos.flownode import create_node

        data = {
            "command": "python --version",
            "save_to": "result",
            "error_handling": "continue",
        }

        node = create_node("toolcall", data)
        assert node is not None
        assert node.command == "python --version"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
