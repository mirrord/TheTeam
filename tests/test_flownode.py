"""Unit tests for flownode module."""

import pytest
from unittest.mock import MagicMock
from pithos.message import NodeInputState, Message, MessageRouter
from pithos.flownode import (
    FlowNode,
    PromptNode,
    CustomNode,
    TextParseNode,
    AgentPromptNode,
    GetHistoryNode,
    SetHistoryNode,
    create_node,
)
from pithos.agent import OllamaAgent


class TestFlowNode:
    """Test FlowNode base class."""

    def test_flownode_creation(self):
        node = FlowNode()
        assert node.extraction == {}
        assert node.set == {}
        assert node.prompt_args == {}

    def test_flownode_with_extraction(self):
        extraction = {"var": r"\d+"}
        node = FlowNode(extraction=extraction)
        assert node.extraction == {"var": r"\d+"}

    def test_parse_extractions(self):
        node = FlowNode(extraction={"number": r"\d+"})
        result = node.parse_extractions("The number is 42")
        assert result == {"number": "42"}

    def test_parse_extractions_no_match(self):
        node = FlowNode(extraction={"number": r"\d+"})
        result = node.parse_extractions("No numbers here")
        assert result == {}

    def test_set_values(self):
        node = FlowNode(set={"output": "test_value"})
        state = {}
        result = node.set_values(state)
        assert result["output"] == "test_value"

    def test_stateful_format(self):
        node = FlowNode()
        state = {"name": "World"}
        result = node._stateful_format("Hello {name}", state)
        assert result == "Hello World"

    def test_stateful_format_no_vars(self):
        node = FlowNode()
        state = {"unused": "value"}
        result = node._stateful_format("Hello World", state)
        assert result == "Hello World"

    def test_stateful_format_non_string(self):
        node = FlowNode()
        state = {}
        result = node._stateful_format(42, state)
        assert result == 42

    def test_to_dict(self):
        node = FlowNode(extraction={"var": r"\d+"}, set={"key": "value"})
        d = node.to_dict()
        assert d["extraction"] == {"var": r"\d+"}
        assert d["set"] == {"key": "value"}
        assert d["type"] == "flow"

    def test_from_dict(self):
        data = {
            "type": "flow",
            "extraction": {"var": r"\d+"},
            "set": {"key": "value"},
        }
        node = FlowNode.from_dict(data)
        assert node.extraction == {"var": r"\d+"}
        assert node.set == {"key": "value"}


class TestPromptNode:
    """Test PromptNode for prompting."""

    def test_prompt_node_creation(self):
        node = PromptNode(extraction={}, prompt="Test prompt")
        assert node.prompt == "Test prompt"

    def test_prompt_node_do(self):
        node = PromptNode(
            extraction={},
            prompt="Hello {name}",
            inputs=["default"],
            outputs=["default"],
        )
        router = MessageRouter()
        router.shared_context["name"] = "World"

        input_state = NodeInputState(node_id="test", required_inputs=["default"])
        msg = Message(data="input", input_key="default")
        input_state.receive_message(msg)

        # Build context and execute to get result
        context = node._build_context_from_messages(input_state, router)
        result = node._execute(context)

        assert result["formatted_prompt"] == "Hello World"

    def test_prompt_node_do_with_extraction(self):
        node = PromptNode(
            extraction={"number": r"\d+"},
            prompt="The number is {number}",
            inputs=["default"],
            outputs=["default"],
        )
        router = MessageRouter()

        input_state = NodeInputState(node_id="test", required_inputs=["default"])
        msg = Message(data="Found 42 items", input_key="default")
        input_state.receive_message(msg)

        # Build context and execute to get result
        context = node._build_context_from_messages(input_state, router)
        result = node._execute(context)

        assert context["number"] == "42"  # Extracted during context building
        assert result["formatted_prompt"] == "The number is 42"

    def test_prompt_node_to_dict(self):
        node = PromptNode(extraction={}, prompt="Test")
        d = node.to_dict()
        assert d["type"] == "prompt"
        assert d["prompt"] == "Test"

    def test_prompt_node_from_dict(self):
        data = {"type": "prompt", "prompt": "Test prompt", "extraction": {}}
        node = PromptNode.from_dict(data)
        assert node.prompt == "Test prompt"


class TestCustomNode:
    """Test CustomNode for custom code execution."""

    def test_custom_node_creation(self):
        code = "state['output'] = 'test'"
        node = CustomNode(extraction={}, custom_code=code)
        assert node.custom_code == code

    def test_custom_node_do(self):
        code = "context['result'] = context.get('value', 0) * 2"
        node = CustomNode(
            extraction={}, custom_code=code, inputs=["default"], outputs=["default"]
        )
        router = MessageRouter()
        router.shared_context["value"] = 21

        input_state = NodeInputState(node_id="test", required_inputs=["default"])
        msg = Message(data="input", input_key="default")
        input_state.receive_message(msg)

        # Build context and execute to get result
        context = node._build_context_from_messages(input_state, router)
        result = node._execute(context)

        assert result["result"] == 42

    def test_custom_node_do_modifies_state(self):
        code = """
context['a'] = 10
context['b'] = 20
context['sum'] = context['a'] + context['b']
"""
        node = CustomNode(
            extraction={}, custom_code=code, inputs=["default"], outputs=["default"]
        )
        router = MessageRouter()

        input_state = NodeInputState(node_id="test", required_inputs=["default"])
        msg = Message(data="input", input_key="default")
        input_state.receive_message(msg)

        # Build context and execute to get result
        context = node._build_context_from_messages(input_state, router)
        result = node._execute(context)

        assert result["a"] == 10
        assert result["b"] == 20
        assert result["sum"] == 30

    def test_custom_node_to_dict(self):
        code = "state['x'] = 1"
        node = CustomNode(extraction={}, custom_code=code)
        d = node.to_dict()
        assert d["type"] == "custom"
        assert d["custom_code"] == code

    def test_custom_node_from_dict(self):
        data = {"type": "custom", "custom_code": "state['x'] = 1", "extraction": {}}
        node = CustomNode.from_dict(data)
        assert node.custom_code == "state['x'] = 1"

    # ------------------------------------------------------------------
    # Sandbox security tests
    # ------------------------------------------------------------------

    @pytest.mark.parametrize(
        "code",
        [
            "import os",
            "import sys",
            "import subprocess",
            "from os import path",
            "from subprocess import run",
        ],
    )
    def test_import_statements_are_blocked(self, code):
        """Import statements must be rejected at construction time."""
        with pytest.raises(ValueError, match="Import statements are not allowed"):
            CustomNode(extraction={}, custom_code=code)

    @pytest.mark.parametrize(
        "code",
        [
            "x = ().__class__.__bases__[0].__subclasses__()",
            "x = context.__class__",
            "x = {}.__class__",
        ],
    )
    def test_dunder_attribute_access_is_blocked(self, code):
        """Dunder attribute access must be rejected at construction time."""
        with pytest.raises(ValueError, match="dunder attribute"):
            CustomNode(extraction={}, custom_code=code)

    @pytest.mark.parametrize(
        "code",
        [
            "eval('1+1')",
            "exec('x=1')",
            "compile('x=1', '<string>', 'exec')",
            "open('/etc/passwd')",
            "getattr(context, '__class__')",
            "setattr(context, 'x', 1)",
            "delattr(context, 'x')",
        ],
    )
    def test_dangerous_calls_are_blocked(self, code):
        """Calls to dangerous built-ins must be rejected at construction time."""
        with pytest.raises(ValueError, match="not allowed in custom code"):
            CustomNode(extraction={}, custom_code=code)

    @pytest.mark.parametrize(
        "name",
        [
            "globals()",
            "locals()",
            "vars()",
            "dir()",
            "__import__('os')",
        ],
    )
    def test_blocked_names_are_blocked(self, name):
        """References to blocked names must be rejected at construction time."""
        with pytest.raises(ValueError, match="not allowed in custom code"):
            CustomNode(extraction={}, custom_code=name)

    def test_safe_builtins_work(self):
        """Standard safe built-ins like len, range, str should work fine."""
        code = """
items = list(range(5))
context['count'] = len(items)
context['joined'] = ','.join(str(i) for i in items)
"""
        node = CustomNode(extraction={}, custom_code=code)
        result = node._execute({})
        assert result["count"] == 5
        assert result["joined"] == "0,1,2,3,4"

    def test_open_not_in_builtins(self):
        """'open' must not be present in the sandbox execution environment."""
        code = "context['has_open'] = 'open' in dir()"
        # 'dir' is blocked at AST level, so construction should raise
        with pytest.raises(ValueError):
            CustomNode(extraction={}, custom_code=code)

    def test_restricted_builtins_excludes_open(self):
        """Verify that 'open' is absent from the safe builtins dict at module level."""
        from pithos.flownode import _SAFE_BUILTINS_DICT

        assert "open" not in _SAFE_BUILTINS_DICT
        assert "__import__" not in _SAFE_BUILTINS_DICT
        assert "eval" not in _SAFE_BUILTINS_DICT
        assert "exec" not in _SAFE_BUILTINS_DICT

    def test_timeout_raises_on_infinite_loop(self):
        """Execution timeout must fire for non-terminating code."""
        # Use a tight timeout so the test finishes quickly
        node = CustomNode(extraction={}, custom_code="while True: pass", timeout=0.25)
        with pytest.raises(TimeoutError, match="timed out"):
            node._execute({})

    def test_syntax_error_raises_value_error(self):
        """Invalid Python syntax must raise ValueError at construction time."""
        with pytest.raises(ValueError, match="Syntax error"):
            CustomNode(extraction={}, custom_code="def (: broken syntax")


class TestCreateNode:
    """Test create_node factory function."""

    def test_create_prompt_node(self):
        data = {"type": "prompt", "prompt": "Test", "extraction": {}}
        node = create_node("prompt", data)
        assert isinstance(node, PromptNode)
        assert node.prompt == "Test"

    def test_create_custom_node(self):
        data = {"type": "custom", "custom_code": "pass", "extraction": {}}
        node = create_node("custom", data)
        assert isinstance(node, CustomNode)

    def test_create_node_case_insensitive(self):
        data = {"type": "PROMPT", "prompt": "Test", "extraction": {}}
        node = create_node("PROMPT", data)
        assert isinstance(node, PromptNode)

    def test_create_node_with_underscore(self):
        data = {"type": "prompt_node", "prompt": "Test", "extraction": {}}
        node = create_node("prompt_node", data)
        assert isinstance(node, PromptNode)

    def test_create_node_invalid_type_raises(self):
        data = {"type": "invalid", "extraction": {}}
        with pytest.raises(ValueError, match="Unknown node type"):
            create_node("invalid", data)


class TestFlowNodeIntegration:
    """Integration tests for FlowNode processing."""

    def test_prompt_node_full_pipeline(self):
        """Test complete prompt node processing."""
        node = PromptNode(
            extraction={"var": r"value: (\d+)"},
            prompt="Extracted {var}",
            set={"step": "extraction"},
            inputs=["default"],
            outputs=["default"],
        )

        router = MessageRouter()

        input_state = NodeInputState(node_id="test", required_inputs=["default"])
        msg = Message(data="The value: 42 was found", input_key="default")
        input_state.receive_message(msg)

        # Build context and execute to get result
        context = node._build_context_from_messages(input_state, router)
        result = node._execute(context)

        assert context["var"] == "42"
        assert context["step"] == "extraction"
        assert result["formatted_prompt"] == "Extracted 42"

    def test_custom_node_complex_logic(self):
        """Test custom node with complex logic."""
        code = """
# Parse input and process
items = context.get('current_input', '').split(',')
context['count'] = len(items)
context['processed'] = [item.strip().upper() for item in items]
"""
        node = CustomNode(
            extraction={}, custom_code=code, inputs=["default"], outputs=["default"]
        )
        router = MessageRouter()
        router.shared_context["current_input"] = "apple, banana, cherry"

        input_state = NodeInputState(node_id="test", required_inputs=["default"])
        msg = Message(data="apple, banana, cherry", input_key="default")
        input_state.receive_message(msg)

        # Build context and execute to get result
        context = node._build_context_from_messages(input_state, router)
        result = node._execute(context)

        assert result["count"] == 3
        assert result["processed"] == ["APPLE", "BANANA", "CHERRY"]

    def test_node_serialization_roundtrip(self):
        """Test that nodes can be serialized and deserialized."""
        original = PromptNode(
            extraction={"x": r"\d+"}, prompt="Test {x}", set={"key": "val"}
        )
        data = original.to_dict()
        restored = PromptNode.from_dict(data)

        assert restored.prompt == original.prompt
        assert restored.extraction == original.extraction
        assert restored.set == original.set

    def test_extraction_with_multiple_patterns(self):
        """Test extraction with multiple regex patterns."""
        node = FlowNode(extraction={"number": r"\d+", "word": r"[A-Za-z]+"})

        result = node.parse_extractions("Found 123 apples")
        assert result["number"] == "123"
        assert result["word"] == "Found"


class TestTextParseNode:
    """Test TextParseNode class."""

    def test_textparse_node_creation(self):
        """Test creating a TextParseNode."""
        node = TextParseNode()
        assert node.extraction == {}
        assert node.set == {}
        assert node.transform is None

    def test_textparse_node_with_set(self):
        """Test TextParseNode with set configuration."""
        node = TextParseNode(
            set={"original_question": "{current_input}"},
            inputs=["default"],
            outputs=["default"],
        )
        router = MessageRouter()

        input_state = NodeInputState(node_id="test", required_inputs=["default"])
        msg = Message(data="What is 2+2?", input_key="default")
        input_state.receive_message(msg)
        router.shared_context["current_input"] = "What is 2+2?"

        # Build context and execute to get result
        context = node._build_context_from_messages(input_state, router)
        result = node._execute(context)

        assert context["original_question"] == "What is 2+2?"
        # TextParseNode returns context which includes the input
        assert context.get("default") or context.get("current_input") == "What is 2+2?"

    def test_textparse_node_with_extraction(self):
        """Test TextParseNode with regex extraction."""
        node = TextParseNode(
            extraction={"answer": r"Answer:\s*([A-D])"},
            set={"captured": "true"},
            inputs=["default"],
            outputs=["default"],
        )
        router = MessageRouter()

        input_state = NodeInputState(node_id="test", required_inputs=["default"])
        msg = Message(data="The Answer: B is correct", input_key="default")
        input_state.receive_message(msg)

        # Build context and execute to get result
        context = node._build_context_from_messages(input_state, router)
        result = node._execute(context)

        assert context["answer"] == "B"
        assert context["captured"] == "true"

    def test_textparse_node_multiple_extractions(self):
        """Test TextParseNode with multiple regex patterns."""
        node = TextParseNode(
            extraction={
                "letter": r"Answer:\s*([A-D])",
                "confidence": r"Confidence:\s*(\d+)%",
            },
            inputs=["default"],
            outputs=["default"],
        )
        router = MessageRouter()

        input_state = NodeInputState(node_id="test", required_inputs=["default"])
        msg = Message(data="Answer: C with Confidence: 95%", input_key="default")
        input_state.receive_message(msg)

        # Build context and execute to get result
        context = node._build_context_from_messages(input_state, router)
        result = node._execute(context)

        assert context["letter"] == "C"
        assert context["confidence"] == "95"

    def test_textparse_node_passes_through_input(self):
        """Test that TextParseNode passes through current_input unchanged."""
        node = TextParseNode(
            set={"saved": "{current_input}"}, inputs=["default"], outputs=["default"]
        )
        original_input = "Some text to parse"
        router = MessageRouter()
        router.shared_context["current_input"] = original_input

        input_state = NodeInputState(node_id="test", required_inputs=["default"])
        msg = Message(data=original_input, input_key="default")
        input_state.receive_message(msg)

        # Build context and execute to get result
        context = node._build_context_from_messages(input_state, router)
        result = node._execute(context)

        # current_input should remain in context
        assert (
            context.get("default") == original_input
            or context.get("current_input") == original_input
        )
        assert context["saved"] == original_input

    def test_textparse_node_from_dict(self):
        """Test creating TextParseNode from dictionary."""
        data = {
            "set": {"var": "value"},
            "extraction": {"num": r"\d+"},
            "transform": "upper",
        }
        node = TextParseNode.from_dict(data)

        assert node.set == {"var": "value"}
        assert node.extraction == {"num": r"\d+"}
        assert node.transform == "upper"

    def test_textparse_node_to_dict(self):
        """Test serializing TextParseNode to dictionary."""
        node = TextParseNode(
            extraction={"x": r"\d+"}, set={"y": "val"}, transform="lower"
        )
        data = node.to_dict()

        assert data["type"] == "textparse"
        assert data["extraction"] == {"x": r"\d+"}
        assert data["set"] == {"y": "val"}
        assert data["transform"] == "lower"

    def test_create_node_textparse(self):
        """Test creating TextParseNode via create_node factory."""
        data = {"set": {"question": "{current_input}"}, "extraction": {}}
        node = create_node("textparse", data)

        assert isinstance(node, TextParseNode)
        assert node.set == {"question": "{current_input}"}


class TestAgentPromptNode:
    """Test AgentPromptNode for team flowcharts."""

    def test_agent_prompt_node_creation(self):
        """Test creating an AgentPromptNode."""
        node = AgentPromptNode(
            agent="creative",
            prompt="Write a story about {topic}",
            extraction={},
        )
        assert node.agent == "creative"
        assert node.prompt == "Write a story about {topic}"
        assert node.model is None
        assert node.context_name is None

    def test_agent_prompt_node_with_model_override(self):
        """Test AgentPromptNode with model override."""
        node = AgentPromptNode(
            agent="analyst",
            prompt="Analyze {data}",
            extraction={},
            model="llama3.2:latest",
        )
        assert node.model == "llama3.2:latest"

    def test_agent_prompt_node_with_context_name(self):
        """Test AgentPromptNode with specific context."""
        node = AgentPromptNode(
            agent="writer",
            prompt="Continue the story",
            extraction={},
            context_name="story_context",
        )
        assert node.context_name == "story_context"

    def test_agent_prompt_node_do_executes_agent(self):
        """Test that AgentPromptNode executes agent's send method."""
        # Create mock agent
        mock_agent = MagicMock(spec=OllamaAgent)
        mock_agent.send.return_value = "Agent response"

        node = AgentPromptNode(
            agent="test_agent",
            prompt="Prompt with {variable}",
            extraction={},
            inputs=["default"],
            outputs=["default"],
        )

        router = MessageRouter()
        router.shared_context["agents"] = {"test_agent": mock_agent}
        router.shared_context["variable"] = "value"

        input_state = NodeInputState(node_id="test", required_inputs=["default"])
        msg = Message(data="input", input_key="default")
        input_state.receive_message(msg)

        # Build context and execute to get result
        context = node._build_context_from_messages(input_state, router)
        result = node._execute(context)

        # Verify agent.send was called with formatted prompt
        mock_agent.send.assert_called_once_with(
            "Prompt with value",
            context_name=None,
            model=None,
        )

        # Verify result contains expected values
        assert result["current_input"] == "Agent response"
        assert result["test_agent_response"] == "Agent response"
        assert result["formatted_prompt"] == "Prompt with value"

    def test_agent_prompt_node_do_missing_agent_raises(self):
        """Test that missing agent raises ValueError."""
        node = AgentPromptNode(
            agent="missing_agent",
            prompt="Test",
            extraction={},
            inputs=["default"],
            outputs=["default"],
        )

        router = MessageRouter()
        router.shared_context["agents"] = {}

        input_state = NodeInputState(node_id="test", required_inputs=["default"])
        msg = Message(data="input", input_key="default")
        input_state.receive_message(msg)

        with pytest.raises(ValueError, match="Agent 'missing_agent' not found"):
            node.execute_with_messages(input_state, message_router=router)

    def test_agent_prompt_node_to_dict(self):
        """Test serializing AgentPromptNode to dictionary."""
        node = AgentPromptNode(
            agent="writer",
            prompt="Write about {topic}",
            extraction={"key": r"\\d+"},
            model="gpt-4",
            context_name="ctx",
        )
        data = node.to_dict()

        assert data["type"] == "agentprompt"
        assert data["agent"] == "writer"
        assert data["prompt"] == "Write about {topic}"
        assert data["model"] == "gpt-4"
        assert data["context_name"] == "ctx"

    def test_agent_prompt_node_from_dict(self):
        """Test creating AgentPromptNode from dictionary."""
        data = {
            "type": "agentprompt",
            "agent": "analyst",
            "prompt": "Analyze {data}",
            "extraction": {},
            "model": "llama3.2",
        }
        node = AgentPromptNode.from_dict(data)

        assert node.agent == "analyst"
        assert node.prompt == "Analyze {data}"
        assert node.model == "llama3.2"


class TestGetHistoryNode:
    """Test GetHistoryNode for extracting agent history."""

    def test_get_history_node_creation(self):
        """Test creating a GetHistoryNode."""
        node = GetHistoryNode(
            agent="agent1",
            save_to="history",
            extraction={},
        )
        assert node.agent == "agent1"
        assert node.save_to == "history"
        assert node.context_name is None

    def test_get_history_node_do_extracts_history(self):
        """Test that GetHistoryNode extracts agent history."""
        # Create mock agent with history
        mock_agent = MagicMock(spec=OllamaAgent)
        mock_agent.current_context = "default"
        mock_context = MagicMock()
        mock_context.message_history = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
        ]
        mock_agent.contexts = {"default": mock_context}

        node = GetHistoryNode(
            agent="agent1",
            save_to="history",
            extraction={},
            inputs=["default"],
            outputs=["default"],
        )

        router = MessageRouter()
        router.shared_context["agents"] = {"agent1": mock_agent}

        input_state = NodeInputState(node_id="test", required_inputs=["default"])
        msg = Message(data="input", input_key="default")
        input_state.receive_message(msg)

        # Build context and execute to get result
        context = node._build_context_from_messages(input_state, router)
        result = node._execute(context)

        assert "history" in result
        assert len(result["history"]) == 2
        assert result["history"][0]["role"] == "user"
        assert result["history"][1]["role"] == "assistant"

    def test_get_history_node_missing_agent_raises(self):
        """Test that missing agent raises ValueError."""
        node = GetHistoryNode(
            agent="missing",
            save_to="history",
            extraction={},
            inputs=["default"],
            outputs=["default"],
        )

        router = MessageRouter()
        router.shared_context["agents"] = {}

        input_state = NodeInputState(node_id="test", required_inputs=["default"])
        msg = Message(data="input", input_key="default")
        input_state.receive_message(msg)

        with pytest.raises(ValueError, match="Agent 'missing' not found"):
            node.execute_with_messages(input_state, message_router=router)

    def test_get_history_node_to_dict(self):
        """Test serializing GetHistoryNode to dictionary."""
        node = GetHistoryNode(
            agent="agent1",
            save_to="history_var",
            context_name="ctx",
            extraction={},
        )
        data = node.to_dict()

        assert data["type"] == "gethistory"
        assert data["agent"] == "agent1"
        assert data["save_to"] == "history_var"
        assert data["context_name"] == "ctx"


class TestSetHistoryNode:
    """Test SetHistoryNode for injecting agent history."""

    def test_set_history_node_creation(self):
        """Test creating a SetHistoryNode."""
        node = SetHistoryNode(
            agent="agent1",
            history_from="history_data",
            extraction={},
        )
        assert node.agent == "agent1"
        assert node.history_from == "history_data"
        assert node.mode == "replace"

    def test_set_history_node_with_append_mode(self):
        """Test SetHistoryNode in append mode."""
        node = SetHistoryNode(
            agent="agent1",
            history_from="history",
            mode="append",
            extraction={},
        )
        assert node.mode == "append"

    def test_set_history_node_do_replaces_history(self):
        """Test that SetHistoryNode replaces agent history."""
        # Create mock agent
        mock_agent = MagicMock(spec=OllamaAgent)
        mock_agent.current_context = "default"
        mock_context = MagicMock()
        mock_context.message_history = []
        mock_agent.contexts = {"default": mock_context}

        node = SetHistoryNode(
            agent="agent1",
            history_from="new_history",
            mode="replace",
            extraction={},
            inputs=["default"],
            outputs=["default"],
        )

        new_history = [
            {"role": "user", "content": "Question"},
            {"role": "assistant", "content": "Answer"},
        ]

        router = MessageRouter()
        router.shared_context["agents"] = {"agent1": mock_agent}
        router.shared_context["new_history"] = new_history

        input_state = NodeInputState(node_id="test", required_inputs=["default"])
        msg = Message(data="input", input_key="default")
        input_state.receive_message(msg)

        outputs = node.execute_with_messages(input_state, message_router=router)

        # Verify history was set (copy method should have been called on the list)
        assert mock_context.message_history == new_history

    def test_set_history_node_missing_agent_raises(self):
        """Test that missing agent raises ValueError."""
        from pithos.message import NodeInputState, Message, MessageRouter

        node = SetHistoryNode(
            agent="missing",
            history_from="history",
            extraction={},
            inputs=["default"],
            outputs=["default"],
        )

        router = MessageRouter()
        router.shared_context["agents"] = {}
        router.shared_context["history"] = []

        input_state = NodeInputState(node_id="test", required_inputs=["default"])
        msg = Message(data="input", input_key="default")
        input_state.receive_message(msg)

        with pytest.raises(ValueError, match="Agent 'missing' not found"):
            node.execute_with_messages(input_state, message_router=router)

    def test_set_history_node_invalid_mode_raises(self):
        """Test that invalid mode raises ValueError."""
        from pithos.message import NodeInputState, Message, MessageRouter

        mock_agent = MagicMock(spec=OllamaAgent)
        mock_agent.current_context = "default"
        mock_agent.contexts = {"default": MagicMock()}

        node = SetHistoryNode(
            agent="agent1",
            history_from="history",
            mode="invalid",
            extraction={},
            inputs=["default"],
            outputs=["default"],
        )

        router = MessageRouter()
        router.shared_context["agents"] = {"agent1": mock_agent}
        router.shared_context["history"] = []

        input_state = NodeInputState(node_id="test", required_inputs=["default"])
        msg = Message(data="input", input_key="default")
        input_state.receive_message(msg)

        with pytest.raises(ValueError, match="Unknown mode"):
            node.execute_with_messages(input_state, message_router=router)

    def test_set_history_node_to_dict(self):
        """Test serializing SetHistoryNode to dictionary."""
        node = SetHistoryNode(
            agent="agent1",
            history_from="history_var",
            context_name="ctx",
            mode="append",
            extraction={},
        )
        data = node.to_dict()

        assert data["type"] == "sethistory"
        assert data["agent"] == "agent1"
        assert data["history_from"] == "history_var"
        assert data["mode"] == "append"


class TestCreateNodeNewTypes:
    """Test create_node factory with new node types."""

    def test_create_agent_prompt_node(self):
        """Test creating AgentPromptNode via factory."""
        data = {
            "type": "agentprompt",
            "agent": "writer",
            "prompt": "Write",
            "extraction": {},
        }
        node = create_node("agentprompt", data)

        assert isinstance(node, AgentPromptNode)
        assert node.agent == "writer"

    def test_create_get_history_node(self):
        """Test creating GetHistoryNode via factory."""
        data = {
            "type": "gethistory",
            "agent": "agent1",
            "save_to": "history",
            "extraction": {},
        }
        node = create_node("gethistory", data)

        assert isinstance(node, GetHistoryNode)
        assert node.agent == "agent1"

    def test_create_set_history_node(self):
        """Test creating SetHistoryNode via factory."""
        data = {
            "type": "sethistory",
            "agent": "agent1",
            "history_from": "history",
            "extraction": {},
        }
        node = create_node("sethistory", data)

        assert isinstance(node, SetHistoryNode)
        assert node.agent == "agent1"
