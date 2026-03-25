"""Tests for InputNode and OutputNode implementations."""

import pytest
import tempfile
import os

from pithos import (
    InputNode,
    OutputNode,
    ChatInputNode,
    ChatOutputNode,
    FileInputNode,
    FileOutputNode,
    ConfigManager,
    Flowchart,
)
from pithos.flownode import create_node


class TestInputNode:
    """Tests for InputNode base class."""

    def test_input_node_initialization(self):
        """Test InputNode can be initialized."""
        node = InputNode()
        assert node.extraction == {}
        assert isinstance(node, InputNode)

    def test_input_node_not_implemented(self):
        """Test InputNode._execute raises NotImplementedError."""
        node = InputNode()

        with pytest.raises(NotImplementedError):
            node._execute({"message_router": None})


class TestOutputNode:
    """Tests for OutputNode base class."""

    def test_output_node_initialization(self):
        """Test OutputNode can be initialized."""
        node = OutputNode()
        assert node.extraction == {}
        assert isinstance(node, OutputNode)

    def test_output_node_not_implemented(self):
        """Test OutputNode._execute raises NotImplementedError."""
        node = OutputNode()

        with pytest.raises(NotImplementedError):
            node._execute({"message_router": None})


class TestChatInputNode:
    """Tests for ChatInputNode."""

    def test_chat_input_initialization(self):
        """Test ChatInputNode initialization with defaults."""
        node = ChatInputNode()
        assert node.prompt_message == "Enter your input:"
        assert node.save_to == "user_input"

    def test_chat_input_custom_params(self):
        """Test ChatInputNode with custom parameters."""
        node = ChatInputNode(prompt_message="Custom prompt", save_to="custom_var")
        assert node.prompt_message == "Custom prompt"
        assert node.save_to == "custom_var"

    def test_chat_input_execution(self):
        """Test ChatInputNode execution."""
        from pithos.message import NodeInputState, Message, MessageRouter

        node = ChatInputNode(
            save_to="user_input", inputs=["default"], outputs=["default"]
        )
        router = MessageRouter()
        router.shared_context["current_input"] = "Hello, world!"

        input_state = NodeInputState(node_id="test", required_inputs=["default"])
        msg = Message(data="Hello, world!", input_key="default")
        input_state.receive_message(msg)

        outputs = node.execute_with_messages(input_state, message_router=router)

        assert len(outputs) > 0
        assert outputs[0].data == "Hello, world!"

    def test_chat_input_empty_state(self):
        """Test ChatInputNode with empty state."""
        from pithos.message import NodeInputState, Message, MessageRouter

        node = ChatInputNode(
            save_to="user_input", inputs=["default"], outputs=["default"]
        )
        router = MessageRouter()

        input_state = NodeInputState(node_id="test", required_inputs=["default"])
        msg = Message(data="", input_key="default")
        input_state.receive_message(msg)

        outputs = node.execute_with_messages(input_state, message_router=router)

        assert len(outputs) > 0
        assert outputs[0].data == ""

    def test_chat_input_from_dict(self):
        """Test ChatInputNode creation from dictionary."""
        data = {
            "type": "chatinput",
            "prompt_message": "Test prompt",
            "save_to": "test_var",
        }
        node = create_node("chatinput", data)
        assert isinstance(node, ChatInputNode)
        assert node.prompt_message == "Test prompt"
        assert node.save_to == "test_var"


class TestChatOutputNode:
    """Tests for ChatOutputNode."""

    def test_chat_output_initialization(self):
        """Test ChatOutputNode initialization with defaults."""
        node = ChatOutputNode()
        assert node.source == "current_input"
        assert node.format_template is None

    def test_chat_output_custom_params(self):
        """Test ChatOutputNode with custom parameters."""
        node = ChatOutputNode(
            source="custom_var", format_template="Result: {custom_var}"
        )
        assert node.source == "custom_var"
        assert node.format_template == "Result: {custom_var}"

    def test_chat_output_execution(self):
        """Test ChatOutputNode execution."""
        from pithos.message import NodeInputState, Message, MessageRouter

        node = ChatOutputNode(source="default", inputs=["default"], outputs=["default"])
        router = MessageRouter()

        input_state = NodeInputState(node_id="test", required_inputs=["default"])
        msg = Message(data="Test output", input_key="default")
        input_state.receive_message(msg)

        outputs = node.execute_with_messages(input_state, message_router=router)

        assert len(outputs) > 0
        assert outputs[0].data == "Test output"

    def test_chat_output_with_template(self):
        """Test ChatOutputNode with format template."""
        from pithos.message import NodeInputState, Message, MessageRouter

        node = ChatOutputNode(
            source="result",
            format_template="Final answer: {result}",
            inputs=["default"],
            outputs=["default"],
        )
        router = MessageRouter()
        router.shared_context["result"] = "42"

        input_state = NodeInputState(node_id="test", required_inputs=["default"])
        msg = Message(data="input", input_key="default")
        input_state.receive_message(msg)

        outputs = node.execute_with_messages(input_state, message_router=router)

        assert len(outputs) > 0
        assert outputs[0].data == "Final answer: 42"

    def test_chat_output_from_dict(self):
        """Test ChatOutputNode creation from dictionary."""
        data = {
            "type": "chatoutput",
            "source": "test_var",
            "format_template": "Output: {test_var}",
        }
        node = create_node("chatoutput", data)
        assert isinstance(node, ChatOutputNode)
        assert node.source == "test_var"
        assert node.format_template == "Output: {test_var}"


class TestFileInputNode:
    """Tests for FileInputNode."""

    def test_file_input_initialization(self):
        """Test FileInputNode initialization."""
        node = FileInputNode(file_path="test.txt")
        assert node.file_path == "test.txt"
        assert node.save_to == "file_content"
        assert node.encoding == "utf-8"

    def test_file_input_read_file(self):
        """Test FileInputNode reads file correctly."""
        from pithos.message import NodeInputState, Message, MessageRouter

        # Create a temporary file
        with tempfile.NamedTemporaryFile(mode="w", delete=False, encoding="utf-8") as f:
            f.write("Test file content\nLine 2")
            temp_path = f.name

        try:
            node = FileInputNode(
                file_path=temp_path,
                save_to="content",
                inputs=["default"],
                outputs=["default"],
            )
            router = MessageRouter()

            input_state = NodeInputState(node_id="test", required_inputs=["default"])
            msg = Message(data="", input_key="default")
            input_state.receive_message(msg)

            outputs = node.execute_with_messages(input_state, message_router=router)

            assert len(outputs) > 0
            assert outputs[0].data == "Test file content\nLine 2"
        finally:
            os.unlink(temp_path)

    def test_file_input_file_not_found(self):
        """Test FileInputNode raises error for missing file."""
        from pithos.message import NodeInputState, Message, MessageRouter

        node = FileInputNode(
            file_path="nonexistent.txt", inputs=["default"], outputs=["default"]
        )
        router = MessageRouter()

        input_state = NodeInputState(node_id="test", required_inputs=["default"])
        msg = Message(data="", input_key="default")
        input_state.receive_message(msg)

        with pytest.raises(FileNotFoundError):
            node.execute_with_messages(input_state, message_router=router)

    def test_file_input_path_formatting(self):
        """Test FileInputNode formats file path with state variables."""
        from pithos.message import NodeInputState, Message, MessageRouter

        with tempfile.NamedTemporaryFile(mode="w", delete=False, encoding="utf-8") as f:
            f.write("Dynamic path content")
            temp_path = f.name

        try:
            node = FileInputNode(
                file_path="{file_name}",
                save_to="content",
                inputs=["default"],
                outputs=["default"],
            )
            router = MessageRouter()
            router.shared_context["file_name"] = temp_path

            input_state = NodeInputState(node_id="test", required_inputs=["default"])
            msg = Message(data="", input_key="default")
            input_state.receive_message(msg)

            outputs = node.execute_with_messages(input_state, message_router=router)

            assert len(outputs) > 0
            assert outputs[0].data == "Dynamic path content"
        finally:
            os.unlink(temp_path)

    def test_file_input_from_dict(self):
        """Test FileInputNode creation from dictionary."""
        data = {
            "type": "fileinput",
            "file_path": "data.txt",
            "save_to": "data",
            "encoding": "utf-8",
        }
        node = create_node("fileinput", data)
        assert isinstance(node, FileInputNode)
        assert node.file_path == "data.txt"
        assert node.save_to == "data"


class TestFileOutputNode:
    """Tests for FileOutputNode."""

    def test_file_output_initialization(self):
        """Test FileOutputNode initialization."""
        node = FileOutputNode(file_path="output.txt")
        assert node.file_path == "output.txt"
        assert node.source == "current_input"
        assert node.mode == "w"
        assert node.encoding == "utf-8"

    def test_file_output_write_file(self):
        """Test FileOutputNode writes file correctly."""
        from pithos.message import NodeInputState, Message, MessageRouter

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = os.path.join(temp_dir, "test_output.txt")
            node = FileOutputNode(
                file_path=output_path,
                source="data",
                inputs=["default"],
                outputs=["default"],
            )
            router = MessageRouter()
            router.shared_context["data"] = "Output content"

            input_state = NodeInputState(node_id="test", required_inputs=["default"])
            msg = Message(data="Output content", input_key="default")
            input_state.receive_message(msg)

            outputs = node.execute_with_messages(input_state, message_router=router)

            assert len(outputs) > 0
            # Verify file was written
            with open(output_path, "r", encoding="utf-8") as f:
                content = f.read()
            assert content == "Output content"

    def test_file_output_append_mode(self):
        """Test FileOutputNode in append mode."""
        from pithos.message import NodeInputState, Message, MessageRouter

        with tempfile.NamedTemporaryFile(mode="w", delete=False, encoding="utf-8") as f:
            f.write("Initial content\n")
            temp_path = f.name

        try:
            node = FileOutputNode(
                file_path=temp_path,
                source="data",
                mode="a",
                inputs=["default"],
                outputs=["default"],
            )
            router = MessageRouter()
            router.shared_context["data"] = "Appended content"

            input_state = NodeInputState(node_id="test", required_inputs=["default"])
            msg = Message(data="Appended content", input_key="default")
            input_state.receive_message(msg)

            node.execute_with_messages(input_state, message_router=router)

            # Verify content was appended
            with open(temp_path, "r", encoding="utf-8") as f:
                content = f.read()
            assert content == "Initial content\nAppended content"
        finally:
            os.unlink(temp_path)

    def test_file_output_path_formatting(self):
        """Test FileOutputNode formats file path with state variables."""
        from pithos.message import NodeInputState, Message, MessageRouter

        with tempfile.TemporaryDirectory() as temp_dir:
            node = FileOutputNode(
                file_path="{dir}/output.txt",
                source="data",
                inputs=["default"],
                outputs=["default"],
            )
            router = MessageRouter()
            router.shared_context["dir"] = temp_dir
            router.shared_context["data"] = "Test"

            input_state = NodeInputState(node_id="test", required_inputs=["default"])
            msg = Message(data="Test", input_key="default")
            input_state.receive_message(msg)

            node.execute_with_messages(input_state, message_router=router)

            expected_path = f"{temp_dir}/output.txt"
            # Verify file exists
            assert os.path.exists(expected_path)

    def test_file_output_from_dict(self):
        """Test FileOutputNode creation from dictionary."""
        data = {
            "type": "fileoutput",
            "file_path": "output.txt",
            "source": "result",
            "mode": "w",
        }
        node = create_node("fileoutput", data)
        assert isinstance(node, FileOutputNode)
        assert node.file_path == "output.txt"
        assert node.source == "result"
        assert node.mode == "w"


class TestFlowchartIOValidation:
    """Tests for flowchart I/O node validation."""

    def test_flowchart_auto_adds_input_node(self):
        """Test flowchart automatically adds ChatInputNode if missing."""
        config_manager = ConfigManager()

        # Create flowchart with no input nodes
        flowchart_data = {
            "nodes": {
                "process": {"type": "prompt", "prompt": "Process: {current_input}"}
            },
            "edges": [],
            "start_node": "process",
        }

        flowchart = Flowchart.from_dict(flowchart_data, config_manager)

        # Check that an input node was auto-added
        assert "__auto_chat_input__" in flowchart.graph.nodes
        # Check that start node is now the auto-added input
        assert flowchart.start_node == "__auto_chat_input__"

    def test_flowchart_auto_adds_output_node(self):
        """Test flowchart automatically adds ChatOutputNode if missing."""
        config_manager = ConfigManager()

        # Create flowchart with no output nodes
        flowchart_data = {
            "nodes": {
                "input": {"type": "chatinput"},
                "process": {"type": "prompt", "prompt": "Process: {current_input}"},
            },
            "edges": [{"from": "input", "to": "process"}],
            "start_node": "input",
        }

        flowchart = Flowchart.from_dict(flowchart_data, config_manager)

        # Check that an output node was auto-added
        assert "__auto_chat_output__" in flowchart.graph.nodes

    def test_flowchart_with_existing_io_nodes(self):
        """Test flowchart doesn't add nodes when I/O nodes exist."""
        config_manager = ConfigManager()

        # Create flowchart with both input and output nodes
        flowchart_data = {
            "nodes": {
                "input": {"type": "chatinput"},
                "process": {"type": "prompt", "prompt": "Process: {current_input}"},
                "output": {"type": "chatoutput"},
            },
            "edges": [
                {"from": "input", "to": "process"},
                {"from": "process", "to": "output"},
            ],
            "start_node": "input",
        }

        flowchart = Flowchart.from_dict(flowchart_data, config_manager)

        # Check that no auto nodes were added
        assert "__auto_chat_input__" not in flowchart.graph.nodes
        assert "__auto_chat_output__" not in flowchart.graph.nodes
        # Original start node should be preserved
        assert flowchart.start_node == "input"

    def test_flowchart_with_file_io_nodes(self):
        """Test flowchart recognizes FileInputNode and FileOutputNode."""
        config_manager = ConfigManager()

        # Create flowchart with file I/O nodes
        flowchart_data = {
            "nodes": {
                "input": {"type": "fileinput", "file_path": "input.txt"},
                "process": {"type": "prompt", "prompt": "Process: {current_input}"},
                "output": {
                    "type": "fileoutput",
                    "file_path": "output.txt",
                    "source": "current_input",
                },
            },
            "edges": [
                {"from": "input", "to": "process"},
                {"from": "process", "to": "output"},
            ],
            "start_node": "input",
        }

        flowchart = Flowchart.from_dict(flowchart_data, config_manager)

        # Check that no auto nodes were added
        assert "__auto_chat_input__" not in flowchart.graph.nodes
        assert "__auto_chat_output__" not in flowchart.graph.nodes
