"""Tests for message-based flowchart execution."""

import pytest
from pithos import Flowchart, Message, AlwaysCondition
from pithos.conditions import Condition
from unittest.mock import patch


class TestMessageBasedExecution:
    """Test message-based flowchart execution."""

    @patch("pithos.flowchart.ConfigManager")
    def test_message_creation(self, mock_config):
        """Test creating messages."""
        msg = Message(
            data="Hello, world!",
            source_node="node1",
            target_node="node2",
            input_key="input1",
        )

        assert msg.data == "Hello, world!"
        assert msg.source_node == "node1"
        assert msg.target_node == "node2"
        assert msg.input_key == "input1"
        assert msg.message_id is not None

    @patch("pithos.flowchart.ConfigManager")
    def test_node_input_state(self, mock_config):
        """Test node input state tracking."""
        from pithos.message import NodeInputState

        state = NodeInputState(node_id="node1", required_inputs=["input1", "input2"])

        assert not state.is_ready()

        # Receive first input
        msg1 = Message(data="data1", input_key="input1")
        state.receive_message(msg1)
        assert not state.is_ready()

        # Receive second input
        msg2 = Message(data="data2", input_key="input2")
        state.receive_message(msg2)
        assert state.is_ready()

        # Get input data
        assert state.get_input_data("input1") == "data1"
        assert state.get_input_data("input2") == "data2"

    @patch("pithos.flowchart.ConfigManager")
    def test_message_router_registration(self, mock_config):
        """Test registering nodes with message router."""
        from pithos.message import MessageRouter

        router = MessageRouter()
        router.register_node("node1", required_inputs=["input1"])
        router.register_node("node2", required_inputs=["input1", "input2"])

        assert "node1" in router.node_states
        assert "node2" in router.node_states
        assert len(router.node_states["node1"].required_inputs) == 1
        assert len(router.node_states["node2"].required_inputs) == 2

    @patch("pithos.flowchart.ConfigManager")
    def test_message_routing(self, mock_config):
        """Test message routing to nodes."""
        from pithos.message import MessageRouter

        router = MessageRouter()
        router.register_node("node1", required_inputs=["default"])
        router.register_node("node2", required_inputs=["default"])

        # Send message to node1
        msg = Message(data="test", target_node="node1", input_key="default")
        router.send_message(msg)

        # Check node1 is ready
        ready = router.get_ready_nodes()
        assert "node1" in ready
        assert "node2" not in ready

    @patch("pithos.flowchart.ConfigManager")
    def test_flowchart_message_mode_enable(self, mock_config):
        """Test that message-based mode is the default and always enabled."""
        flow = Flowchart(mock_config)
        flow.add_node("node1", type="prompt", prompt="Test", extraction={})
        flow.add_node("node2", type="prompt", prompt="Test2", extraction={})

        # Initialize message routing
        flow._initialize_message_routing()

        # Check nodes are registered
        assert "node1" in flow.message_router.node_states
        assert "node2" in flow.message_router.node_states

    @patch("pithos.flowchart.ConfigManager")
    def test_simple_message_flow(self, mock_config):
        """Test simple message-based flowchart execution."""
        flow = Flowchart(mock_config)

        # Create simple linear flowchart
        flow.add_node("start", type="textparse", extraction={})
        flow.add_node("end", type="textparse", extraction={})
        flow.add_edge("start", "end", AlwaysCondition)
        flow.set_start_node("start")

        # Initialize message routing
        flow._initialize_message_routing()

        # Run flowchart
        result = flow.run_message_based(initial_data="Hello")

        assert result["completed"]
        assert result["steps"] > 0

    @patch("pithos.flowchart.ConfigManager")
    def test_multi_input_node(self, mock_config):
        """Test node that requires multiple inputs."""
        flow = Flowchart(mock_config)

        # Create flowchart with merge node
        flow.add_node(
            "source1",
            type="prompt",
            prompt="Source 1",
            extraction={},
            inputs=["default"],
            outputs=["output1"],
        )
        flow.add_node(
            "source2",
            type="prompt",
            prompt="Source 2",
            extraction={},
            inputs=["default"],
            outputs=["output2"],
        )
        flow.add_node(
            "merge",
            type="prompt",
            prompt="Merge: {input1} + {input2}",
            extraction={},
            inputs=["input1", "input2"],
        )

        flow.add_edge(
            "source1",
            "merge",
            AlwaysCondition,
            output_key="output1",
            input_key="input1",
        )
        flow.add_edge(
            "source2",
            "merge",
            AlwaysCondition,
            output_key="output2",
            input_key="input2",
        )

        flow.set_start_node("source1")
        flow._initialize_message_routing()

        # The merge node should only execute when both inputs are received
        # Note: This test demonstrates the concept but may need adjustment
        # based on how we handle multiple start nodes

    @patch("pithos.flowchart.ConfigManager")
    def test_message_history(self, mock_config):
        """Test that message history is maintained."""
        flow = Flowchart(mock_config)

        flow.add_node("node1", type="prompt", prompt="Node 1", extraction={})
        flow.add_node("node2", type="prompt", prompt="Node 2", extraction={})
        flow.add_edge("node1", "node2", AlwaysCondition)
        flow.set_start_node("node1")

        flow._initialize_message_routing()

        result = flow.run_message_based(initial_data="test")

        # Check message history is available
        assert len(result["message_history"]) > 0
        assert all(isinstance(msg, Message) for msg in result["message_history"])

    @patch("pithos.flowchart.ConfigManager")
    def test_reset_clears_messages(self, mock_config):
        """Test that reset clears message router state."""
        flow = Flowchart(mock_config)

        flow.add_node("node1", type="prompt", prompt="Node 1", extraction={})
        flow.set_start_node("node1")
        flow._initialize_message_routing()

        # Send initial message
        msg = Message(data="test", target_node="node1", input_key="default")
        flow.message_router.send_message(msg)

        assert len(flow.message_router.pending_messages) > 0

        # Reset should clear everything
        flow.reset()

        assert len(flow.message_router.pending_messages) == 0
        assert len(flow.message_router.message_history) == 0

    @patch("pithos.flowchart.ConfigManager")
    def test_serialization_with_message_routing(self, mock_config):
        """Test that message routing settings are preserved in serialization."""
        flow = Flowchart(mock_config)

        flow.add_node("node1", type="prompt", prompt="Test", extraction={})
        flow.add_node("node2", type="prompt", prompt="Test2", extraction={})
        flow.add_edge(
            "node1", "node2", AlwaysCondition, output_key="out1", input_key="in1"
        )
        flow.set_start_node("node1")
        flow._initialize_message_routing()

        # Serialize
        data = flow.to_dict()

        assert any(e.get("output_key") == "out1" for e in data["edges"])
        assert any(e.get("input_key") == "in1" for e in data["edges"])

        # Deserialize
        flow2 = Flowchart.from_dict(data, mock_config)

    @patch("pithos.flowchart.ConfigManager")
    def test_backward_compatibility(self, mock_config):
        """Test message-based execution is always used."""
        flow = Flowchart(mock_config)

        flow.add_node("node1", type="textparse", extraction={}, set={"value": "hello"})
        flow.add_node("node2", type="textparse", extraction={})
        flow.add_edge("node1", "node2", AlwaysCondition)
        flow.set_start_node("node1")

        # Run with message-based execution
        flow._initialize_message_routing()
        result = flow.run_message_based(initial_data="test")

        assert result["completed"]

    @patch("pithos.flowchart.ConfigManager")
    def test_conditional_routing_with_messages(self, mock_config):
        """Test conditional edge routing with messages."""
        flow = Flowchart(mock_config)

        flow.add_node("check", type="prompt", prompt="Check", extraction={})
        flow.add_node("path_a", type="prompt", prompt="Path A", extraction={})
        flow.add_node("path_b", type="prompt", prompt="Path B", extraction={})

        cond_a = Condition(condition=lambda s: "A" in str(s.get("default", "")))
        cond_b = Condition(condition=lambda s: "B" in str(s.get("default", "")))

        flow.add_edge("check", "path_a", cond_a, priority=1)
        flow.add_edge("check", "path_b", cond_b, priority=2)

        flow.set_start_node("check")
        flow._initialize_message_routing()

        # Test with data that should go to path A
        result_a = flow.run_message_based(initial_data="Choose A")
        assert result_a["completed"]


class TestNodeMessageExecution:
    """Test individual node execution with messages."""

    @patch("pithos.flowchart.ConfigManager")
    def test_prompt_node_message_execution(self, mock_config):
        """Test PromptNode execution with messages."""
        from pithos import PromptNode
        from pithos.message import NodeInputState, Message

        node = PromptNode(
            prompt="Hello {default}!",
            extraction={},
            inputs=["default"],
            outputs=["default"],
        )

        # Create input state
        input_state = NodeInputState(node_id="test", required_inputs=["default"])

        msg = Message(data="World", input_key="default")
        input_state.receive_message(msg)

        # Execute node
        outputs = node.execute_with_messages(input_state)

        assert len(outputs) > 0
        assert any("World" in str(msg.data) for msg in outputs)

    @patch("pithos.flowchart.ConfigManager")
    def test_node_with_extractions(self, mock_config):
        """Test node with extraction patterns."""
        from pithos import PromptNode
        from pithos.message import NodeInputState, Message

        node = PromptNode(
            prompt="Result: {value}",
            extraction={"value": r"Value: (\d+)"},
            inputs=["default"],
            outputs=["default"],
        )

        input_state = NodeInputState(node_id="test", required_inputs=["default"])

        msg = Message(data="Value: 42", input_key="default")
        input_state.receive_message(msg)

        outputs = node.execute_with_messages(input_state)

        assert len(outputs) > 0


class TestMessageHistoryWindowUnit:
    """Unit tests for MessageRouter rolling history window."""

    def test_default_max_history_is_zero(self):
        """max_history defaults to 0 (unlimited)."""
        from pithos.message import MessageRouter

        router = MessageRouter()
        assert router._max_history == 0

    def test_custom_max_history(self):
        """Constructor accepts a custom max_history."""
        from pithos.message import MessageRouter

        router = MessageRouter(max_history=10)
        assert router._max_history == 10

    def test_unlimited_keeps_all_messages(self):
        """With max_history=0 all messages accumulate."""
        from pithos.message import MessageRouter, Message

        router = MessageRouter(max_history=0)
        router.register_node("n", required_inputs=["default"])
        for i in range(20):
            router.send_message(
                Message(data=f"msg{i}", target_node="n", input_key="default")
            )

        assert len(router.message_history) == 20

    def test_rolling_window_caps_history(self):
        """history is capped at max_history oldest messages are discarded."""
        from pithos.message import MessageRouter, Message

        router = MessageRouter(max_history=5)
        router.register_node("n", required_inputs=["default"])
        for i in range(10):
            router.send_message(
                Message(data=f"msg{i}", target_node="n", input_key="default")
            )

        assert len(router.message_history) == 5
        # Only the most recent 5 are kept
        assert router.message_history[0].data == "msg5"
        assert router.message_history[-1].data == "msg9"

    def test_rolling_window_retains_latest_on_overflow(self):
        """The retained messages are always the most recently sent ones."""
        from pithos.message import MessageRouter, Message

        router = MessageRouter(max_history=3)
        router.register_node("n", required_inputs=["default"])
        payloads = ["a", "b", "c", "d", "e"]
        for p in payloads:
            router.send_message(Message(data=p, target_node="n", input_key="default"))

        retained = [m.data for m in router.message_history]
        assert retained == ["c", "d", "e"]

    def test_history_window_one(self):
        """max_history=1 keeps exactly the last message."""
        from pithos.message import MessageRouter, Message

        router = MessageRouter(max_history=1)
        router.register_node("n", required_inputs=["default"])
        for i in range(5):
            router.send_message(
                Message(data=f"m{i}", target_node="n", input_key="default")
            )

        assert len(router.message_history) == 1
        assert router.message_history[0].data == "m4"

    def test_reset_clears_history_keeps_max_history_setting(self):
        """reset() empties message_history but preserves _max_history."""
        from pithos.message import MessageRouter, Message

        router = MessageRouter(max_history=3)
        router.register_node("n", required_inputs=["default"])
        router.send_message(Message(data="x", target_node="n", input_key="default"))
        assert len(router.message_history) == 1

        router.reset()

        assert len(router.message_history) == 0
        assert router._max_history == 3  # setting is preserved

    def test_max_history_set_after_init(self):
        """_max_history can be updated after construction and applies immediately."""
        from pithos.message import MessageRouter, Message

        router = MessageRouter(max_history=0)
        router.register_node("n", required_inputs=["default"])
        # Send 5 messages without a limit
        for i in range(5):
            router.send_message(
                Message(data=f"p{i}", target_node="n", input_key="default")
            )
        assert len(router.message_history) == 5

        # Now enforce a tighter limit
        router._max_history = 2
        router.send_message(Message(data="new", target_node="n", input_key="default"))
        # The new message pushed out the oldest; only 2 remain
        assert len(router.message_history) == 2
        assert router.message_history[-1].data == "new"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
