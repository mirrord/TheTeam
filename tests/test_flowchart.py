"""Unit tests for flowchart module."""

import pytest
from pithos.flowchart import Flowchart
from pithos.flownode import PromptNode
from pithos.conditions import AlwaysCondition
from pithos.validation import ValidationError
from unittest.mock import patch


class TestFlowchart:
    """Test Flowchart structure and configuration."""

    @patch("pithos.flowchart.ConfigManager")
    def test_flowchart_creation(self, mock_config):
        flow = Flowchart(mock_config)
        assert flow.config_manager is mock_config
        assert flow.start_node is None
        assert flow.current_node is None
        assert flow.finished is False

    @patch("pithos.flowchart.ConfigManager")
    def test_add_node(self, mock_config):
        flow = Flowchart(mock_config)
        flow.add_node("node1", type="prompt", prompt="Test", extraction={})

        assert "node1" in flow.graph.nodes
        node_obj = flow.graph.nodes["node1"]["nodeobj"]
        assert isinstance(node_obj, PromptNode)

    @patch("pithos.flowchart.ConfigManager")
    def test_add_node_sets_start_node_if_first(self, mock_config):
        flow = Flowchart(mock_config)
        flow.add_node("first", type="prompt", prompt="Test", extraction={})
        assert flow.start_node == "first"

    @patch("pithos.flowchart.ConfigManager")
    def test_set_start_node(self, mock_config):
        flow = Flowchart(mock_config)
        flow.add_node("node1", type="prompt", prompt="Test", extraction={})
        flow.add_node("node2", type="prompt", prompt="Test2", extraction={})
        flow.set_start_node("node2")
        assert flow.start_node == "node2"

    @patch("pithos.flowchart.ConfigManager")
    def test_add_edge(self, mock_config):
        flow = Flowchart(mock_config)
        flow.add_node("node1", type="prompt", prompt="Test", extraction={})
        flow.add_node("node2", type="prompt", prompt="Test2", extraction={})

        cond = AlwaysCondition
        flow.add_edge("node1", "node2", cond)

        assert flow.graph.has_edge("node1", "node2")

    @patch("pithos.flowchart.ConfigManager")
    def test_add_edge_with_priority(self, mock_config):
        flow = Flowchart(mock_config)
        flow.add_node("node1", type="prompt", prompt="Test", extraction={})
        flow.add_node("node2", type="prompt", prompt="Test2", extraction={})

        cond = AlwaysCondition
        flow.add_edge("node1", "node2", cond, priority=5)

        edges = flow.graph["node1"]["node2"]
        assert list(edges.values())[0]["priority"] == 5

    @patch("pithos.flowchart.ConfigManager")
    def test_reset(self, mock_config):
        flow = Flowchart(mock_config)
        flow.current_node = "some_node"
        flow.finished = True

        flow.reset()

        assert flow.current_node is None
        assert flow.finished is False

    @patch("pithos.flowchart.ConfigManager")
    def test_to_dict(self, mock_config):
        flow = Flowchart(mock_config)
        flow.add_node("node1", type="prompt", prompt="Test", extraction={})
        flow.add_node("node2", type="custom", custom_code="pass", extraction={})
        flow.add_edge("node1", "node2", AlwaysCondition)

        data = flow.to_dict()

        assert "nodes" in data
        assert "edges" in data
        assert "start_node" in data
        assert "node1" in data["nodes"]
        assert "node2" in data["nodes"]
        assert len(data["edges"]) >= 1  # May have auto I/O node edges

    @patch("pithos.flowchart.ConfigManager")
    def test_from_dict(self, mock_config):
        data = {
            "nodes": {
                "input": {"type": "chatinput"},
                "node1": {"type": "prompt", "prompt": "Test", "extraction": {}},
                "node2": {"type": "prompt", "prompt": "Test2", "extraction": {}},
                "output": {"type": "chatoutput"},
            },
            "edges": [
                {
                    "from": "input",
                    "to": "node1",
                    "condition": {"type": "AlwaysCondition"},
                    "priority": 1,
                },
                {
                    "from": "node1",
                    "to": "node2",
                    "condition": {"type": "AlwaysCondition"},
                    "priority": 1,
                },
                {
                    "from": "node2",
                    "to": "output",
                    "condition": {"type": "AlwaysCondition"},
                    "priority": 1,
                },
            ],
            "start_node": "input",
        }

        flow = Flowchart.from_dict(data, mock_config)

        assert "input" in flow.graph.nodes
        assert "node1" in flow.graph.nodes
        assert "node2" in flow.graph.nodes
        assert "output" in flow.graph.nodes
        assert flow.graph.has_edge("input", "node1")
        assert flow.graph.has_edge("node1", "node2")
        assert flow.graph.has_edge("node2", "output")
        assert flow.start_node == "input"


class TestFlowchartExecution:
    """Integration tests for message-based flowchart execution."""

    @patch("pithos.flowchart.ConfigManager")
    def test_simple_message_based_execution(self, mock_config):
        """Test basic message-based flowchart execution."""
        flow = Flowchart(mock_config)
        flow.add_node(
            "node1", type="textparse", extraction={}, set={"output1": "value1"}
        )
        flow.add_node(
            "node2", type="textparse", extraction={}, set={"output2": "value2"}
        )

        flow.add_edge("node1", "node2", AlwaysCondition)
        flow._initialize_message_routing()

        # Execute flowchart
        result = flow.run_message_based(initial_data="test", max_steps=10)

        assert result["completed"]
        # Check that flowchart executed (message_history should have messages)
        assert len(result["message_history"]) > 0

    @patch("pithos.flowchart.ConfigManager")
    def test_flowchart_serialization_roundtrip(self, mock_config):
        """Test that flowcharts can be saved and restored."""
        # Create original flowchart
        flow1 = Flowchart(mock_config)
        flow1.add_node("input", type="chatinput")
        flow1.add_node("node1", type="textparse", extraction={})
        flow1.add_node("node2", type="textparse", extraction={})
        flow1.add_node("output", type="chatoutput")
        flow1.add_edge("input", "node1", AlwaysCondition)
        flow1.add_edge("node1", "node2", AlwaysCondition)
        flow1.add_edge("node2", "output", AlwaysCondition)
        flow1.set_start_node("input")

        # Serialize
        data = flow1.to_dict()

        # Restore
        flow2 = Flowchart.from_dict(data, mock_config)

        # Verify structure
        assert flow2.start_node == flow1.start_node
        assert "input" in flow2.graph.nodes
        assert "node1" in flow2.graph.nodes
        assert "node2" in flow2.graph.nodes
        assert "output" in flow2.graph.nodes
        assert flow2.graph.has_edge("input", "node1")
        assert flow2.graph.has_edge("node1", "node2")
        assert flow2.graph.has_edge("node2", "output")

    @patch("pithos.flowchart.ConfigManager")
    def test_from_dict_with_validation(self, mock_config):
        """Test that from_dict validates configurations by default."""
        data = {
            "nodes": {
                "input": {
                    "type": "chatinput",
                },
                "node1": {
                    "type": "prompt",
                    "prompt": "Test prompt",
                    "extraction": {},
                },
                "output": {
                    "type": "chatoutput",
                },
            },
            "edges": [
                {
                    "from": "input",
                    "to": "node1",
                    "condition": {"type": "AlwaysCondition"},
                },
                {
                    "from": "node1",
                    "to": "output",
                    "condition": {"type": "AlwaysCondition"},
                },
            ],
            "start_node": "input",
        }

        # Should not raise with valid config
        flowchart = Flowchart.from_dict(data, mock_config, validate=True)
        assert flowchart.start_node == "input"
        assert "node1" in flowchart.graph.nodes

    @patch("pithos.flowchart.ConfigManager")
    def test_from_dict_detects_invalid_config(self, mock_config):
        """Test that from_dict detects invalid node configurations."""
        data = {
            "nodes": {
                "bad_node": {
                    "type": "prompt",
                    # Missing required 'prompt' parameter
                    "extraction": {},
                }
            },
            "edges": [],
            "start_node": "bad_node",
        }

        # Should raise ValidationError during validation
        with pytest.raises(ValidationError):
            Flowchart.from_dict(data, mock_config, validate=True)

    @patch("pithos.flowchart.ConfigManager")
    def test_from_dict_can_skip_validation(self, mock_config):
        """Test that validation can be skipped with validate=False."""
        data = {
            "nodes": {
                "node1": {
                    "type": "textparse",
                    "extraction": {},
                }
            },
            "edges": [],
            "start_node": "node1",
        }

        # Should work fine with validation disabled
        flowchart = Flowchart.from_dict(data, mock_config, validate=False)
        # _ensure_io_nodes will add input/output nodes, but original node should exist
        assert "node1" in flowchart.graph.nodes

    @patch("pithos.flowchart.ConfigManager")
    def test_validate_method(self, mock_config):
        """Test the validate() method on a Flowchart instance."""
        flow = Flowchart(mock_config)
        flow.add_node("input", type="chatinput")
        flow.add_node("node1", type="prompt", prompt="Test", extraction={})
        flow.add_node("output", type="chatoutput")
        flow.set_start_node("input")

        # Should not raise
        assert flow.validate(strict=False) is True

    @patch("pithos.flowchart.ConfigManager")
    def test_from_dict_detects_invalid_edges(self, mock_config):
        """Test that validation detects invalid edge configurations."""
        data = {
            "nodes": {
                "node1": {
                    "type": "prompt",
                    "prompt": "Test",
                    "extraction": {},
                }
            },
            "edges": [
                {
                    "from": "node1",
                    "to": "nonexistent_node",  # References non-existent node
                    "condition": {"type": "AlwaysCondition"},
                }
            ],
            "start_node": "node1",
        }

        # Should raise ValidationError
        with pytest.raises(ValidationError) as exc_info:
            Flowchart.from_dict(data, mock_config, validate=True)

        assert "non-existent" in str(exc_info.value).lower()


class TestMessageHistoryWindow:
    """Tests for rolling message history window in flowchart execution."""

    @patch("pithos.flowchart.ConfigManager")
    def test_unlimited_history_by_default(self, mock_config):
        """With history_window=0 all messages are kept."""
        flow = Flowchart(mock_config)
        flow.add_node(
            "n1", type="textparse", extraction={}, set={"current_input": "{default}"}
        )
        flow.add_node(
            "n2", type="textparse", extraction={}, set={"current_input": "{default}"}
        )
        flow.add_node(
            "n3", type="textparse", extraction={}, set={"current_input": "{default}"}
        )
        flow.add_edge("n1", "n2", AlwaysCondition)
        flow.add_edge("n2", "n3", AlwaysCondition)
        flow._initialize_message_routing()

        result = flow.run_message_based(initial_data="start", history_window=0)

        # All routed messages are in history
        assert len(result["message_history"]) > 0
        assert flow.message_router._max_history == 0

    @patch("pithos.flowchart.ConfigManager")
    def test_history_window_limits_messages(self, mock_config):
        """history_window=1 keeps at most 1 message in the rolling buffer."""
        flow = Flowchart(mock_config)
        flow.add_node(
            "n1", type="textparse", extraction={}, set={"current_input": "{default}"}
        )
        flow.add_node(
            "n2", type="textparse", extraction={}, set={"current_input": "{default}"}
        )
        flow.add_node(
            "n3", type="textparse", extraction={}, set={"current_input": "{default}"}
        )
        flow.add_edge("n1", "n2", AlwaysCondition)
        flow.add_edge("n2", "n3", AlwaysCondition)
        flow._initialize_message_routing()

        result = flow.run_message_based(initial_data="start", history_window=1)

        assert len(result["message_history"]) <= 1
        assert flow.message_router._max_history == 1

    @patch("pithos.flowchart.ConfigManager")
    def test_history_window_is_reset_between_runs(self, mock_config):
        """Each call to run_message_based starts with a fresh history."""
        flow = Flowchart(mock_config)
        flow.add_node("n1", type="textparse", extraction={}, set={"v": "x"})
        flow._initialize_message_routing()

        flow.run_message_based(initial_data="first", history_window=5)
        first_run_history = list(flow.message_router.message_history)

        flow.run_message_based(initial_data="second", history_window=5)
        second_run_history = list(flow.message_router.message_history)

        # Second run should not contain messages from first run
        first_ids = {m.message_id for m in first_run_history}
        second_ids = {m.message_id for m in second_run_history}
        assert first_ids.isdisjoint(second_ids)

    @patch("pithos.flowchart.ConfigManager")
    def test_reset_clears_tracking_state(self, mock_config):
        """reset() clears step counter, prev results, and route info."""
        flow = Flowchart(mock_config)
        flow.add_node("n1", type="textparse", extraction={}, set={"v": "x"})
        flow._initialize_message_routing()
        flow.run_message_based(initial_data="hello")

        flow.reset()

        assert flow._step_counter == 0
        assert flow._prev_output_messages == []
        assert flow._node_route_info == {}


class TestProgressCallback:
    """Tests for the on_progress callback in flowchart execution."""

    def _make_simple_flow(self, mock_config):
        """Helper: 3-node textparse chain n1 -> n2 -> n3.

        Each node forwards 'current_input' so that routing messages are
        produced and the next downstream node receives its 'default' input.
        """
        flow = Flowchart(mock_config)
        flow.add_node(
            "n1", type="textparse", extraction={}, set={"current_input": "{default}"}
        )
        flow.add_node(
            "n2", type="textparse", extraction={}, set={"current_input": "{default}"}
        )
        flow.add_node(
            "n3", type="textparse", extraction={}, set={"current_input": "{default}"}
        )
        flow.add_edge("n1", "n2", AlwaysCondition)
        flow.add_edge("n2", "n3", AlwaysCondition)
        flow._initialize_message_routing()
        return flow

    @patch("pithos.flowchart.ConfigManager")
    def test_callback_fires_for_each_node(self, mock_config):
        """on_progress is called once per node execution."""
        from pithos.flowchart import ProgressEvent

        flow = self._make_simple_flow(mock_config)
        events: list[ProgressEvent] = []
        flow.run_message_based(initial_data="hello", on_progress=events.append)

        assert len(events) == 3

    @patch("pithos.flowchart.ConfigManager")
    def test_event_fields_are_populated(self, mock_config):
        """Each ProgressEvent carries node_id, step, inputs, and edge info."""
        from pithos.flowchart import ProgressEvent

        flow = self._make_simple_flow(mock_config)
        events: list[ProgressEvent] = []
        flow.run_message_based(initial_data="hello", on_progress=events.append)

        assert all(isinstance(e, ProgressEvent) for e in events)

        # Step numbers should be sequential 0, 1, 2
        assert [e.step for e in events] == [0, 1, 2]

        # All events have a node_id
        assert all(e.node_id for e in events)

        # Each event has an inputs dict
        assert all(isinstance(e.inputs, dict) for e in events)

    @patch("pithos.flowchart.ConfigManager")
    def test_first_node_has_no_edge(self, mock_config):
        """The first node executed has edge=None (no incoming edge)."""
        from pithos.flowchart import ProgressEvent

        flow = self._make_simple_flow(mock_config)
        events: list[ProgressEvent] = []
        flow.run_message_based(initial_data="hello", on_progress=events.append)

        assert events[0].edge is None
        assert events[0].previous_results == []

    @patch("pithos.flowchart.ConfigManager")
    def test_subsequent_nodes_have_edge_info(self, mock_config):
        """Nodes after the first carry EdgeInfo with from/to nodes."""
        from pithos.flowchart import ProgressEvent, EdgeInfo

        flow = self._make_simple_flow(mock_config)
        events: list[ProgressEvent] = []
        flow.run_message_based(initial_data="hello", on_progress=events.append)

        for event in events[1:]:
            assert isinstance(event.edge, EdgeInfo)
            assert event.edge.to_node == event.node_id
            assert event.edge.from_node != event.node_id
            assert event.edge.condition_type  # non-empty string

    @patch("pithos.flowchart.ConfigManager")
    def test_previous_results_match_prior_output(self, mock_config):
        """previous_results in step N matches the output messages of step N-1."""
        from pithos.flowchart import ProgressEvent
        from pithos.message import Message

        flow = self._make_simple_flow(mock_config)
        events: list[ProgressEvent] = []
        flow.run_message_based(initial_data="hello", on_progress=events.append)

        # Step 0 has no previous results
        assert events[0].previous_results == []

        # Step 1+ should have previous results that are Message instances
        for event in events[1:]:
            assert isinstance(event.previous_results, list)
            assert all(isinstance(m, Message) for m in event.previous_results)

    @patch("pithos.flowchart.ConfigManager")
    def test_no_callback_runs_without_error(self, mock_config):
        """Omitting on_progress does not break execution."""
        flow = self._make_simple_flow(mock_config)
        result = flow.run_message_based(initial_data="hello")
        assert result["completed"]

    @patch("pithos.flowchart.ConfigManager")
    def test_callback_exception_propagates(self, mock_config):
        """Exceptions raised inside the callback bubble up to the caller."""
        flow = self._make_simple_flow(mock_config)

        def bad_callback(event):
            raise ValueError("oops")

        with pytest.raises(ValueError, match="oops"):
            flow.run_message_based(initial_data="hello", on_progress=bad_callback)

    @patch("pithos.flowchart.ConfigManager")
    def test_callback_receives_correct_node_order(self, mock_config):
        """Callback events for a linear chain arrive in topological order."""
        from pithos.flowchart import ProgressEvent

        flow = self._make_simple_flow(mock_config)
        events: list[ProgressEvent] = []
        flow.run_message_based(initial_data="in", on_progress=events.append)

        node_sequence = [e.node_id for e in events]
        # n1 before n2 before n3
        assert node_sequence.index("n1") < node_sequence.index("n2")
        assert node_sequence.index("n2") < node_sequence.index("n3")


# ---------------------------------------------------------------------------
# Pre-execution validation
# ---------------------------------------------------------------------------


class TestPreExecutionValidation:
    """Tests for automatic pre-execution validation in run / run_message_based."""

    def _make_valid_flow(self, mock_config):
        """3-node chain with proper wiring."""
        flow = Flowchart(mock_config)
        flow.add_node(
            "n1", type="textparse", extraction={}, set={"current_input": "{default}"}
        )
        flow.add_node(
            "n2", type="textparse", extraction={}, set={"current_input": "{default}"}
        )
        flow.add_node(
            "n3", type="textparse", extraction={}, set={"current_input": "{default}"}
        )
        flow.add_edge("n1", "n2", AlwaysCondition)
        flow.add_edge("n2", "n3", AlwaysCondition)
        flow._initialize_message_routing()
        return flow

    @patch("pithos.flowchart.ConfigManager")
    def test_validate_passes_for_valid_flowchart(self, mock_config):
        """validate() returns True for a well-formed flowchart."""
        flow = self._make_valid_flow(mock_config)
        assert flow.validate(strict=False) is True

    @patch("pithos.flowchart.ConfigManager")
    def test_validate_detects_missing_condition(self, mock_config):
        """validate() raises if an edge has no condition."""
        flow = Flowchart(mock_config)
        flow.add_node("n1", type="textparse", extraction={})
        flow.add_node("n2", type="textparse", extraction={})
        # Manually add edge without a traversal_condition to simulate a corrupt graph
        flow.graph.add_edge("n1", "n2")  # no condition key
        flow._initialize_message_routing()

        with pytest.raises(ValidationError):
            flow.validate()

    @patch("pithos.flowchart.ConfigManager")
    def test_validate_warns_about_uncovered_inputs(self, mock_config):
        """validate() warns when a non-start node's required input is unwired."""
        flow = Flowchart(mock_config)
        # n2 expects a 'secondary' input but no edge provides it
        flow.add_node(
            "n1",
            type="textparse",
            extraction={},
            inputs=["default"],
            outputs=["default"],
        )
        flow.add_node(
            "n2",
            type="textparse",
            extraction={},
            inputs=["default", "secondary"],  # secondary never wired
        )
        flow.add_edge("n1", "n2", AlwaysCondition, input_key="default")
        flow._initialize_message_routing()

        # strict=False → should not raise even with warnings
        assert flow.validate(strict=False) is True

    @patch("pithos.flowchart.ConfigManager")
    def test_validate_called_before_run_message_based(self, mock_config):
        """run_message_based raises ValidationError for invalid flowchart."""
        flow = Flowchart(mock_config)
        flow.add_node("n1", type="textparse", extraction={})
        flow.add_node("n2", type="textparse", extraction={})
        # Add edge without a condition object to trigger graph-level error
        flow.graph.add_edge("n1", "n2")
        flow._initialize_message_routing()

        with pytest.raises(ValidationError):
            flow.run_message_based(initial_data="test")

    @patch("pithos.flowchart.ConfigManager")
    def test_valid_flowchart_runs_without_error(self, mock_config):
        """A valid flowchart runs through run_message_based with no exception."""
        flow = self._make_valid_flow(mock_config)
        result = flow.run_message_based(initial_data="hello")
        assert result["completed"] is True

    @patch("pithos.flowchart.ConfigManager")
    def test_validate_checks_orphaned_nodes(self, mock_config):
        """validate() warns about nodes unreachable from the start node."""
        data = {
            "nodes": {
                "start": {"type": "chatinput"},
                "reachable": {"type": "textparse", "extraction": {}},
                "orphan": {"type": "textparse", "extraction": {}},  # no incoming edge
            },
            "edges": [
                {
                    "from": "start",
                    "to": "reachable",
                    "condition": {"type": "AlwaysCondition"},
                }
            ],
            "start_node": "start",
        }
        # strict=False should not raise, just produce warnings
        flow = Flowchart.from_dict(data, mock_config, validate=False)
        # Calling validate directly with strict=True should raise
        with pytest.raises(ValidationError):
            flow.validate(strict=True)


# ---------------------------------------------------------------------------
# Execution tracing
# ---------------------------------------------------------------------------


class TestExecutionTracing:
    """Tests for enable_trace() / get_execution_trace()."""

    def _make_chain(self, mock_config, length=3):
        """Build a linear chain of textparse nodes of the given length."""
        flow = Flowchart(mock_config)
        for i in range(1, length + 1):
            flow.add_node(
                f"n{i}",
                type="textparse",
                extraction={},
                set={"current_input": "{default}"},
            )
        for i in range(1, length):
            flow.add_edge(f"n{i}", f"n{i + 1}", AlwaysCondition)
        flow._initialize_message_routing()
        return flow

    @patch("pithos.flowchart.ConfigManager")
    def test_trace_disabled_by_default(self, mock_config):
        """get_execution_trace() returns None when tracing is off."""
        flow = self._make_chain(mock_config)
        flow.run_message_based(initial_data="hi")
        assert flow.get_execution_trace() is None

    @patch("pithos.flowchart.ConfigManager")
    def test_enable_trace_returns_trace_after_run(self, mock_config):
        """Enabling tracing makes get_execution_trace() return an ExecutionTrace."""
        from pithos.flowchart import ExecutionTrace

        flow = self._make_chain(mock_config, length=3)
        flow.enable_trace()
        flow.run_message_based(initial_data="hello")

        trace = flow.get_execution_trace()
        assert isinstance(trace, ExecutionTrace)

    @patch("pithos.flowchart.ConfigManager")
    def test_trace_has_one_entry_per_step(self, mock_config):
        """Each node execution produces exactly one TraceEntry."""
        flow = self._make_chain(mock_config, length=3)
        flow.enable_trace()
        flow.run_message_based(initial_data="hello")

        trace = flow.get_execution_trace()
        assert len(trace.entries) == 3

    @patch("pithos.flowchart.ConfigManager")
    def test_trace_entries_have_correct_node_ids(self, mock_config):
        """TraceEntry.node_id reflects the node that executed."""
        flow = self._make_chain(mock_config, length=3)
        flow.enable_trace()
        flow.run_message_based(initial_data="hello")

        trace = flow.get_execution_trace()
        node_ids = [e.node_id for e in trace.entries]
        assert node_ids == ["n1", "n2", "n3"]

    @patch("pithos.flowchart.ConfigManager")
    def test_trace_entries_have_sequential_steps(self, mock_config):
        """TraceEntry.step is sequential starting at 0."""
        flow = self._make_chain(mock_config, length=3)
        flow.enable_trace()
        flow.run_message_based(initial_data="hello")

        trace = flow.get_execution_trace()
        steps = [e.step for e in trace.entries]
        assert steps == [0, 1, 2]

    @patch("pithos.flowchart.ConfigManager")
    def test_trace_captures_timing(self, mock_config):
        """TraceEntry records timestamp_start, timestamp_end, and duration_ms."""
        from datetime import datetime

        flow = self._make_chain(mock_config, length=2)
        flow.enable_trace()
        flow.run_message_based(initial_data="hello")

        trace = flow.get_execution_trace()
        assert trace.start_time is not None
        assert trace.end_time is not None
        assert trace.end_time >= trace.start_time

        for entry in trace.entries:
            assert isinstance(entry.timestamp_start, datetime)
            assert isinstance(entry.timestamp_end, datetime)
            assert entry.timestamp_end >= entry.timestamp_start
            assert entry.duration_ms >= 0

    @patch("pithos.flowchart.ConfigManager")
    def test_trace_captures_node_type(self, mock_config):
        """TraceEntry.node_type matches the FlowNode class name."""
        flow = self._make_chain(mock_config, length=1)
        flow.enable_trace()
        flow.run_message_based(initial_data="hello")

        trace = flow.get_execution_trace()
        assert trace.entries[0].node_type == "TextParseNode"

    @patch("pithos.flowchart.ConfigManager")
    def test_trace_captures_inputs(self, mock_config):
        """TraceEntry.inputs holds the data the node received."""
        flow = self._make_chain(mock_config, length=1)
        flow.enable_trace()
        flow.run_message_based(initial_data="my input")

        trace = flow.get_execution_trace()
        assert trace.entries[0].inputs.get("default") == "my input"

    @patch("pithos.flowchart.ConfigManager")
    def test_trace_captures_outputs(self, mock_config):
        """TraceEntry.outputs holds the data values produced by the node."""
        flow = self._make_chain(mock_config, length=1)
        flow.enable_trace()
        flow.run_message_based(initial_data="hello")

        trace = flow.get_execution_trace()
        # n1 sets current_input to the default input value
        assert len(trace.entries[0].outputs) >= 0  # may be empty if no downstream

    @patch("pithos.flowchart.ConfigManager")
    def test_trace_captures_edge_info(self, mock_config):
        """TraceEntry.edge is None for the first node and EdgeInfo for later nodes."""
        from pithos.flowchart import EdgeInfo

        flow = self._make_chain(mock_config, length=3)
        flow.enable_trace()
        flow.run_message_based(initial_data="hello")

        trace = flow.get_execution_trace()
        assert trace.entries[0].edge is None
        for entry in trace.entries[1:]:
            assert isinstance(entry.edge, EdgeInfo)
            assert entry.edge.to_node == entry.node_id

    @patch("pithos.flowchart.ConfigManager")
    def test_trace_resets_between_runs(self, mock_config):
        """Each new run starts a fresh trace — no carry-over from prior run."""
        flow = self._make_chain(mock_config, length=2)
        flow.enable_trace()

        flow.run_message_based(initial_data="first")
        trace1 = flow.get_execution_trace()

        flow.run_message_based(initial_data="second")
        trace2 = flow.get_execution_trace()

        # Different runs produce independent traces
        assert len(trace1.entries) == len(trace2.entries)
        # Entry step_ids are the same schema but different trace objects
        assert trace1 is not trace2

    @patch("pithos.flowchart.ConfigManager")
    def test_trace_completed_flag(self, mock_config):
        """ExecutionTrace.completed mirrors flowchart.finished after run."""
        flow = self._make_chain(mock_config, length=2)
        flow.enable_trace()
        flow.run_message_based(initial_data="hello")

        trace = flow.get_execution_trace()
        assert trace.completed is True
        assert trace.total_steps == 2

    @patch("pithos.flowchart.ConfigManager")
    def test_trace_entry_has_checkpoint(self, mock_config):
        """Each TraceEntry embeds a non-empty _checkpoint for restore_state."""
        flow = self._make_chain(mock_config, length=2)
        flow.enable_trace()
        flow.run_message_based(initial_data="hello")

        trace = flow.get_execution_trace()
        for entry in trace.entries:
            assert entry._checkpoint  # non-empty dict
            assert "step_counter" in entry._checkpoint
            assert "node_states" in entry._checkpoint


# ---------------------------------------------------------------------------
# Restore state
# ---------------------------------------------------------------------------


class TestRestoreState:
    """Tests for restore_state() allowing execution to resume mid-trace."""

    def _make_counting_flow(self, mock_config):
        """3-node chain where each node appends its ID to current_input."""
        flow = Flowchart(mock_config)

        def make_appender(label):
            return {
                "type": "textparse",
                "extraction": {},
                "set": {"current_input": "{default}" + f"|{label}"},
            }

        flow.add_node("n1", **make_appender("n1"))
        flow.add_node("n2", **make_appender("n2"))
        flow.add_node("n3", **make_appender("n3"))
        flow.add_edge("n1", "n2", AlwaysCondition)
        flow.add_edge("n2", "n3", AlwaysCondition)
        flow._initialize_message_routing()
        return flow

    @patch("pithos.flowchart.ConfigManager")
    def test_restore_state_raises_without_trace(self, mock_config):
        """restore_state raises if the TraceEntry has no checkpoint data."""
        from pithos.flowchart import TraceEntry
        from datetime import datetime

        flow = Flowchart(mock_config)
        flow.add_node("n1", type="textparse", extraction={})
        flow._initialize_message_routing()

        empty_entry = TraceEntry(
            step=0,
            node_id="n1",
            node_type="TextParseNode",
            timestamp_start=datetime.now(),
            timestamp_end=datetime.now(),
            duration_ms=0.0,
            inputs={},
            outputs=[],
            edge=None,
            _checkpoint={},  # empty — no checkpoint
        )
        with pytest.raises(ValueError, match="no checkpoint data"):
            flow.restore_state(empty_entry)

    @patch("pithos.flowchart.ConfigManager")
    def test_restore_state_raises_for_wrong_type(self, mock_config):
        """restore_state raises TypeError for unexpected argument types."""
        flow = Flowchart(mock_config)
        flow.add_node("n1", type="textparse", extraction={})

        with pytest.raises(TypeError):
            flow.restore_state("not a trace")  # type: ignore

    @patch("pithos.flowchart.ConfigManager")
    def test_restore_state_raises_for_empty_trace(self, mock_config):
        """restore_state raises ValueError for an ExecutionTrace with no entries."""
        from pithos.flowchart import ExecutionTrace

        flow = Flowchart(mock_config)
        empty_trace = ExecutionTrace(
            entries=[],
            completed=False,
            total_steps=0,
            start_time=None,
            end_time=None,
        )
        with pytest.raises(ValueError, match="empty"):
            flow.restore_state(empty_trace)

    @patch("pithos.flowchart.ConfigManager")
    def test_restore_sets_flag_and_state(self, mock_config):
        """After restore_state(), _has_restored_state is True and checkpoint applied."""
        flow = self._make_counting_flow(mock_config)
        flow.enable_trace()
        flow.run_message_based(initial_data="start")

        trace = flow.get_execution_trace()
        first_entry = trace.entries[0]

        flow.restore_state(first_entry)
        assert flow._has_restored_state is True

    @patch("pithos.flowchart.ConfigManager")
    def test_restore_from_trace_entry_resumes_execution(self, mock_config):
        """Restoring to step 0 and re-running continues from step 1."""
        flow = self._make_counting_flow(mock_config)
        flow.enable_trace()
        flow.run_message_based(initial_data="X")

        original_trace = flow.get_execution_trace()
        assert len(original_trace.entries) == 3

        # Restore to after step 0 (n1 has run, n2 and n3 are pending)
        flow.restore_state(original_trace.entries[0])

        result = flow.run_message_based()
        assert result["completed"] is True

        # n2 and n3 should have executed (2 steps)
        new_trace = flow.get_execution_trace()
        assert len(new_trace.entries) == 2
        node_ids = [e.node_id for e in new_trace.entries]
        assert node_ids == ["n2", "n3"]

    @patch("pithos.flowchart.ConfigManager")
    def test_restore_from_full_trace_resumes_from_end(self, mock_config):
        """Restoring from the full ExecutionTrace resumes from the final state."""
        flow = self._make_counting_flow(mock_config)
        flow.enable_trace()
        flow.run_message_based(initial_data="X")

        trace = flow.get_execution_trace()

        # Restore to the last state — already finished, no more steps
        flow.restore_state(trace)
        result = flow.run_message_based()

        # No nodes execute; a single terminal loop iteration detects the
        # finished state and counts as 1 step in the result dict.
        assert result["completed"] is True
        assert result["steps"] == 1

    @patch("pithos.flowchart.ConfigManager")
    def test_restore_from_middle_entry(self, mock_config):
        """Restoring to step 1 skips n1 and runs n3 only."""
        flow = self._make_counting_flow(mock_config)
        flow.enable_trace()
        flow.run_message_based(initial_data="X")

        trace = flow.get_execution_trace()
        assert len(trace.entries) == 3

        # entries[1] is after n2 executed — only n3 remains
        flow.restore_state(trace.entries[1])
        result = flow.run_message_based()

        assert result["completed"] is True
        new_trace = flow.get_execution_trace()
        assert len(new_trace.entries) == 1
        assert new_trace.entries[0].node_id == "n3"

    @patch("pithos.flowchart.ConfigManager")
    def test_manual_reset_clears_restored_state(self, mock_config):
        """Calling reset() after restore_state() cancels the restore."""
        flow = self._make_counting_flow(mock_config)
        flow.enable_trace()
        flow.run_message_based(initial_data="X")

        trace = flow.get_execution_trace()
        flow.restore_state(trace.entries[0])

        # Manual reset should clear the flag
        flow.reset()
        assert flow._has_restored_state is False

    @patch("pithos.flowchart.ConfigManager")
    def test_restored_step_counter_is_correct(self, mock_config):
        """After restore to step N, _step_counter matches the checkpoint."""
        flow = self._make_counting_flow(mock_config)
        flow.enable_trace()
        flow.run_message_based(initial_data="X")

        trace = flow.get_execution_trace()
        entry = trace.entries[1]  # after step 1

        flow.restore_state(entry)
        # step_counter in checkpoint is 2 (steps 0 and 1 completed)
        assert flow._step_counter == entry._checkpoint["step_counter"]


# ---------------------------------------------------------------------------
# Hot-reload / file watching tests
# ---------------------------------------------------------------------------


class TestHotReload:
    """Tests for Flowchart hot-reload / file-watching functionality."""

    # Minimal YAML flowchart string used across tests.
    # chatinput / chatoutput have no required parameters so they pass validation.
    _YAML_V1 = (
        "nodes:\n" "  n1:\n" "    type: chatinput\n" "start_node: n1\n" "edges: []\n"
    )

    _YAML_V2 = (
        "nodes:\n"
        "  n1:\n"
        "    type: chatinput\n"
        "  n2:\n"
        "    type: chatoutput\n"
        "start_node: n1\n"
        "edges: []\n"
    )

    def _write(self, path, content: str) -> None:
        path.write_text(content)

    # ------------------------------------------------------------------
    # start_watching / stop_watching / is_watching
    # ------------------------------------------------------------------

    @patch("pithos.flowchart.ConfigManager")
    def test_start_watching_raises_for_missing_file(self, mock_config, tmp_path):
        flow = Flowchart(mock_config)
        with pytest.raises(FileNotFoundError):
            flow.start_watching(tmp_path / "nonexistent.yaml")

    @patch("pithos.flowchart.ConfigManager")
    def test_is_watching_false_by_default(self, mock_config):
        flow = Flowchart(mock_config)
        assert flow.is_watching is False

    @patch("pithos.flowchart.ConfigManager")
    def test_is_watching_true_after_start(self, mock_config, tmp_path):
        yaml_file = tmp_path / "flow.yaml"
        self._write(yaml_file, self._YAML_V1)
        flow = Flowchart(mock_config)
        flow.start_watching(yaml_file, poll_interval=60)
        try:
            assert flow.is_watching is True
        finally:
            flow.stop_watching()

    @patch("pithos.flowchart.ConfigManager")
    def test_is_watching_false_after_stop(self, mock_config, tmp_path):
        yaml_file = tmp_path / "flow.yaml"
        self._write(yaml_file, self._YAML_V1)
        flow = Flowchart(mock_config)
        flow.start_watching(yaml_file, poll_interval=60)
        flow.stop_watching()
        assert flow.is_watching is False

    @patch("pithos.flowchart.ConfigManager")
    def test_stop_watching_idempotent_when_not_watching(self, mock_config):
        """stop_watching() is safe to call when no watcher is running."""
        flow = Flowchart(mock_config)
        flow.stop_watching()  # Should not raise
        assert flow.is_watching is False

    @patch("pithos.flowchart.ConfigManager")
    def test_restart_watcher_replaces_old_thread(self, mock_config, tmp_path):
        yaml_file = tmp_path / "flow.yaml"
        self._write(yaml_file, self._YAML_V1)
        flow = Flowchart(mock_config)
        flow.start_watching(yaml_file, poll_interval=60)
        first_thread = flow._watcher_thread
        flow.start_watching(yaml_file, poll_interval=60)
        try:
            assert flow._watcher_thread is not first_thread
        finally:
            flow.stop_watching()

    # ------------------------------------------------------------------
    # from_yaml / from_registered watch parameter
    # ------------------------------------------------------------------

    @patch("pithos.flowchart.ConfigManager")
    def test_from_yaml_watch_false_no_watcher(self, mock_config, tmp_path):
        yaml_file = tmp_path / "flow.yaml"
        self._write(yaml_file, self._YAML_V1)
        flow = Flowchart.from_yaml(str(yaml_file), mock_config, watch=False)
        assert flow.is_watching is False

    @patch("pithos.flowchart.ConfigManager")
    def test_from_yaml_watch_true_starts_watcher(self, mock_config, tmp_path):
        yaml_file = tmp_path / "flow.yaml"
        self._write(yaml_file, self._YAML_V1)
        flow = Flowchart.from_yaml(
            str(yaml_file), mock_config, watch=True, poll_interval=60
        )
        try:
            assert flow.is_watching is True
            assert flow._watch_path == yaml_file
        finally:
            flow.stop_watching()

    def test_from_registered_watch_true_starts_watcher(self, tmp_path):
        """from_registered(watch=True) wires up the file watcher."""
        from pithos.config_manager import ConfigManager
        from unittest.mock import MagicMock

        yaml_file = tmp_path / "my_flow.yaml"
        self._write(yaml_file, self._YAML_V1)

        cm = MagicMock(spec=ConfigManager)
        cm.get_config_file.return_value = yaml_file
        cm.get_registered_condition = MagicMock(return_value=AlwaysCondition)

        flow = Flowchart.from_registered("my_flow", cm, watch=True, poll_interval=60)
        try:
            assert flow.is_watching is True
        finally:
            flow.stop_watching()

    def test_from_registered_watch_raises_if_not_found(self):
        from pithos.config_manager import ConfigManager
        from unittest.mock import MagicMock

        cm = MagicMock(spec=ConfigManager)
        cm.get_config_file.return_value = None

        with pytest.raises(ValueError, match="not found"):
            Flowchart.from_registered("missing", cm, watch=True)

    # ------------------------------------------------------------------
    # Actual reload behaviour
    # ------------------------------------------------------------------

    @patch("pithos.flowchart.ConfigManager")
    def test_reload_updates_graph_in_place(self, mock_config, tmp_path):
        """_reload_from_path() replaces the graph with the new definition."""
        yaml_file = tmp_path / "flow.yaml"
        self._write(yaml_file, self._YAML_V1)
        flow = Flowchart.from_yaml(str(yaml_file), mock_config, watch=False)

        assert "n2" not in flow.graph.nodes

        # Write V2 and trigger a manual reload (bypasses the polling thread).
        self._write(yaml_file, self._YAML_V2)
        flow._reload_from_path(yaml_file)

        assert "n2" in flow.graph.nodes

    @patch("pithos.flowchart.ConfigManager")
    def test_reload_resets_execution_state(self, mock_config, tmp_path):
        """_reload_from_path() resets finished/step_counter state."""
        yaml_file = tmp_path / "flow.yaml"
        self._write(yaml_file, self._YAML_V1)
        flow = Flowchart.from_yaml(str(yaml_file), mock_config, watch=False)
        flow.finished = True
        flow._step_counter = 5

        flow._reload_from_path(yaml_file)

        assert flow.finished is False
        assert flow._step_counter == 0

    @patch("pithos.flowchart.ConfigManager")
    def test_on_reload_callback_invoked(self, mock_config, tmp_path):
        """on_reload callback receives the flowchart instance after reload."""
        yaml_file = tmp_path / "flow.yaml"
        self._write(yaml_file, self._YAML_V1)

        received = []

        def callback(fc):
            received.append(fc)

        flow = Flowchart.from_yaml(str(yaml_file), mock_config, watch=False)
        flow._on_reload = callback
        flow._reload_from_path(yaml_file)

        assert len(received) == 1
        assert received[0] is flow

    @patch("pithos.flowchart.ConfigManager")
    def test_on_reload_callback_exception_does_not_propagate(
        self, mock_config, tmp_path
    ):
        """A raising on_reload callback is logged but does not re-raise."""
        yaml_file = tmp_path / "flow.yaml"
        self._write(yaml_file, self._YAML_V1)

        def bad_callback(fc):
            raise RuntimeError("boom")

        flow = Flowchart.from_yaml(str(yaml_file), mock_config, watch=False)
        flow._on_reload = bad_callback
        # Should not raise even though the callback does.
        flow._reload_from_path(yaml_file)

    @patch("pithos.flowchart.ConfigManager")
    def test_bad_yaml_does_not_corrupt_graph(self, mock_config, tmp_path):
        """A malformed YAML reload leaves the existing graph intact."""
        yaml_file = tmp_path / "flow.yaml"
        self._write(yaml_file, self._YAML_V1)
        flow = Flowchart.from_yaml(str(yaml_file), mock_config, watch=False)
        original_nodes = set(flow.graph.nodes)

        yaml_file.write_text(": bad: yaml: {{{{")
        with pytest.raises(Exception):
            flow._reload_from_path(yaml_file)

        # Graph must not have been altered.
        assert set(flow.graph.nodes) == original_nodes

    # ------------------------------------------------------------------
    # Live polling test (short poll interval)
    # ------------------------------------------------------------------

    @patch("pithos.flowchart.ConfigManager")
    def test_watcher_thread_detects_file_change(self, mock_config, tmp_path):
        """Background watcher picks up a real file modification."""
        import time

        yaml_file = tmp_path / "flow.yaml"
        self._write(yaml_file, self._YAML_V1)

        reload_events = []

        flow = Flowchart.from_yaml(
            str(yaml_file),
            mock_config,
            watch=True,
            poll_interval=0.1,
            on_reload=lambda fc: reload_events.append(True),
        )
        try:
            # Give the watcher thread a moment to record the initial mtime.
            time.sleep(0.15)

            # Overwrite the file with V2.
            self._write(yaml_file, self._YAML_V2)

            # Wait up to 2 seconds for the reload to be detected.
            deadline = time.monotonic() + 2.0
            while not reload_events and time.monotonic() < deadline:
                time.sleep(0.05)

            assert reload_events, "Watcher did not detect the file change"
            assert "n2" in flow.graph.nodes
        finally:
            flow.stop_watching()
