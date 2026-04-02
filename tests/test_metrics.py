"""Tests for pithos.metrics.MetricsCollector."""

import json
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from pithos.metrics import (
    MetricsCollector,
    TokenMetrics,
    ToolCallMetrics,
    MemoryMetrics,
    FlowchartPathEntry,
)


# ---------------------------------------------------------------------------
# TokenMetrics unit tests
# ---------------------------------------------------------------------------


class TestTokenMetrics:
    def test_initial_state(self):
        m = TokenMetrics()
        assert m.prompt_tokens == 0
        assert m.completion_tokens == 0
        assert m.total_calls == 0
        assert m.avg_response_time_ms == 0.0
        assert m.min_response_time_ms is None
        assert m.max_response_time_ms is None

    def test_single_record(self):
        m = TokenMetrics()
        m.record(100, 50, 200.0)
        assert m.prompt_tokens == 100
        assert m.completion_tokens == 50
        assert m.total_calls == 1
        assert m.avg_response_time_ms == 200.0
        assert m.min_response_time_ms == 200.0
        assert m.max_response_time_ms == 200.0

    def test_multiple_records_accumulate(self):
        m = TokenMetrics()
        m.record(100, 50, 200.0)
        m.record(200, 100, 400.0)
        assert m.prompt_tokens == 300
        assert m.completion_tokens == 150
        assert m.total_calls == 2
        assert m.avg_response_time_ms == 300.0
        assert m.min_response_time_ms == 200.0
        assert m.max_response_time_ms == 400.0

    def test_to_dict_roundtrip(self):
        m = TokenMetrics()
        m.record(10, 20, 100.0)
        d = m.to_dict()
        m2 = TokenMetrics.from_dict(d)
        assert m2.prompt_tokens == 10
        assert m2.completion_tokens == 20
        assert m2.total_calls == 1
        assert m2.min_response_time_ms == 100.0
        assert m2.max_response_time_ms == 100.0

    def test_from_dict_missing_keys(self):
        m = TokenMetrics.from_dict({})
        assert m.prompt_tokens == 0
        assert m.total_calls == 0
        assert m.min_response_time_ms is None


# ---------------------------------------------------------------------------
# ToolCallMetrics unit tests
# ---------------------------------------------------------------------------


class TestToolCallMetrics:
    def test_initial_state(self):
        m = ToolCallMetrics()
        assert m.successes == 0
        assert m.failures == 0
        assert m.total_calls == 0
        assert m.success_rate == 0.0
        assert m.avg_execution_time_ms == 0.0

    def test_record_success(self):
        m = ToolCallMetrics()
        m.record(success=True, execution_time_ms=50.0)
        assert m.successes == 1
        assert m.failures == 0
        assert m.success_rate == 1.0

    def test_record_failure(self):
        m = ToolCallMetrics()
        m.record(success=False, execution_time_ms=10.0)
        assert m.successes == 0
        assert m.failures == 1
        assert m.success_rate == 0.0

    def test_mixed_success_rate(self):
        m = ToolCallMetrics()
        m.record(True, 10.0)
        m.record(True, 20.0)
        m.record(False, 5.0)
        assert m.total_calls == 3
        assert pytest.approx(m.success_rate) == 2 / 3
        assert m.avg_execution_time_ms == pytest.approx(35.0 / 3)

    def test_to_dict_roundtrip(self):
        m = ToolCallMetrics()
        m.record(True, 100.0)
        m.record(False, 50.0)
        d = m.to_dict()
        m2 = ToolCallMetrics.from_dict(d)
        assert m2.successes == 1
        assert m2.failures == 1
        assert m2.total_execution_time_ms == 150.0


# ---------------------------------------------------------------------------
# MemoryMetrics unit tests
# ---------------------------------------------------------------------------


class TestMemoryMetrics:
    def test_initial_state(self):
        m = MemoryMetrics()
        assert m.retrieve_hits == 0
        assert m.retrieve_misses == 0
        assert m.hit_rate == 0.0
        assert m.store_count == 0

    def test_record_retrieve_hit(self):
        m = MemoryMetrics()
        m.record_retrieve(hit=True, result_count=3)
        assert m.retrieve_hits == 1
        assert m.retrieve_misses == 0
        assert m.hit_rate == 1.0
        assert m.total_results_returned == 3

    def test_record_retrieve_miss(self):
        m = MemoryMetrics()
        m.record_retrieve(hit=False, result_count=0)
        assert m.retrieve_hits == 0
        assert m.retrieve_misses == 1
        assert m.hit_rate == 0.0

    def test_hit_rate_mixed(self):
        m = MemoryMetrics()
        m.record_retrieve(True, 2)
        m.record_retrieve(False, 0)
        assert m.hit_rate == pytest.approx(0.5)

    def test_record_store(self):
        m = MemoryMetrics()
        m.record_store()
        m.record_store()
        assert m.store_count == 2

    def test_to_dict_roundtrip(self):
        m = MemoryMetrics()
        m.record_retrieve(True, 5)
        m.record_store()
        d = m.to_dict()
        m2 = MemoryMetrics.from_dict(d)
        assert m2.retrieve_hits == 1
        assert m2.store_count == 1
        assert m2.total_results_returned == 5


# ---------------------------------------------------------------------------
# FlowchartPathEntry unit tests
# ---------------------------------------------------------------------------


class TestFlowchartPathEntry:
    def test_to_dict_roundtrip(self):
        e = FlowchartPathEntry(
            "flow1", "node_a", "PromptNode", 42.0, "start", "2025-01-01T00:00:00"
        )
        d = e.to_dict()
        e2 = FlowchartPathEntry.from_dict(d)
        assert e2.flowchart_name == "flow1"
        assert e2.node_id == "node_a"
        assert e2.node_type == "PromptNode"
        assert e2.duration_ms == 42.0
        assert e2.from_node == "start"

    def test_from_dict_missing_keys(self):
        e = FlowchartPathEntry.from_dict({})
        assert e.flowchart_name == ""
        assert e.from_node is None


# ---------------------------------------------------------------------------
# MetricsCollector unit tests
# ---------------------------------------------------------------------------


class TestMetricsCollector:
    def test_initial_snapshot_empty(self):
        c = MetricsCollector()
        snap = c.get_snapshot()
        assert snap["token_usage"] == {}
        assert snap["tool_calls"] == {}
        assert snap["memory"]["retrieve_total"] == 0
        assert snap["flowchart_paths"] == []

    def test_record_token_usage(self):
        c = MetricsCollector()
        c.record_token_usage("glm-4.7-flash", 100, 50, 200.0)
        snap = c.get_snapshot()
        m = snap["token_usage"]["glm-4.7-flash"]
        assert m["prompt_tokens"] == 100
        assert m["completion_tokens"] == 50
        assert m["total_calls"] == 1
        assert m["avg_response_time_ms"] == 200.0

    def test_record_token_usage_multiple_models(self):
        c = MetricsCollector()
        c.record_token_usage("model_a", 10, 5, 100.0)
        c.record_token_usage("model_b", 20, 10, 200.0)
        c.record_token_usage("model_a", 30, 15, 300.0)
        snap = c.get_snapshot()
        assert snap["token_usage"]["model_a"]["total_calls"] == 2
        assert snap["token_usage"]["model_b"]["total_calls"] == 1

    def test_record_tool_call(self):
        c = MetricsCollector()
        c.record_tool_call("python", True, 50.0)
        c.record_tool_call("python", False, 10.0)
        snap = c.get_snapshot()
        tool = snap["tool_calls"]["python"]
        assert tool["successes"] == 1
        assert tool["failures"] == 1
        assert tool["success_rate"] == pytest.approx(0.5)

    def test_record_memory_retrieve_hit(self):
        c = MetricsCollector()
        c.record_memory_retrieve(result_count=3)
        snap = c.get_snapshot()
        assert snap["memory"]["retrieve_hits"] == 1
        assert snap["memory"]["retrieve_misses"] == 0
        assert snap["memory"]["hit_rate"] == 1.0

    def test_record_memory_retrieve_miss(self):
        c = MetricsCollector()
        c.record_memory_retrieve(result_count=0)
        snap = c.get_snapshot()
        assert snap["memory"]["retrieve_hits"] == 0
        assert snap["memory"]["retrieve_misses"] == 1

    def test_record_memory_store(self):
        c = MetricsCollector()
        c.record_memory_store()
        c.record_memory_store()
        snap = c.get_snapshot()
        assert snap["memory"]["store_count"] == 2

    def test_record_flowchart_step(self):
        c = MetricsCollector()
        c.record_flowchart_step("my_flow", "node1", "PromptNode", 150.0, None)
        c.record_flowchart_step("my_flow", "node2", "AgentPromptNode", 300.0, "node1")
        snap = c.get_snapshot()
        paths = snap["flowchart_paths"]
        assert len(paths) == 2
        assert paths[0]["node_id"] == "node1"
        assert paths[0]["from_node"] is None
        assert paths[1]["from_node"] == "node1"

    def test_reset(self):
        c = MetricsCollector()
        c.record_token_usage("m", 1, 1, 1.0)
        c.record_tool_call("t", True, 1.0)
        c.record_memory_store()
        c.record_flowchart_step("f", "n", "T", 1.0)
        c.reset()
        snap = c.get_snapshot()
        assert snap["token_usage"] == {}
        assert snap["tool_calls"] == {}
        assert snap["memory"]["store_count"] == 0
        assert snap["flowchart_paths"] == []

    def test_max_path_entries_ring_buffer(self):
        c = MetricsCollector(max_path_entries=5)
        for i in range(10):
            c.record_flowchart_step("f", f"n{i}", "T", float(i))
        snap = c.get_snapshot()
        paths = snap["flowchart_paths"]
        assert len(paths) == 5
        # Most recent 5 entries should be retained
        assert paths[-1]["node_id"] == "n9"

    def test_thread_safety(self):
        c = MetricsCollector()
        errors = []

        def worker():
            try:
                for _ in range(100):
                    c.record_token_usage("m", 1, 1, 1.0)
                    c.record_tool_call("t", True, 1.0)
                    c.record_memory_retrieve(1)
                    c.record_flowchart_step("f", "n", "T", 1.0)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == []
        snap = c.get_snapshot()
        assert snap["token_usage"]["m"]["total_calls"] == 400

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def test_save_and_reload(self, tmp_path):
        c = MetricsCollector()
        c.record_token_usage("glm-4.7-flash", 50, 25, 100.0)
        c.record_tool_call("git", True, 200.0)
        c.record_memory_retrieve(3)
        c.record_memory_store()
        c.record_flowchart_step("flow", "a", "PromptNode", 50.0)

        path = str(tmp_path / "metrics.json")
        c.save(path)
        assert Path(path).is_file()

        c2 = MetricsCollector()
        c2.load(path)
        snap = c2.get_snapshot()
        assert snap["token_usage"]["glm-4.7-flash"]["prompt_tokens"] == 50
        assert snap["tool_calls"]["git"]["successes"] == 1
        assert snap["memory"]["retrieve_hits"] == 1
        assert snap["memory"]["store_count"] == 1
        assert len(snap["flowchart_paths"]) == 1

    def test_load_merges_with_existing(self, tmp_path):
        c = MetricsCollector()
        c.record_token_usage("model", 100, 50, 200.0)
        path = str(tmp_path / "metrics.json")
        c.save(path)

        c2 = MetricsCollector()
        c2.record_token_usage("model", 50, 25, 100.0)
        c2.load(path)
        snap = c2.get_snapshot()
        # Loaded 100 + existing 50 = 150
        assert snap["token_usage"]["model"]["prompt_tokens"] == 150
        assert snap["token_usage"]["model"]["total_calls"] == 2

    def test_load_creates_new_model_entries(self, tmp_path):
        c = MetricsCollector()
        c.record_token_usage("model_a", 10, 5, 50.0)
        path = str(tmp_path / "metrics.json")
        c.save(path)

        c2 = MetricsCollector()
        c2.record_token_usage("model_b", 20, 10, 100.0)
        c2.load(path)
        snap = c2.get_snapshot()
        assert "model_a" in snap["token_usage"]
        assert "model_b" in snap["token_usage"]

    def test_load_file_not_found(self):
        c = MetricsCollector()
        with pytest.raises(FileNotFoundError):
            c.load("/nonexistent/path/metrics.json")

    def test_save_creates_parent_directories(self, tmp_path):
        c = MetricsCollector()
        path = str(tmp_path / "nested" / "deep" / "metrics.json")
        c.save(path)
        assert Path(path).is_file()

    def test_save_is_valid_json(self, tmp_path):
        c = MetricsCollector()
        c.record_token_usage("m", 1, 1, 1.0)
        path = str(tmp_path / "metrics.json")
        c.save(path)
        with open(path) as f:
            data = json.load(f)
        assert "token_usage" in data
        assert "tool_calls" in data
        assert "memory" in data
        assert "flowchart_paths" in data

    def test_load_path_entries_respect_max(self, tmp_path):
        c = MetricsCollector(max_path_entries=5)
        for i in range(4):
            c.record_flowchart_step("f", f"n{i}", "T", float(i))
        path = str(tmp_path / "metrics.json")
        c.save(path)

        c2 = MetricsCollector(max_path_entries=5)
        for i in range(4):
            c2.record_flowchart_step("f", f"m{i}", "T", float(i))
        c2.load(path)
        snap = c2.get_snapshot()
        # 4 existing + 4 from file = 8, capped at 5
        assert len(snap["flowchart_paths"]) == 5

    # ------------------------------------------------------------------
    # Auto-save
    # ------------------------------------------------------------------

    def test_start_stop_auto_save(self, tmp_path):
        c = MetricsCollector()
        path = str(tmp_path / "auto.json")
        c.start_auto_save(path, interval_seconds=0.05)
        assert c.is_auto_saving
        time.sleep(0.15)
        c.stop_auto_save()
        assert not c.is_auto_saving
        # stop_auto_save performs a final save
        assert Path(path).is_file()

    def test_auto_save_writes_metrics(self, tmp_path):
        c = MetricsCollector()
        path = str(tmp_path / "auto.json")
        c.record_token_usage("m", 5, 5, 50.0)
        c.start_auto_save(path, interval_seconds=0.05)
        time.sleep(0.15)
        c.stop_auto_save()
        with open(path) as f:
            data = json.load(f)
        assert data["token_usage"]["m"]["total_calls"] == 1

    def test_invalid_interval_raises(self):
        c = MetricsCollector()
        with pytest.raises(ValueError):
            c.start_auto_save("/tmp/m.json", interval_seconds=0)
        with pytest.raises(ValueError):
            c.start_auto_save("/tmp/m.json", interval_seconds=-1)

    def test_restart_auto_save_stops_old_thread(self, tmp_path):
        c = MetricsCollector()
        path = str(tmp_path / "auto.json")
        c.start_auto_save(path, interval_seconds=60)
        first_thread = c._save_thread
        c.start_auto_save(path, interval_seconds=60)
        # The first thread should have been replaced
        assert c._save_thread is not first_thread
        c.stop_auto_save()


# ---------------------------------------------------------------------------
# Integration: Agent.attach_metrics()
# ---------------------------------------------------------------------------


class TestAgentMetricsIntegration:
    """Verify that OllamaAgent hooks call MetricsCollector correctly."""

    def _make_agent(self):
        from pithos import OllamaAgent

        return OllamaAgent("test-model")

    def test_attach_metrics_sets_attribute(self):
        agent = self._make_agent()
        c = MetricsCollector()
        agent.attach_metrics(c)
        assert agent.metrics is c

    def test_token_usage_recorded_on_send(self):
        agent = self._make_agent()
        c = MetricsCollector()
        agent.attach_metrics(c)

        mock_chunk = MagicMock()
        mock_chunk.message.content = "Hello"
        mock_usage = MagicMock()
        mock_usage.prompt_tokens = 10
        mock_usage.completion_tokens = 5
        mock_chunk.usage = mock_usage

        with patch("pithos.agent.ollama_agent.chat", return_value=[mock_chunk]):
            agent.send("Hi")

        snap = c.get_snapshot()
        assert "test-model" in snap["token_usage"]
        assert snap["token_usage"]["test-model"]["prompt_tokens"] == 10
        assert snap["token_usage"]["test-model"]["completion_tokens"] == 5
        assert snap["token_usage"]["test-model"]["total_calls"] == 1

    def test_response_time_recorded_on_send(self):
        agent = self._make_agent()
        c = MetricsCollector()
        agent.attach_metrics(c)

        mock_chunk = MagicMock()
        mock_chunk.message.content = "Hi"
        mock_usage = MagicMock()
        mock_usage.prompt_tokens = 1
        mock_usage.completion_tokens = 1
        mock_chunk.usage = mock_usage

        with patch("pithos.agent.ollama_agent.chat", return_value=[mock_chunk]):
            agent.send("test")

        snap = c.get_snapshot()
        assert snap["token_usage"]["test-model"]["total_response_time_ms"] >= 0
        assert snap["token_usage"]["test-model"]["min_response_time_ms"] is not None

    def test_send_without_metrics_does_not_raise(self):
        agent = self._make_agent()
        mock_chunk = MagicMock()
        mock_chunk.message.content = "Hello"
        mock_chunk.usage = None

        with patch("pithos.agent.ollama_agent.chat", return_value=[mock_chunk]):
            result = agent.send("Hi")
        assert result == "Hello"

    def test_tool_call_metrics_recorded(self):
        agent = self._make_agent()
        c = MetricsCollector()
        agent.attach_metrics(c)

        from pithos.tools import ToolResult, ToolCallRequest

        mock_executor = MagicMock()
        mock_registry = MagicMock()
        mock_result = ToolResult(
            success=True,
            stdout="output",
            stderr="",
            exit_code=0,
            execution_time=0.1,
            command="python --version",
        )
        mock_executor.run.return_value = mock_result
        agent.tool_executor = mock_executor
        agent.tool_registry = mock_registry

        requests = [
            ToolCallRequest(command="python --version", format="cli", raw_text="")
        ]
        agent._execute_tools(requests)

        snap = c.get_snapshot()
        assert "python" in snap["tool_calls"]
        assert snap["tool_calls"]["python"]["successes"] == 1
        assert snap["tool_calls"]["python"]["total_execution_time_ms"] == pytest.approx(
            100.0
        )

    def test_tool_call_failure_recorded(self):
        agent = self._make_agent()
        c = MetricsCollector()
        agent.attach_metrics(c)

        from pithos.tools import ToolResult

        mock_executor = MagicMock()
        mock_registry = MagicMock()
        mock_result = ToolResult(
            success=False,
            stdout="",
            stderr="error",
            exit_code=1,
            execution_time=0.05,
            command="badtool arg",
        )
        mock_executor.run.return_value = mock_result
        agent.tool_executor = mock_executor
        agent.tool_registry = mock_registry

        requests = [MagicMock()]
        requests[0].command = "badtool arg"
        agent._execute_tools(requests)

        snap = c.get_snapshot()
        assert snap["tool_calls"]["badtool"]["failures"] == 1

    def test_memory_store_metric_recorded(self):
        from pithos import OllamaAgent
        from pithos.tools import MemoryOpRequest

        agent = OllamaAgent("test-model")
        c = MetricsCollector()
        agent.attach_metrics(c)

        mock_store = MagicMock()
        mock_store.store.return_value = "id123"
        agent.memory_store = mock_store

        ops = [
            MemoryOpRequest(
                operation="store", category="facts", content="hello world", query=None
            )
        ]
        agent._execute_memory_ops(ops)

        snap = c.get_snapshot()
        assert snap["memory"]["store_count"] == 1

    def test_memory_retrieve_hit_recorded(self):
        from pithos import OllamaAgent
        from pithos.tools import MemoryOpRequest

        agent = OllamaAgent("test-model")
        c = MetricsCollector()
        agent.attach_metrics(c)

        mock_result = MagicMock()
        mock_result.relevance_score = 0.9
        mock_result.content = "some fact"

        mock_store = MagicMock()
        mock_store.retrieve.return_value = [mock_result, mock_result]
        agent.memory_store = mock_store

        ops = [
            MemoryOpRequest(
                operation="retrieve", category="facts", query="hello", content=None
            )
        ]
        agent._execute_memory_ops(ops)

        snap = c.get_snapshot()
        assert snap["memory"]["retrieve_hits"] == 1
        assert snap["memory"]["retrieve_misses"] == 0

    def test_memory_retrieve_miss_recorded(self):
        from pithos import OllamaAgent
        from pithos.tools import MemoryOpRequest

        agent = OllamaAgent("test-model")
        c = MetricsCollector()
        agent.attach_metrics(c)

        mock_store = MagicMock()
        mock_store.retrieve.return_value = []
        agent.memory_store = mock_store

        ops = [
            MemoryOpRequest(
                operation="retrieve", category="facts", query="nothing", content=None
            )
        ]
        agent._execute_memory_ops(ops)

        snap = c.get_snapshot()
        assert snap["memory"]["retrieve_hits"] == 0
        assert snap["memory"]["retrieve_misses"] == 1


# ---------------------------------------------------------------------------
# Integration: Flowchart.attach_metrics()
# ---------------------------------------------------------------------------


class TestFlowchartMetricsIntegration:
    def test_attach_metrics_sets_attribute(self):
        from pithos import Flowchart, ConfigManager

        cm = MagicMock(spec=ConfigManager)
        fc = Flowchart(cm)
        collector = MetricsCollector()
        fc.attach_metrics(collector, name="test_flow")
        assert fc.metrics is collector
        assert fc._metrics_name == "test_flow"

    def test_attach_metrics_default_name(self):
        from pithos import Flowchart, ConfigManager

        cm = MagicMock(spec=ConfigManager)
        fc = Flowchart(cm, registered_name="my_flow")
        collector = MetricsCollector()
        fc.attach_metrics(collector)
        # Name defaults to registered_name
        assert fc._metrics_name == "my_flow"

    def test_flowchart_step_recorded_during_execution(self):
        """A minimal flowchart run records path entries in the collector."""
        from pithos import Flowchart, ConfigManager
        from pithos.flownode import FlowNode
        from pithos.message import Message, NodeInputState

        cm = MagicMock(spec=ConfigManager)
        cm.get_config.return_value = None

        # Build a one-node flowchart with a mock node
        class _EchoNode(FlowNode):
            required_inputs = ["default"]

            def execute_with_messages(
                self, input_state: NodeInputState, router
            ) -> list:
                data = input_state.get_input_data("default")
                return [Message(data=data, source_node="echo")]

            def to_dict(self):
                return {"type": "echo"}

        fc = Flowchart(cm)
        echo = _EchoNode({})
        fc.graph.add_node("echo", nodeobj=echo)
        fc.start_node = "echo"

        collector = MetricsCollector()
        fc.attach_metrics(collector, name="test_flow")

        with patch.object(fc, "validate", return_value=True):
            fc._initialize_message_routing()
            fc.run_message_based(initial_data="hello", max_steps=5)

        snap = collector.get_snapshot()
        paths = snap["flowchart_paths"]
        assert len(paths) >= 1
        assert paths[0]["flowchart_name"] == "test_flow"
        assert paths[0]["node_id"] == "echo"
        assert paths[0]["duration_ms"] >= 0

    def test_flowchart_records_from_node(self):
        """Multi-node flowchart: second node's from_node is set correctly."""
        from pithos import Flowchart, ConfigManager
        from pithos.flownode import FlowNode
        from pithos.message import Message, NodeInputState
        from pithos.conditions import AlwaysCondition

        cm = MagicMock(spec=ConfigManager)
        cm.get_config.return_value = None

        class _PassNode(FlowNode):
            required_inputs = ["default"]

            def execute_with_messages(
                self, input_state: NodeInputState, router
            ) -> list:
                data = input_state.get_input_data("default")
                return [Message(data=data, source_node=self.__class__.__name__)]

            def to_dict(self):
                return {"type": "pass"}

        fc = Flowchart(cm)
        a = _PassNode({})
        b = _PassNode({})
        fc.graph.add_node("a", nodeobj=a)
        fc.graph.add_node("b", nodeobj=b)
        fc.start_node = "a"
        # AlwaysCondition is used as a class (not instance) in this codebase
        fc.add_edge("a", "b", AlwaysCondition)

        collector = MetricsCollector()
        fc.attach_metrics(collector, name="two_step")

        with patch.object(fc, "validate", return_value=True):
            fc._initialize_message_routing()
            fc.run_message_based(initial_data="hi", max_steps=10)

        snap = collector.get_snapshot()
        paths = snap["flowchart_paths"]
        nodes = [p["node_id"] for p in paths]
        assert "a" in nodes
        assert "b" in nodes
        b_entry = next(p for p in paths if p["node_id"] == "b")
        assert b_entry["from_node"] == "a"

    def test_no_metrics_attached_does_not_raise(self):
        """Flowchart without attached metrics executes without error."""
        from pithos import Flowchart, ConfigManager
        from pithos.flownode import FlowNode
        from pithos.message import Message

        cm = MagicMock(spec=ConfigManager)
        cm.get_config.return_value = None

        class _Node(FlowNode):
            required_inputs = ["default"]

            def execute_with_messages(self, input_state, router):
                return [Message(data="ok")]

            def to_dict(self):
                return {"type": "prompt"}

        fc = Flowchart(cm)
        fc.graph.add_node("n", nodeobj=_Node({}))
        fc.start_node = "n"
        # No metrics attached — should not raise
        with patch.object(fc, "validate", return_value=True):
            result = fc.run_message_based(initial_data="x", max_steps=5)
        assert result["steps"] >= 1
