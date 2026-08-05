"""Tests for global streaming flowchart trace-to-file support."""

import pytest
from unittest.mock import patch

from pithos.flowchart import Flowchart
from pithos.flowchart.trace_stream import (
    enable_global_trace,
    disable_global_trace,
    get_global_trace_sink,
    FlowchartTraceSink,
)
from pithos.conditions import AlwaysCondition


@pytest.fixture(autouse=True)
def _reset_global_trace():
    """Ensure the global trace sink never leaks between tests."""
    disable_global_trace()
    yield
    disable_global_trace()


def _make_chain(mock_config, length=3):
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


class TestGlobalTraceToggle:
    def test_no_sink_by_default(self):
        assert get_global_trace_sink() is None

    def test_enable_global_trace_returns_sink(self, tmp_path):
        path = tmp_path / "trace.log"
        sink = enable_global_trace(path)
        assert isinstance(sink, FlowchartTraceSink)
        assert get_global_trace_sink() is sink

    def test_disable_global_trace_clears_sink(self, tmp_path):
        enable_global_trace(tmp_path / "trace.log")
        disable_global_trace()
        assert get_global_trace_sink() is None

    def test_enable_creates_parent_directories(self, tmp_path):
        path = tmp_path / "nested" / "dir" / "trace.log"
        enable_global_trace(path)
        assert path.parent.is_dir()


class TestFlowchartTraceSinkFormatting:
    def test_write_input_line_format(self, tmp_path):
        path = tmp_path / "trace.log"
        sink = FlowchartTraceSink(path)
        sink.write_input("my_node", {"default": "hello"})
        sink.close()

        content = path.read_text(encoding="utf-8")
        lines = content.strip().splitlines()
        assert len(lines) == 1
        assert ": my_node, INPUT" in lines[0]
        assert "hello" in lines[0]

    def test_write_output_line_format(self, tmp_path):
        path = tmp_path / "trace.log"
        sink = FlowchartTraceSink(path)
        sink.write_output("my_node", ["world"])
        sink.close()

        content = path.read_text(encoding="utf-8")
        lines = content.strip().splitlines()
        assert len(lines) == 1
        assert ": my_node, OUTPUT" in lines[0]
        assert "world" in lines[0]

    def test_long_data_is_truncated(self, tmp_path):
        path = tmp_path / "trace.log"
        sink = FlowchartTraceSink(path)
        sink.write_input("n", "x" * 5000)
        sink.close()

        line = path.read_text(encoding="utf-8").strip()
        assert len(line) < 1000
        assert "truncated" in line

    def test_newlines_in_data_are_escaped(self, tmp_path):
        path = tmp_path / "trace.log"
        sink = FlowchartTraceSink(path)
        sink.write_input("n", "line1\nline2")
        sink.close()

        content = path.read_text(encoding="utf-8")
        assert len(content.strip().splitlines()) == 1
        assert "\\n" in content


class TestFlowchartStreamsToGlobalSink:
    @patch("pithos.flowchart.ConfigManager")
    def test_new_flowchart_auto_enables_tracer_when_global_trace_set(
        self, mock_config, tmp_path
    ):
        enable_global_trace(tmp_path / "trace.log")
        flow = _make_chain(mock_config, length=1)
        assert flow.get_execution_trace() is not None  # tracer already enabled
        flow.run_message_based(initial_data="hello")
        trace = flow.get_execution_trace()
        assert trace is not None
        assert len(trace.entries) == 1

    @patch("pithos.flowchart.ConfigManager")
    def test_flowchart_created_before_enable_is_not_traced(self, mock_config, tmp_path):
        flow = _make_chain(mock_config, length=1)
        enable_global_trace(tmp_path / "trace.log")
        flow.run_message_based(initial_data="hello")
        assert flow.get_execution_trace() is None

    @patch("pithos.flowchart.ConfigManager")
    def test_run_streams_input_and_output_lines_per_node(self, mock_config, tmp_path):
        path = tmp_path / "trace.log"
        enable_global_trace(path)
        flow = _make_chain(mock_config, length=3)
        flow.run_message_based(initial_data="hello")

        lines = path.read_text(encoding="utf-8").strip().splitlines()
        # Two lines (INPUT + OUTPUT) per executed node.
        assert len(lines) == 6
        for node_id in ("n1", "n2", "n3"):
            assert any(f": {node_id}, INPUT" in line for line in lines)
            assert any(f": {node_id}, OUTPUT" in line for line in lines)

    @patch("pithos.flowchart.ConfigManager")
    def test_input_line_precedes_output_line_for_each_node(self, mock_config, tmp_path):
        path = tmp_path / "trace.log"
        enable_global_trace(path)
        flow = _make_chain(mock_config, length=1)
        flow.run_message_based(initial_data="hello")

        lines = path.read_text(encoding="utf-8").strip().splitlines()
        assert ": n1, INPUT" in lines[0]
        assert ": n1, OUTPUT" in lines[1]

    @patch("pithos.flowchart.ConfigManager")
    def test_no_stream_when_global_trace_disabled(self, mock_config, tmp_path):
        path = tmp_path / "trace.log"
        flow = _make_chain(mock_config, length=2)
        flow.run_message_based(initial_data="hello")
        assert not path.exists()
