"""Process-global streaming trace sink for flowchart node execution.

When enabled via :func:`enable_global_trace`, every :class:`~pithos.flowchart.Flowchart`
instance created afterwards automatically enables its :class:`ExecutionTracer`
and streams per-node activity — a timestamp, the node id, and the node's
input or output data — to the configured file as execution happens.

This is independent from :meth:`Flowchart.get_execution_trace`, which only
holds the trace in memory for the most recent run. The streaming sink writes
incrementally, one line per node input and one line per node output, so it
can be tailed live and captures activity across every flowchart run in the
process (chain-of-thought inference flowcharts, the ``flowchart`` tool,
nested ``ToolCallNode`` executions, etc.) without threading a config object
through every call site.
"""

from __future__ import annotations

import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Union

_MAX_LINE_LEN = 500

_lock = threading.Lock()
_sink: Optional["FlowchartTraceSink"] = None


class FlowchartTraceSink:
    """Thread-safe writer that appends node activity lines to a file."""

    def __init__(self, path: Union[str, Path]) -> None:
        self.path = Path(path)
        self._write_lock = threading.Lock()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = open(self.path, "a", encoding="utf-8", buffering=1)

    @staticmethod
    def _format_data(data: Any) -> str:
        text = (
            str(data).replace("\r\n", "\\n").replace("\n", "\\n").replace("\r", "\\r")
        )
        if len(text) > _MAX_LINE_LEN:
            text = text[:_MAX_LINE_LEN] + "...(truncated)"
        return text

    def write_input(
        self, node_id: str, data: Any, ts: Optional[datetime] = None
    ) -> None:
        """Write a line reporting that *node_id* has started with *data*."""
        ts = ts or datetime.now()
        line = f"{ts.isoformat(timespec='milliseconds')}: {node_id}, INPUT {self._format_data(data)}\n"
        with self._write_lock:
            self._file.write(line)

    def write_output(
        self, node_id: str, data: Any, ts: Optional[datetime] = None
    ) -> None:
        """Write a line reporting that *node_id* finished, producing *data*."""
        ts = ts or datetime.now()
        line = f"{ts.isoformat(timespec='milliseconds')}: {node_id}, OUTPUT {self._format_data(data)}\n"
        with self._write_lock:
            self._file.write(line)

    def close(self) -> None:
        """Flush and close the underlying file handle."""
        with self._write_lock:
            try:
                self._file.close()
            except Exception:
                pass


def enable_global_trace(path: Union[str, Path]) -> FlowchartTraceSink:
    """Enable process-wide flowchart tracing to *path*.

    Every :class:`Flowchart` constructed after this call automatically
    enables its execution tracer and streams per-node activity to *path*.
    Flowcharts constructed before this call are unaffected.
    """
    global _sink
    with _lock:
        if _sink is not None:
            _sink.close()
        _sink = FlowchartTraceSink(path)
        return _sink


def disable_global_trace() -> None:
    """Disable process-wide flowchart tracing.

    Flowcharts already streaming to the previous sink stop being written to;
    new :class:`Flowchart` instances will not stream.
    """
    global _sink
    with _lock:
        if _sink is not None:
            _sink.close()
        _sink = None


def get_global_trace_sink() -> Optional[FlowchartTraceSink]:
    """Return the current global :class:`FlowchartTraceSink`, or ``None``."""
    with _lock:
        return _sink
