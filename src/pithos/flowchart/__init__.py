"""Flowchart package — directed graph-based workflow execution.

All public names that were previously importable from ``pithos.flowchart``
are re-exported here so that existing code continues to work unchanged.
"""

from .flowchart import Flowchart
from .models import EdgeInfo, ExecutionTrace, ProgressEvent, TraceEntry
from .graph import FlowchartGraph
from .executor import FlowchartExecutor
from .tracer import ExecutionTracer
from .watcher import FlowchartWatcher
from .serialization import FlowchartSerializer

# Keep the module-level helpers that existed in the old flowchart.py
from .models import serialise_message as _serialise_message  # noqa: F401
from .models import deserialise_message as _deserialise_message  # noqa: F401

# Re-export names that tests/callers expect to find on `pithos.flowchart`
from ..validation import ValidationError  # noqa: F401
from ..config_manager import ConfigManager  # noqa: F401
from ..conditions import ConditionManager  # noqa: F401

__all__ = [
    "Flowchart",
    "EdgeInfo",
    "ExecutionTrace",
    "ProgressEvent",
    "TraceEntry",
    "FlowchartGraph",
    "FlowchartExecutor",
    "ExecutionTracer",
    "FlowchartWatcher",
    "FlowchartSerializer",
    "ValidationError",
]
