"""Data classes for flowchart execution tracing and progress events."""

from typing import Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
import uuid

from ..message import Message


@dataclass
class EdgeInfo:
    """Metadata about a flowchart edge that was traversed to reach a node."""

    from_node: str
    """Source node."""

    to_node: str
    """Destination node."""

    condition_type: str
    """Class name of the condition that opened the edge."""

    priority: int
    """Edge priority value."""

    output_key: str
    """Output port on the source node."""

    input_key: str
    """Input port on the destination node."""


@dataclass
class TraceEntry:
    """One execution step captured during flowchart tracing."""

    step: int
    """Zero-based step index."""

    node_id: str
    """ID of the node that executed."""

    node_type: str
    """Class name of the node."""

    timestamp_start: datetime
    """Wall-clock time immediately before the node executed."""

    timestamp_end: datetime
    """Wall-clock time immediately after the node finished."""

    duration_ms: float
    """Execution duration in milliseconds."""

    inputs: dict[str, Any]
    """Input data the node received, keyed by port name."""

    outputs: list[Any]
    """Output data values produced by the node."""

    edge: Optional["EdgeInfo"]
    """Edge taken to reach this node, or ``None`` for the start node."""

    _checkpoint: dict = field(default_factory=dict, repr=False, compare=False)
    """Internal state snapshot used by restore_state()."""


@dataclass
class ExecutionTrace:
    """Full execution trace returned by :meth:`Flowchart.get_execution_trace`.

    Contains the ordered sequence of :class:`TraceEntry` objects from the most
    recent run, plus metadata.  Pass an entry (or the trace itself) to
    :meth:`Flowchart.restore_state` to resume from that point.
    """

    entries: list[TraceEntry]
    """Ordered trace entries, one per executed node step."""

    completed: bool
    """Whether execution reached a natural end state."""

    total_steps: int
    """Number of steps executed."""

    start_time: Optional[datetime]
    """Wall-clock time when the run started."""

    end_time: Optional[datetime]
    """Wall-clock time when the run finished."""


@dataclass
class ProgressEvent:
    """Event fired before each node is executed during a flowchart run.

    Passed to the ``on_progress`` callback supplied to
    :meth:`Flowchart.run_message_based` or :meth:`Flowchart.run`.
    """

    step: int
    """Zero-based step counter for the current execution."""

    node_id: str
    """ID of the node that is about to execute."""

    inputs: dict[str, Any]
    """Input data the node will receive, keyed by input port name."""

    edge: Optional[EdgeInfo]
    """Edge taken to reach this node, or ``None`` for the first (start) node."""

    previous_results: list[Message]
    """Output messages produced by the previous node execution."""


# ---------------------------------------------------------------------------
# Message serialisation helpers for checkpoints
# ---------------------------------------------------------------------------


def serialise_message(msg: Message) -> dict:
    """Convert a Message to a plain dict for checkpoint storage."""
    return {
        "data": msg.data,
        "source_node": msg.source_node,
        "target_node": msg.target_node,
        "input_key": msg.input_key,
        "message_id": msg.message_id,
        "metadata": dict(msg.metadata),
    }


def deserialise_message(data: dict) -> Message:
    """Reconstruct a Message from a serialised checkpoint dict."""
    return Message(
        data=data["data"],
        source_node=data.get("source_node"),
        target_node=data.get("target_node"),
        input_key=data.get("input_key", "default"),
        message_id=data.get("message_id", str(uuid.uuid4())),
        metadata=dict(data.get("metadata", {})),
    )
