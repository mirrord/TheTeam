"""Flowchart module for managing directed graph-based workflows."""

from typing import Optional, Any, Callable, Union
from dataclasses import dataclass, field
from networkx import MultiDiGraph
from datetime import datetime
import copy
import logging
import threading
import uuid
import yaml
import random
import argparse

logger = logging.getLogger(__name__)
from .config_manager import ConfigManager
from pathlib import Path
from .conditions import Condition, AlwaysCondition, CountCondition, ConditionManager
from .flownode import create_node
from .message import Message, MessageRouter
from .validation import validate_flowchart, ValidationError  # noqa: F401


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


# ---------------------------------------------------------------------------
# Message serialisation helpers for checkpoints
# ---------------------------------------------------------------------------


def _serialise_message(msg: Message) -> dict:
    """Convert a Message to a plain dict for checkpoint storage."""
    return {
        "data": msg.data,
        "source_node": msg.source_node,
        "target_node": msg.target_node,
        "input_key": msg.input_key,
        "message_id": msg.message_id,
        "metadata": dict(msg.metadata),
    }


def _deserialise_message(data: dict) -> Message:
    """Reconstruct a Message from a serialised checkpoint dict."""
    return Message(
        data=data["data"],
        source_node=data.get("source_node"),
        target_node=data.get("target_node"),
        input_key=data.get("input_key", "default"),
        message_id=data.get("message_id", str(uuid.uuid4())),
        metadata=dict(data.get("metadata", {})),
    )


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


class Flowchart:
    """Directed graph-based workflow with conditional node transitions."""

    def __init__(
        self, config_manager: ConfigManager, registered_name: Optional[str] = None
    ):
        self.config_manager = config_manager
        self.condition_manager = ConditionManager(config_manager)
        self.graph = MultiDiGraph()
        self.start_node = None
        self.current_node = None
        self.finished = False
        self.registered = registered_name is not None
        self.registered_name = registered_name

        # Message-based execution (always enabled)
        self.message_router = MessageRouter()

        # Progress callback — fired before each node executes.
        self._on_progress: Optional[Callable[[ProgressEvent], None]] = None

        # Per-run tracking state (reset in reset())
        self._step_counter: int = 0
        self._prev_output_messages: list[Message] = []
        # Maps target_node_id -> EdgeInfo for the edge that last routed to it.
        self._node_route_info: dict[str, EdgeInfo] = {}

        # Execution tracing (disabled by default; enable with enable_trace())
        self._trace_enabled: bool = False
        self._trace_entries: list[TraceEntry] = []
        self._trace_start_time: Optional[datetime] = None
        self._trace_end_time: Optional[datetime] = None

        # Restored-state flag: set by restore_state(), consumed by run/_run_message_based
        self._has_restored_state: bool = False

        # Hot-reload state (disabled by default; enabled via start_watching / from_yaml watch=True)
        self._watch_path: Optional[Path] = None
        self._watcher_thread: Optional[threading.Thread] = None
        self._watcher_stop: threading.Event = threading.Event()
        self._reload_lock: threading.Lock = threading.Lock()
        self._on_reload: Optional[Callable[["Flowchart"], None]] = None

    def reset(self) -> None:
        """Reset flowchart execution state."""
        self.current_node = None
        self.finished = False
        self._step_counter = 0
        self._prev_output_messages = []
        self._node_route_info = {}
        self._has_restored_state = False  # manual reset cancels any pending restore
        self.message_router.reset()

    # ------------------------------------------------------------------
    # Hot-reload / file watching
    # ------------------------------------------------------------------

    def start_watching(
        self,
        path: Union[str, Path],
        poll_interval: float = 1.0,
        on_reload: Optional[Callable[["Flowchart"], None]] = None,
    ) -> None:
        """Watch a YAML file and hot-reload the flowchart whenever it changes.

        Starts a daemon background thread that polls the file's modification
        time every *poll_interval* seconds.  When a change is detected the
        flowchart is re-parsed and its internal state is replaced in-place so
        that any existing reference to this object reflects the new definition.

        Ongoing executions are *not* interrupted; the new definition takes
        effect on the next :meth:`run` or :meth:`run_message_based` call.

        Args:
            path: Path to the YAML file to watch.
            poll_interval: Seconds between modification-time checks (default 1).
            on_reload: Optional callback invoked after each successful reload.
                       Receives this :class:`Flowchart` instance as its sole
                       argument.

        Raises:
            FileNotFoundError: If *path* does not exist at call time.
        """
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"Cannot watch non-existent file: {path}")
        self.stop_watching()  # Stop any existing watcher before starting a new one
        self._watch_path = path
        self._on_reload = on_reload
        self._watcher_stop.clear()
        self._watcher_thread = threading.Thread(
            target=self._watch_loop,
            args=(path, poll_interval),
            daemon=True,
            name=f"flowchart-watcher-{path.name}",
        )
        self._watcher_thread.start()

    def stop_watching(self) -> None:
        """Stop the file-change watcher thread, if running."""
        self._watcher_stop.set()
        if self._watcher_thread is not None and self._watcher_thread.is_alive():
            self._watcher_thread.join(timeout=5.0)
        self._watcher_thread = None
        self._watch_path = None
        # Reset the event so start_watching can be called again later.
        self._watcher_stop.clear()

    @property
    def is_watching(self) -> bool:
        """Return ``True`` if a file-change watcher thread is currently active."""
        return self._watcher_thread is not None and self._watcher_thread.is_alive()

    def _watch_loop(self, path: Path, poll_interval: float) -> None:
        """Background thread: poll *path* mtime and reload on change."""
        try:
            last_mtime = path.stat().st_mtime
        except OSError:
            logger.warning("Flowchart watcher: cannot stat %s, stopping.", path)
            return

        while not self._watcher_stop.wait(poll_interval):
            try:
                current_mtime = path.stat().st_mtime
            except OSError:
                continue
            if current_mtime != last_mtime:
                last_mtime = current_mtime
                try:
                    self._reload_from_path(path)
                    logger.debug("Flowchart hot-reloaded from %s", path)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Flowchart hot-reload failed for %s: %s", path, exc)

    def _reload_from_path(self, path: Path) -> None:
        """Reload the flowchart in-place from *path*.

        Parses the YAML, rebuilds the graph, and replaces this instance's
        internal state atomically under ``_reload_lock``.  The execution state
        is reset so the next run uses the refreshed definition.
        """
        with open(path, "r") as f:
            data = yaml.safe_load(f)

        fresh = Flowchart.from_dict(data, self.config_manager)

        with self._reload_lock:
            self.graph = fresh.graph
            self.start_node = fresh.start_node
            self.condition_manager = fresh.condition_manager
            self.message_router = fresh.message_router
            self.reset()

        if self._on_reload is not None:
            try:
                self._on_reload(self)
            except Exception as exc:  # noqa: BLE001
                logger.warning("on_reload callback raised: %s", exc)

    def run(
        self,
        agents: dict[str, Any],
        initial_input: Optional[str] = None,
        max_steps: int = 100,
        history_window: int = 0,
        on_progress: Optional[Callable[[ProgressEvent], None]] = None,
    ) -> str:
        """Execute the flowchart with a set of named agents.

        Args:
            agents: Dictionary mapping agent names to OllamaAgent instances.
            initial_input: Initial input to the flowchart.
            max_steps: Maximum number of steps to prevent infinite loops.
            history_window: Rolling window size for ``message_history``.
                            ``0`` means unlimited (default).
            on_progress: Optional progress callback. See
                         :meth:`run_message_based` for details.

        Returns:
            Final response from the flowchart execution.

        Raises:
            ValueError: If required agents are missing.
        """
        # Verify all required agents are provided
        from .flownode import AgentPromptNode, GetHistoryNode, SetHistoryNode

        required_agents = set()
        for node_id in self.graph.nodes:
            node_obj = self.graph.nodes[node_id]["nodeobj"]
            if isinstance(node_obj, (AgentPromptNode, GetHistoryNode, SetHistoryNode)):
                required_agents.add(node_obj.agent)

        missing_agents = required_agents - set(agents.keys())
        if missing_agents:
            raise ValueError(
                f"Missing required agents for flowchart: {', '.join(missing_agents)}"
            )

        skip_reset = self._has_restored_state
        if not skip_reset:
            # Pre-execution validation (skipped when resuming a restored state)
            self.validate(strict=False)
            self.reset()
            self._initialize_message_routing()

        # Inject agents into message router's shared context
        self.message_router.shared_context["agents"] = agents

        # Run message-based execution
        result = self.run_message_based(
            initial_data=None if skip_reset else (initial_input or ""),
            max_steps=max_steps,
            history_window=history_window,
            on_progress=on_progress,
        )

        # Extract final response from messages
        response = ""
        if result.get("messages"):
            last_message = result["messages"][-1]
            response = str(last_message.data)

        return response

    def _initialize_message_routing(self) -> None:
        """Initialize message router with node requirements."""
        for node_id in self.graph.nodes:
            node_obj = self.graph.nodes[node_id]["nodeobj"]
            self.message_router.register_node(
                node_id=node_id,
                required_inputs=node_obj.required_inputs,
                optional_inputs=[],
            )

    def add_node(self, node_name: str, **kwargs) -> None:
        """Add a node to the flowchart."""
        node_obj = create_node(kwargs.get("type", "prompt"), kwargs)
        if not node_obj:
            raise ValueError(f"Invalid node type: {node_name}")
        self.graph.add_node(node_name, nodeobj=node_obj)
        if not self.start_node:
            self.start_node = node_name

    def set_start_node(self, node_name: str) -> None:
        """Set the starting node for flowchart execution."""
        self.start_node = node_name

    def add_edge(
        self,
        from_node: str,
        to_node: str,
        condition: Condition,
        priority: int = 1,
        output_key: str = "default",
        input_key: str = "default",
    ) -> None:
        """Add a conditional edge between two nodes.

        Args:
            from_node: Source node ID.
            to_node: Target node ID.
            condition: Condition for edge traversal.
            priority: Edge priority (lower = higher priority).
            output_key: Which output from source node to route (for message-based).
            input_key: Which input on target node to connect to (for message-based).
        """
        self.graph.add_edge(
            from_node,
            to_node,
            traversal_condition=condition,
            priority=priority,
            output_key=output_key,
            input_key=input_key,
        )

    def _ensure_io_nodes(self) -> None:
        """Ensure flowchart has at least one input and one output node.

        If no input node exists, automatically adds a ChatInputNode at the beginning.
        If no output node exists, automatically adds a ChatOutputNode at the end.
        """
        from .flownode import InputNode, OutputNode

        # Check for input and output nodes
        has_input = False
        has_output = False
        end_nodes = []  # Nodes with no outgoing edges

        for node_id in self.graph.nodes:
            node_obj = self.graph.nodes[node_id]["nodeobj"]

            # Check if node is an InputNode
            if isinstance(node_obj, InputNode):
                has_input = True

            # Check if node is an OutputNode
            if isinstance(node_obj, OutputNode):
                has_output = True

            # Check if this is an end node (no outgoing edges)
            if self.graph.out_degree(node_id) == 0:
                end_nodes.append(node_id)

        # Add ChatInputNode if no input node exists
        if not has_input:
            input_node_id = "__auto_chat_input__"
            self.add_node(
                input_node_id,
                type="chatinput",
                prompt_message="Enter your input:",
                save_to="user_input",
            )

            # Connect input node to the original start node
            if self.start_node and self.start_node != input_node_id:
                original_start = self.start_node
                # Set the new input node as the start
                self.start_node = input_node_id
                # Add edge from input to original start
                self.add_edge(
                    input_node_id, original_start, AlwaysCondition, priority=1
                )
            else:
                self.start_node = input_node_id

        # Add ChatOutputNode if no output node exists
        if not has_output:
            output_node_id = "__auto_chat_output__"
            self.add_node(output_node_id, type="chatoutput", source="current_input")

            # Connect all end nodes to the output node
            if end_nodes:
                for end_node in end_nodes:
                    self.add_edge(end_node, output_node_id, AlwaysCondition, priority=1)
            # If no end nodes exist (circular graph), just leave output node disconnected
            # It will be accessible but won't be reached unless explicitly connected

    def step_message_based(
        self, initial_message: Optional[Message] = None
    ) -> list[Message]:
        """Execute one step of message-based flowchart execution.

        Args:
            initial_message: Optional initial message to send to start node.

        Returns:
            List of messages produced in this step.
        """
        if self.finished:
            raise RuntimeError("Flowchart is finished. Please reset it.")

        # If initial message provided, send it to start node
        if initial_message and self.current_node is None:
            initial_message.target_node = self.start_node
            self.message_router.send_message(initial_message)

        # Get ready nodes (nodes with all inputs satisfied)
        ready_nodes = self.message_router.get_ready_nodes()

        if not ready_nodes:
            self.finished = True
            return []

        # Execute the first ready node (could be extended to parallel execution)
        node_id = ready_nodes[0]
        self.current_node = node_id

        return self._execute_node_message_based(node_id)

    def _execute_node_message_based(self, node_id: str) -> list[Message]:
        """Execute a node in message-based mode and route its outputs.

        Args:
            node_id: ID of the node to execute.

        Returns:
            List of output messages produced.
        """
        # Get node and its input state
        node_obj = self.graph.nodes[node_id]["nodeobj"]
        input_state = self.message_router.get_node_state(node_id)

        if not input_state:
            return []

        # Snapshot inputs now — used by both the progress callback and the
        # trace entry.  Must be captured before execute_with_messages (which
        # may alter shared state) and before clear_node_inputs().
        node_inputs = input_state.get_all_input_data()

        # Fire progress callback before execution.
        if self._on_progress is not None:
            event = ProgressEvent(
                step=self._step_counter,
                node_id=node_id,
                inputs=node_inputs,
                edge=self._node_route_info.get(node_id),
                previous_results=list(self._prev_output_messages),
            )
            self._on_progress(event)

        # Execute the node with access to shared context
        ts_start = datetime.now()
        output_messages = node_obj.execute_with_messages(
            input_state, self.message_router
        )
        ts_end = datetime.now()

        # Set source node on all output messages
        for msg in output_messages:
            msg.source_node = node_id

        # Route messages to downstream nodes based on edges
        self._route_output_messages(node_id, output_messages)

        # Advance step counter and record outputs for the next callback.
        current_step = self._step_counter
        self._step_counter += 1
        self._prev_output_messages = list(output_messages)

        # Clear this node's inputs after execution
        self.message_router.clear_node_inputs(node_id)

        # Capture trace entry if tracing is enabled
        if self._trace_enabled:
            checkpoint = self._capture_checkpoint()
            duration_ms = (ts_end - ts_start).total_seconds() * 1000
            entry = TraceEntry(
                step=current_step,
                node_id=node_id,
                node_type=type(node_obj).__name__,
                timestamp_start=ts_start,
                timestamp_end=ts_end,
                duration_ms=duration_ms,
                inputs=node_inputs,
                outputs=[msg.data for msg in output_messages],
                edge=self._node_route_info.get(node_id),
                _checkpoint=checkpoint,
            )
            self._trace_entries.append(entry)

        return output_messages

    def _route_output_messages(self, source_node: str, messages: list[Message]) -> None:
        """Route output messages to downstream nodes based on edges and conditions.

        Args:
            source_node: The node that produced the messages.
            messages: Output messages to route.
        """
        # Get outgoing edges from this node
        neighbors = list(self.graph.neighbors(source_node))

        # Build a simple state for condition evaluation (for backward compatibility)
        state = {}
        for msg in messages:
            if msg.input_key == "default":
                state["current_input"] = msg.data
            state[msg.input_key] = msg.data

        # Evaluate edges and route messages
        for neighbor in neighbors:
            for edge in self.graph[source_node][neighbor].values():
                condition = edge["traversal_condition"]

                # Check if condition is satisfied
                if condition.is_open(state):
                    # Record the edge taken to reach this neighbor.
                    self._node_route_info[neighbor] = EdgeInfo(
                        from_node=source_node,
                        to_node=neighbor,
                        condition_type=condition.__class__.__name__,
                        priority=edge.get("priority", 1),
                        output_key=edge.get("output_key", "default"),
                        input_key=edge.get("input_key", "default"),
                    )

                    # Route all messages to this neighbor
                    for msg in messages:
                        routed_msg = Message(
                            data=msg.data,
                            source_node=source_node,
                            target_node=neighbor,
                            input_key=msg.input_key,
                            metadata=msg.metadata.copy(),
                        )
                        self.message_router.send_message(routed_msg)

                    # Traverse the edge (for count conditions, etc.)
                    condition.traverse(state)

                    # Only take the highest priority edge
                    break

    def run_message_based(
        self,
        initial_data: Any = None,
        max_steps: int = 100,
        history_window: int = 0,
        on_progress: Optional[Callable[[ProgressEvent], None]] = None,
    ) -> dict[str, Any]:
        """Run the entire flowchart using message-based execution.

        Args:
            initial_data: Initial data to send to start node.
            max_steps: Maximum number of steps to prevent infinite loops.
            history_window: Rolling window size for ``message_history``.
                            Only the most recent *history_window* messages are
                            kept.  ``0`` means unlimited (default).
            on_progress: Optional callback invoked **before** each node runs.
                         Receives a :class:`ProgressEvent` with the current node
                         ID, its inputs, the edge taken to reach it, and the
                         previous node\'s output messages.

        Returns:
            Dict with execution results and message history.
        """
        self._on_progress = on_progress

        if self._has_restored_state:
            # Resuming from a restored checkpoint — skip reset so the router
            # state installed by _apply_checkpoint() is preserved.
            self._has_restored_state = False
            self.message_router._max_history = history_window
            initial_message = None
        else:
            # Pre-execution validation (guard against calling this entry point
            # directly without going through run(), which already validates).
            self.validate(strict=False)
            self.reset()
            self.message_router._max_history = history_window
            initial_message = None
            if initial_data is not None:
                initial_message = Message(
                    data=initial_data, target_node=self.start_node, input_key="default"
                )

        # Start/reset trace for this run
        if self._trace_enabled:
            self._trace_entries = []
            self._trace_start_time = datetime.now()
            self._trace_end_time = None

        all_messages = []
        step_count = 0

        # Execute until finished or max steps reached
        while not self.finished and step_count < max_steps:
            messages = self.step_message_based(initial_message)
            all_messages.extend(messages)
            initial_message = None  # Only use on first step
            step_count += 1

        if self._trace_enabled:
            self._trace_end_time = datetime.now()

        return {
            "completed": self.finished,
            "steps": step_count,
            "messages": all_messages,
            "message_history": self.message_router.message_history,
        }

    def validate(self, strict: bool = False) -> bool:
        """Validate the flowchart configuration.

        Runs both the YAML-level structural checks and additional graph-level
        checks: all non-start nodes must have required inputs wired by incoming
        edges, and all edge conditions must be callable.

        Args:
            strict: If True, treat warnings as errors.

        Returns:
            True if validation passes.

        Raises:
            ValidationError: If validation fails.
        """
        # Build nodes dict from graph
        nodes = {}
        for node_name, node_data in self.graph.nodes(data=True):
            nodes[node_name] = node_data["nodeobj"].to_dict()

        # Build edges list from graph
        edges = []
        for from_node, to_node, edge_data in self.graph.edges(data=True):
            cond = edge_data.get("traversal_condition")
            edge_dict = {
                "from": from_node,
                "to": to_node,
                "condition": {
                    "type": cond.__class__.__name__ if cond is not None else "unknown"
                },
                "priority": edge_data.get("priority", 1),
                "output_key": edge_data.get("output_key", "default"),
                "input_key": edge_data.get("input_key", "default"),
            }
            edges.append(edge_dict)

        # YAML-level structural validation
        validate_flowchart(nodes, edges, self.start_node, strict=strict)

        # Graph-level checks (work directly on node objects and edge data)
        graph_errors: list[str] = []
        graph_warnings: list[str] = []
        self._validate_graph_input_coverage(graph_errors, graph_warnings)
        self._validate_graph_conditions(graph_errors, graph_warnings)

        if graph_errors:
            raise ValidationError(
                "Pre-execution validation failed:\n"
                + "\n".join(f"  - {e}" for e in graph_errors)
            )
        if graph_warnings and strict:
            raise ValidationError(
                "Pre-execution validation warnings:\n"
                + "\n".join(f"  - {w}" for w in graph_warnings)
            )

        return True

    def _validate_graph_input_coverage(
        self, errors: list[str], warnings: list[str]
    ) -> None:
        """Warn when a non-start node has required inputs not wired by any edge."""
        for node_id in self.graph.nodes:
            if node_id == self.start_node:
                continue
            node_obj = self.graph.nodes[node_id]["nodeobj"]
            required = set(node_obj.required_inputs)
            if not required:
                continue
            provided: set[str] = set()
            for pred in self.graph.predecessors(node_id):
                for edge_data in self.graph[pred][node_id].values():
                    provided.add(edge_data.get("input_key", "default"))
            missing = required - provided
            if missing:
                warnings.append(
                    f"Node '{node_id}' required inputs {sorted(missing)} "
                    "are not covered by any incoming edge"
                )

    def _validate_graph_conditions(
        self, errors: list[str], warnings: list[str]
    ) -> None:
        """Verify every edge condition is a valid, callable Condition object."""
        for from_node, to_node, edge_data in self.graph.edges(data=True):
            condition = edge_data.get("traversal_condition")
            if condition is None:
                errors.append(
                    f"Edge '{from_node}' -> '{to_node}' has no traversal_condition"
                )
                continue
            if not (hasattr(condition, "is_open") and callable(condition.is_open)):
                errors.append(
                    f"Edge '{from_node}' -> '{to_node}' condition "
                    f"'{type(condition).__name__}' has no callable is_open method"
                )

    # ------------------------------------------------------------------
    # Execution tracing
    # ------------------------------------------------------------------

    def enable_trace(self) -> None:
        """Enable execution tracing for subsequent runs.

        Once enabled, every call to :meth:`run` or :meth:`run_message_based`
        will record a :class:`TraceEntry` for each executed node.  Retrieve
        the results with :meth:`get_execution_trace`.  Tracing adds modest
        overhead (timestamp capture + checkpoint copy per step).
        """
        self._trace_enabled = True

    def get_execution_trace(self) -> Optional[ExecutionTrace]:
        """Return the execution trace from the most recent run.

        Returns ``None`` if tracing is not enabled or no run has been
        performed since the last :meth:`enable_trace` call.

        The returned :class:`ExecutionTrace` is a full snapshot: it contains
        ordered :class:`TraceEntry` objects (one per executed node step)
        with timing, inputs, outputs, edge info, and an embedded checkpoint
        suitable for :meth:`restore_state`.
        """
        if not self._trace_enabled:
            return None
        return ExecutionTrace(
            entries=list(self._trace_entries),
            completed=self.finished,
            total_steps=self._step_counter,
            start_time=self._trace_start_time,
            end_time=self._trace_end_time,
        )

    # ------------------------------------------------------------------
    # State restore
    # ------------------------------------------------------------------

    def restore_state(self, state: Union[ExecutionTrace, TraceEntry]) -> None:
        """Restore the flowchart to a previously traced execution state.

        After calling ``restore_state()``, invoke :meth:`run` or
        :meth:`run_message_based` and execution will continue from the point
        captured in *state* rather than starting from the beginning.

        Args:
            state: An :class:`ExecutionTrace` (restores to the final captured
                   state) **or** a specific :class:`TraceEntry` (restores to
                   the state immediately after that step completed).

        Raises:
            TypeError:  If *state* is not an :class:`ExecutionTrace` or
                        :class:`TraceEntry`.
            ValueError: If the trace is empty or has no checkpoint data.
        """
        if isinstance(state, ExecutionTrace):
            if not state.entries:
                raise ValueError("Cannot restore from an empty ExecutionTrace")
            entry = state.entries[-1]
        elif isinstance(state, TraceEntry):
            entry = state
        else:
            raise TypeError(
                f"restore_state() expects ExecutionTrace or TraceEntry, "
                f"got {type(state).__name__}"
            )

        if not entry._checkpoint:
            raise ValueError(
                "TraceEntry has no checkpoint data. "
                "Ensure tracing was enabled before the run."
            )

        self._apply_checkpoint(entry._checkpoint)
        self._has_restored_state = True

    def _capture_checkpoint(self) -> dict:
        """Snapshot the current execution state for later restore."""
        node_states_snap: dict[str, dict] = {}
        for node_id, state in self.message_router.node_states.items():
            if state.received_inputs:
                node_states_snap[node_id] = {
                    key: _serialise_message(msg)
                    for key, msg in state.received_inputs.items()
                }

        return {
            "step_counter": self._step_counter,
            "finished": self.finished,
            "node_route_info": copy.deepcopy(self._node_route_info),
            "prev_output_messages": [
                _serialise_message(m) for m in self._prev_output_messages
            ],
            "node_states": node_states_snap,
            "pending_messages": [
                _serialise_message(m) for m in self.message_router.pending_messages
            ],
            "message_history": [
                _serialise_message(m) for m in self.message_router.message_history
            ],
        }

    def _apply_checkpoint(self, checkpoint: dict) -> None:
        """Restore execution state from a captured checkpoint dict."""
        self.finished = checkpoint["finished"]
        self._step_counter = checkpoint["step_counter"]
        self._node_route_info = copy.deepcopy(checkpoint["node_route_info"])
        self._prev_output_messages = [
            _deserialise_message(m) for m in checkpoint["prev_output_messages"]
        ]

        # Restore router node input states
        for state in self.message_router.node_states.values():
            state.reset()
        for node_id, inputs in checkpoint["node_states"].items():
            if node_id in self.message_router.node_states:
                for input_key, msg_data in inputs.items():
                    msg = _deserialise_message(msg_data)
                    self.message_router.node_states[node_id].receive_message(msg)

        # Restore pending messages and history
        self.message_router.pending_messages.clear()
        for msg_data in checkpoint["pending_messages"]:
            self.message_router.pending_messages.append(_deserialise_message(msg_data))

        self.message_router.message_history.clear()
        for msg_data in checkpoint["message_history"]:
            self.message_router.message_history.append(_deserialise_message(msg_data))

    def to_dict(self) -> dict:
        data = {"nodes": {}, "edges": []}
        for node_name, node_data in self.graph.nodes(data=True):
            data["nodes"][node_name] = node_data["nodeobj"].to_dict()
        data["start_node"] = self.start_node
        for from_node, to_node, edge_data in self.graph.edges(data=True):
            condition = edge_data["traversal_condition"]
            edge_dict = {
                "from": from_node,
                "to": to_node,
                "condition": condition.to_dict(),
                "priority": edge_data.get("priority", 1),
            }
            # Include message routing keys if present
            if "output_key" in edge_data:
                edge_dict["output_key"] = edge_data["output_key"]
            if "input_key" in edge_data:
                edge_dict["input_key"] = edge_data["input_key"]
            data["edges"].append(edge_dict)
        return data

    def to_yaml(self, yaml_path: str) -> None:
        """Serialize flowchart to YAML file."""
        data = self.to_dict()
        with open(yaml_path, "w") as file:
            yaml.safe_dump(data, file)

    @classmethod
    def from_yaml(
        cls,
        yaml_path: Union[str, Path],
        config_manager: ConfigManager,
        watch: bool = False,
        poll_interval: float = 1.0,
        on_reload: Optional[Callable[["Flowchart"], None]] = None,
    ) -> "Flowchart":
        """Load flowchart from YAML file.

        Args:
            yaml_path: Path to the YAML file.
            config_manager: Configuration manager instance.
            watch: If ``True``, start a background watcher that hot-reloads
                   the flowchart whenever the file changes on disk.
            poll_interval: Seconds between file-modification checks when
                           *watch* is ``True`` (default 1 second).
            on_reload: Optional callback invoked after each successful
                       hot-reload.  Only used when *watch* is ``True``.

        Returns:
            :class:`Flowchart` instance.
        """
        with open(yaml_path, "r") as file:
            data = yaml.safe_load(file)
        flowchart = cls.from_dict(data, config_manager)
        if watch:
            flowchart.start_watching(
                yaml_path, poll_interval=poll_interval, on_reload=on_reload
            )
        return flowchart

    @classmethod
    def from_registered(
        cls,
        config_name: str,
        config_manager: ConfigManager,
        watch: bool = False,
        poll_interval: float = 1.0,
        on_reload: Optional[Callable[["Flowchart"], None]] = None,
    ) -> "Flowchart":
        """Load flowchart from registered configuration.

        Args:
            config_name: Registered name of the flowchart.
            config_manager: Configuration manager instance.
            watch: If ``True``, start a background watcher that hot-reloads
                   the flowchart whenever the source YAML file changes on disk.
            poll_interval: Seconds between file-modification checks when
                           *watch* is ``True`` (default 1 second).
            on_reload: Optional callback invoked after each successful
                       hot-reload.  Only used when *watch* is ``True``.

        Returns:
            :class:`Flowchart` instance.

        Raises:
            ValueError: If no flowchart with *config_name* is registered.
        """
        fname = config_manager.get_config_file(config_name, "flowcharts")
        if not fname:
            raise ValueError(f"Flowchart {config_name} not found.")
        return cls.from_yaml(
            fname,
            config_manager,
            watch=watch,
            poll_interval=poll_interval,
            on_reload=on_reload,
        )

    def register(self, registered_name: Optional[str] = None) -> None:
        """Register this flowchart configuration."""
        self.registered = True
        self.registered_name = registered_name or self.registered_name
        self.config_manager.register_config(
            self.to_dict(), self.registered_name, "flowcharts"
        )

    @classmethod
    def from_dict(
        cls, data: dict, config_manager: ConfigManager, validate: bool = True
    ) -> "Flowchart":
        """Deserialize flowchart from dictionary.

        Args:
            data: Flowchart configuration dictionary.
            config_manager: Configuration manager instance.
            validate: If True, validate configuration before creating flowchart.

        Returns:
            Flowchart instance.

        Raises:
            ValidationError: If configuration is invalid and validate=True.
        """
        # Validate configuration before building flowchart
        if validate:
            nodes = data.get("nodes", {})
            edges = data.get("edges", [])
            start_node = data.get("start_node")
            validate_flowchart(nodes, edges, start_node, strict=False)

        flowchart = cls(config_manager)
        condition_manager = flowchart.condition_manager
        for node_name, node_dict in data.get("nodes", {}).items():
            flowchart.add_node(node_name, **node_dict)

        if "start_node" in data:
            flowchart.set_start_node(data["start_node"])

        for edge in data["edges"]:
            condition_dict = edge.get("condition", {"type": "AlwaysCondition"})
            cond_type = condition_dict.get("type", "AlwaysCondition")
            condition = condition_manager.get_registered_condition(
                cond_type, **condition_dict
            )
            priority = 9 if cond_type == "AlwaysCondition" else edge.get("priority", 1)
            output_key = edge.get("output_key", "default")
            input_key = edge.get("input_key", "default")
            flowchart.add_edge(
                edge["from"],
                edge["to"],
                condition,
                priority,
                output_key=output_key,
                input_key=input_key,
            )

        # Ensure flowchart has required input/output nodes
        flowchart._ensure_io_nodes()

        # Initialize message routing
        flowchart._initialize_message_routing()

        return flowchart


def create_random_flowchart(config_manager: ConfigManager) -> None:
    """Create a random flowchart for testing purposes."""
    flowchart = Flowchart(config_manager)

    # Create random nodes
    for i in range(5):
        flowchart.add_node(
            f"Node{i}", type="prompt", prompt=f"Prompt for Node {i}", extraction={}
        )

    # Set a random start node
    start_node = random.choice(list(flowchart.graph.nodes))
    flowchart.set_start_node(start_node)

    # Create random edges with conditions
    for i in range(5):
        from_node = random.choice(list(flowchart.graph.nodes))
        to_node = random.choice(list(flowchart.graph.nodes))
        condition = (
            AlwaysCondition
            if random.choice([True, False])
            else CountCondition(random.randint(1, 3))
        )
        flowchart.add_edge(from_node, to_node, condition)

    # Save the flowchart to a YAML file
    flowchart.to_yaml("random_flowchart.yaml")


def display_flowchart(yaml_path: str, config_manager: ConfigManager) -> None:
    """Display flowchart information (visualization removed)."""
    if Path(yaml_path).exists():
        flowchart = Flowchart.from_yaml(yaml_path, config_manager)
    else:
        flowchart = Flowchart.from_registered(yaml_path, config_manager)
    print(f"Flowchart loaded: {flowchart.start_node}")
    print(f"Nodes: {list(flowchart.graph.nodes)}")
    print(f"Edges: {len(list(flowchart.graph.edges))}")


def main() -> None:
    """CLI entry point for flowchart operations."""
    parser = argparse.ArgumentParser(description="Flowchart CLI")
    parser.add_argument(
        "action",
        choices=["create", "show", "register", "list"],
        help="Action to perform",
    )
    parser.add_argument(
        "-y",
        "--yaml_path",
        type=str,
        help="Path to the YAML file",
        default="random_flowchart.yaml",
    )
    args = parser.parse_args()
    config_manager = ConfigManager()

    if args.action == "create":
        create_random_flowchart(config_manager)
    elif args.action == "show":
        display_flowchart(args.yaml_path, config_manager)
    elif args.action == "register":
        flowchart = Flowchart.from_yaml(args.yaml_path, config_manager)
        rname = Path(args.yaml_path).stem
        flowchart.register(rname)
        print(f"Flowchart registered with name {rname}.")
    elif args.action == "list":
        print("Registered flowcharts:")
        for name in config_manager.get_registered_flowchart_names():
            print(name)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
