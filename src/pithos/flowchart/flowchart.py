"""Flowchart — thin facade composing graph, executor, tracer, watcher, and serializer.

This module preserves the exact public API of the original monolithic
``Flowchart`` class so that **all existing imports continue to work**.
"""

from __future__ import annotations

from typing import Any, Callable, Optional, Union
from pathlib import Path

from ..conditions import Condition, ConditionManager
from ..config_manager import ConfigManager
from ..message import Message, MessageRouter
from ..metrics import MetricsCollector
from ..validation import validate_flowchart, ValidationError  # noqa: F401

from .models import EdgeInfo, ExecutionTrace, ProgressEvent, TraceEntry
from .graph import FlowchartGraph
from .executor import FlowchartExecutor
from .tracer import ExecutionTracer
from .watcher import FlowchartWatcher
from .serialization import FlowchartSerializer


class Flowchart:
    """Directed graph-based workflow with conditional node transitions.

    Composes :class:`FlowchartGraph`, :class:`FlowchartExecutor`,
    :class:`ExecutionTracer`, :class:`FlowchartWatcher`, and
    :class:`FlowchartSerializer` behind the original public interface.
    """

    def __init__(
        self, config_manager: ConfigManager, registered_name: Optional[str] = None
    ):
        self.config_manager = config_manager
        self.condition_manager = ConditionManager(config_manager)
        self.registered = registered_name is not None
        self.registered_name = registered_name

        # Composed components
        self._graph_manager = FlowchartGraph()
        self.message_router = MessageRouter()
        self._tracer = ExecutionTracer()
        self._watcher = FlowchartWatcher()
        self._executor = FlowchartExecutor(
            graph=self._graph_manager.graph,
            message_router=self.message_router,
            tracer=self._tracer,
            metrics_name=registered_name or "unnamed",
        )

    # ==================================================================
    # Properties that proxy to composed objects
    # ==================================================================

    @property
    def graph(self):
        """The underlying ``networkx.MultiDiGraph``."""
        return self._graph_manager.graph

    @graph.setter
    def graph(self, value):
        self._graph_manager.graph = value
        # Keep the executor's graph reference in sync
        self._executor._graph = value

    @property
    def start_node(self):
        return self._graph_manager.start_node

    @start_node.setter
    def start_node(self, value):
        self._graph_manager.start_node = value

    @property
    def current_node(self):
        return self._executor.current_node

    @current_node.setter
    def current_node(self, value):
        self._executor.current_node = value

    @property
    def finished(self):
        return self._executor.finished

    @finished.setter
    def finished(self, value):
        self._executor.finished = value

    @property
    def metrics(self) -> Optional[MetricsCollector]:
        return self._executor.metrics

    @metrics.setter
    def metrics(self, value):
        self._executor.metrics = value

    # Proxy internal state accessed by tests
    @property
    def _step_counter(self) -> int:
        return self._executor._step_counter

    @_step_counter.setter
    def _step_counter(self, value: int):
        self._executor._step_counter = value

    @property
    def _prev_output_messages(self) -> list:
        return self._executor._prev_output_messages

    @_prev_output_messages.setter
    def _prev_output_messages(self, value):
        self._executor._prev_output_messages = value

    @property
    def _node_route_info(self) -> dict:
        return self._executor._node_route_info

    @_node_route_info.setter
    def _node_route_info(self, value):
        self._executor._node_route_info = value

    @property
    def _has_restored_state(self) -> bool:
        return self._executor._has_restored_state

    @_has_restored_state.setter
    def _has_restored_state(self, value: bool):
        self._executor._has_restored_state = value

    @property
    def _on_progress(self):
        return self._executor._on_progress

    @_on_progress.setter
    def _on_progress(self, value):
        self._executor._on_progress = value

    @property
    def _trace_enabled(self) -> bool:
        return self._tracer.enabled

    @property
    def _trace_entries(self) -> list:
        return self._tracer._entries

    @property
    def _watcher_thread(self):
        return self._watcher._watcher_thread

    @property
    def _watch_path(self):
        return self._watcher._watch_path

    @property
    def _reload_lock(self):
        return self._watcher._reload_lock

    @property
    def _on_reload(self):
        return self._watcher._on_reload

    @_on_reload.setter
    def _on_reload(self, value):
        self._watcher._on_reload = value

    def _reload_from_path(self, path):
        """Proxy to watcher's reload (used by tests)."""
        self._watcher._reload_from_path(self, path)

    @property
    def _metrics_name(self) -> str:
        return self._executor._metrics_name

    @_metrics_name.setter
    def _metrics_name(self, value: str):
        self._executor._metrics_name = value

    # ==================================================================
    # Metrics
    # ==================================================================

    def attach_metrics(
        self, collector: MetricsCollector, name: Optional[str] = None
    ) -> None:
        """Attach a :class:`~pithos.metrics.MetricsCollector` to this flowchart."""
        self._executor.metrics = collector
        if name is not None:
            self._executor._metrics_name = name

    # ==================================================================
    # Reset
    # ==================================================================

    def reset(self) -> None:
        """Reset flowchart execution state."""
        self._executor.reset()

    # ==================================================================
    # Hot-reload / file watching  (delegates to FlowchartWatcher)
    # ==================================================================

    def start_watching(
        self,
        path: Union[str, Path],
        poll_interval: float = 1.0,
        on_reload: Optional[Callable[["Flowchart"], None]] = None,
    ) -> None:
        """Watch a YAML file and hot-reload the flowchart whenever it changes."""
        self._watcher.start_watching(
            self, path, poll_interval=poll_interval, on_reload=on_reload
        )

    def stop_watching(self) -> None:
        """Stop the file-change watcher thread, if running."""
        self._watcher.stop_watching()

    @property
    def is_watching(self) -> bool:
        """Return ``True`` if a file-change watcher thread is currently active."""
        return self._watcher.is_watching

    # ==================================================================
    # Graph building  (delegates to FlowchartGraph)
    # ==================================================================

    def add_node(self, node_name: str, **kwargs) -> None:
        """Add a node to the flowchart."""
        self._graph_manager.add_node(node_name, **kwargs)

    def set_start_node(self, node_name: str) -> None:
        """Set the starting node for flowchart execution."""
        self._graph_manager.set_start_node(node_name)

    def add_edge(
        self,
        from_node: str,
        to_node: str,
        condition: Condition,
        priority: int = 1,
        output_key: str = "default",
        input_key: str = "default",
    ) -> None:
        """Add a conditional edge between two nodes."""
        self._graph_manager.add_edge(
            from_node, to_node, condition, priority, output_key, input_key
        )

    def _ensure_io_nodes(self) -> None:
        """Ensure flowchart has at least one input and one output node."""
        self._graph_manager.ensure_io_nodes()

    def _initialize_message_routing(self) -> None:
        """Initialize message router with node requirements."""
        self._graph_manager.initialize_message_routing(self.message_router)

    # ==================================================================
    # Execution  (delegates to FlowchartExecutor)
    # ==================================================================

    def run(
        self,
        agents: dict[str, Any],
        initial_input: Optional[str] = None,
        max_steps: int = 100,
        history_window: int = 0,
        on_progress: Optional[Callable[[ProgressEvent], None]] = None,
    ) -> str:
        """Execute the flowchart with a set of named agents."""
        return self._executor.run(
            agents=agents,
            start_node=self.start_node,
            initial_input=initial_input,
            max_steps=max_steps,
            history_window=history_window,
            on_progress=on_progress,
            validate_fn=lambda: self.validate(strict=False),
            init_routing_fn=self._initialize_message_routing,
        )

    def step_message_based(
        self, initial_message: Optional[Message] = None
    ) -> list[Message]:
        """Execute one step of message-based flowchart execution."""
        return self._executor.step_message_based(self.start_node, initial_message)

    def run_message_based(
        self,
        initial_data: Any = None,
        max_steps: int = 100,
        history_window: int = 0,
        on_progress: Optional[Callable[[ProgressEvent], None]] = None,
    ) -> dict[str, Any]:
        """Run the entire flowchart using message-based execution."""
        return self._executor.run_message_based(
            start_node=self.start_node,
            initial_data=initial_data,
            max_steps=max_steps,
            history_window=history_window,
            on_progress=on_progress,
            validate_fn=lambda: self.validate(strict=False),
            init_routing_fn=self._initialize_message_routing,
        )

    # ==================================================================
    # Validation
    # ==================================================================

    def validate(self, strict: bool = False) -> bool:
        """Validate the flowchart configuration."""
        nodes = {}
        for node_name, node_data in self.graph.nodes(data=True):
            nodes[node_name] = node_data["nodeobj"].to_dict()

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

        validate_flowchart(nodes, edges, self.start_node, strict=strict)

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

    # ==================================================================
    # Tracing  (delegates to ExecutionTracer)
    # ==================================================================

    def enable_trace(self) -> None:
        """Enable execution tracing for subsequent runs."""
        self._tracer.enable()

    def get_execution_trace(self) -> Optional[ExecutionTrace]:
        """Return the execution trace from the most recent run."""
        return self._tracer.get_execution_trace(
            finished=self.finished,
            total_steps=self._executor._step_counter,
        )

    # ==================================================================
    # State restore
    # ==================================================================

    def restore_state(self, state: Union[ExecutionTrace, TraceEntry]) -> None:
        """Restore the flowchart to a previously traced execution state."""
        entry = self._tracer.validate_restore_source(state)
        restored = ExecutionTracer.apply_checkpoint(
            entry._checkpoint, self.message_router
        )
        self._executor.finished = restored["finished"]
        self._executor._step_counter = restored["step_counter"]
        self._executor._node_route_info = restored["node_route_info"]
        self._executor._prev_output_messages = restored["prev_output_messages"]
        self._executor._has_restored_state = True

    # ==================================================================
    # Serialization  (delegates to FlowchartSerializer)
    # ==================================================================

    def to_dict(self) -> dict:
        return FlowchartSerializer.to_dict(self)

    def to_yaml(self, yaml_path: str) -> None:
        """Serialize flowchart to YAML file."""
        FlowchartSerializer.to_yaml(self, yaml_path)

    @classmethod
    def from_yaml(
        cls,
        yaml_path: Union[str, Path],
        config_manager: ConfigManager,
        watch: bool = False,
        poll_interval: float = 1.0,
        on_reload: Optional[Callable[["Flowchart"], None]] = None,
    ) -> "Flowchart":
        """Load flowchart from YAML file."""
        return FlowchartSerializer.from_yaml(
            yaml_path,
            config_manager,
            watch=watch,
            poll_interval=poll_interval,
            on_reload=on_reload,
        )

    @classmethod
    def from_registered(
        cls,
        config_name: str,
        config_manager: ConfigManager,
        watch: bool = False,
        poll_interval: float = 1.0,
        on_reload: Optional[Callable[["Flowchart"], None]] = None,
    ) -> "Flowchart":
        """Load flowchart from registered configuration."""
        return FlowchartSerializer.from_registered(
            config_name,
            config_manager,
            watch=watch,
            poll_interval=poll_interval,
            on_reload=on_reload,
        )

    def register(self, registered_name: Optional[str] = None) -> None:
        """Register this flowchart configuration."""
        FlowchartSerializer.register(self, registered_name)

    @classmethod
    def from_dict(
        cls, data: dict, config_manager: ConfigManager, validate: bool = True
    ) -> "Flowchart":
        """Deserialize flowchart from dictionary."""
        return FlowchartSerializer.from_dict(data, config_manager, validate=validate)
