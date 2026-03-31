"""FlowchartExecutor — execution engine for message-based flowchart runs."""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Callable, Optional

from ..message import Message, MessageRouter
from ..metrics import MetricsCollector
from .models import EdgeInfo, ProgressEvent
from .tracer import ExecutionTracer

logger = logging.getLogger(__name__)


class FlowchartExecutor:
    """Runs a flowchart graph step-by-step using message-based data flow."""

    def __init__(
        self,
        graph: Any,  # networkx.MultiDiGraph (stored on FlowchartGraph)
        message_router: MessageRouter,
        tracer: ExecutionTracer,
        metrics: Optional[MetricsCollector] = None,
        metrics_name: str = "unnamed",
    ) -> None:
        self._graph = graph
        self._router = message_router
        self._tracer = tracer
        self.metrics = metrics
        self._metrics_name = metrics_name

        # Per-run tracking state
        self._step_counter: int = 0
        self._prev_output_messages: list[Message] = []
        self._node_route_info: dict[str, EdgeInfo] = {}
        self._on_progress: Optional[Callable[[ProgressEvent], None]] = None

        self.current_node: Optional[str] = None
        self.finished: bool = False

        # Restored-state flag: set externally, consumed by run_message_based
        self._has_restored_state: bool = False

    # ------------------------------------------------------------------
    # State management
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Reset execution state for a fresh run."""
        self.current_node = None
        self.finished = False
        self._step_counter = 0
        self._prev_output_messages = []
        self._node_route_info = {}
        self._has_restored_state = False
        self._router.reset()

    # ------------------------------------------------------------------
    # High-level run
    # ------------------------------------------------------------------

    def run(
        self,
        agents: dict[str, Any],
        start_node: Optional[str],
        initial_input: Optional[str] = None,
        max_steps: int = 100,
        history_window: int = 0,
        on_progress: Optional[Callable[[ProgressEvent], None]] = None,
        validate_fn: Optional[Callable[[], None]] = None,
        init_routing_fn: Optional[Callable[[], None]] = None,
    ) -> str:
        """Execute the flowchart with a set of named agents.

        Args:
            agents: Dictionary mapping agent names to agent instances.
            start_node: ID of the start node.
            initial_input: Initial input to the flowchart.
            max_steps: Maximum number of steps to prevent infinite loops.
            history_window: Rolling window size for ``message_history``.
            on_progress: Optional progress callback.
            validate_fn: Callable to run pre-execution validation.
            init_routing_fn: Callable to initialise message routing.

        Returns:
            Final response from the flowchart execution.
        """
        from ..flownode import AgentPromptNode, GetHistoryNode, SetHistoryNode

        required_agents: set[str] = set()
        for node_id in self._graph.nodes:
            node_obj = self._graph.nodes[node_id]["nodeobj"]
            if isinstance(node_obj, (AgentPromptNode, GetHistoryNode, SetHistoryNode)):
                required_agents.add(node_obj.agent)

        missing_agents = required_agents - set(agents.keys())
        if missing_agents:
            raise ValueError(
                f"Missing required agents for flowchart: {', '.join(missing_agents)}"
            )

        skip_reset = self._has_restored_state
        if not skip_reset:
            if validate_fn:
                validate_fn()
            self.reset()
            if init_routing_fn:
                init_routing_fn()

        self._router.shared_context["agents"] = agents

        result = self.run_message_based(
            start_node=start_node,
            initial_data=None if skip_reset else (initial_input or ""),
            max_steps=max_steps,
            history_window=history_window,
            on_progress=on_progress,
            validate_fn=None if skip_reset else validate_fn,
            init_routing_fn=None if skip_reset else init_routing_fn,
        )

        response = ""
        if result.get("messages"):
            last_message = result["messages"][-1]
            response = str(last_message.data)

        return response

    # ------------------------------------------------------------------
    # Message-based execution
    # ------------------------------------------------------------------

    def run_message_based(
        self,
        start_node: Optional[str],
        initial_data: Any = None,
        max_steps: int = 100,
        history_window: int = 0,
        on_progress: Optional[Callable[[ProgressEvent], None]] = None,
        validate_fn: Optional[Callable[[], None]] = None,
        init_routing_fn: Optional[Callable[[], None]] = None,
    ) -> dict[str, Any]:
        """Run the entire flowchart using message-based execution.

        Args:
            start_node: ID of the start node.
            initial_data: Initial data to send to start node.
            max_steps: Maximum number of steps.
            history_window: Rolling window size for ``message_history``.
            on_progress: Optional callback invoked before each node runs.
            validate_fn: Callable to run pre-execution validation.
            init_routing_fn: Callable to initialise message routing.

        Returns:
            Dict with execution results and message history.
        """
        self._on_progress = on_progress

        if self._has_restored_state:
            self._has_restored_state = False
            self._router._max_history = history_window
            initial_message = None
        else:
            if validate_fn:
                validate_fn()
            self.reset()
            if init_routing_fn:
                init_routing_fn()
            self._router._max_history = history_window
            initial_message = None
            if initial_data is not None:
                initial_message = Message(
                    data=initial_data,
                    target_node=start_node,
                    input_key="default",
                )

        self._tracer.begin_run()

        all_messages: list[Message] = []
        step_count = 0

        while not self.finished and step_count < max_steps:
            messages = self.step_message_based(start_node, initial_message)
            all_messages.extend(messages)
            initial_message = None
            step_count += 1

        self._tracer.end_run()

        return {
            "completed": self.finished,
            "steps": step_count,
            "messages": all_messages,
            "message_history": self._router.message_history,
        }

    def step_message_based(
        self,
        start_node: Optional[str],
        initial_message: Optional[Message] = None,
    ) -> list[Message]:
        """Execute one step of message-based flowchart execution.

        Args:
            start_node: ID of the start node.
            initial_message: Optional initial message to send to start node.

        Returns:
            List of messages produced in this step.
        """
        if self.finished:
            raise RuntimeError("Flowchart is finished. Please reset it.")

        if initial_message and self.current_node is None:
            initial_message.target_node = start_node
            self._router.send_message(initial_message)

        ready_nodes = self._router.get_ready_nodes()

        if not ready_nodes:
            self.finished = True
            return []

        node_id = ready_nodes[0]
        self.current_node = node_id

        return self._execute_node(node_id)

    # ------------------------------------------------------------------
    # Internal node execution
    # ------------------------------------------------------------------

    def _execute_node(self, node_id: str) -> list[Message]:
        """Execute a node and route its outputs."""
        node_obj = self._graph.nodes[node_id]["nodeobj"]
        input_state = self._router.get_node_state(node_id)

        if not input_state:
            return []

        node_inputs = input_state.get_all_input_data()

        if self._on_progress is not None:
            event = ProgressEvent(
                step=self._step_counter,
                node_id=node_id,
                inputs=node_inputs,
                edge=self._node_route_info.get(node_id),
                previous_results=list(self._prev_output_messages),
            )
            self._on_progress(event)

        ts_start = datetime.now()
        output_messages = node_obj.execute_with_messages(input_state, self._router)
        ts_end = datetime.now()

        for msg in output_messages:
            msg.source_node = node_id

        self._route_output_messages(node_id, output_messages)

        current_step = self._step_counter
        self._step_counter += 1
        self._prev_output_messages = list(output_messages)

        self._router.clear_node_inputs(node_id)

        duration_ms = (ts_end - ts_start).total_seconds() * 1000

        if self._tracer.enabled:
            checkpoint = ExecutionTracer.capture_checkpoint(
                step_counter=self._step_counter,
                finished=self.finished,
                node_route_info=self._node_route_info,
                prev_output_messages=self._prev_output_messages,
                message_router=self._router,
            )
            self._tracer.record_step(
                step=current_step,
                node_id=node_id,
                node_type=type(node_obj).__name__,
                ts_start=ts_start,
                ts_end=ts_end,
                duration_ms=duration_ms,
                inputs=node_inputs,
                outputs=[msg.data for msg in output_messages],
                edge=self._node_route_info.get(node_id),
                checkpoint=checkpoint,
            )

        if self.metrics is not None:
            try:
                edge_info = self._node_route_info.get(node_id)
                self.metrics.record_flowchart_step(
                    flowchart_name=self._metrics_name,
                    node_id=node_id,
                    node_type=type(node_obj).__name__,
                    duration_ms=duration_ms,
                    from_node=edge_info.from_node if edge_info is not None else None,
                )
            except Exception:
                pass

        return output_messages

    def _route_output_messages(self, source_node: str, messages: list[Message]) -> None:
        """Route output messages to downstream nodes based on edges and conditions."""
        neighbors = list(self._graph.neighbors(source_node))

        state: dict[str, Any] = {}
        for msg in messages:
            if msg.input_key == "default":
                state["current_input"] = msg.data
            state[msg.input_key] = msg.data

        for neighbor in neighbors:
            for edge in self._graph[source_node][neighbor].values():
                condition = edge["traversal_condition"]

                if condition.is_open(state):
                    self._node_route_info[neighbor] = EdgeInfo(
                        from_node=source_node,
                        to_node=neighbor,
                        condition_type=condition.__class__.__name__,
                        priority=edge.get("priority", 1),
                        output_key=edge.get("output_key", "default"),
                        input_key=edge.get("input_key", "default"),
                    )

                    for msg in messages:
                        routed_msg = Message(
                            data=msg.data,
                            source_node=source_node,
                            target_node=neighbor,
                            input_key=msg.input_key,
                            metadata=msg.metadata.copy(),
                        )
                        self._router.send_message(routed_msg)

                    condition.traverse(state)
                    break
