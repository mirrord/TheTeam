"""FlowchartGraph — node/edge management and graph structure."""

from __future__ import annotations

from typing import Any, Optional

from networkx import MultiDiGraph

from ..conditions import Condition, AlwaysCondition
from ..flownode import create_node, InputNode, OutputNode
from ..message import MessageRouter


class FlowchartGraph:
    """Owns the directed multi-graph, nodes, edges, and start-node pointer."""

    def __init__(self) -> None:
        self.graph: MultiDiGraph = MultiDiGraph()
        self.start_node: Optional[str] = None

    # ------------------------------------------------------------------
    # Node / edge manipulation
    # ------------------------------------------------------------------

    def add_node(self, node_name: str, **kwargs: Any) -> None:
        """Add a node to the flowchart.

        Args:
            node_name: Unique identifier for the node.
            **kwargs: Node configuration (must include ``type``).

        Raises:
            ValueError: If the node type is invalid.
        """
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
            output_key: Which output from source node to route.
            input_key: Which input on target node to connect to.
        """
        self.graph.add_edge(
            from_node,
            to_node,
            traversal_condition=condition,
            priority=priority,
            output_key=output_key,
            input_key=input_key,
        )

    # ------------------------------------------------------------------
    # Automatic IO node injection
    # ------------------------------------------------------------------

    def ensure_io_nodes(self) -> None:
        """Ensure flowchart has at least one input and one output node.

        If no input node exists, automatically adds a ChatInputNode at the
        beginning.  If no output node exists, automatically adds a
        ChatOutputNode at the end.
        """
        has_input = False
        has_output = False
        end_nodes: list[str] = []

        for node_id in self.graph.nodes:
            node_obj = self.graph.nodes[node_id]["nodeobj"]
            if isinstance(node_obj, InputNode):
                has_input = True
            if isinstance(node_obj, OutputNode):
                has_output = True
            if self.graph.out_degree(node_id) == 0:
                end_nodes.append(node_id)

        if not has_input:
            input_node_id = "__auto_chat_input__"
            self.add_node(
                input_node_id,
                type="chatinput",
                prompt_message="Enter your input:",
                save_to="user_input",
            )
            if self.start_node and self.start_node != input_node_id:
                original_start = self.start_node
                self.start_node = input_node_id
                self.add_edge(
                    input_node_id, original_start, AlwaysCondition, priority=1
                )
            else:
                self.start_node = input_node_id

        if not has_output:
            output_node_id = "__auto_chat_output__"
            self.add_node(output_node_id, type="chatoutput", source="current_input")
            if end_nodes:
                for end_node in end_nodes:
                    self.add_edge(end_node, output_node_id, AlwaysCondition, priority=1)

    # ------------------------------------------------------------------
    # Message routing initialisation
    # ------------------------------------------------------------------

    def initialize_message_routing(self, router: MessageRouter) -> None:
        """Register every node's input requirements with the router."""
        for node_id in self.graph.nodes:
            node_obj = self.graph.nodes[node_id]["nodeobj"]
            router.register_node(
                node_id=node_id,
                required_inputs=node_obj.required_inputs,
                optional_inputs=[],
            )
