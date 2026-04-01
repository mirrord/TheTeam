"""Flowchart serialization — to/from dict, YAML, and config registry."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Callable, Optional, Union

import yaml

from ..conditions import ConditionManager
from ..config_manager import ConfigManager
from ..validation import validate_flowchart

if TYPE_CHECKING:
    from .flowchart import Flowchart


class FlowchartSerializer:
    """Handles all serialization and deserialization for flowcharts."""

    @staticmethod
    def to_dict(flowchart: "Flowchart") -> dict:
        """Serialize a flowchart to a plain dictionary."""
        data: dict = {"nodes": {}, "edges": []}
        for node_name, node_data in flowchart.graph.nodes(data=True):
            data["nodes"][node_name] = node_data["nodeobj"].to_dict()
        data["start_node"] = flowchart.start_node
        for from_node, to_node, edge_data in flowchart.graph.edges(data=True):
            condition = edge_data["traversal_condition"]
            edge_dict = {
                "from": from_node,
                "to": to_node,
                "condition": condition.to_dict(),
                "priority": edge_data.get("priority", 1),
            }
            if "output_key" in edge_data:
                edge_dict["output_key"] = edge_data["output_key"]
            if "input_key" in edge_data:
                edge_dict["input_key"] = edge_data["input_key"]
            data["edges"].append(edge_dict)
        return data

    @staticmethod
    def to_yaml(flowchart: "Flowchart", yaml_path: str) -> None:
        """Serialize flowchart to YAML file."""
        data = FlowchartSerializer.to_dict(flowchart)
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
            watch: If ``True``, start a background watcher.
            poll_interval: Seconds between file-modification checks.
            on_reload: Optional callback invoked after each reload.

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
            watch: If ``True``, start a background watcher.
            poll_interval: Seconds between file-modification checks.
            on_reload: Optional callback invoked after each reload.

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

    @staticmethod
    def register(flowchart: "Flowchart", registered_name: Optional[str] = None) -> None:
        """Register this flowchart configuration."""
        flowchart.registered = True
        flowchart.registered_name = registered_name or flowchart.registered_name
        flowchart.config_manager.register_config(
            FlowchartSerializer.to_dict(flowchart),
            flowchart.registered_name,
            "flowcharts",
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
        from .flowchart import Flowchart

        if validate:
            nodes = data.get("nodes", {})
            edges = data.get("edges", [])
            start_node = data.get("start_node")
            validate_flowchart(nodes, edges, start_node, strict=False)

        flowchart = Flowchart(config_manager)
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

        flowchart._graph_manager.ensure_io_nodes()
        flowchart._graph_manager.initialize_message_routing(flowchart.message_router)

        return flowchart
