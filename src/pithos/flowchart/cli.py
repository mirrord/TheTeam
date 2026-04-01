"""Standalone CLI utilities for flowchart operations (preserved from original module)."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

from ..conditions import AlwaysCondition, CountCondition
from ..config_manager import ConfigManager
from .flowchart import Flowchart


def create_random_flowchart(config_manager: ConfigManager) -> None:
    """Create a random flowchart for testing purposes."""
    flowchart = Flowchart(config_manager)

    for i in range(5):
        flowchart.add_node(
            f"Node{i}", type="prompt", prompt=f"Prompt for Node {i}", extraction={}
        )

    start_node = random.choice(list(flowchart.graph.nodes))
    flowchart.set_start_node(start_node)

    for i in range(5):
        from_node = random.choice(list(flowchart.graph.nodes))
        to_node = random.choice(list(flowchart.graph.nodes))
        condition = (
            AlwaysCondition
            if random.choice([True, False])
            else CountCondition(random.randint(1, 3))
        )
        flowchart.add_edge(from_node, to_node, condition)

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
