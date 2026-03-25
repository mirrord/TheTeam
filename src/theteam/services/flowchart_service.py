"""
Flowchart service - manages flowchart operations and execution.
"""

import logging
import uuid
from pathlib import Path
from typing import Optional
import yaml
import threading
from datetime import datetime

logger = logging.getLogger(__name__)


class FlowchartService:
    """Service for managing flowcharts and their execution."""

    def __init__(self, config_dir: Optional[Path] = None):
        """Initialize flowchart service."""
        if config_dir is None:
            config_dir = Path.cwd() / "configs" / "flowcharts"
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(parents=True, exist_ok=True)

        # Runtime flowcharts storage
        self.runtime_flowcharts: dict[str, dict] = {}

        # Active executions
        self.executions: dict[str, dict] = {}
        self.execution_lock = threading.Lock()

    def list_flowcharts(self) -> list[dict]:
        """List all available flowcharts."""
        flowcharts = []

        # Load from YAML files
        if self.config_dir.exists():
            for yaml_file in self.config_dir.glob("*.yaml"):
                try:
                    with open(yaml_file, "r") as f:
                        config = yaml.safe_load(f)

                    flowchart_id = yaml_file.stem
                    flowcharts.append(
                        {
                            "id": flowchart_id,
                            "name": config.get("name", flowchart_id),
                            "description": config.get("description", ""),
                            "node_count": len(config.get("nodes", {})),
                            "source": "file",
                            "path": str(yaml_file),
                        }
                    )
                except Exception as e:
                    logger.error(f"Error loading flowchart from {yaml_file}: {e}")

        # Add runtime flowcharts
        for flowchart_id, config in self.runtime_flowcharts.items():
            flowcharts.append(
                {
                    "id": flowchart_id,
                    "name": config.get("name", flowchart_id),
                    "description": config.get("description", ""),
                    "node_count": len(config.get("nodes", {})),
                    "source": "runtime",
                }
            )

        return flowcharts

    def _add_auto_io_nodes(self, config: dict) -> dict:
        """Add automatic I/O nodes to a flowchart config if needed.

        This ensures the frontend displays all nodes including auto-added ones.

        Args:
            config: Flowchart configuration dict

        Returns:
            Modified config with auto I/O nodes added if needed
        """
        # Make a copy to avoid modifying original
        import copy

        config = copy.deepcopy(config)

        # Get nodes and edges
        nodes = config.get("nodes", {})
        edges = config.get("edges", [])

        if not nodes:
            return config

        # Check for input and output nodes
        has_input = False
        has_output = False
        end_nodes = []  # Nodes with no outgoing edges

        # Track which nodes have outgoing edges
        nodes_with_outgoing = set()
        for edge in edges:
            nodes_with_outgoing.add(edge.get("from"))

        for node_id, node_config in nodes.items():
            node_type = node_config.get("type", "")

            # Check if node is an InputNode
            if node_type in ["chatinput", "fileinput"]:
                has_input = True

            # Check if node is an OutputNode
            if node_type in ["chatoutput", "fileoutput"]:
                has_output = True

            # Check if this is an end node (no outgoing edges)
            # Exclude input/output nodes since they can't be edge sources/targets
            if node_id not in nodes_with_outgoing:
                # Only add to end_nodes if it's not already an I/O node
                if node_type not in [
                    "chatinput",
                    "fileinput",
                    "chatoutput",
                    "fileoutput",
                ]:
                    end_nodes.append(node_id)

        # Add ChatInputNode if no input node exists
        if not has_input:
            input_node_id = "__auto_chat_input__"
            nodes[input_node_id] = {
                "type": "chatinput",
                "label": "Auto Input",
                "prompt_message": "Enter your input:",
                "save_to": "user_input",
                "position": {"x": 50, "y": 50},  # Default position in top-left
            }

            # Connect input node to the original start node
            original_start = config.get("start_node")
            if original_start and original_start != input_node_id:
                # Set the new input node as the start
                config["start_node"] = input_node_id
                # Add edge from input to original start
                edges.insert(
                    0,
                    {
                        "from": input_node_id,
                        "to": original_start,
                        "condition": {"type": "AlwaysCondition"},
                        "priority": 1,
                    },
                )
            else:
                config["start_node"] = input_node_id

        # Add ChatOutputNode if no output node exists
        if not has_output:
            output_node_id = "__auto_chat_output__"
            # Calculate position: place to the right of existing nodes
            max_x = max(
                (node.get("position", {}).get("x", 0) for node in nodes.values()),
                default=0,
            )
            nodes[output_node_id] = {
                "type": "chatoutput",
                "label": "Auto Output",
                "source": "current_input",
                "position": {"x": max_x + 200, "y": 50},  # Position to right
            }

            # Connect all end nodes to the output node
            for end_node in end_nodes:
                edges.append(
                    {
                        "from": end_node,
                        "to": output_node_id,
                        "condition": {"type": "AlwaysCondition"},
                        "priority": 1,
                    }
                )

        return config

    def get_flowchart(
        self, flowchart_id: str, include_auto_nodes: bool = True
    ) -> Optional[dict]:
        """Get a specific flowchart.

        Args:
            flowchart_id: ID of the flowchart to retrieve
            include_auto_nodes: If True, automatically add I/O nodes to the config

        Returns:
            Flowchart dict with id, config, source, and optionally path
        """
        # Check runtime flowcharts first
        if flowchart_id in self.runtime_flowcharts:
            config = self.runtime_flowcharts[flowchart_id]
            if include_auto_nodes:
                config = self._add_auto_io_nodes(config)
            return {
                "id": flowchart_id,
                "config": config,
                "source": "runtime",
            }

        # Check file-based flowcharts
        yaml_file = self.config_dir / f"{flowchart_id}.yaml"
        if yaml_file.exists():
            try:
                with open(yaml_file, "r") as f:
                    config = yaml.safe_load(f)
                if include_auto_nodes:
                    config = self._add_auto_io_nodes(config)
                return {
                    "id": flowchart_id,
                    "config": config,
                    "source": "file",
                    "path": str(yaml_file),
                }
            except Exception as e:
                logger.error(f"Error loading flowchart {flowchart_id}: {e}")
                return None

        return None

    def create_flowchart(self, config: dict) -> str:
        """Create a new flowchart."""
        flowchart_id = config.get("id")
        if not flowchart_id:
            name = config.get("name", "unnamed")
            flowchart_id = name.lower().replace(" ", "-")

        # Validate required fields
        if "nodes" not in config:
            raise ValueError("Flowchart configuration must include 'nodes'")

        save_to_file = config.pop("save_to_file", False)

        if save_to_file:
            yaml_file = self.config_dir / f"{flowchart_id}.yaml"
            try:
                with open(yaml_file, "w") as f:
                    yaml.dump(config, f, default_flow_style=False)
                logger.info(f"Created flowchart {flowchart_id} in {yaml_file}")
            except Exception as e:
                logger.error(f"Error saving flowchart {flowchart_id}: {e}")
                raise
        else:
            self.runtime_flowcharts[flowchart_id] = config
            logger.info(f"Created runtime flowchart {flowchart_id}")

        return flowchart_id

    def update_flowchart(self, flowchart_id: str, config: dict) -> bool:
        """Update an existing flowchart."""
        flowchart = self.get_flowchart(flowchart_id)
        if not flowchart:
            return False

        if flowchart["source"] == "runtime":
            self.runtime_flowcharts[flowchart_id] = config
            logger.info(f"Updated runtime flowchart {flowchart_id}")
            return True
        elif flowchart["source"] == "file":
            yaml_file = Path(flowchart["path"])
            try:
                with open(yaml_file, "w") as f:
                    yaml.dump(config, f, default_flow_style=False)
                logger.info(f"Updated flowchart {flowchart_id} in {yaml_file}")
                return True
            except Exception as e:
                logger.error(f"Error updating flowchart {flowchart_id}: {e}")
                raise

        return False

    def delete_flowchart(self, flowchart_id: str) -> bool:
        """Delete a flowchart."""
        flowchart = self.get_flowchart(flowchart_id)
        if not flowchart:
            return False

        if flowchart["source"] == "runtime":
            del self.runtime_flowcharts[flowchart_id]
            logger.info(f"Deleted runtime flowchart {flowchart_id}")
            return True
        elif flowchart["source"] == "file":
            yaml_file = Path(flowchart["path"])
            try:
                yaml_file.unlink()
                logger.info(f"Deleted flowchart {flowchart_id} file {yaml_file}")
                return True
            except Exception as e:
                logger.error(f"Error deleting flowchart {flowchart_id}: {e}")
                raise

        return False

    def import_from_yaml(self, yaml_content: str, name: Optional[str] = None) -> str:
        """Import a flowchart from YAML string."""
        try:
            config = yaml.safe_load(yaml_content)
            if name:
                config["name"] = name
            return self.create_flowchart(config)
        except Exception as e:
            logger.error(f"Error importing flowchart: {e}")
            raise ValueError(f"Invalid YAML format: {e}")

    def export_to_yaml(self, flowchart_id: str) -> Optional[str]:
        """Export a flowchart to YAML string.

        Note: Auto-added I/O nodes are not included in the export to keep YAML clean.
        """
        flowchart = self.get_flowchart(flowchart_id, include_auto_nodes=False)
        if not flowchart:
            return None

        try:
            return yaml.dump(flowchart["config"], default_flow_style=False)
        except Exception as e:
            logger.error(f"Error exporting flowchart {flowchart_id}: {e}")
            raise

    def validate_flowchart(self, flowchart_id: str) -> dict:
        """Validate a flowchart structure."""
        flowchart = self.get_flowchart(flowchart_id)
        if not flowchart:
            return {"valid": False, "errors": ["Flowchart not found"]}

        errors = []
        warnings = []
        config = flowchart["config"]

        # Check for required fields
        if "nodes" not in config:
            errors.append("Missing 'nodes' field")
        else:
            nodes = config["nodes"]
            if not nodes:
                warnings.append("Flowchart has no nodes")

            # Check node structure
            for node_id, node_config in nodes.items():
                if "type" not in node_config:
                    errors.append(f"Node '{node_id}' missing 'type' field")

                # Check for dangling references
                if "next" in node_config:
                    next_nodes = node_config["next"]
                    if isinstance(next_nodes, str):
                        next_nodes = [next_nodes]
                    for next_node in next_nodes:
                        if next_node not in nodes and next_node != "END":
                            errors.append(
                                f"Node '{node_id}' references non-existent node '{next_node}'"
                            )

        return {"valid": len(errors) == 0, "errors": errors, "warnings": warnings}

    def start_execution(
        self,
        flowchart_id: str,
        initial_context: dict,
        client_id: Optional[str] = None,
        socketio=None,
    ) -> str:
        """Start flowchart execution in a background thread."""
        flowchart = self.get_flowchart(flowchart_id)
        if not flowchart:
            raise ValueError(f"Flowchart {flowchart_id} not found")

        execution_id = str(uuid.uuid4())

        with self.execution_lock:
            self.executions[execution_id] = {
                "id": execution_id,
                "flowchart_id": flowchart_id,
                "status": "starting",
                "started_at": datetime.now().isoformat(),
                "client_id": client_id,
                "context": initial_context,
                "outputs": {},
                "current_node": None,
                "stop_requested": False,
            }

        # Start execution in background thread
        thread = threading.Thread(
            target=self._execute_flowchart,
            args=(execution_id, flowchart, initial_context, socketio, client_id),
        )
        thread.daemon = True
        thread.start()

        logger.info(f"Started execution {execution_id} for flowchart {flowchart_id}")
        return execution_id

    def _execute_flowchart(
        self,
        execution_id: str,
        flowchart: dict,
        context: dict,
        socketio,
        client_id: Optional[str],
    ):
        """Execute flowchart (runs in background thread)."""
        from pithos.flowchart import Flowchart
        from pithos.config_manager import ConfigManager
        from theteam.api.socketio_handlers import emit_to_room

        try:
            # Update status
            with self.execution_lock:
                if execution_id in self.executions:
                    self.executions[execution_id]["status"] = "running"

            if socketio:
                emit_to_room(
                    socketio,
                    f"execution_{execution_id}",
                    "execution_update",
                    {
                        "execution_id": execution_id,
                        "status": "running",
                        "timestamp": datetime.now().isoformat(),
                    },
                )

            # Create config manager and flowchart
            config_manager = ConfigManager()
            fc = Flowchart.from_dict(flowchart["config"], config_manager)

            # Hook into node execution for real-time updates
            original_execute = fc.execute_node

            def wrapped_execute(node_id, context):
                # Check for stop request
                if (
                    execution_id in self.executions
                    and self.executions[execution_id]["stop_requested"]
                ):
                    raise InterruptedError("Execution stopped by user")

                # Notify about current node
                with self.execution_lock:
                    if execution_id in self.executions:
                        self.executions[execution_id]["current_node"] = node_id

                if socketio:
                    emit_to_room(
                        socketio,
                        f"execution_{execution_id}",
                        "node_execution",
                        {
                            "execution_id": execution_id,
                            "node_id": node_id,
                            "status": "executing",
                            "timestamp": datetime.now().isoformat(),
                        },
                    )

                # Execute node
                result = original_execute(node_id, context)

                # Store output
                with self.execution_lock:
                    if execution_id in self.executions:
                        self.executions[execution_id]["outputs"][node_id] = result

                if socketio:
                    emit_to_room(
                        socketio,
                        f"execution_{execution_id}",
                        "node_complete",
                        {
                            "execution_id": execution_id,
                            "node_id": node_id,
                            "output": result,
                            "timestamp": datetime.now().isoformat(),
                        },
                    )

                return result

            fc.execute_node = wrapped_execute

            # Execute
            final_context = fc.run(context)

            # Mark as completed
            with self.execution_lock:
                if execution_id in self.executions:
                    self.executions[execution_id]["status"] = "completed"
                    self.executions[execution_id][
                        "completed_at"
                    ] = datetime.now().isoformat()
                    self.executions[execution_id]["final_context"] = final_context

            if socketio:
                emit_to_room(
                    socketio,
                    f"execution_{execution_id}",
                    "execution_complete",
                    {
                        "execution_id": execution_id,
                        "status": "completed",
                        "final_context": final_context,
                        "timestamp": datetime.now().isoformat(),
                    },
                )

        except InterruptedError as e:
            logger.info(f"Execution {execution_id} stopped: {e}")
            with self.execution_lock:
                if execution_id in self.executions:
                    self.executions[execution_id]["status"] = "stopped"

            if socketio:
                emit_to_room(
                    socketio,
                    f"execution_{execution_id}",
                    "execution_stopped",
                    {
                        "execution_id": execution_id,
                        "message": str(e),
                        "timestamp": datetime.now().isoformat(),
                    },
                )

        except Exception as e:
            logger.error(
                f"Error executing flowchart {execution_id}: {e}", exc_info=True
            )
            with self.execution_lock:
                if execution_id in self.executions:
                    self.executions[execution_id]["status"] = "failed"
                    self.executions[execution_id]["error"] = str(e)

            if socketio:
                emit_to_room(
                    socketio,
                    f"execution_{execution_id}",
                    "execution_error",
                    {
                        "execution_id": execution_id,
                        "error": str(e),
                        "timestamp": datetime.now().isoformat(),
                    },
                )

    def stop_execution(self, execution_id: str) -> bool:
        """Request to stop a running execution."""
        with self.execution_lock:
            if execution_id in self.executions:
                self.executions[execution_id]["stop_requested"] = True
                logger.info(f"Stop requested for execution {execution_id}")
                return True
        return False

    def get_execution_status(self, execution_id: str) -> Optional[dict]:
        """Get the status of an execution."""
        with self.execution_lock:
            return self.executions.get(execution_id)
