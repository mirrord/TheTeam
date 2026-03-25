"""Validation module for flowchart node configurations."""

from typing import Any, Optional
from pathlib import Path
import re


class ValidationError(Exception):
    """Exception raised when flowchart validation fails."""

    pass


class FlowchartValidator:
    """Validates flowchart configurations before execution."""

    # Required parameters for each node type
    NODE_REQUIREMENTS = {
        "prompt": ["prompt"],
        "promptnode": ["prompt"],
        "custom": ["custom_code"],
        "customnode": ["custom_code"],
        "toolcall": ["command"],
        "toolcallnode": ["command"],
        "textparse": [],
        "textparsenode": [],
        "agentprompt": ["agent", "prompt"],
        "agentpromptnode": ["agent", "prompt"],
        "gethistory": ["agent"],
        "gethistorynode": ["agent"],
        "sethistory": ["agent"],
        "sethistorynode": ["agent"],
        "chatinput": [],
        "chatinputnode": [],
        "chatoutput": [],
        "chatoutputnode": [],
        "fileinput": ["file_path"],
        "fileinputnode": ["file_path"],
        "fileoutput": ["file_path", "source"],
        "fileoutputnode": ["file_path", "source"],
    }

    def __init__(self, strict: bool = True):
        """Initialize validator.

        Args:
            strict: If True, raise errors for warnings. If False, only report them.
        """
        self.strict = strict
        self.errors: list[str] = []
        self.warnings: list[str] = []

    def validate_flowchart(
        self,
        nodes: dict[str, dict[str, Any]],
        edges: list[dict[str, Any]],
        start_node: Optional[str] = None,
    ) -> None:
        """Validate complete flowchart configuration.

        Args:
            nodes: Dictionary of node configurations.
            edges: List of edge configurations.
            start_node: Optional start node ID.

        Raises:
            ValidationError: If validation fails.
            ValueError: If nodes or edges are None.
        """
        if nodes is None or edges is None:
            raise ValueError("nodes and edges cannot be None")

        self.errors = []
        self.warnings = []

        # Basic structural validation
        self._validate_structure(nodes, edges, start_node)

        # Validate all nodes
        for node_id, node_config in nodes.items():
            self._validate_node(node_id, node_config)

        # Validate all edges
        self._validate_edges(nodes, edges)

        # Check for unreachable nodes
        self._check_reachability(nodes, edges, start_node)

        # Check for cycles (warning only)
        self._check_cycles(nodes, edges, start_node)

        # Report results
        if self.errors:
            error_msg = "Flowchart validation failed:\n" + "\n".join(
                f"  - {error}" for error in self.errors
            )
            raise ValidationError(error_msg)

        if self.warnings and self.strict:
            warning_msg = "Flowchart validation warnings:\n" + "\n".join(
                f"  - {warning}" for warning in self.warnings
            )
            raise ValidationError(warning_msg)

    def _validate_structure(
        self,
        nodes: dict[str, dict[str, Any]],
        edges: list[dict[str, Any]],
        start_node: Optional[str],
    ) -> None:
        """Validate basic flowchart structure.

        Args:
            nodes: Dictionary of node configurations.
            edges: List of edge configurations.
            start_node: Optional start node ID.
        """
        if not nodes:
            self.errors.append("Flowchart must have at least one node")
            return

        if start_node and start_node not in nodes:
            self.errors.append(
                f"Start node '{start_node}' does not exist in flowchart nodes"
            )

    def _validate_node(self, node_id: str, node_config: dict[str, Any]) -> None:
        """Validate a single node configuration.

        Args:
            node_id: Node identifier.
            node_config: Node configuration dictionary.
        """
        # Check for node type
        if "type" not in node_config:
            self.errors.append(f"Node '{node_id}' is missing 'type' field")
            return

        node_type = node_config["type"].replace("_", "").lower()

        # Check if node type is valid
        if node_type not in self.NODE_REQUIREMENTS:
            self.errors.append(
                f"Node '{node_id}' has unknown type '{node_config['type']}'"
            )
            return

        # Check required parameters for this node type
        required_params = self.NODE_REQUIREMENTS[node_type]
        for param in required_params:
            if param not in node_config:
                self.errors.append(
                    f"Node '{node_id}' (type: {node_config['type']}) is missing required parameter '{param}'"
                )

        # Type-specific validation
        self._validate_node_specific(node_id, node_type, node_config)

        # Validate extraction patterns if present
        if "extraction" in node_config:
            self._validate_extraction(node_id, node_config["extraction"])

        # Validate inputs/outputs if present
        if "inputs" in node_config:
            if not isinstance(node_config["inputs"], list):
                self.errors.append(
                    f"Node '{node_id}' 'inputs' must be a list, got {type(node_config['inputs']).__name__}"
                )

        if "outputs" in node_config:
            if not isinstance(node_config["outputs"], list):
                self.errors.append(
                    f"Node '{node_id}' 'outputs' must be a list, got {type(node_config['outputs']).__name__}"
                )

    def _validate_node_specific(
        self, node_id: str, node_type: str, node_config: dict[str, Any]
    ) -> None:
        """Perform type-specific node validation.

        Args:
            node_id: Node identifier.
            node_type: Normalized node type.
            node_config: Node configuration.
        """
        # Validate CustomNode for security concerns
        if node_type in ["custom", "customnode"]:
            if "custom_code" in node_config:
                code = node_config["custom_code"]
                # Check for potentially dangerous operations
                dangerous_patterns = [
                    (r"\bexec\s*\(", "nested exec() calls"),
                    (r"\beval\s*\(", "eval() usage"),
                    (r"\b__import__\s*\(", "__import__() usage"),
                    (r"\bos\.system\s*\(", "os.system() calls"),
                    (r"\bsubprocess\.", "subprocess module usage"),
                ]
                for pattern, description in dangerous_patterns:
                    if re.search(pattern, code):
                        self.warnings.append(
                            f"Node '{node_id}' custom_code contains {description} - security risk"
                        )

        # Validate SetHistoryNode mode
        if node_type in ["sethistory", "sethistorynode"]:
            if "mode" in node_config:
                mode = node_config["mode"]
                if mode not in ["replace", "append"]:
                    self.errors.append(
                        f"Node '{node_id}' has invalid mode '{mode}'. Must be 'replace' or 'append'"
                    )

        # Validate ToolCallNode error handling
        if node_type in ["toolcall", "toolcallnode"]:
            if "error_handling" in node_config:
                error_handling = node_config["error_handling"]
                if error_handling not in ["continue", "stop", "retry"]:
                    self.warnings.append(
                        f"Node '{node_id}' has non-standard error_handling '{error_handling}'. "
                        "Expected: 'continue', 'stop', or 'retry'"
                    )

        # Validate FileInputNode/FileOutputNode paths
        if node_type in ["fileinput", "fileinputnode"]:
            if "file_path" in node_config:
                file_path = node_config["file_path"]
                # Check if it's a template (contains {})
                if not re.search(r"\{.*?\}", file_path):
                    # Not a template, check if file exists
                    path = Path(file_path)
                    if not path.exists():
                        self.warnings.append(
                            f"Node '{node_id}' file_path '{file_path}' does not exist"
                        )

        if node_type in ["fileoutput", "fileoutputnode"]:
            if "file_path" in node_config:
                file_path = node_config["file_path"]
                # Check if directory exists (if not a template)
                if not re.search(r"\{.*?\}", file_path):
                    path = Path(file_path)
                    if not path.parent.exists():
                        self.warnings.append(
                            f"Node '{node_id}' output directory '{path.parent}' does not exist"
                        )

    def _validate_extraction(self, node_id: str, extraction: Any) -> None:
        """Validate extraction patterns.

        Args:
            node_id: Node identifier.
            extraction: Extraction configuration.

        Raises:
            None, but adds errors to self.errors list.
        """
        if extraction and not isinstance(extraction, dict):
            self.errors.append(
                f"Node '{node_id}' 'extraction' must be a dict, got {type(extraction).__name__}"
            )
            return

        if not extraction:
            return

        for var_name, pattern in extraction.items():
            if not isinstance(pattern, str):
                self.errors.append(
                    f"Node '{node_id}' extraction pattern for '{var_name}' must be a string"
                )
                continue

            # Try to compile regex pattern
            try:
                re.compile(pattern)
            except re.error as e:
                self.errors.append(
                    f"Node '{node_id}' extraction pattern for '{var_name}' is invalid regex: {e}"
                )

    def _validate_edges(
        self, nodes: dict[str, dict[str, Any]], edges: list[dict[str, Any]]
    ) -> None:
        """Validate edge configurations.

        Args:
            nodes: Dictionary of node configurations.
            edges: List of edge configurations.
        """
        for i, edge in enumerate(edges):
            edge_id = f"Edge {i}"

            # Check required fields
            if "from" not in edge:
                self.errors.append(f"{edge_id} is missing 'from' field")
                continue
            if "to" not in edge:
                self.errors.append(f"{edge_id} is missing 'to' field")
                continue

            from_node = edge["from"]
            to_node = edge["to"]

            # Check nodes exist
            if from_node not in nodes:
                self.errors.append(
                    f"{edge_id} references non-existent 'from' node '{from_node}'"
                )
            if to_node not in nodes:
                self.errors.append(
                    f"{edge_id} references non-existent 'to' node '{to_node}'"
                )

            # Check self-loops
            if from_node == to_node:
                self.warnings.append(
                    f"{edge_id} creates a self-loop on node '{from_node}'"
                )

            # Validate condition if present
            if "condition" in edge:
                self._validate_condition(edge_id, edge["condition"])

            # Validate output_key and input_key match between nodes
            if from_node in nodes and to_node in nodes:
                output_key = edge.get("output_key", "default")
                input_key = edge.get("input_key", "default")

                from_outputs = nodes[from_node].get("outputs", ["default"])
                to_inputs = nodes[to_node].get("inputs", ["default"])

                if output_key not in from_outputs:
                    self.warnings.append(
                        f"{edge_id} uses output_key '{output_key}' not in node '{from_node}' outputs {from_outputs}"
                    )

                if input_key not in to_inputs:
                    self.warnings.append(
                        f"{edge_id} uses input_key '{input_key}' not in node '{to_node}' inputs {to_inputs}"
                    )

    def _validate_condition(self, edge_id: str, condition: Any) -> None:
        """Validate edge condition.

        Args:
            edge_id: Edge identifier for error reporting.
            condition: Condition configuration.
        """
        if not isinstance(condition, dict):
            self.errors.append(
                f"{edge_id} condition must be a dict, got {type(condition).__name__}"
            )
            return

        if "type" not in condition:
            self.errors.append(f"{edge_id} condition is missing 'type' field")
            return

        cond_type = condition["type"]

        # Validate known condition types
        known_conditions = [
            "AlwaysCondition",
            "CountCondition",
            "StateCondition",
            "RegexCondition",
            "LambdaCondition",
        ]

        if cond_type not in known_conditions:
            self.warnings.append(f"{edge_id} uses unknown condition type '{cond_type}'")

        # Type-specific validation
        if cond_type == "CountCondition":
            if "count" not in condition:
                self.errors.append(
                    f"{edge_id} CountCondition is missing 'count' parameter"
                )

    def _check_reachability(
        self,
        nodes: dict[str, dict[str, Any]],
        edges: list[dict[str, Any]],
        start_node: Optional[str],
    ) -> None:
        """Check for unreachable nodes.

        Args:
            nodes: Dictionary of node configurations.
            edges: List of edge configurations.
            start_node: Start node ID.
        """
        if not start_node or start_node not in nodes:
            # Can't check reachability without a valid start node
            return

        # Build adjacency list
        graph: dict[str, set[str]] = {node_id: set() for node_id in nodes}
        for edge in edges:
            if "from" in edge and "to" in edge:
                from_node = edge["from"]
                to_node = edge["to"]
                if from_node in graph:
                    graph[from_node].add(to_node)

        # BFS from start node
        reachable = {start_node}
        queue = [start_node]

        while queue:
            current = queue.pop(0)
            for neighbor in graph.get(current, set()):
                if neighbor not in reachable:
                    reachable.add(neighbor)
                    queue.append(neighbor)

        # Check for unreachable nodes
        unreachable = set(nodes.keys()) - reachable
        if unreachable:
            self.warnings.append(
                f"Unreachable nodes from start '{start_node}': {', '.join(sorted(unreachable))}"
            )

    def _check_cycles(
        self,
        nodes: dict[str, dict[str, Any]],
        edges: list[dict[str, Any]],
        start_node: Optional[str],
    ) -> None:
        """Check for cycles in the flowchart (informational only).

        Args:
            nodes: Dictionary of node configurations.
            edges: List of edge configurations.
            start_node: Start node ID.
        """
        # Build adjacency list
        graph: dict[str, set[str]] = {node_id: set() for node_id in nodes}
        for edge in edges:
            if "from" in edge and "to" in edge:
                from_node = edge["from"]
                to_node = edge["to"]
                if from_node in graph:
                    graph[from_node].add(to_node)

        # Detect cycles using DFS
        visited: set[str] = set()
        rec_stack: set[str] = set()
        cycles_found = False

        def has_cycle(node: str) -> bool:
            nonlocal cycles_found
            visited.add(node)
            rec_stack.add(node)

            for neighbor in graph.get(node, set()):
                if neighbor not in visited:
                    if has_cycle(neighbor):
                        return True
                elif neighbor in rec_stack:
                    cycles_found = True
                    return True

            rec_stack.remove(node)
            return False

        for node_id in nodes:
            if node_id not in visited:
                has_cycle(node_id)

        if cycles_found:
            self.warnings.append(
                "Flowchart contains cycles. This may cause infinite loops without proper exit conditions."
            )


def validate_flowchart(
    nodes: dict[str, dict[str, Any]],
    edges: list[dict[str, Any]],
    start_node: Optional[str] = None,
    strict: bool = True,
) -> tuple[list[str], list[str]]:
    """Validate flowchart configuration.

    Args:
        nodes: Dictionary of node configurations.
        edges: List of edge configurations.
        start_node: Optional start node ID.
        strict: If True, raise errors for warnings.

    Returns:
        Tuple of (errors, warnings) lists.

    Raises:
        ValidationError: If validation fails.
    """
    validator = FlowchartValidator(strict=strict)
    validator.validate_flowchart(nodes, edges, start_node)
    return validator.errors, validator.warnings
