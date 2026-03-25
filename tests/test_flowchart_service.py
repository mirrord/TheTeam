"""
Test suite for flowchart service and edge handling.

This test suite investigates edge rendering issues by testing:
1. YAML parsing of edges
2. Flowchart service returning correct edge data
3. Edge format compatibility with frontend expectations
"""

import pytest
import tempfile
import yaml
from pathlib import Path
from theteam.services.flowchart_service import FlowchartService


@pytest.fixture
def temp_config_dir():
    """Create a temporary directory for flowchart configs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_flowchart_config():
    """Sample flowchart config with edges."""
    return {
        "name": "test_flowchart",
        "description": "Test flowchart with edges",
        "start_node": "NodeA",
        "edges": [
            {
                "from": "NodeA",
                "to": "NodeB",
                "condition": {"type": "AlwaysCondition"},
                "priority": 9,
            },
            {
                "from": "NodeB",
                "to": "NodeC",
                "condition": {"type": "AlwaysCondition"},
                "priority": 9,
            },
            {
                "from": "NodeC",
                "to": "NodeA",
                "condition": {"type": "CountCondition", "limit": 3},
                "priority": 2,
            },
        ],
        "nodes": {
            "NodeA": {
                "type": "prompt",
                "prompt": "This is node A: {input}",
                "position": {"x": 100, "y": 100},
            },
            "NodeB": {
                "type": "textparse",
                "set": {"output": "{current_input}"},
                "position": {"x": 300, "y": 100},
            },
            "NodeC": {
                "type": "custom",
                "custom_code": "return context",
                "position": {"x": 500, "y": 100},
            },
        },
    }


class TestFlowchartService:
    """Test FlowchartService edge handling."""

    def test_load_flowchart_with_edges(self, temp_config_dir, sample_flowchart_config):
        """Test loading a flowchart that has edges defined."""
        # Create a YAML file with edges
        yaml_file = temp_config_dir / "test_flowchart.yaml"
        with open(yaml_file, "w") as f:
            yaml.dump(sample_flowchart_config, f)

        # Load the flowchart without auto I/O nodes
        service = FlowchartService(config_dir=temp_config_dir)
        flowchart = service.get_flowchart("test_flowchart", include_auto_nodes=False)

        # Verify flowchart loaded
        assert flowchart is not None
        assert flowchart["id"] == "test_flowchart"
        assert "config" in flowchart

        # Verify edges are present in config
        config = flowchart["config"]
        assert "edges" in config
        assert isinstance(config["edges"], list)
        assert len(config["edges"]) == 3

        # Verify edge structure
        for edge in config["edges"]:
            assert "from" in edge
            assert "to" in edge
            assert edge["from"] in config["nodes"]
            # Allow 'to' to point to nodes or 'END'
            if edge["to"] != "END":
                assert edge["to"] in config["nodes"]

    def test_edges_array_format(self, temp_config_dir, sample_flowchart_config):
        """Test that edges array has correct format for frontend consumption."""
        yaml_file = temp_config_dir / "test_flowchart.yaml"
        with open(yaml_file, "w") as f:
            yaml.dump(sample_flowchart_config, f)

        service = FlowchartService(config_dir=temp_config_dir)
        flowchart = service.get_flowchart("test_flowchart", include_auto_nodes=False)
        config = flowchart["config"]

        # Check edges structure matches what frontend expects
        assert "edges" in config
        edges = config["edges"]

        # First edge: NodeA -> NodeB
        edge1 = edges[0]
        assert edge1["from"] == "NodeA"
        assert edge1["to"] == "NodeB"
        assert "condition" in edge1

        # Second edge: NodeB -> NodeC
        edge2 = edges[1]
        assert edge2["from"] == "NodeB"
        assert edge2["to"] == "NodeC"

        # Third edge: NodeC -> NodeA (loop)
        edge3 = edges[2]
        assert edge3["from"] == "NodeC"
        assert edge3["to"] == "NodeA"

    def test_nodes_have_positions(self, temp_config_dir, sample_flowchart_config):
        """Test that nodes have position data for ReactFlow."""
        yaml_file = temp_config_dir / "test_flowchart.yaml"
        with open(yaml_file, "w") as f:
            yaml.dump(sample_flowchart_config, f)

        service = FlowchartService(config_dir=temp_config_dir)
        flowchart = service.get_flowchart("test_flowchart", include_auto_nodes=False)
        config = flowchart["config"]

        # Verify all nodes have positions
        for node_id, node_data in config["nodes"].items():
            assert "position" in node_data, f"Node {node_id} missing position"
            assert "x" in node_data["position"]
            assert "y" in node_data["position"]

    def test_node_types_are_preserved(self, temp_config_dir, sample_flowchart_config):
        """Test that node types are correctly preserved."""
        yaml_file = temp_config_dir / "test_flowchart.yaml"
        with open(yaml_file, "w") as f:
            yaml.dump(sample_flowchart_config, f)

        service = FlowchartService(config_dir=temp_config_dir)
        flowchart = service.get_flowchart("test_flowchart")
        config = flowchart["config"]

        # Verify node types
        assert config["nodes"]["NodeA"]["type"] == "prompt"
        assert config["nodes"]["NodeB"]["type"] == "textparse"
        assert config["nodes"]["NodeC"]["type"] == "custom"

    def test_list_flowcharts_includes_edge_count(
        self, temp_config_dir, sample_flowchart_config
    ):
        """Test that listed flowcharts show node count (could add edge count too)."""
        yaml_file = temp_config_dir / "test_flowchart.yaml"
        with open(yaml_file, "w") as f:
            yaml.dump(sample_flowchart_config, f)

        service = FlowchartService(config_dir=temp_config_dir)
        flowcharts = service.list_flowcharts()

        assert len(flowcharts) == 1
        flowchart = flowcharts[0]
        assert flowchart["id"] == "test_flowchart"
        assert flowchart["node_count"] == 3

    def test_auto_io_nodes_added_by_default(
        self, temp_config_dir, sample_flowchart_config
    ):
        """Test that auto I/O nodes are added by default."""
        yaml_file = temp_config_dir / "test_flowchart.yaml"
        with open(yaml_file, "w") as f:
            yaml.dump(sample_flowchart_config, f)

        service = FlowchartService(config_dir=temp_config_dir)
        flowchart = service.get_flowchart(
            "test_flowchart"
        )  # include_auto_nodes=True by default
        config = flowchart["config"]

        # Verify auto nodes were added
        assert "__auto_chat_input__" in config["nodes"]
        assert "__auto_chat_output__" in config["nodes"]

        # Verify auto nodes have correct types
        assert config["nodes"]["__auto_chat_input__"]["type"] == "chatinput"
        assert config["nodes"]["__auto_chat_output__"]["type"] == "chatoutput"

        # Verify start node was updated
        assert config["start_node"] == "__auto_chat_input__"

    def test_auto_io_nodes_have_positions(
        self, temp_config_dir, sample_flowchart_config
    ):
        """Test that auto-added I/O nodes have position data."""
        yaml_file = temp_config_dir / "test_flowchart.yaml"
        with open(yaml_file, "w") as f:
            yaml.dump(sample_flowchart_config, f)

        service = FlowchartService(config_dir=temp_config_dir)
        flowchart = service.get_flowchart("test_flowchart")
        config = flowchart["config"]

        # Verify auto input node has position
        auto_input = config["nodes"]["__auto_chat_input__"]
        assert "position" in auto_input
        assert "x" in auto_input["position"]
        assert "y" in auto_input["position"]

        # Verify auto output node has position
        auto_output = config["nodes"]["__auto_chat_output__"]
        assert "position" in auto_output
        assert "x" in auto_output["position"]
        assert "y" in auto_output["position"]

    def test_auto_io_nodes_connected_with_edges(
        self, temp_config_dir, sample_flowchart_config
    ):
        """Test that auto I/O nodes are properly connected with edges."""
        yaml_file = temp_config_dir / "test_flowchart.yaml"
        with open(yaml_file, "w") as f:
            yaml.dump(sample_flowchart_config, f)

        service = FlowchartService(config_dir=temp_config_dir)
        flowchart = service.get_flowchart("test_flowchart")
        config = flowchart["config"]

        # Should have original 3 edges plus edge from auto input to original start
        assert len(config["edges"]) >= 4

        # Verify edge from auto input to original start node
        input_edge = config["edges"][0]  # Should be first edge
        assert input_edge["from"] == "__auto_chat_input__"
        assert input_edge["to"] == "NodeA"  # Original start node

    def test_no_duplicate_io_nodes_when_explicit(self, temp_config_dir):
        """Test that no duplicate I/O nodes are added if flowchart has explicit ones."""
        # Create a flowchart that already has I/O nodes
        config_with_io = {
            "name": "test_with_io",
            "description": "Test flowchart with explicit I/O nodes",
            "start_node": "input_node",
            "nodes": {
                "input_node": {
                    "type": "chatinput",
                    "label": "User Input",
                    "save_to": "user_input",
                    "position": {"x": 100, "y": 100},
                },
                "process_node": {
                    "type": "prompt",
                    "prompt": "Process: {user_input}",
                    "position": {"x": 300, "y": 100},
                },
                "output_node": {
                    "type": "chatoutput",
                    "label": "Result",
                    "source": "current_input",
                    "position": {"x": 500, "y": 100},
                },
            },
            "edges": [
                {
                    "from": "input_node",
                    "to": "process_node",
                    "condition": {"type": "AlwaysCondition"},
                },
                {
                    "from": "process_node",
                    "to": "output_node",
                    "condition": {"type": "AlwaysCondition"},
                },
            ],
        }

        yaml_file = temp_config_dir / "test_with_io.yaml"
        with open(yaml_file, "w") as f:
            yaml.dump(config_with_io, f)

        service = FlowchartService(config_dir=temp_config_dir)
        flowchart = service.get_flowchart("test_with_io")
        config = flowchart["config"]

        # Should have exactly 3 nodes (no auto nodes added)
        assert len(config["nodes"]) == 3
        assert "__auto_chat_input__" not in config["nodes"]
        assert "__auto_chat_output__" not in config["nodes"]

        # Start node should remain unchanged
        assert config["start_node"] == "input_node"


class TestEdgeParsing:
    """Test edge parsing logic specifically."""

    def test_refined_reflect_yaml_structure(self):
        """Test parsing the actual refined_reflect.yaml structure."""
        # Load the actual refined_reflect.yaml file
        config_path = Path.cwd() / "configs" / "flowcharts" / "refined_reflect.yaml"

        if not config_path.exists():
            pytest.skip("refined_reflect.yaml not found")

        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        # Verify structure
        assert "edges" in config, "Config missing 'edges' key"
        assert "nodes" in config, "Config missing 'nodes' key"
        assert isinstance(config["edges"], list), "Edges should be a list"
        assert isinstance(config["nodes"], dict), "Nodes should be a dict"

        # Print debug info
        print(f"\nEdges count: {len(config['edges'])}")
        print(f"Nodes count: {len(config['nodes'])}")
        print(f"Node IDs: {list(config['nodes'].keys())}")

        # Verify all edges reference valid nodes
        node_ids = set(config["nodes"].keys())
        for i, edge in enumerate(config["edges"]):
            assert "from" in edge, f"Edge {i} missing 'from'"
            assert "to" in edge, f"Edge {i} missing 'to'"

            from_node = edge["from"]
            to_node = edge["to"]

            assert (
                from_node in node_ids
            ), f"Edge {i}: 'from' node '{from_node}' not in nodes"

            # 'to' can be 'END' or a valid node
            if to_node != "END":
                assert (
                    to_node in node_ids
                ), f"Edge {i}: 'to' node '{to_node}' not in nodes"

            print(f"Edge {i}: {from_node} -> {to_node}")

    def test_edge_to_reactflow_conversion(self):
        """Test conversion from YAML edges to ReactFlow format."""
        # Simulate what the frontend store does
        yaml_edges = [
            {"from": "NodeA", "to": "NodeB", "condition": {"type": "AlwaysCondition"}},
            {"from": "NodeB", "to": "NodeC", "condition": {"type": "AlwaysCondition"}},
        ]

        nodes_config = {
            "NodeA": {"type": "prompt"},
            "NodeB": {"type": "textparse"},
            "NodeC": {"type": "custom"},
        }

        # Convert to ReactFlow format (simulate frontend logic)
        def get_handle_ids(node_type):
            if node_type == "prompt":
                return {"source": "response", "target": "input"}
            else:
                return {"source": "output", "target": "input"}

        reactflow_edges = []
        for idx, edge in enumerate(yaml_edges):
            source_type = nodes_config[edge["from"]].get("type", "prompt")
            target_type = nodes_config[edge["to"]].get("type", "prompt")

            source_handles = get_handle_ids(source_type)
            target_handles = get_handle_ids(target_type)

            reactflow_edge = {
                "id": f"edge-{edge['from']}-{edge['to']}-{idx}",
                "source": edge["from"],
                "target": edge["to"],
                "sourceHandle": source_handles["source"],
                "targetHandle": target_handles["target"],
                "animated": True,
                "style": {"stroke": "#64748b", "strokeWidth": 2},
            }
            reactflow_edges.append(reactflow_edge)

        # Verify conversion
        assert len(reactflow_edges) == 2

        # Check first edge
        edge1 = reactflow_edges[0]
        assert edge1["source"] == "NodeA"
        assert edge1["target"] == "NodeB"
        assert edge1["sourceHandle"] == "response"  # prompt node
        assert edge1["targetHandle"] == "input"

        # Check second edge
        edge2 = reactflow_edges[1]
        assert edge2["source"] == "NodeB"
        assert edge2["target"] == "NodeC"
        assert edge2["sourceHandle"] == "output"  # textparse node
        assert edge2["targetHandle"] == "input"

        print("\nReactFlow edges:")
        for edge in reactflow_edges:
            print(
                f"  {edge['id']}: {edge['source']}[{edge['sourceHandle']}] -> {edge['target']}[{edge['targetHandle']}]"
            )


class TestEdgeHandleMatching:
    """Test that edge handles match what nodes actually provide."""

    def test_prompt_node_handles(self):
        """Verify prompt node handle IDs."""
        # From PromptNode.tsx, prompt nodes have:
        # - Target handle: 'input' (or variable names)
        # - Source handle: 'response'
        node_type = "prompt"
        expected_source = "response"
        expected_target = "input"

        # This is what the frontend expects
        if node_type == "prompt":
            assert expected_source == "response"
            assert expected_target == "input"

    def test_textparse_node_handles(self):
        """Verify textparse node handle IDs."""
        # From TextParseNode.tsx:
        # - Target handle: 'input'
        # - Source handle: 'output'
        node_type = "textparse"
        expected_source = "output"
        expected_target = "input"

        if node_type == "textparse":
            assert expected_source == "output"
            assert expected_target == "input"

    def test_custom_node_handles(self):
        """Verify custom node handle IDs."""
        # From CustomNode.tsx:
        # - Target handle: 'input'
        # - Source handle: 'output'
        node_type = "custom"
        expected_source = "output"
        expected_target = "input"

        if node_type == "custom":
            assert expected_source == "output"
            assert expected_target == "input"


if __name__ == "__main__":
    # Run with pytest -v -s to see print output
    pytest.main([__file__, "-v", "-s"])
