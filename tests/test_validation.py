"""Tests for flowchart validation."""

import pytest
from pithos.validation import (
    FlowchartValidator,
    ValidationError,
    validate_flowchart,
)


class TestFlowchartValidator:
    """Test suite for FlowchartValidator."""

    def test_valid_simple_flowchart(self):
        """Test validation of a simple valid flowchart."""
        nodes = {
            "start": {
                "type": "prompt",
                "prompt": "What is your question?",
                "extraction": {},
                "inputs": ["default"],
                "outputs": ["default"],
            },
            "process": {
                "type": "textparse",
                "extraction": {},
                "inputs": ["default"],
                "outputs": ["default"],
            },
        }
        edges = [
            {
                "from": "start",
                "to": "process",
                "condition": {"type": "AlwaysCondition"},
                "output_key": "default",
                "input_key": "default",
            }
        ]

        validator = FlowchartValidator(strict=False)
        validator.validate_flowchart(nodes, edges, start_node="start")

        assert len(validator.errors) == 0

    def test_missing_node_type(self):
        """Test that missing node type is detected."""
        nodes = {
            "bad_node": {
                "prompt": "Some prompt",
                # Missing 'type' field
            }
        }
        edges = []

        validator = FlowchartValidator(strict=False)

        with pytest.raises(ValidationError) as exc_info:
            validator.validate_flowchart(nodes, edges)

        assert "missing 'type' field" in str(exc_info.value)

    def test_unknown_node_type(self):
        """Test that unknown node types are detected."""
        nodes = {
            "bad_node": {
                "type": "invalid_node_type",
                "prompt": "Some prompt",
            }
        }
        edges = []

        validator = FlowchartValidator(strict=False)

        with pytest.raises(ValidationError) as exc_info:
            validator.validate_flowchart(nodes, edges)

        assert "unknown type" in str(exc_info.value).lower()

    def test_missing_required_parameter_prompt_node(self):
        """Test that missing required parameters are detected for PromptNode."""
        nodes = {
            "incomplete": {
                "type": "prompt",
                # Missing 'prompt' parameter
                "extraction": {},
            }
        }
        edges = []

        validator = FlowchartValidator(strict=False)

        with pytest.raises(ValidationError) as exc_info:
            validator.validate_flowchart(nodes, edges)

        assert "missing required parameter 'prompt'" in str(exc_info.value)

    def test_missing_required_parameter_agent_prompt_node(self):
        """Test that missing required parameters are detected for AgentPromptNode."""
        nodes = {
            "incomplete": {
                "type": "agentprompt",
                "prompt": "Some prompt",
                # Missing 'agent' parameter
            }
        }
        edges = []

        validator = FlowchartValidator(strict=False)

        with pytest.raises(ValidationError) as exc_info:
            validator.validate_flowchart(nodes, edges)

        assert "missing required parameter 'agent'" in str(exc_info.value)

    def test_missing_required_parameter_custom_node(self):
        """Test that missing required parameters are detected for CustomNode."""
        nodes = {
            "incomplete": {
                "type": "custom",
                # Missing 'custom_code' parameter
            }
        }
        edges = []

        validator = FlowchartValidator(strict=False)

        with pytest.raises(ValidationError) as exc_info:
            validator.validate_flowchart(nodes, edges)

        assert "missing required parameter 'custom_code'" in str(exc_info.value)

    def test_missing_required_parameter_toolcall_node(self):
        """Test that missing required parameters are detected for ToolCallNode."""
        nodes = {
            "incomplete": {
                "type": "toolcall",
                # Missing 'command' parameter
                "save_to": "result",
            }
        }
        edges = []

        validator = FlowchartValidator(strict=False)

        with pytest.raises(ValidationError) as exc_info:
            validator.validate_flowchart(nodes, edges)

        assert "missing required parameter 'command'" in str(exc_info.value)

    def test_invalid_edge_missing_from(self):
        """Test that edges missing 'from' field are detected."""
        nodes = {
            "node1": {"type": "prompt", "prompt": "Test"},
            "node2": {"type": "prompt", "prompt": "Test"},
        }
        edges = [
            {
                # Missing 'from' field
                "to": "node2",
                "condition": {"type": "AlwaysCondition"},
            }
        ]

        validator = FlowchartValidator(strict=False)

        with pytest.raises(ValidationError) as exc_info:
            validator.validate_flowchart(nodes, edges, start_node="node1")

        assert "missing 'from' field" in str(exc_info.value)

    def test_invalid_edge_missing_to(self):
        """Test that edges missing 'to' field are detected."""
        nodes = {
            "node1": {"type": "prompt", "prompt": "Test"},
            "node2": {"type": "prompt", "prompt": "Test"},
        }
        edges = [
            {
                "from": "node1",
                # Missing 'to' field
                "condition": {"type": "AlwaysCondition"},
            }
        ]

        validator = FlowchartValidator(strict=False)

        with pytest.raises(ValidationError) as exc_info:
            validator.validate_flowchart(nodes, edges, start_node="node1")

        assert "missing 'to' field" in str(exc_info.value)

    def test_edge_references_nonexistent_from_node(self):
        """Test that edges referencing non-existent nodes are detected."""
        nodes = {
            "node1": {"type": "prompt", "prompt": "Test"},
        }
        edges = [
            {
                "from": "nonexistent_node",
                "to": "node1",
                "condition": {"type": "AlwaysCondition"},
            }
        ]

        validator = FlowchartValidator(strict=False)

        with pytest.raises(ValidationError) as exc_info:
            validator.validate_flowchart(nodes, edges, start_node="node1")

        assert "non-existent 'from' node" in str(exc_info.value)

    def test_edge_references_nonexistent_to_node(self):
        """Test that edges referencing non-existent nodes are detected."""
        nodes = {
            "node1": {"type": "prompt", "prompt": "Test"},
        }
        edges = [
            {
                "from": "node1",
                "to": "nonexistent_node",
                "condition": {"type": "AlwaysCondition"},
            }
        ]

        validator = FlowchartValidator(strict=False)

        with pytest.raises(ValidationError) as exc_info:
            validator.validate_flowchart(nodes, edges, start_node="node1")

        assert "non-existent 'to' node" in str(exc_info.value)

    def test_invalid_start_node(self):
        """Test that invalid start nodes are detected."""
        nodes = {
            "node1": {"type": "prompt", "prompt": "Test"},
        }
        edges = []

        validator = FlowchartValidator(strict=False)

        with pytest.raises(ValidationError) as exc_info:
            validator.validate_flowchart(nodes, edges, start_node="nonexistent")

        assert "Start node 'nonexistent' does not exist" in str(exc_info.value)

    def test_empty_flowchart(self):
        """Test that empty flowcharts are rejected."""
        nodes = {}
        edges = []

        validator = FlowchartValidator(strict=False)

        with pytest.raises(ValidationError) as exc_info:
            validator.validate_flowchart(nodes, edges)

        assert "at least one node" in str(exc_info.value)

    def test_invalid_extraction_pattern(self):
        """Test that invalid regex patterns are detected."""
        nodes = {
            "bad_regex": {
                "type": "prompt",
                "prompt": "Test",
                "extraction": {
                    "value": "[invalid(regex",  # Invalid regex
                },
            }
        }
        edges = []

        validator = FlowchartValidator(strict=False)

        with pytest.raises(ValidationError) as exc_info:
            validator.validate_flowchart(nodes, edges, start_node="bad_regex")

        assert "invalid regex" in str(exc_info.value).lower()

    def test_extraction_not_dict(self):
        """Test that non-dict extraction fields are detected."""
        nodes = {
            "bad_extraction": {
                "type": "prompt",
                "prompt": "Test",
                "extraction": "not a dict",  # Should be a dict
            }
        }
        edges = []

        validator = FlowchartValidator(strict=False)

        with pytest.raises(ValidationError) as exc_info:
            validator.validate_flowchart(nodes, edges, start_node="bad_extraction")

        assert "'extraction' must be a dict" in str(exc_info.value)

    def test_inputs_not_list(self):
        """Test that non-list inputs fields are detected."""
        nodes = {
            "bad_inputs": {
                "type": "prompt",
                "prompt": "Test",
                "inputs": "not a list",  # Should be a list
            }
        }
        edges = []

        validator = FlowchartValidator(strict=False)

        with pytest.raises(ValidationError) as exc_info:
            validator.validate_flowchart(nodes, edges, start_node="bad_inputs")

        assert "'inputs' must be a list" in str(exc_info.value)

    def test_outputs_not_list(self):
        """Test that non-list outputs fields are detected."""
        nodes = {
            "bad_outputs": {
                "type": "prompt",
                "prompt": "Test",
                "outputs": "not a list",  # Should be a list
            }
        }
        edges = []

        validator = FlowchartValidator(strict=False)

        with pytest.raises(ValidationError) as exc_info:
            validator.validate_flowchart(nodes, edges, start_node="bad_outputs")

        assert "'outputs' must be a list" in str(exc_info.value)

    def test_unreachable_nodes_warning(self):
        """Test that unreachable nodes generate warnings."""
        nodes = {
            "start": {"type": "prompt", "prompt": "Test"},
            "reachable": {"type": "prompt", "prompt": "Test"},
            "unreachable": {"type": "prompt", "prompt": "Test"},
        }
        edges = [
            {
                "from": "start",
                "to": "reachable",
                "condition": {"type": "AlwaysCondition"},
            }
            # No edge to 'unreachable'
        ]

        validator = FlowchartValidator(strict=False)
        validator.validate_flowchart(nodes, edges, start_node="start")

        assert len(validator.warnings) > 0
        assert any("unreachable" in w.lower() for w in validator.warnings)

    def test_cycle_detection_warning(self):
        """Test that cycles generate warnings."""
        nodes = {
            "node1": {"type": "prompt", "prompt": "Test"},
            "node2": {"type": "prompt", "prompt": "Test"},
        }
        edges = [
            {
                "from": "node1",
                "to": "node2",
                "condition": {"type": "AlwaysCondition"},
            },
            {
                "from": "node2",
                "to": "node1",
                "condition": {"type": "CountCondition", "count": 3},
            },
        ]

        validator = FlowchartValidator(strict=False)
        validator.validate_flowchart(nodes, edges, start_node="node1")

        assert len(validator.warnings) > 0
        assert any("cycle" in w.lower() for w in validator.warnings)

    def test_self_loop_warning(self):
        """Test that self-loops generate warnings."""
        nodes = {
            "loop_node": {"type": "prompt", "prompt": "Test"},
        }
        edges = [
            {
                "from": "loop_node",
                "to": "loop_node",
                "condition": {"type": "CountCondition", "count": 5},
            }
        ]

        validator = FlowchartValidator(strict=False)
        validator.validate_flowchart(nodes, edges, start_node="loop_node")

        assert len(validator.warnings) > 0
        assert any("self-loop" in w.lower() for w in validator.warnings)

    def test_custom_node_security_warnings(self):
        """Test that dangerous code in CustomNode generates warnings."""
        dangerous_codes = [
            "exec('malicious code')",
            "eval('2 + 2')",
            "__import__('os')",
            "os.system('rm -rf /')",
            "subprocess.run(['ls'])",
        ]

        for code in dangerous_codes:
            nodes = {
                "dangerous": {
                    "type": "custom",
                    "custom_code": code,
                }
            }
            edges = []

            validator = FlowchartValidator(strict=False)
            validator.validate_flowchart(nodes, edges, start_node="dangerous")

            assert len(validator.warnings) > 0, f"No warning for: {code}"
            assert any("security risk" in w.lower() for w in validator.warnings)

    def test_invalid_sethistory_mode(self):
        """Test that invalid SetHistoryNode modes are detected."""
        nodes = {
            "bad_mode": {
                "type": "sethistory",
                "agent": "test_agent",
                "mode": "invalid_mode",
            }
        }
        edges = []

        validator = FlowchartValidator(strict=False)

        with pytest.raises(ValidationError) as exc_info:
            validator.validate_flowchart(nodes, edges, start_node="bad_mode")

        assert "invalid mode" in str(exc_info.value).lower()

    def test_mismatched_output_input_keys_warning(self):
        """Test that mismatched output/input keys generate warnings."""
        nodes = {
            "producer": {
                "type": "prompt",
                "prompt": "Test",
                "outputs": ["result"],
            },
            "consumer": {
                "type": "prompt",
                "prompt": "Test",
                "inputs": ["input"],
            },
        }
        edges = [
            {
                "from": "producer",
                "to": "consumer",
                "condition": {"type": "AlwaysCondition"},
                "output_key": "wrong_key",  # Not in producer's outputs
                "input_key": "wrong_input",  # Not in consumer's inputs
            }
        ]

        validator = FlowchartValidator(strict=False)
        validator.validate_flowchart(nodes, edges, start_node="producer")

        assert len(validator.warnings) >= 2
        assert any("output_key" in w for w in validator.warnings)
        assert any("input_key" in w for w in validator.warnings)

    def test_count_condition_missing_count_parameter(self):
        """Test that CountCondition without count parameter is detected."""
        nodes = {
            "node1": {"type": "prompt", "prompt": "Test"},
            "node2": {"type": "prompt", "prompt": "Test"},
        }
        edges = [
            {
                "from": "node1",
                "to": "node2",
                "condition": {
                    "type": "CountCondition",
                    # Missing 'count' parameter
                },
            }
        ]

        validator = FlowchartValidator(strict=False)

        with pytest.raises(ValidationError) as exc_info:
            validator.validate_flowchart(nodes, edges, start_node="node1")

        assert "missing 'count' parameter" in str(exc_info.value).lower()

    def test_condition_not_dict(self):
        """Test that non-dict condition fields are detected."""
        nodes = {
            "node1": {"type": "prompt", "prompt": "Test"},
            "node2": {"type": "prompt", "prompt": "Test"},
        }
        edges = [
            {
                "from": "node1",
                "to": "node2",
                "condition": "not a dict",  # Should be a dict
            }
        ]

        validator = FlowchartValidator(strict=False)

        with pytest.raises(ValidationError) as exc_info:
            validator.validate_flowchart(nodes, edges, start_node="node1")

        assert "condition must be a dict" in str(exc_info.value)

    def test_condition_missing_type(self):
        """Test that conditions without type field are detected."""
        nodes = {
            "node1": {"type": "prompt", "prompt": "Test"},
            "node2": {"type": "prompt", "prompt": "Test"},
        }
        edges = [
            {
                "from": "node1",
                "to": "node2",
                "condition": {
                    # Missing 'type' field
                    "count": 3,
                },
            }
        ]

        validator = FlowchartValidator(strict=False)

        with pytest.raises(ValidationError) as exc_info:
            validator.validate_flowchart(nodes, edges, start_node="node1")

        assert "missing 'type' field" in str(exc_info.value)

    def test_validate_flowchart_function(self):
        """Test the standalone validate_flowchart function."""
        nodes = {
            "test": {"type": "prompt", "prompt": "Test prompt"},
        }
        edges = []

        # Should not raise
        errors, warnings = validate_flowchart(nodes, edges, "test", strict=False)

        assert len(errors) == 0

    def test_strict_mode_treats_warnings_as_errors(self):
        """Test that strict mode raises errors for warnings."""
        nodes = {
            "start": {"type": "prompt", "prompt": "Test"},
            "unreachable": {"type": "prompt", "prompt": "Test"},
        }
        edges = []

        validator = FlowchartValidator(strict=True)

        with pytest.raises(ValidationError) as exc_info:
            validator.validate_flowchart(nodes, edges, start_node="start")

        # Should fail on unreachable node warning
        assert (
            "unreachable" in str(exc_info.value).lower()
            or "warning" in str(exc_info.value).lower()
        )

    def test_complex_valid_team_flowchart(self):
        """Test validation of a complex multi-agent team flowchart."""
        nodes = {
            "InitialResearch": {
                "type": "agentprompt",
                "agent": "researcher",
                "prompt": "Research the topic: {current_input}",
                "extraction": {},
                "inputs": ["default"],
                "outputs": ["default"],
            },
            "CaptureResearch": {
                "type": "textparse",
                "extraction": {},
                "set": {"research_findings": "{current_input}"},
                "inputs": ["default"],
                "outputs": ["default"],
            },
            "WriteDraft": {
                "type": "agentprompt",
                "agent": "writer",
                "prompt": "Write a summary: {research_findings}",
                "extraction": {},
                "inputs": ["default"],
                "outputs": ["default"],
            },
        }
        edges = [
            {
                "from": "InitialResearch",
                "to": "CaptureResearch",
                "condition": {"type": "AlwaysCondition"},
                "output_key": "default",
                "input_key": "default",
            },
            {
                "from": "CaptureResearch",
                "to": "WriteDraft",
                "condition": {"type": "AlwaysCondition"},
                "output_key": "default",
                "input_key": "default",
            },
        ]

        validator = FlowchartValidator(strict=False)
        validator.validate_flowchart(nodes, edges, start_node="InitialResearch")

        assert len(validator.errors) == 0
