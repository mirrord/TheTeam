"""Tests for flowchart tool execution and expanded tool registry features."""

import pytest
from unittest.mock import Mock, patch, MagicMock

from pithos.tools import ToolMetadata, ToolResult, ToolRegistry
from pithos.tools.flowchart_tool import FlowchartToolExecutor
from pithos.config_manager import ConfigManager


# ---------------------------------------------------------------------------
# ToolMetadata.tool_type
# ---------------------------------------------------------------------------


class TestToolMetadataType:
    """Tests for the tool_type field on ToolMetadata."""

    def test_default_tool_type_is_cli(self):
        tool = ToolMetadata(
            name="echo",
            path="/bin/echo",
            description="echo",
            platform="unix",
            source="system",
        )
        assert tool.tool_type == "cli"

    def test_explicit_flowchart_tool_type(self):
        tool = ToolMetadata(
            name="flowchart:my_fc",
            path="",
            description="Flowchart workflow",
            platform="cross-platform",
            source="flowchart",
            tool_type="flowchart",
        )
        assert tool.tool_type == "flowchart"


# ---------------------------------------------------------------------------
# FlowchartToolExecutor
# ---------------------------------------------------------------------------


class TestFlowchartToolExecutor:
    """Tests for FlowchartToolExecutor."""

    @pytest.fixture
    def mock_config_manager(self):
        cm = Mock(spec=ConfigManager)
        cm.get_registered_flowchart_names.return_value = iter(
            ["simple_reflect", "multi_agent_research"]
        )
        return cm

    def test_list_flowcharts(self, mock_config_manager):
        executor = FlowchartToolExecutor(mock_config_manager)
        names = executor.list_flowcharts()
        assert "simple_reflect" in names
        assert "multi_agent_research" in names

    def test_discover_flowcharts(self, mock_config_manager):
        executor = FlowchartToolExecutor(mock_config_manager)
        tools = executor.discover_flowcharts()
        assert "flowchart:simple_reflect" in tools
        assert "flowchart:multi_agent_research" in tools
        for meta in tools.values():
            assert meta.tool_type == "flowchart"
            assert meta.source == "flowchart"

    def test_run_flowchart_not_found(self, mock_config_manager):
        """Running a non-existent flowchart returns an error ToolResult."""
        mock_config_manager.get_config_file.return_value = None
        executor = FlowchartToolExecutor(mock_config_manager)
        result = executor.run("nonexistent", "hello", agents={})
        assert result.success is False
        assert result.error_hint is not None
        assert "nonexistent" in result.error_hint or "not found" in result.stderr

    @patch("pithos.flowchart.Flowchart")
    def test_run_flowchart_success(self, MockFlowchart, mock_config_manager):
        """A successful flowchart run yields a ToolResult with output."""
        fc_instance = MagicMock()
        fc_instance.run.return_value = "The answer is 42"
        MockFlowchart.from_registered.return_value = fc_instance

        executor = FlowchartToolExecutor(mock_config_manager)
        result = executor.run(
            "simple_reflect", "What is 6*7?", agents={"agent": Mock()}
        )

        assert result.success is True
        assert result.stdout == "The answer is 42"
        assert result.exit_code == 0
        MockFlowchart.from_registered.assert_called_once_with(
            "simple_reflect", mock_config_manager
        )

    @patch("pithos.flowchart.Flowchart")
    def test_run_flowchart_execution_error(self, MockFlowchart, mock_config_manager):
        """A failing flowchart run yields an error ToolResult."""
        fc_instance = MagicMock()
        fc_instance.run.side_effect = RuntimeError("agent not responding")
        MockFlowchart.from_registered.return_value = fc_instance

        executor = FlowchartToolExecutor(mock_config_manager)
        result = executor.run("simple_reflect", "hello", agents={})

        assert result.success is False
        assert "agent not responding" in result.stderr


# ---------------------------------------------------------------------------
# ToolRegistry — flowchart discovery
# ---------------------------------------------------------------------------


class TestRegistryFlowchartDiscovery:
    """Tests that ToolRegistry registers flowcharts when configured."""

    @pytest.fixture
    def mock_cm_with_flowcharts(self):
        cm = Mock(spec=ConfigManager)
        cm.get_config.return_value = {
            "enabled": True,
            "timeout": 30,
            "max_output_size": 10000,
            "mode": "include",
            "include": ["echo", "flowchart"],
            "exclude": [],
            "descriptions": {},
            "flowcharts": {"enabled": True, "timeout": 120, "max_steps": 100},
        }
        cm.get_registered_flowchart_names.return_value = iter(
            ["simple_reflect", "teacher_student"]
        )
        return cm

    def test_flowchart_virtual_tool_registered(self, mock_cm_with_flowcharts):
        ToolRegistry.invalidate_cache()
        registry = ToolRegistry(mock_cm_with_flowcharts)
        assert "flowchart" in registry.tools
        assert registry.tools["flowchart"].tool_type == "flowchart"

    def test_individual_flowcharts_registered(self, mock_cm_with_flowcharts):
        ToolRegistry.invalidate_cache()
        registry = ToolRegistry(mock_cm_with_flowcharts)
        assert "flowchart:simple_reflect" in registry.tools
        assert "flowchart:teacher_student" in registry.tools

    def test_flowchart_not_registered_when_disabled(self):
        cm = Mock(spec=ConfigManager)
        cm.get_config.return_value = {
            "enabled": True,
            "timeout": 30,
            "max_output_size": 10000,
            "mode": "include",
            "include": ["echo", "flowchart"],
            "exclude": [],
            "descriptions": {},
            "flowcharts": {"enabled": False},
        }
        ToolRegistry.invalidate_cache()
        registry = ToolRegistry(cm)
        assert "flowchart" not in registry.tools

    def test_tool_list_text_groups_flowcharts(self, mock_cm_with_flowcharts):
        ToolRegistry.invalidate_cache()
        registry = ToolRegistry(mock_cm_with_flowcharts)
        text = registry.get_tool_list_text()
        assert "Flowchart tools" in text
        assert "simple_reflect" in text


# ---------------------------------------------------------------------------
# ToolRegistry — expanded CLI tools
# ---------------------------------------------------------------------------


class TestRegistryExpandedTools:
    """Tests that the expanded include list gates the right tools."""

    @pytest.fixture
    def cm_with_shells(self):
        cm = Mock(spec=ConfigManager)
        cm.get_config.return_value = {
            "enabled": True,
            "timeout": 30,
            "max_output_size": 10000,
            "mode": "include",
            "include": ["powershell", "pwsh", "cmd", "bash", "sh", "wsl", "ping"],
            "exclude": [],
            "descriptions": {
                "powershell": "Windows PowerShell",
                "pwsh": "PowerShell Core",
                "cmd": "Windows Command Prompt",
                "bash": "Bash shell",
            },
        }
        return cm

    def test_shell_tools_allowed(self, cm_with_shells):
        ToolRegistry.invalidate_cache()
        registry = ToolRegistry(cm_with_shells)
        for name in ("powershell", "pwsh", "cmd", "bash", "sh", "wsl", "ping"):
            assert registry.is_allowed(name), f"{name} should be allowed"

    def test_excluded_tools_still_blocked(self, cm_with_shells):
        """Dangerous commands stay blocked even with generous include list."""
        ToolRegistry.invalidate_cache()
        registry = ToolRegistry(cm_with_shells)
        for name in ("rm", "del", "format", "shutdown"):
            assert not registry.is_allowed(name), f"{name} should be blocked"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
