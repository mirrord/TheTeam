"""Tests for pithos tool calling system."""

import os
import pytest
from unittest.mock import Mock, patch
import subprocess

from pithos.tools import (
    ToolMetadata,
    ToolResult,
    ToolRegistry,
    ToolExecutor,
)
from pithos.config_manager import ConfigManager


class TestToolMetadata:
    """Tests for ToolMetadata dataclass."""

    def test_tool_metadata_creation(self):
        """Test creating ToolMetadata."""
        tool = ToolMetadata(
            name="python",
            path="/usr/bin/python",
            description="Python interpreter",
            platform="unix",
            source="system",
        )
        assert tool.name == "python"
        assert tool.path == "/usr/bin/python"
        assert tool.description == "Python interpreter"
        assert tool.platform == "unix"
        assert tool.source == "system"


class TestToolResult:
    """Tests for ToolResult dataclass."""

    def test_tool_result_creation(self):
        """Test creating ToolResult."""
        result = ToolResult(
            success=True,
            stdout="Python 3.10.0",
            stderr="",
            exit_code=0,
            execution_time=0.123,
            command="python --version",
        )
        assert result.success is True
        assert result.stdout == "Python 3.10.0"
        assert result.stderr == ""
        assert result.exit_code == 0
        assert result.execution_time == 0.123
        assert result.command == "python --version"


class TestToolRegistry:
    """Tests for ToolRegistry."""

    @pytest.fixture
    def mock_config_manager(self):
        """Create a mock ConfigManager."""
        cm = Mock(spec=ConfigManager)
        cm.get_config.return_value = {
            "enabled": True,
            "timeout": 30,
            "max_output_size": 10000,
            "mode": "include",
            "include": ["echo", "python", "git"],
            "exclude": ["rm", "del"],
            "descriptions": {
                "echo": "Display a line of text",
                "python": "Python interpreter",
            },
        }
        return cm

    def test_tool_registry_initialization(self, mock_config_manager):
        """Test ToolRegistry initialization."""
        registry = ToolRegistry(mock_config_manager)
        assert registry.config_manager == mock_config_manager
        assert isinstance(registry.tools, dict)
        assert isinstance(registry.config, dict)

    def test_is_allowed_present_in_tools(self, mock_config_manager):
        """is_allowed returns True for tools that exist in the registry dict."""
        registry = ToolRegistry(mock_config_manager)
        registry.tools["python"] = ToolMetadata(
            "python", "/usr/bin/python", "Python", "unix", "system"
        )
        assert registry.is_allowed("python") is True
        assert registry.is_allowed("curl") is False

    def test_is_allowed_empty_raises(self, mock_config_manager):
        """is_allowed raises ValueError for empty tool name."""
        registry = ToolRegistry(mock_config_manager)
        import pytest

        with pytest.raises(ValueError):
            registry.is_allowed("")

    def test_requires_confirmation_confirm_mode_in_list(self, mock_config_manager):
        """requires_confirmation returns True for a tool in confirm list."""
        registry = ToolRegistry(mock_config_manager)
        registry.config = {
            "mode": "confirm",
            "confirm": ["bash", "powershell"],
        }
        assert registry.requires_confirmation("bash") is True
        assert registry.requires_confirmation("powershell") is True

    def test_requires_confirmation_confirm_mode_not_in_list(self, mock_config_manager):
        """requires_confirmation returns False for a tool NOT in confirm list."""
        registry = ToolRegistry(mock_config_manager)
        registry.config = {
            "mode": "confirm",
            "confirm": ["bash"],
        }
        assert registry.requires_confirmation("python") is False

    def test_requires_confirmation_other_modes(self, mock_config_manager):
        """requires_confirmation returns False for all non-confirm modes."""
        registry = ToolRegistry(mock_config_manager)
        for mode in ("include", "exclude", "all"):
            registry.config = {
                "mode": mode,
                "include": ["python"],
                "exclude": [],
                "confirm": ["python"],  # should be ignored in non-confirm modes
            }
            assert (
                registry.requires_confirmation("python") is False
            ), f"expected False for mode={mode!r}"

    def test_get_tool(self, mock_config_manager):
        """Test getting a specific tool."""
        registry = ToolRegistry(mock_config_manager)
        # Manually add a tool to test
        registry.tools["python"] = ToolMetadata(
            name="python",
            path="/usr/bin/python",
            description="Python interpreter",
            platform="unix",
            source="system",
        )
        tool = registry.get_tool("python")
        assert tool is not None
        assert tool.name == "python"

        # Test non-existent tool
        assert registry.get_tool("nonexistent") is None

    def test_list_tools(self, mock_config_manager):
        """Test listing all tools."""
        registry = ToolRegistry(mock_config_manager)
        registry.tools = {
            "python": Mock(),
            "git": Mock(),
            "echo": Mock(),
        }
        tools = registry.list_tools()
        assert tools == ["echo", "git", "python"]  # Should be sorted

    def test_get_tool_list_text(self, mock_config_manager):
        """Test getting formatted tool list."""
        registry = ToolRegistry(mock_config_manager)
        registry.tools = {
            "python": ToolMetadata(
                "python", "/usr/bin/python", "Python interpreter", "unix", "system"
            ),
            "git": ToolMetadata(
                "git", "/usr/bin/git", "Version control", "unix", "system"
            ),
        }
        text = registry.get_tool_list_text()
        assert "python: Python interpreter" in text
        assert "git: Version control" in text

    def test_refresh(self, mock_config_manager):
        """Test refreshing tool registry."""
        registry = ToolRegistry(mock_config_manager)
        registry.refresh()
        # Should clear and reload
        assert isinstance(registry.tools, dict)


class TestToolExecutor:
    """Tests for ToolExecutor."""

    def test_tool_executor_initialization(self):
        """Test ToolExecutor initialization."""
        executor = ToolExecutor(timeout=10, max_output_size=5000)
        assert executor.timeout == 10
        assert executor.max_output_size == 5000

    def test_parse_command(self):
        """Test command parsing."""
        executor = ToolExecutor()

        # Simple command
        tool_name, args = executor._parse_command("python --version")
        assert tool_name == "python"
        assert args == ["--version"]

        # Command with multiple arguments
        tool_name, args = executor._parse_command("git log -n 5")
        assert tool_name == "git"
        assert args == ["log", "-n", "5"]

        # Command with quoted arguments
        tool_name, args = executor._parse_command('echo "hello world"')
        assert tool_name == "echo"
        assert args == ["hello world"]

    def test_parse_command_empty(self):
        """Test parsing empty command."""
        executor = ToolExecutor()
        tool_name, args = executor._parse_command("")
        assert tool_name is None
        assert args == []

    def test_run_successful_command(self):
        """Test successful command execution via a mock provider."""
        expected = ToolResult(
            success=True,
            stdout="Python 3.10.0",
            stderr="",
            exit_code=0,
            execution_time=0.05,
            command="python --version",
        )
        mock_provider = Mock()
        mock_provider.execute.return_value = expected

        executor = ToolExecutor()
        registry = Mock()
        registry.is_allowed.return_value = True
        registry.get_provider.return_value = mock_provider
        registry.requires_confirmation.return_value = False

        result = executor.run("python --version", registry)

        assert result.success is True
        assert result.stdout == "Python 3.10.0"
        assert result.exit_code == 0
        mock_provider.execute.assert_called_once()

    def test_run_failed_command(self):
        """Test failed command execution via a mock provider."""
        expected = ToolResult(
            success=False,
            stdout="",
            stderr="Error: command not found",
            exit_code=1,
            execution_time=0.01,
            command="badcmd",
        )
        mock_provider = Mock()
        mock_provider.execute.return_value = expected

        executor = ToolExecutor()
        registry = Mock()
        registry.is_allowed.return_value = True
        registry.get_provider.return_value = mock_provider
        registry.requires_confirmation.return_value = False

        result = executor.run("badcmd", registry)

        assert result.success is False
        assert result.stderr == "Error: command not found"
        assert result.exit_code == 1

    def test_run_tool_not_allowed(self):
        """Test running a tool that is not allowed."""
        executor = ToolExecutor()
        registry = Mock()
        registry.is_allowed.return_value = False
        registry.list_tools.return_value = []

        result = executor.run("rm -rf /", registry)

        assert result.success is False
        assert "not available or not allowed" in result.stderr
        assert result.exit_code == -1

    def test_run_invalid_command(self):
        """Test running an invalid command."""
        executor = ToolExecutor()
        registry = Mock()

        result = executor.run("", registry)

        assert result.success is False
        assert "Invalid command format" in result.stderr
        assert result.exit_code == -1

    def test_run_command_timeout(self):
        """Test command timeout is reported when a provider raises the error."""
        from pithos.tools.models import ToolResult

        timeout_result = ToolResult(
            success=False,
            stdout="",
            stderr="Command timed out after 1s",
            exit_code=-1,
            execution_time=1.0,
            command="sleep 10",
        )
        mock_provider = Mock()
        mock_provider.execute.return_value = timeout_result

        executor = ToolExecutor(timeout=1)
        registry = Mock()
        registry.is_allowed.return_value = True
        registry.get_provider.return_value = mock_provider
        registry.requires_confirmation.return_value = False

        result = executor.run("sleep 10", registry)

        assert result.success is False
        assert "timed out" in result.stderr
        assert result.exit_code == -1


class TestToolIntegration:
    """Integration tests for tool system."""

    @pytest.fixture
    def config_manager(self, tmp_path):
        """Create a ConfigManager with test configuration."""
        # Create test config directory
        config_dir = tmp_path / "configs" / "tools"
        config_dir.mkdir(parents=True)

        # Create test config file
        config_file = config_dir / "tool_config.yaml"
        config_content = """
enabled: true
timeout: 5
max_output_size: 1000
mode: include
include:
  - echo
  - python
descriptions:
  echo: "Display a line of text"
exclude: []
"""
        config_file.write_text(config_content)

        # Create ConfigManager pointing to test directory
        cm = ConfigManager(str(tmp_path / "configs"))
        return cm

    def test_registry_with_real_config(self, config_manager):
        """Test ToolRegistry with real configuration."""
        registry = ToolRegistry(config_manager)
        assert registry.config["enabled"] is True
        assert registry.config["timeout"] == 5
        assert "echo" in registry.config["include"]

    def test_end_to_end_tool_execution(self, config_manager):
        """Test complete tool execution flow."""
        # Create registry and register a mock provider
        registry = ToolRegistry(config_manager)
        registry.tools["echo"] = ToolMetadata(
            "echo", "/bin/echo", "Display a line of text", "unix", "system"
        )

        mock_provider = Mock()
        mock_provider.can_execute.side_effect = lambda name: name == "echo"
        mock_provider.execute.return_value = ToolResult(
            success=True,
            stdout="Hello World",
            stderr="",
            exit_code=0,
            execution_time=0.05,
            command="echo Hello World",
        )
        registry._providers.append(mock_provider)

        executor = ToolExecutor()

        # Execute command
        result = executor.run("echo Hello World", registry)

        # Verify
        assert result.success is True
        assert "Hello World" in result.stdout
        assert result.exit_code == 0


class TestToolExecutorConfirm:
    """Tests for ToolExecutor confirmation behaviour."""

    def _make_registry_with_confirm(self, tools: list[str], provider=None):
        """Return a mock registry where the given tools require confirmation."""
        registry = Mock()
        registry.is_allowed.side_effect = lambda name: name in tools
        registry.requires_confirmation.side_effect = lambda name: name in tools
        registry.list_tools.return_value = list(tools)
        if provider is None:
            default_provider = Mock()
            default_provider.execute.return_value = ToolResult(
                success=True,
                stdout="ok",
                stderr="",
                exit_code=0,
                execution_time=0.01,
                command="echo hello",
            )
            provider = default_provider
        registry.get_provider.return_value = provider
        return registry

    def test_confirm_approved_via_callback(self):
        """When callback returns True, the tool executes normally."""
        callback = Mock(return_value=True)
        executor = ToolExecutor(confirm_callback=callback)
        registry = self._make_registry_with_confirm(["echo"])

        result = executor.run("echo hello", registry)

        callback.assert_called_once_with("echo hello")
        assert result.success is True

    def test_confirm_denied_via_callback(self):
        """When callback returns False, execution is skipped and agent receives denial."""
        callback = Mock(return_value=False)
        executor = ToolExecutor(confirm_callback=callback)
        registry = self._make_registry_with_confirm(["bash"])

        result = executor.run("bash -c 'echo hi'", registry)

        callback.assert_called_once_with("bash -c 'echo hi'")
        assert result.success is False
        assert result.stdout == "Denied by user."
        assert result.exit_code == -1
        registry.get_provider.return_value.execute.assert_not_called()

    @patch("builtins.input", return_value="n")
    def test_confirm_denied_stdin_fallback(self, mock_input):
        """Without callback, 'n' via input() denies the tool call."""
        executor = ToolExecutor()  # no callback
        registry = self._make_registry_with_confirm(["bash"])

        with patch.dict("sys.modules", {"rich": None, "rich.prompt": None}):
            result = executor.run("bash -c 'ls'", registry)

        assert result.success is False
        assert result.stdout == "Denied by user."
        registry.get_provider.return_value.execute.assert_not_called()

    @patch("builtins.input", return_value="y")
    def test_confirm_approved_stdin_fallback(self, mock_input):
        """Without callback, 'y' via input() approves the tool call."""
        executor = ToolExecutor()
        registry = self._make_registry_with_confirm(["echo"])

        with patch.dict("sys.modules", {"rich": None, "rich.prompt": None}):
            result = executor.run("echo ok", registry)

        assert result.success is True

    def test_no_confirmation_for_non_confirm_mode(self):
        """Tools that don't require confirmation execute without prompting."""
        mock_provider = Mock()
        mock_provider.execute.return_value = ToolResult(
            success=True,
            stdout="ok",
            stderr="",
            exit_code=0,
            execution_time=0.01,
            command="echo ok",
        )

        callback = Mock(return_value=False)
        executor = ToolExecutor(confirm_callback=callback)

        registry = Mock()
        registry.is_allowed.return_value = True
        registry.requires_confirmation.return_value = False
        registry.get_provider.return_value = mock_provider

        result = executor.run("echo ok", registry)

        callback.assert_not_called()
        assert result.success is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
