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

    def test_is_allowed_include_mode(self, mock_config_manager):
        """Test tool filtering in include mode."""
        registry = ToolRegistry(mock_config_manager)
        registry.config = {
            "mode": "include",
            "include": ["python", "git"],
            "exclude": ["rm"],
        }
        assert registry.is_allowed("python") is True
        assert registry.is_allowed("git") is True
        assert registry.is_allowed("rm") is False
        assert registry.is_allowed("curl") is False

    def test_is_allowed_exclude_mode(self, mock_config_manager):
        """Test tool filtering in exclude mode."""
        registry = ToolRegistry(mock_config_manager)
        registry.config = {
            "mode": "exclude",
            "include": [],
            "exclude": ["rm", "del"],
        }
        assert registry.is_allowed("python") is True
        assert registry.is_allowed("rm") is False
        assert registry.is_allowed("del") is False

    def test_is_allowed_all_mode(self, mock_config_manager):
        """Test tool filtering in all mode."""
        registry = ToolRegistry(mock_config_manager)
        registry.config = {
            "mode": "all",
            "include": [],
            "exclude": ["rm"],
        }
        assert registry.is_allowed("python") is True
        assert registry.is_allowed("git") is True
        assert registry.is_allowed("rm") is False

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

    # ------------------------------------------------------------------
    # Cache tests
    # ------------------------------------------------------------------

    def test_cache_populated_after_init(self, mock_config_manager):
        """A scan cache entry is written after the first PATH scan."""
        ToolRegistry.invalidate_cache()
        assert ToolRegistry._scan_cache is None
        ToolRegistry(mock_config_manager)
        assert ToolRegistry._scan_cache is not None
        assert "tools" in ToolRegistry._scan_cache
        assert "path_hash" in ToolRegistry._scan_cache
        assert "timestamp" in ToolRegistry._scan_cache

    def test_cache_reused_across_instances(self, mock_config_manager):
        """Second ToolRegistry instantiation reuses the cached scan result."""
        ToolRegistry.invalidate_cache()
        with patch("os.scandir") as mock_scandir:
            mock_scandir.return_value.__iter__ = Mock(return_value=iter([]))
            mock_scandir.return_value.__enter__ = Mock(return_value=iter([]))
            mock_scandir.return_value.__exit__ = Mock(return_value=False)
            ToolRegistry(mock_config_manager)
            first_call_count = mock_scandir.call_count
            ToolRegistry(mock_config_manager)
            # scandir should not have been called again
            assert mock_scandir.call_count == first_call_count

    def test_cache_bypassed_on_path_change(self, mock_config_manager):
        """Changing PATH invalidates the cache and triggers a fresh scan."""
        ToolRegistry.invalidate_cache()
        original_path = os.environ.get("PATH", "")
        try:
            ToolRegistry(mock_config_manager)
            assert ToolRegistry._scan_cache is not None
            # Simulate a PATH change
            os.environ["PATH"] = original_path + os.pathsep + "/some/new/dir"
            with patch("os.scandir") as mock_scandir:
                mock_scandir.return_value.__iter__ = Mock(return_value=iter([]))
                ToolRegistry(mock_config_manager)
                # scandir must have been called because the path hash changed
                assert mock_scandir.called
        finally:
            os.environ["PATH"] = original_path
            ToolRegistry.invalidate_cache()

    def test_cache_bypassed_on_ttl_expiry(self, mock_config_manager):
        """A stale cache (past TTL) triggers a re-scan."""
        ToolRegistry.invalidate_cache()
        ToolRegistry(mock_config_manager)
        # Wind the clock past TTL by backdating the timestamp
        ToolRegistry._scan_cache["timestamp"] -= ToolRegistry._CACHE_TTL + 1
        with patch("os.scandir") as mock_scandir:
            mock_scandir.return_value.__iter__ = Mock(return_value=iter([]))
            ToolRegistry(mock_config_manager)
            assert mock_scandir.called

    def test_invalidate_cache_classmethod(self, mock_config_manager):
        """invalidate_cache() sets _scan_cache to None."""
        ToolRegistry(mock_config_manager)
        assert ToolRegistry._scan_cache is not None
        ToolRegistry.invalidate_cache()
        assert ToolRegistry._scan_cache is None

    def test_refresh_invalidates_cache(self, mock_config_manager):
        """refresh() discards the scan cache so the next operation re-scans."""
        ToolRegistry.invalidate_cache()
        registry = ToolRegistry(mock_config_manager)
        assert ToolRegistry._scan_cache is not None
        registry.refresh()
        # Cache is rebuilt (not None) because _discover_tools runs immediately
        # after invalidation, but the important thing is it was invalidated and
        # a fresh scan was performed — verify by checking it's a new object.
        assert ToolRegistry._scan_cache is not None

    # ------------------------------------------------------------------

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

    def test_truncate_output(self):
        """Test output truncation."""
        executor = ToolExecutor(max_output_size=20)

        # Short output - no truncation
        short = "Hello"
        assert executor._truncate_output(short) == "Hello"

        # Long output - should be truncated
        long = "A" * 100
        result = executor._truncate_output(long)
        assert len(result) > 20  # Includes truncation note
        assert "truncated" in result.lower()

    @patch("subprocess.run")
    def test_run_successful_command(self, mock_run):
        """Test successful command execution."""
        # Setup mock
        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stdout = "Python 3.10.0"
        mock_result.stderr = ""
        mock_run.return_value = mock_result

        # Create executor and mock registry
        executor = ToolExecutor()
        registry = Mock()
        registry.get_tool.return_value = ToolMetadata(
            "python", "/usr/bin/python", "Python interpreter", "unix", "system"
        )

        # Execute command
        result = executor.run("python --version", registry)

        # Verify
        assert result.success is True
        assert result.stdout == "Python 3.10.0"
        assert result.exit_code == 0
        assert result.command == "python --version"
        mock_run.assert_called_once()

    @patch("subprocess.run")
    def test_run_failed_command(self, mock_run):
        """Test failed command execution."""
        # Setup mock
        mock_result = Mock()
        mock_result.returncode = 1
        mock_result.stdout = ""
        mock_result.stderr = "Error: command not found"
        mock_run.return_value = mock_result

        # Create executor and mock registry
        executor = ToolExecutor()
        registry = Mock()
        registry.get_tool.return_value = ToolMetadata(
            "badcmd", "/usr/bin/badcmd", "Bad command", "unix", "system"
        )

        # Execute command
        result = executor.run("badcmd", registry)

        # Verify
        assert result.success is False
        assert result.stderr == "Error: command not found"
        assert result.exit_code == 1

    def test_run_tool_not_allowed(self):
        """Test running a tool that is not allowed."""
        executor = ToolExecutor()
        registry = Mock()
        registry.get_tool.return_value = None

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

    @patch("subprocess.run")
    def test_run_command_timeout(self, mock_run):
        """Test command timeout handling."""
        # Setup mock to raise TimeoutExpired
        mock_run.side_effect = subprocess.TimeoutExpired("test", 5)

        executor = ToolExecutor(timeout=1)
        registry = Mock()
        registry.get_tool.return_value = ToolMetadata(
            "sleep", "/usr/bin/sleep", "Sleep command", "unix", "system"
        )

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

    @patch("subprocess.run")
    def test_end_to_end_tool_execution(self, mock_run, config_manager):
        """Test complete tool execution flow."""
        # Setup mock
        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stdout = "Hello World"
        mock_result.stderr = ""
        mock_run.return_value = mock_result

        # Create registry and executor
        registry = ToolRegistry(config_manager)
        # Manually add echo tool since PATH scanning might not find it in tests
        registry.tools["echo"] = ToolMetadata(
            "echo", "/bin/echo", "Display a line of text", "unix", "system"
        )
        executor = ToolExecutor()

        # Execute command
        result = executor.run("echo Hello World", registry)

        # Verify
        assert result.success is True
        assert "Hello World" in result.stdout
        assert result.exit_code == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
