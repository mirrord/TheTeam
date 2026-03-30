"""Tool registry for discovering, storing, and filtering CLI tools."""

import hashlib
import os
import platform
import subprocess
import shutil
import time
from typing import Any, Optional

from ..config_manager import ConfigManager
from .models import ToolMetadata


class ToolRegistry:
    """Registry for discovering, storing, and filtering CLI tools."""

    # Class-level cache shared across all instances; PATH is a process-global resource.
    _scan_cache: Optional[dict[str, Any]] = None
    # How long (seconds) a cache entry stays valid before a re-scan is forced.
    _CACHE_TTL: float = 300.0

    def __init__(self, config_manager: ConfigManager):
        """Initialize tool registry.

        Args:
            config_manager: Configuration manager for loading tool configs.

        Raises:
            ValueError: If config_manager is None.
        """
        if config_manager is None:
            raise ValueError("config_manager cannot be None")

        self.config_manager = config_manager
        self.tools: dict[str, ToolMetadata] = {}
        self.config = self._load_config()
        self._discover_tools()

    def _load_config(self) -> dict[str, Any]:
        """Load tool configuration."""
        config = self.config_manager.get_config("tool_config", "tools")
        if not config:
            # Default configuration
            return {
                "enabled": True,
                "timeout": 30,
                "max_output_size": 10000,
                "mode": "include",
                "include": [
                    "python",
                    "pip",
                    "git",
                    "curl",
                    "wget",
                    "node",
                    "npm",
                    "echo",
                    "cat",
                    "ls",
                    "dir",
                    "pwd",
                    "cd",
                ],
                "exclude": ["rm", "del", "format", "shutdown", "reboot", "kill"],
                "descriptions": {},
            }
        return config

    def _get_platform(self) -> str:
        """Get current platform identifier."""
        system = platform.system().lower()
        if system == "windows":
            return "windows"
        elif system in ["linux", "darwin"]:
            return "unix"
        return "unknown"

    # ------------------------------------------------------------------
    # Cache helpers
    # ------------------------------------------------------------------

    def _get_path_hash(self) -> str:
        """Return an MD5 digest of the current PATH environment variable."""
        path_env = os.environ.get("PATH", "")
        return hashlib.md5(path_env.encode(), usedforsecurity=False).hexdigest()

    def _is_cache_valid(self) -> bool:
        """Return True when the scan cache exists and is still fresh."""
        cache = ToolRegistry._scan_cache
        if cache is None:
            return False
        # Invalidate on TTL expiry.
        if time.monotonic() - cache["timestamp"] > self._CACHE_TTL:
            return False
        # Invalidate when PATH has changed since the last scan.
        if cache["path_hash"] != self._get_path_hash():
            return False
        return True

    @classmethod
    def invalidate_cache(cls) -> None:
        """Explicitly discard the scan cache, forcing a fresh PATH scan next time."""
        cls._scan_cache = None

    def _discover_tools(self) -> None:
        """Discover CLI tools from system PATH."""
        # Get manual descriptions
        manual_descriptions = self.config.get("descriptions", {})

        # Get tools from PATH
        discovered = self._scan_path()

        # Apply filters and create ToolMetadata objects
        for tool_name, tool_path in discovered.items():
            if self.is_allowed(tool_name):
                description = self._get_tool_description(
                    tool_name, tool_path, manual_descriptions
                )
                self.tools[tool_name] = ToolMetadata(
                    name=tool_name,
                    path=tool_path,
                    description=description,
                    platform=self._get_platform(),
                    source="system",
                )

        # Add manual tools from descriptions that weren't discovered
        for tool_name, description in manual_descriptions.items():
            if tool_name not in self.tools and self.is_allowed(tool_name):
                # Try to find it
                tool_path = shutil.which(tool_name)
                if tool_path:
                    self.tools[tool_name] = ToolMetadata(
                        name=tool_name,
                        path=tool_path,
                        description=description,
                        platform=self._get_platform(),
                        source="manual",
                    )

        # Discover flowcharts as virtual tools
        self._discover_flowcharts(manual_descriptions)

    def _discover_flowcharts(self, manual_descriptions: dict[str, str]) -> None:
        """Register each available flowchart as a virtual tool.

        The ``flowchart`` entry in the config include-list gates this
        feature. Individual flowcharts are registered as
        ``flowchart:<name>`` so agents can call
        ``RUN: flowchart <name> <input>``.
        """
        fc_config = self.config.get("flowcharts", {})
        if not fc_config.get("enabled", False):
            return
        if not self.is_allowed("flowchart"):
            return

        # Register the dispatcher tool itself
        fc_description = manual_descriptions.get(
            "flowchart",
            "Run a registered pithos flowchart. Usage: flowchart <name> [input text]",
        )
        self.tools["flowchart"] = ToolMetadata(
            name="flowchart",
            path="",
            description=fc_description,
            platform="cross-platform",
            source="virtual",
            tool_type="flowchart",
        )

        # Register each individual flowchart for discoverability
        try:
            for name in self.config_manager.get_registered_flowchart_names():
                key = f"flowchart:{name}"
                self.tools[key] = ToolMetadata(
                    name=key,
                    path="",
                    description=f"Flowchart workflow '{name}'",
                    platform="cross-platform",
                    source="flowchart",
                    tool_type="flowchart",
                )
        except Exception:
            pass  # graceful degradation if no flowcharts directory

    def _scan_path(self) -> dict[str, str]:
        """Scan system PATH for executable tools.

        Results are cached at the class level and reused across instances until
        the cache is invalidated (PATH change, TTL expiry, or explicit call to
        :meth:`invalidate_cache`).

        Returns:
            Dictionary mapping tool name to full path.
        """
        if self._is_cache_valid():
            # Return a shallow copy so callers cannot mutate the cached dict.
            return dict(ToolRegistry._scan_cache["tools"])  # type: ignore[index]

        tools = {}

        # Get PATH directories
        path_env = os.environ.get("PATH", "")
        path_dirs = path_env.split(os.pathsep)

        # Common additional directories based on platform
        sys_platform = self._get_platform()
        if sys_platform == "windows":
            additional_dirs = [
                r"C:\Windows\System32",
                r"C:\Windows",
                r"C:\Program Files\Git\cmd",
            ]
        else:
            additional_dirs = ["/usr/bin", "/usr/local/bin", "/bin"]

        all_dirs = path_dirs + additional_dirs

        # Scan directories
        for dir_path in all_dirs:
            if not os.path.isdir(dir_path):
                continue

            try:
                for entry in os.scandir(dir_path):
                    if entry.is_file() and os.access(entry.path, os.X_OK):
                        # Get base name without extension
                        name = entry.name
                        if sys_platform == "windows":
                            # Remove common Windows executable extensions
                            for ext in [".exe", ".bat", ".cmd", ".ps1"]:
                                if name.lower().endswith(ext):
                                    name = name[: -len(ext)]
                                    break

                        # Add to tools if not already present (first occurrence wins)
                        if name not in tools:
                            tools[name] = entry.path
            except (PermissionError, OSError):
                # Skip directories we can't read
                continue

        # Persist results so subsequent calls (from this or other instances)
        # skip the expensive directory walk.
        ToolRegistry._scan_cache = {
            "tools": tools,
            "path_hash": self._get_path_hash(),
            "timestamp": time.monotonic(),
        }
        return tools

    def _get_tool_description(
        self, tool_name: str, tool_path: str, manual_descriptions: dict[str, str]
    ) -> str:
        """Get description for a tool.

        Priority order:
        1. Manual description from config
        2. Parse from --help output
        3. Parse from --version output
        4. Fallback generic description

        Args:
            tool_name: Name of the tool.
            tool_path: Full path to tool executable.
            manual_descriptions: Dictionary of manual descriptions.

        Returns:
            Tool description string.
        """
        # 1. Check manual descriptions
        if tool_name in manual_descriptions:
            return manual_descriptions[tool_name]

        # 2. Try to get description from --help
        description = self._extract_description_from_help(tool_path)
        if description:
            return description

        # 3. Try --version
        description = self._extract_description_from_version(tool_path)
        if description:
            return description

        # 4. Fallback
        return f"Command-line utility: {tool_name}"

    def _extract_description_from_help(self, tool_path: str) -> Optional[str]:
        """Try to extract description from tool --help output."""
        try:
            result = subprocess.run(
                [tool_path, "--help"],
                capture_output=True,
                text=True,
                timeout=2,
                errors="ignore",
            )
            if result.returncode == 0 and result.stdout:
                # Get first non-empty line that looks like a description
                lines = result.stdout.strip().split("\n")
                for line in lines[:5]:  # Check first 5 lines
                    line = line.strip()
                    if line and len(line) > 10 and not line.startswith("-"):
                        # Clean up and truncate
                        return line[:100]
        except (subprocess.TimeoutExpired, OSError, FileNotFoundError):
            pass
        return None

    def _extract_description_from_version(self, tool_path: str) -> Optional[str]:
        """Try to extract description from tool --version output."""
        try:
            result = subprocess.run(
                [tool_path, "--version"],
                capture_output=True,
                text=True,
                timeout=2,
                errors="ignore",
            )
            if result.returncode == 0 and result.stdout:
                # Get first line
                first_line = result.stdout.strip().split("\n")[0]
                if first_line and len(first_line) > 5:
                    return first_line[:100]
        except (subprocess.TimeoutExpired, OSError, FileNotFoundError):
            pass
        return None

    def is_allowed(self, tool_name: str) -> bool:
        """Check if a tool is allowed based on configuration.

        Args:
            tool_name: Name of the tool to check.

        Returns:
            True if tool is allowed, False otherwise.

        Raises:
            ValueError: If tool_name is empty.
        """
        if not tool_name or not tool_name.strip():
            raise ValueError("tool_name cannot be empty")

        mode = self.config.get("mode", "include")
        include_list = self.config.get("include", [])
        exclude_list = self.config.get("exclude", [])

        if mode == "all":
            # Allow all except those in exclude list
            return tool_name not in exclude_list
        elif mode == "include":
            # Only allow tools in include list
            return tool_name in include_list
        elif mode == "exclude":
            # Allow all except those in exclude list
            return tool_name not in exclude_list
        else:
            # Unknown mode, default to restrictive
            return False

    def get_tool(self, name: str) -> Optional[ToolMetadata]:
        """Get metadata for a specific tool.

        Args:
            name: Tool name.

        Returns:
            ToolMetadata if found, None otherwise.
        """
        return self.tools.get(name)

    def list_tools(self) -> list[str]:
        """Get list of all available tool names.

        Returns:
            Sorted list of tool names.
        """
        return sorted(self.tools.keys())

    def refresh(self) -> None:
        """Refresh tool discovery and reload configuration.

        Invalidates the shared PATH scan cache so the next discovery performs a
        full re-scan rather than returning stale results.
        """
        self.invalidate_cache()
        self.tools.clear()
        self.config = self._load_config()
        self._discover_tools()

    def get_tool_list_text(self) -> str:
        """Get formatted list of tools for system prompt.

        Returns:
            Formatted string listing available tools grouped by type.
        """
        if not self.tools:
            return "No tools available."

        cli_lines: list[str] = []
        flowchart_lines: list[str] = []

        for tool_name in sorted(self.tools.keys()):
            tool = self.tools[tool_name]
            if tool.tool_type == "flowchart":
                # Skip the individual flowchart:* entries in the summary;
                # only list the dispatcher + available names.
                if not tool_name.startswith("flowchart:"):
                    flowchart_lines.append(f"  - {tool_name}: {tool.description}")
                else:
                    short = tool_name.removeprefix("flowchart:")
                    flowchart_lines.append(f"      {short}")
            else:
                cli_lines.append(f"  - {tool_name}: {tool.description}")

        sections = []
        if cli_lines:
            sections.append("CLI tools:\n" + "\n".join(cli_lines))
        if flowchart_lines:
            sections.append(
                "Flowchart tools (use: flowchart <name> [input]):\n"
                + "\n".join(flowchart_lines)
            )

        return "\n\n".join(sections)
