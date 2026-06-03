"""CLI tool provider — discovers executables from the system PATH and runs them.

This provider owns the full lifecycle for system CLI tools:
- discovery via a cached PATH scan (class-level cache shared across instances)
- allowlist/blocklist filtering sourced from the tool_config
- optional heuristic safety analysis via
  :class:`~pithos.tools.safety.CommandSafetyChecker`
- subprocess execution with timeout and output truncation

Safety/subprocess logic was previously split between
:class:`~pithos.tools.registry.ToolRegistry` (discovery) and
:class:`~pithos.tools.executor.ToolExecutor` (execution).  It is unified here
so that the registry and executor become generic coordinators.
"""

import hashlib
import os
import platform
import shlex
import shutil
import subprocess
import time
from typing import Any, Optional

from .models import RiskLevel, ToolMetadata, ToolResult
from .provider import ToolProvider


class CLIToolProvider(ToolProvider):
    """Discovers and executes system PATH executables as pithos tools.

    Attributes:
        config: Tool configuration dict (from ``tool_config.yaml`` / defaults).
        timeout: Per-command subprocess timeout in seconds.
        max_output_size: Byte cap on captured stdout/stderr.
        safety_checker: Optional
            :class:`~pithos.tools.safety.CommandSafetyChecker`.  When set,
            BLOCK verdicts deny execution outright; REVIEW verdicts are flagged
            on the returned :class:`~pithos.tools.models.ToolResult` but do
            **not** prompt for confirmation here — confirmation is the
            :class:`~pithos.tools.executor.ToolExecutor`'s responsibility.
    """

    # Class-level cache shared across all instances; PATH is a process-global
    # resource so scanning it once per TTL is sufficient.
    _scan_cache: Optional[dict] = None
    _CACHE_TTL: float = 300.0

    def __init__(
        self,
        config: dict[str, Any],
        timeout: int = 30,
        max_output_size: int = 10000,
        safety_checker=None,
    ) -> None:
        """Initialise the provider.

        Args:
            config: Tool config dict (keys: mode, include, exclude, confirm,
                timeout, max_output_size, descriptions).
            timeout: Subprocess timeout in seconds.
            max_output_size: Maximum bytes to keep from stdout/stderr.
            safety_checker: Optional
                :class:`~pithos.tools.safety.CommandSafetyChecker` instance.
        """
        self.config = config
        self.timeout = config.get("timeout", timeout)
        self.max_output_size = config.get("max_output_size", max_output_size)
        self.safety_checker = safety_checker
        self._platform = self._get_platform()
        self._discovered: dict[str, ToolMetadata] = {}

    # ------------------------------------------------------------------
    # ToolProvider interface
    # ------------------------------------------------------------------

    def discover(self) -> dict[str, ToolMetadata]:
        """Scan the system PATH and return allowed CLI tools as metadata.

        Results are cached at the class level; the cache is invalidated when
        PATH changes or :attr:`_CACHE_TTL` seconds have elapsed.

        Returns:
            Mapping of tool name → :class:`~pithos.tools.models.ToolMetadata`.
        """
        manual_descriptions = self.config.get("descriptions", {})
        raw = self._scan_path()
        tools: dict[str, ToolMetadata] = {}

        for name, path in raw.items():
            if self._is_allowed(name):
                desc = self._get_tool_description(name, path, manual_descriptions)
                tools[name] = ToolMetadata(
                    name=name,
                    path=path,
                    description=desc,
                    platform=self._platform,
                    source="system",
                    tool_type="cli",
                )

        # Add manually-described tools not found via scan.
        for name, desc in manual_descriptions.items():
            if name not in tools and self._is_allowed(name):
                path = shutil.which(name)
                if path:
                    tools[name] = ToolMetadata(
                        name=name,
                        path=path,
                        description=desc,
                        platform=self._platform,
                        source="manual",
                        tool_type="cli",
                    )

        self._discovered = tools
        return dict(tools)

    def can_execute(self, tool_name: str) -> bool:
        """Return True when *tool_name* was discovered as a CLI tool.

        Args:
            tool_name: Leading token of the command string.
        """
        return tool_name in self._discovered

    def execute(
        self,
        command: str,
        context: Optional[dict[str, Any]] = None,
    ) -> ToolResult:
        """Run a CLI command via subprocess.

        Safety analysis is applied before execution:
        - BLOCK verdict → return a failed :class:`~pithos.tools.models.ToolResult`
          immediately (no subprocess).
        - REVIEW verdict → set ``safety_verdict`` on the result so the
          :class:`~pithos.tools.executor.ToolExecutor` can prompt for
          confirmation; execution itself is still attempted here if the
          executor already approved it.  The executor must check
          ``safety_verdict`` **before** calling this method when it needs to
          interrupt for confirmation — the split is deliberate: CLIToolProvider
          does the analysis; ToolExecutor owns user interaction.

        Args:
            command: Full command string (e.g. ``"python --version"``).
            context: Unused for CLI tools; accepted for interface consistency.

        Returns:
            :class:`~pithos.tools.models.ToolResult` with stdout/stderr/timing.
        """
        start = time.time()

        tool_name, args = self._parse_command(command)
        if not tool_name:
            return ToolResult(
                success=False,
                stdout="",
                stderr=f"Invalid command format: '{command}'",
                exit_code=-1,
                execution_time=0.0,
                command=command,
                error_hint=(
                    "Command should be in format: toolname [arguments]\n"
                    "Example: python --version"
                ),
            )

        tool_meta = self._discovered.get(tool_name)
        if not tool_meta:
            available = sorted(self._discovered.keys())[:10]
            hint = (
                f"Tool '{tool_name}' not found or not allowed.\n"
                f"Available tools include: {', '.join(available)}...\n"
                "Use exact tool names from the available list."
            )
            return ToolResult(
                success=False,
                stdout="",
                stderr=f"Tool '{tool_name}' is not available or not allowed",
                exit_code=-1,
                execution_time=0.0,
                command=command,
                error_hint=hint,
            )

        # Safety analysis.
        safety_verdict = None
        if self.safety_checker is not None:
            safety_verdict = self.safety_checker.check(command)
            if safety_verdict.level == RiskLevel.BLOCK:
                return ToolResult(
                    success=False,
                    stdout="",
                    stderr=f"Command blocked by safety checker: {safety_verdict.reason}",
                    exit_code=-1,
                    execution_time=0.0,
                    command=command,
                    error_hint=(
                        "This command was blocked because it contains a pattern considered "
                        "too dangerous to execute. Rephrase your request without using "
                        "shell injection operators or known-destructive commands."
                    ),
                    safety_verdict=safety_verdict,
                )

        # Execute via subprocess.
        try:
            result = subprocess.run(
                [tool_meta.path] + args,
                capture_output=True,
                text=True,
                timeout=self.timeout,
                errors="ignore",
            )
            execution_time = time.time() - start

            stdout = self._truncate_output(result.stdout)
            stderr = self._truncate_output(result.stderr)
            error_hint = (
                f"Command exited with code {result.returncode}. Check stderr for details."
                if result.returncode != 0
                else None
            )

            return ToolResult(
                success=result.returncode == 0,
                stdout=stdout,
                stderr=stderr,
                exit_code=result.returncode,
                execution_time=execution_time,
                command=command,
                error_hint=error_hint,
                safety_verdict=safety_verdict,
            )

        except subprocess.TimeoutExpired:
            return ToolResult(
                success=False,
                stdout="",
                stderr=f"Command timed out after {self.timeout} seconds",
                exit_code=-1,
                execution_time=time.time() - start,
                command=command,
                error_hint="Command took too long. Try simplifying or use a faster operation.",
            )
        except (OSError, FileNotFoundError) as exc:
            return ToolResult(
                success=False,
                stdout="",
                stderr=f"Failed to execute command: {exc}",
                exit_code=-1,
                execution_time=time.time() - start,
                command=command,
                error_hint="Tool path may be invalid or tool may not be installed properly.",
            )

    # ------------------------------------------------------------------
    # Cache helpers
    # ------------------------------------------------------------------

    @classmethod
    def invalidate_cache(cls) -> None:
        """Discard the shared PATH scan cache, forcing a re-scan next time."""
        cls._scan_cache = None

    def _get_path_hash(self) -> str:
        path_env = os.environ.get("PATH", "")
        return hashlib.md5(path_env.encode(), usedforsecurity=False).hexdigest()

    def _is_cache_valid(self) -> bool:
        cache = CLIToolProvider._scan_cache
        if cache is None:
            return False
        if time.monotonic() - cache["timestamp"] > self._CACHE_TTL:
            return False
        if cache["path_hash"] != self._get_path_hash():
            return False
        return True

    # ------------------------------------------------------------------
    # PATH scanning
    # ------------------------------------------------------------------

    def _scan_path(self) -> dict[str, str]:
        """Scan PATH directories for executable files.

        Returns a ``{name: path}`` dict.  Results are cached class-wide.
        """
        if self._is_cache_valid():
            return dict(CLIToolProvider._scan_cache["tools"])  # type: ignore[index]

        tools: dict[str, str] = {}
        path_dirs = os.environ.get("PATH", "").split(os.pathsep)

        if self._platform == "windows":
            additional = [
                r"C:\Windows\System32",
                r"C:\Windows",
                r"C:\Program Files\Git\cmd",
            ]
        else:
            additional = ["/usr/bin", "/usr/local/bin", "/bin"]

        for dir_path in path_dirs + additional:
            if not os.path.isdir(dir_path):
                continue
            try:
                for entry in os.scandir(dir_path):
                    if entry.is_file() and os.access(entry.path, os.X_OK):
                        name = entry.name
                        if self._platform == "windows":
                            for ext in (".exe", ".bat", ".cmd", ".ps1"):
                                if name.lower().endswith(ext):
                                    name = name[: -len(ext)]
                                    break
                        if name not in tools:
                            tools[name] = entry.path
            except (PermissionError, OSError):
                continue

        CLIToolProvider._scan_cache = {
            "tools": tools,
            "path_hash": self._get_path_hash(),
            "timestamp": time.monotonic(),
        }
        return tools

    # ------------------------------------------------------------------
    # Filtering helpers
    # ------------------------------------------------------------------

    def _is_allowed(self, tool_name: str) -> bool:
        """Apply include/exclude/all/confirm mode filtering."""
        mode = self.config.get("mode", "include")
        include_list = self.config.get("include", [])
        exclude_list = self.config.get("exclude", [])
        confirm_list = self.config.get("confirm", [])

        if mode == "all":
            return tool_name not in exclude_list
        elif mode == "include":
            return tool_name in include_list
        elif mode == "exclude":
            return tool_name not in exclude_list
        elif mode == "confirm":
            return tool_name in confirm_list
        return False

    # ------------------------------------------------------------------
    # Description helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _get_platform() -> str:
        system = platform.system().lower()
        if system == "windows":
            return "windows"
        elif system in ("linux", "darwin"):
            return "unix"
        return "unknown"

    def _get_tool_description(
        self,
        tool_name: str,
        tool_path: str,
        manual_descriptions: dict[str, str],
    ) -> str:
        if tool_name in manual_descriptions:
            return manual_descriptions[tool_name]
        desc = self._extract_description_from_help(tool_path)
        if desc:
            return desc
        desc = self._extract_description_from_version(tool_path)
        if desc:
            return desc
        return f"Command-line utility: {tool_name}"

    @staticmethod
    def _extract_description_from_help(tool_path: str) -> Optional[str]:
        try:
            result = subprocess.run(
                [tool_path, "--help"],
                capture_output=True,
                text=True,
                timeout=2,
                errors="ignore",
            )
            if result.returncode == 0 and result.stdout:
                for line in result.stdout.strip().split("\n")[:5]:
                    line = line.strip()
                    if line and len(line) > 10 and not line.startswith("-"):
                        return line[:100]
        except (subprocess.TimeoutExpired, OSError, FileNotFoundError):
            pass
        return None

    @staticmethod
    def _extract_description_from_version(tool_path: str) -> Optional[str]:
        try:
            result = subprocess.run(
                [tool_path, "--version"],
                capture_output=True,
                text=True,
                timeout=2,
                errors="ignore",
            )
            if result.returncode == 0 and result.stdout:
                first_line = result.stdout.strip().split("\n")[0]
                if first_line and len(first_line) > 5:
                    return first_line[:100]
        except (subprocess.TimeoutExpired, OSError, FileNotFoundError):
            pass
        return None

    # ------------------------------------------------------------------
    # Command parsing & output utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_command(command: str) -> tuple[Optional[str], list[str]]:
        command = command.strip()
        if not command:
            return None, []
        try:
            parts = shlex.split(command)
            if not parts:
                return None, []
            return parts[0], parts[1:]
        except ValueError:
            parts = command.split()
            return (parts[0], parts[1:]) if parts else (None, [])

    def _truncate_output(self, output: str) -> str:
        if len(output) <= self.max_output_size:
            return output
        return (
            output[: self.max_output_size]
            + f"\n\n[Output truncated — exceeded {self.max_output_size} bytes]"
        )
