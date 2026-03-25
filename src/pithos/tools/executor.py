"""Tool executor for running CLI tools safely with timeout and output capture."""

import platform
import subprocess
import time
from typing import Optional

from .models import ToolResult
from .registry import ToolRegistry


class ToolExecutor:
    """Executes CLI tools safely with timeout and output capture."""

    def __init__(self, timeout: int = 30, max_output_size: int = 10000):
        """Initialize tool executor.

        Args:
            timeout: Maximum execution time in seconds.
            max_output_size: Maximum size of captured output in bytes.

        Raises:
            ValueError: If timeout or max_output_size is invalid.
        """
        if timeout <= 0:
            raise ValueError("timeout must be > 0")
        if max_output_size <= 0:
            raise ValueError("max_output_size must be > 0")

        self.timeout = timeout
        self.max_output_size = max_output_size
        self.platform = platform.system().lower()

    def run(self, command: str, tool_registry: ToolRegistry) -> ToolResult:
        """Execute a command safely with detailed error feedback.

        Args:
            command: Command string to execute (e.g., "python --version").
            tool_registry: ToolRegistry for validation.

        Returns:
            ToolResult with execution details and error hints.
        """
        start_time = time.time()

        # Parse and validate command
        tool_name, args = self._parse_command(command)
        if not tool_name:
            return ToolResult(
                success=False,
                stdout="",
                stderr=f"Invalid command format: '{command}'",
                exit_code=-1,
                execution_time=0.0,
                command=command,
                error_hint="Command should be in format: toolname [arguments]\nExample: python --version",
            )

        # Validate tool is allowed
        tool_meta = tool_registry.get_tool(tool_name)
        if not tool_meta:
            # Try to get available tools for hint
            try:
                available = tool_registry.list_tools()
                available_tools = (
                    ", ".join(available[:10]) if available else "No tools available"
                )
            except (AttributeError, TypeError):
                available_tools = "Unable to list available tools"

            hint = f"Tool '{tool_name}' not found or not allowed.\n"
            hint += f"Available tools include: {available_tools}...\n"
            hint += "Use exact tool names from the available list."

            return ToolResult(
                success=False,
                stdout="",
                stderr=f"Tool '{tool_name}' is not available or not allowed",
                exit_code=-1,
                execution_time=0.0,
                command=command,
                error_hint=hint,
            )

        # Execute command
        try:
            result = subprocess.run(
                [tool_meta.path] + args,
                capture_output=True,
                text=True,
                timeout=self.timeout,
                errors="ignore",
            )

            execution_time = time.time() - start_time

            # Truncate output if necessary
            stdout = self._truncate_output(result.stdout)
            stderr = self._truncate_output(result.stderr)

            # Add hint for non-zero exit codes
            error_hint = None
            if result.returncode != 0:
                error_hint = f"Command exited with code {result.returncode}. Check stderr for details."

            return ToolResult(
                success=result.returncode == 0,
                stdout=stdout,
                stderr=stderr,
                exit_code=result.returncode,
                execution_time=execution_time,
                command=command,
                error_hint=error_hint,
            )

        except subprocess.TimeoutExpired:
            execution_time = time.time() - start_time
            return ToolResult(
                success=False,
                stdout="",
                stderr=f"Command timed out after {self.timeout} seconds",
                exit_code=-1,
                execution_time=execution_time,
                command=command,
                error_hint="Command took too long. Try simplifying or use a faster operation.",
            )
        except (OSError, FileNotFoundError) as e:
            execution_time = time.time() - start_time
            return ToolResult(
                success=False,
                stdout="",
                stderr=f"Failed to execute command: {str(e)}",
                exit_code=-1,
                execution_time=execution_time,
                command=command,
                error_hint="Tool path may be invalid or tool may not be installed properly.",
            )

    def _parse_command(self, command: str) -> tuple[str | None, list[str]]:
        """Parse command string into tool name and arguments.

        Args:
            command: Command string (e.g., "python --version").

        Returns:
            Tuple of (tool_name, args_list) or (None, []) if invalid.
        """
        command = command.strip()
        if not command:
            return None, []

        # Split command respecting quotes
        try:
            parts = self._split_args(command)
            if not parts:
                return None, []
            return parts[0], parts[1:]
        except ValueError:
            return None, []

    def _split_args(self, command: str) -> list[str]:
        """Split command string into arguments, respecting quotes.

        Args:
            command: Command string.

        Returns:
            List of argument strings.
        """
        import shlex

        try:
            return shlex.split(command)
        except ValueError:
            # If shlex fails, fall back to simple split
            return command.split()

    def _truncate_output(self, output: str) -> str:
        """Truncate output to maximum size.

        Args:
            output: Output string.

        Returns:
            Truncated output with note if truncated.
        """
        if len(output) <= self.max_output_size:
            return output

        truncated = output[: self.max_output_size]
        note = f"\n\n[Output truncated - exceeded {self.max_output_size} bytes]"
        return truncated + note


def format_tool_result_for_agent(result: ToolResult) -> str:
    """Format a tool result exactly as agents see it.

    Args:
        result: ToolResult from execution.

    Returns:
        Formatted string with agent-style output.
    """
    lines = [f"Tool execution: {result.command}"]
    lines.append(f"Status: {'✓ Success' if result.success else '✗ Failed'}")
    lines.append(f"Exit code: {result.exit_code}")

    if result.stdout:
        lines.append(f"\nOutput:\n{result.stdout}")

    if result.stderr:
        lines.append(f"\nStderr:\n{result.stderr}")

    # Add error hints if present
    if result.error_hint:
        lines.append(f"\n💡 Hint: {result.error_hint}")

    return "\n".join(lines)
