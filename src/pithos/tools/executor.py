"""Tool executor - routes tool calls to the appropriate ToolProvider.

ToolExecutor is the single entry point for all tool execution.  It:

1. Parses the leading token from the command string to identify the tool.
2. Validates the tool is registered.
3. Looks up the owning ToolProvider via the ToolRegistry.
4. Prompts for user confirmation when the registry or a REVIEW safety verdict
   requires it.
5. Delegates the actual execution to provider.execute(command, context).

Safety analysis (BLOCK/REVIEW) is the responsibility of each provider; the
CLI provider applies heuristic checks.  A safety_verdict of REVIEW on the
returned ToolResult causes ToolExecutor to prompt for confirmation before
delegating to the next call.
"""

import shlex
from typing import Any, Callable, Optional

from .models import RiskLevel, ToolResult
from .registry import ToolRegistry


class ToolExecutor:
    """Routes tool calls to registered ToolProviders with confirmation support."""

    def __init__(
        self,
        confirm_callback: Optional[Callable[[str], bool]] = None,
    ) -> None:
        """Initialise the executor.

        Args:
            confirm_callback: Optional callable invoked when a tool requires
                confirmation.  Receives the full command string and must return
                True (approved) or False (denied).  When None, falls back to
                interactive CLI prompting.
        """
        self.confirm_callback = confirm_callback

    def run(
        self,
        command: str,
        tool_registry: ToolRegistry,
        context: Optional[dict[str, Any]] = None,
    ) -> ToolResult:
        """Execute command by dispatching to the appropriate ToolProvider.

        Args:
            command: Full command string (e.g. "python --version").
            tool_registry: Registry used to look up the owning provider.
            context: Optional runtime context passed through to the provider
                (e.g. {"agent": agent_instance}).

        Returns:
            ToolResult from the provider.
        """
        tool_name, _args = self._parse_command(command)
        if not tool_name:
            return ToolResult(
                success=False,
                stdout="",
                stderr=f"Invalid command format: {command!r}",
                exit_code=-1,
                execution_time=0.0,
                command=command,
                error_hint=(
                    "Command should be in format: toolname [arguments]\n"
                    "Example: python --version"
                ),
            )

        # Validate tool is registered.
        if not tool_registry.is_allowed(tool_name):
            try:
                available = tool_registry.list_tools()
                available_str = (
                    ", ".join(available[:10]) if available else "No tools available"
                )
            except Exception:
                available_str = "Unable to list available tools"

            return ToolResult(
                success=False,
                stdout="",
                stderr=f"Tool {tool_name!r} is not available or not allowed",
                exit_code=-1,
                execution_time=0.0,
                command=command,
                error_hint=(
                    f"Tool {tool_name!r} not found or not allowed.\n"
                    f"Available tools include: {available_str}...\n"
                    "Use exact tool names from the available list."
                ),
            )

        provider = tool_registry.get_provider(tool_name)
        if provider is None:
            return ToolResult(
                success=False,
                stdout="",
                stderr=f"No provider found for tool {tool_name!r}",
                exit_code=-1,
                execution_time=0.0,
                command=command,
                error_hint="The tool is registered but has no execution provider.",
            )

        # Registry-level confirmation check (mode == confirm).
        needs_confirm = tool_registry.requires_confirmation(tool_name)
        if needs_confirm and not self._prompt_confirm(command):
            return ToolResult(
                success=False,
                stdout="Denied by user.",
                stderr="",
                exit_code=-1,
                execution_time=0.0,
                command=command,
            )

        result = provider.execute(command, context)

        # Provider may have flagged the result as REVIEW (e.g. CLIToolProvider
        # detected a destructive flag).  Prompt if not already confirmed.
        if (
            result.safety_verdict is not None
            and result.safety_verdict.level == RiskLevel.REVIEW
            and not needs_confirm
        ):
            if not self._prompt_confirm(command):
                return ToolResult(
                    success=False,
                    stdout="Denied by user.",
                    stderr="",
                    exit_code=-1,
                    execution_time=result.execution_time,
                    command=command,
                    safety_verdict=result.safety_verdict,
                )

        return result

    # ------------------------------------------------------------------
    # Confirmation helpers
    # ------------------------------------------------------------------

    def _prompt_confirm(self, command: str) -> bool:
        """Ask the user whether to allow a tool call.

        Resolution order:
        1. self.confirm_callback - injected at construction time.
        2. rich.prompt.Confirm - styled interactive CLI prompt.
        3. Plain input() fallback.

        Returns False automatically when stdin is unavailable.
        """
        if self.confirm_callback is not None:
            return self.confirm_callback(command)

        prompt_text = f"Allow tool call: {command!r}?"
        try:
            from rich.prompt import Confirm

            return Confirm.ask(prompt_text)
        except ImportError:
            pass
        except Exception:
            pass

        try:
            answer = input(f"{prompt_text} [y/N]: ").strip().lower()
            return answer in ("y", "yes")
        except (EOFError, OSError):
            return False

    # ------------------------------------------------------------------
    # Command parsing
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_command(command: str) -> tuple[Optional[str], list[str]]:
        """Parse command string into (tool_name, args_list)."""
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

    @staticmethod
    def _split_args(command: str) -> list[str]:
        """Split command string into arguments, respecting quotes."""
        try:
            return shlex.split(command)
        except ValueError:
            return command.split()


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

    if result.error_hint:
        lines.append(f"\n💡 Hint: {result.error_hint}")

    return "\n".join(lines)
