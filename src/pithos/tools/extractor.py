"""Tool call extractor for parsing tool invocations from agent output."""

import re

from .models import ToolCallRequest


class ToolCallExtractor:
    """Extracts tool calls from agent output using multiple formats.

    Supports multiple formats to make tool calling more reliable:
    - CLI-style: RUN: command args
    - Function-style: run(command args) or tool(command args)
    - Bracket-style: [RUN]command args[/RUN]
    - Legacy: runcommand("command args")
    """

    def __init__(self):
        """Initialize extractor with patterns for each format."""
        # CLI-style patterns: RUN: command, EXEC: command, TOOL: command
        # More flexible - can appear anywhere in text, not just at line start
        self.cli_patterns = [
            (r"\bRUN:\s*(.+?)(?:\n|$)", "cli"),
            (r"\bEXEC:\s*(.+?)(?:\n|$)", "cli"),
            (r"\bTOOL:\s*(.+?)(?:\n|$)", "cli"),
        ]

        # Function-style patterns: run(...), tool(...), execute(...)
        self.function_patterns = [
            (r"run\s*\(([^)]+)\)", "function"),
            (r"tool\s*\(([^)]+)\)", "function"),
            (r"execute\s*\(([^)]+)\)", "function"),
        ]

        # Bracket-style patterns: [RUN]...[/RUN], <RUN>...</RUN>
        self.bracket_patterns = [
            (r"\[RUN\](.+?)\[/RUN\]", "bracket"),
            (r"<RUN>(.+?)</RUN>", "bracket"),
            (r"\[EXEC\](.+?)\[/EXEC\]", "bracket"),
        ]

        # Legacy pattern: runcommand("...")
        self.legacy_pattern = (r'runcommand\s*\(["\']([^"\']+)["\']\)', "legacy")

    def extract(self, content: str) -> list[ToolCallRequest]:
        """Extract all tool calls from content using all supported formats.

        Args:
            content: Text to extract tool calls from.

        Returns:
            List of ToolCallRequest objects.
        """
        requests = []

        # Try CLI patterns
        for pattern, fmt in self.cli_patterns:
            for match in re.finditer(pattern, content, re.MULTILINE):
                command = match.group(1).strip()
                if command:
                    requests.append(
                        ToolCallRequest(
                            command=command, format=fmt, raw_text=match.group(0)
                        )
                    )

        # Try function patterns
        for pattern, fmt in self.function_patterns:
            for match in re.finditer(pattern, content, re.IGNORECASE):
                command = match.group(1).strip()
                # Remove quotes if present
                command = command.strip("\"'")
                if command:
                    requests.append(
                        ToolCallRequest(
                            command=command, format=fmt, raw_text=match.group(0)
                        )
                    )

        # Try bracket patterns
        for pattern, fmt in self.bracket_patterns:
            for match in re.finditer(pattern, content, re.DOTALL):
                command = match.group(1).strip()
                if command:
                    requests.append(
                        ToolCallRequest(
                            command=command, format=fmt, raw_text=match.group(0)
                        )
                    )

        # Try legacy pattern
        pattern, fmt = self.legacy_pattern
        for match in re.finditer(pattern, content):
            command = match.group(1).strip()
            if command:
                requests.append(
                    ToolCallRequest(
                        command=command, format=fmt, raw_text=match.group(0)
                    )
                )

        return requests

    def get_usage_examples(self) -> str:
        """Get formatted examples of all supported formats.

        Returns:
            Formatted string with examples.
        """
        examples = """
Tool Call Formats (all are supported):

1. CLI-style (simplest):
   RUN: python --version
   EXEC: git status

2. Function-style:
   run(python --version)
   tool(git status)

3. Bracket-style:
   [RUN]python --version[/RUN]
   <RUN>git status</RUN>

4. Legacy (still supported):
   runcommand("python --version")
""".strip()
        return examples
