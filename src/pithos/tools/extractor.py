"""Tool call extractor for parsing tool invocations from agent output."""

import re

from .models import ToolCallRequest


class ToolCallExtractor:
    """Extracts tool calls from agent output using the bracket format.

    Only the bracket-style format is supported:
    - [RUN]command args[/RUN]
    - <RUN>command args</RUN>
    - [EXEC]command args[/EXEC]

    A bracket call requires an explicit closing tag, which keeps extraction
    unambiguous and safe for mid-stream detection (a partial call cannot match
    until its closing tag has been produced).
    """

    def __init__(self):
        """Initialize extractor with the bracket-style patterns."""
        # Bracket-style patterns: [RUN]...[/RUN], <RUN>...</RUN>, [EXEC]...[/EXEC]
        self.bracket_patterns = [
            (r"\[RUN\](.+?)\[/RUN\]", "bracket"),
            (r"<RUN>(.+?)</RUN>", "bracket"),
            (r"\[EXEC\](.+?)\[/EXEC\]", "bracket"),
        ]

    def extract(self, content: str) -> list[ToolCallRequest]:
        """Extract all bracket-style tool calls from content.

        Args:
            content: Text to extract tool calls from.

        Returns:
            List of ToolCallRequest objects.
        """
        requests = []

        for pattern, fmt in self.bracket_patterns:
            for match in re.finditer(pattern, content, re.DOTALL):
                command = match.group(1).strip()
                if command:
                    requests.append(
                        ToolCallRequest(
                            command=command, format=fmt, raw_text=match.group(0)
                        )
                    )

        return requests

    def get_usage_examples(self) -> str:
        """Get formatted examples of the supported format.

        Returns:
            Formatted string with examples.
        """
        examples = """
Tool Call Format (bracket-style — use exactly this syntax):

   [RUN]python --version[/RUN]
   [RUN]git status[/RUN]

Wrap the command between an opening [RUN] tag and a closing [/RUN] tag.
The closing tag is required; the command is not executed until it appears.
""".strip()
        return examples
