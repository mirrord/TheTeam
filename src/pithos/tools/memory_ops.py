"""Memory operation request and extraction for agent memory interactions."""

from dataclasses import dataclass
from typing import Optional
import re

_VALID_FORMATS = {"cli", "function", "legacy"}


@dataclass
class MemoryOpRequest:
    """Represents a parsed memory operation request from agent output."""

    operation: str  # 'store' or 'retrieve'
    category: str
    content: Optional[str] = None  # For store operations
    query: Optional[str] = None  # For retrieve operations
    format: str = "unknown"  # Which format was matched
    raw_text: str = ""  # Original matched text


class MemoryOpExtractor:
    """Extracts memory operations from agent output using a single configured format.

    Supported formats (select one at construction time):
    - ``"cli"`` (default): ``STORE[category]: content`` / ``RETRIEVE[category]: query``
    - ``"function"``: ``store(category, content)`` / ``retrieve(category, query)``
    - ``"legacy"``: ``storemem(category, "content")`` / ``retrievemem(category, "query")``
    """

    def __init__(self, format: str = "cli") -> None:
        """Initialize the extractor for a single format.

        Args:
            format: The format to use.  Must be one of ``"cli"``, ``"function"``,
                or ``"legacy"``.  Defaults to ``"cli"``.

        Raises:
            ValueError: If *format* is not a recognised value.
        """
        if format not in _VALID_FORMATS:
            raise ValueError(
                f"Unknown memory operation format {format!r}. "
                f"Valid options are: {sorted(_VALID_FORMATS)}"
            )
        self.format = format

    def extract(self, content: str) -> list[MemoryOpRequest]:
        """Extract memory operations from *content* using the configured format.

        Args:
            content: Text to extract memory operations from.

        Returns:
            List of MemoryOpRequest objects.
        """
        if self.format == "cli":
            return self._extract_cli(content)
        if self.format == "function":
            return self._extract_function(content)
        # self.format == "legacy"
        return self._extract_legacy(content)

    # ------------------------------------------------------------------
    # Private per-format extractors
    # ------------------------------------------------------------------

    def _extract_cli(self, content: str) -> list[MemoryOpRequest]:
        operations: list[MemoryOpRequest] = []

        store_pattern = r"\bSTORE\[([^\]]+)\]:\s*(.+?)(?:\n|$)"
        for match in re.finditer(store_pattern, content, re.MULTILINE):
            operations.append(
                MemoryOpRequest(
                    operation="store",
                    category=match.group(1).strip(),
                    content=match.group(2).strip(),
                    format="cli",
                    raw_text=match.group(0),
                )
            )

        retrieve_pattern = r"\bRETRIEVE\[([^\]]+)\]:\s*(.+?)(?:\n|$)"
        for match in re.finditer(retrieve_pattern, content, re.MULTILINE):
            operations.append(
                MemoryOpRequest(
                    operation="retrieve",
                    category=match.group(1).strip(),
                    query=match.group(2).strip(),
                    format="cli",
                    raw_text=match.group(0),
                )
            )

        return operations

    def _extract_function(self, content: str) -> list[MemoryOpRequest]:
        operations: list[MemoryOpRequest] = []

        store_pattern = r"store\s*\(\s*([^,]+?)\s*,\s*([^)]+)\)"
        for match in re.finditer(store_pattern, content, re.IGNORECASE):
            operations.append(
                MemoryOpRequest(
                    operation="store",
                    category=match.group(1).strip().strip("\"'"),
                    content=match.group(2).strip().strip("\"'"),
                    format="function",
                    raw_text=match.group(0),
                )
            )

        retrieve_pattern = r"retrieve\s*\(\s*([^,]+?)\s*,\s*([^)]+)\)"
        for match in re.finditer(retrieve_pattern, content, re.IGNORECASE):
            operations.append(
                MemoryOpRequest(
                    operation="retrieve",
                    category=match.group(1).strip().strip("\"'"),
                    query=match.group(2).strip().strip("\"'"),
                    format="function",
                    raw_text=match.group(0),
                )
            )

        return operations

    def _extract_legacy(self, content: str) -> list[MemoryOpRequest]:
        operations: list[MemoryOpRequest] = []

        store_pattern = r'storemem\s*\(\s*([^,]+?)\s*,\s*["\']([^"\']+)["\']\s*\)'
        for match in re.finditer(store_pattern, content):
            operations.append(
                MemoryOpRequest(
                    operation="store",
                    category=match.group(1).strip("\"'"),
                    content=match.group(2),
                    format="legacy",
                    raw_text=match.group(0),
                )
            )

        retrieve_pattern = r'retrievemem\s*\(\s*([^,]+?)\s*,\s*["\']([^"\']+)["\']\s*\)'
        for match in re.finditer(retrieve_pattern, content):
            operations.append(
                MemoryOpRequest(
                    operation="retrieve",
                    category=match.group(1).strip("\"'"),
                    query=match.group(2),
                    format="legacy",
                    raw_text=match.group(0),
                )
            )

        return operations

    # ------------------------------------------------------------------
    # Usage examples
    # ------------------------------------------------------------------

    def get_usage_examples(self) -> str:
        """Get formatted examples for the configured format.

        Returns:
            Formatted string with examples.
        """
        if self.format == "cli":
            return (
                "Memory Operation Format (CLI-style):\n\n"
                "  STORE[facts]: Important information here\n"
                "  RETRIEVE[facts]: search query here"
            )
        if self.format == "function":
            return (
                "Memory Operation Format (function-style):\n\n"
                "  store(facts, important information)\n"
                "  retrieve(facts, search query)"
            )
        # legacy
        return (
            "Memory Operation Format (legacy):\n\n"
            '  storemem(facts, "important information")\n'
            '  retrievemem(facts, "search query")'
        )
