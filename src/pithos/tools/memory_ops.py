"""Memory operation request and extraction for agent memory interactions."""

from dataclasses import dataclass
from typing import Optional
import re


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
    """Extracts memory operations from agent output using multiple formats.

    Supports multiple formats to make memory operations more reliable:
    - CLI-style: STORE[category]: content, RETRIEVE[category]: query
    - Function-style: store(category, content), retrieve(category, query)
    - Bracket-style: [STORE:category]content[/STORE]
    - Legacy: storemem(category, "content"), retrievemem(category, "query")
    """

    def __init__(self):
        """Initialize extractor with patterns for each format."""
        pass

    def extract(self, content: str) -> list[MemoryOpRequest]:
        """Extract all memory operations from content using all supported formats.

        Args:
            content: Text to extract memory operations from.

        Returns:
            List of MemoryOpRequest objects.
        """
        operations = []

        # CLI-style: STORE[category]: content
        # More flexible - can appear anywhere in text
        cli_store_pattern = r"\bSTORE\[([^\]]+)\]:\s*(.+?)(?:\n|$)"
        for match in re.finditer(cli_store_pattern, content, re.MULTILINE):
            category = match.group(1).strip()
            content_text = match.group(2).strip()
            operations.append(
                MemoryOpRequest(
                    operation="store",
                    category=category,
                    content=content_text,
                    format="cli",
                    raw_text=match.group(0),
                )
            )

        # CLI-style: RETRIEVE[category]: query
        cli_retrieve_pattern = r"\bRETRIEVE\[([^\]]+)\]:\s*(.+?)(?:\n|$)"
        for match in re.finditer(cli_retrieve_pattern, content, re.MULTILINE):
            category = match.group(1).strip()
            query = match.group(2).strip()
            operations.append(
                MemoryOpRequest(
                    operation="retrieve",
                    category=category,
                    query=query,
                    format="cli",
                    raw_text=match.group(0),
                )
            )

        # Function-style: store(category, content)
        func_store_pattern = r"store\s*\(\s*([^,]+?)\s*,\s*([^)]+)\)"
        for match in re.finditer(func_store_pattern, content, re.IGNORECASE):
            category = match.group(1).strip().strip("\"'")
            content_text = match.group(2).strip().strip("\"'")
            operations.append(
                MemoryOpRequest(
                    operation="store",
                    category=category,
                    content=content_text,
                    format="function",
                    raw_text=match.group(0),
                )
            )

        # Function-style: retrieve(category, query)
        func_retrieve_pattern = r"retrieve\s*\(\s*([^,]+?)\s*,\s*([^)]+)\)"
        for match in re.finditer(func_retrieve_pattern, content, re.IGNORECASE):
            category = match.group(1).strip().strip("\"'")
            query = match.group(2).strip().strip("\"'")
            operations.append(
                MemoryOpRequest(
                    operation="retrieve",
                    category=category,
                    query=query,
                    format="function",
                    raw_text=match.group(0),
                )
            )

        # Legacy patterns: storemem(category, "content")
        legacy_store_pattern = (
            r'storemem\s*\(\s*([^,]+?)\s*,\s*["\']([^"\']+)["\']\s*\)'
        )
        for match in re.finditer(legacy_store_pattern, content):
            category = match.group(1).strip("\"'")
            content_text = match.group(2)
            operations.append(
                MemoryOpRequest(
                    operation="store",
                    category=category,
                    content=content_text,
                    format="legacy",
                    raw_text=match.group(0),
                )
            )

        # Legacy patterns: retrievemem(category, "query")
        legacy_retrieve_pattern = (
            r'retrievemem\s*\(\s*([^,]+?)\s*,\s*["\']([^"\']+)["\']\s*\)'
        )
        for match in re.finditer(legacy_retrieve_pattern, content):
            category = match.group(1).strip("\"'")
            query = match.group(2)
            operations.append(
                MemoryOpRequest(
                    operation="retrieve",
                    category=category,
                    query=query,
                    format="legacy",
                    raw_text=match.group(0),
                )
            )

        return operations

    def get_usage_examples(self) -> str:
        """Get formatted examples of all supported formats.

        Returns:
            Formatted string with examples.
        """
        examples = """
Memory Operation Formats (all are supported):

1. CLI-style (simplest):
   STORE[facts]: Important information here
   RETRIEVE[facts]: search query here

2. Function-style:
   store(facts, important information)
   retrieve(facts, search query)

3. Legacy (still supported):
   storemem(facts, "important information")
   retrievemem(facts, "search query")
""".strip()
        return examples
