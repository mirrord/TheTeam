"""Memory operation request and extraction for agent memory interactions.

Memory operations use the project-wide JSON tool-call syntax (see
:mod:`pithos.tools.tool_calls`): the agent expresses them as ``memory`` tool
calls inside fenced ``tool_call`` code blocks, parsed by the same shared parser
that handles every other tool call::

    ```tool_call
    {"name": "memory", "arguments": {"operation": "store", "category": "facts",
                                      "content": "..."}}
    ```
    ```tool_call
    {"name": "memory", "arguments": {"operation": "retrieve", "category": "facts",
                                      "query": "..."}}
    ```

:class:`MemoryOpExtractor` is a thin adapter that maps those parsed calls onto
:class:`MemoryOpRequest` objects; execution lives in
:class:`pithos.tools.memory_provider.MemoryToolProvider`.
"""

from dataclasses import dataclass
from typing import Optional

from .tool_calls import extract_tool_calls, lookup, to_str

#: Call names that denote a memory operation (``memory:store`` etc. also work).
_MEMORY_NAMES = frozenset({"memory", "remember", "recall"})

#: Operation aliases -> canonical ``store`` / ``retrieve``.
_STORE_OPS = frozenset({"store", "add", "save", "remember", "write", "put"})
_RETRIEVE_OPS = frozenset(
    {"retrieve", "search", "recall", "query", "get", "find", "read"}
)


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
    """Extracts memory operations from agent output as JSON ``memory`` tool calls.

    A memory operation is a JSON tool call whose name is ``memory`` (or
    ``memory:store`` / ``memory:retrieve``, or the aliases ``remember`` /
    ``recall``).  The ``operation`` argument selects store vs. retrieve; the
    ``category`` argument names the memory bucket; ``content`` carries the text
    to store and ``query`` the search text to retrieve.
    """

    def extract(self, content: str) -> list[MemoryOpRequest]:
        """Extract memory operations from *content*.

        Args:
            content: Text to extract memory operations from.

        Returns:
            List of MemoryOpRequest objects (empty when none are present).
        """
        operations: list[MemoryOpRequest] = []
        for call in extract_tool_calls(content):
            op = self._call_to_op(call)
            if op is not None:
                operations.append(op)
        return operations

    @staticmethod
    def _call_to_op(call) -> Optional[MemoryOpRequest]:
        base, _, sub = call.name.strip().lower().partition(":")
        if base not in _MEMORY_NAMES:
            return None

        args = call.arguments
        operation = to_str(
            lookup(args, ("operation", "op", "action", "mode"))
        ).strip().lower()
        if not operation and sub:
            operation = sub.strip().lower()
        if not operation:
            if base == "remember":
                operation = "store"
            elif base == "recall":
                operation = "retrieve"

        category = to_str(
            lookup(args, ("category", "cat", "namespace", "collection"))
        ).strip()

        if operation in _STORE_OPS:
            content_val = to_str(
                lookup(args, ("content", "text", "value", "data", "information"))
            ).strip()
            return MemoryOpRequest(
                operation="store",
                category=category,
                content=content_val,
                format="tool_call",
                raw_text=call.raw,
            )
        if operation in _RETRIEVE_OPS:
            query_val = to_str(
                lookup(args, ("query", "q", "search", "text", "question"))
            ).strip()
            return MemoryOpRequest(
                operation="retrieve",
                category=category,
                query=query_val,
                format="tool_call",
                raw_text=call.raw,
            )
        return None

    # ------------------------------------------------------------------
    # Usage examples
    # ------------------------------------------------------------------

    def get_usage_examples(self) -> str:
        """Get formatted examples of the JSON memory tool-call format."""
        return (
            "Memory Operation Format (JSON tool calls):\n\n"
            "Store an important fact:\n"
            "   ```tool_call\n"
            '   {"name": "memory", "arguments": {"operation": "store", '
            '"category": "facts", "content": "important information here"}}\n'
            "   ```\n\n"
            "Retrieve relevant memories:\n"
            "   ```tool_call\n"
            '   {"name": "memory", "arguments": {"operation": "retrieve", '
            '"category": "facts", "query": "search query here"}}\n'
            "   ```"
        )
