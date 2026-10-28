"""Tool call extractor for parsing tool invocations from agent output.

Agent tool calls use the project-wide JSON tool-call syntax defined in
:mod:`pithos.tools.tool_calls`: one or more JSON objects inside fenced
``tool_call`` code blocks.  This extractor adapts those parsed calls into the
command-string :class:`~pithos.tools.models.ToolCallRequest` that the
:class:`~pithos.tools.executor.ToolExecutor` dispatches on.
"""

from .models import ToolCallRequest
from .tool_calls import extract_tool_calls, lookup, to_str

#: Tool names that denote "run this command line" (as opposed to naming a
#: specific CLI/virtual tool directly).
_RUN_ALIASES = frozenset({"run", "exec", "shell", "bash", "command", "cli"})

#: Argument keys that carry the command line for a run-style call.
_COMMAND_KEYS = ("command", "cmd", "shell", "args", "run", "input")


class ToolCallExtractor:
    """Extracts tool calls from agent output using the JSON tool-call format.

    A tool call is a JSON object with a ``name`` and an ``arguments`` object
    inside a fenced ``tool_call`` block::

        ```tool_call
        {"name": "run", "arguments": {"command": "python --version"}}
        ```

    The ``run`` tool executes the command line in its ``command`` argument.  A
    call that instead names a tool directly (``{"name": "git", "arguments":
    {"command": "status"}}``) is flattened to the command line ``git status``.
    Requiring the closing ``` fence keeps extraction unambiguous and safe for
    mid-stream detection (a partial call cannot match until it is complete).
    """

    def extract(self, content: str) -> list[ToolCallRequest]:
        """Extract all JSON tool calls from *content*.

        Args:
            content: Text to extract tool calls from.

        Returns:
            List of ToolCallRequest objects (empty when none are present).
        """
        requests: list[ToolCallRequest] = []
        for call in extract_tool_calls(content):
            command = self._command_for(call)
            if command:
                requests.append(
                    ToolCallRequest(
                        command=command,
                        format="tool_call",
                        raw_text=call.raw,
                        name=call.name,
                        arguments=dict(call.arguments),
                    )
                )
        return requests

    @staticmethod
    def _command_for(call) -> str:
        """Map a parsed tool call to the command string the executor runs."""
        name = call.name.strip()
        arg_command = to_str(lookup(call.arguments, _COMMAND_KEYS)).strip()
        if name.lower() in _RUN_ALIASES:
            return arg_command
        # A directly-named tool: prepend the tool name to any argument string.
        if arg_command:
            return f"{name} {arg_command}".strip()
        return name

    def get_usage_examples(self) -> str:
        """Get formatted examples of the supported format.

        Returns:
            Formatted string with examples.
        """
        examples = """
Tool Call Format (JSON — use exactly this syntax):

Emit each tool call as a JSON object inside a fenced ```tool_call code block.
Use the "run" tool and put the full command line in its "command" argument:

   ```tool_call
   {"name": "run", "arguments": {"command": "python --version"}}
   ```
   ```tool_call
   {"name": "run", "arguments": {"command": "git status"}}
   ```

Put several calls in one block as a JSON array, or use one block per call. The
command is not executed until the closing ``` fence has been produced.
""".strip()
        return examples
