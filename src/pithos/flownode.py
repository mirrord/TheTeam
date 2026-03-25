"""Flow nodes for flowchart execution."""

import ast
import builtins as _builtins_module
import re
import threading
from typing import Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .message import Message, NodeInputState

# ---------------------------------------------------------------------------
# CustomNode sandbox helpers
# ---------------------------------------------------------------------------

# Safe subset of built-in names available inside sandboxed custom code.
# Excludes anything that can access the file system, spawn processes,
# load modules, or manipulate Python internals (open, eval, exec, compile,
# __import__, getattr/setattr/delattr, globals, locals, vars, dir, …).
_SAFE_BUILTIN_NAMES: frozenset[str] = frozenset(
    [
        "abs",
        "all",
        "any",
        "bool",
        "bytes",
        "callable",
        "chr",
        "dict",
        "divmod",
        "enumerate",
        "filter",
        "float",
        "format",
        "frozenset",
        "hash",
        "hex",
        "int",
        "isinstance",
        "issubclass",
        "iter",
        "len",
        "list",
        "map",
        "max",
        "min",
        "next",
        "oct",
        "ord",
        "pow",
        "print",
        "range",
        "repr",
        "reversed",
        "round",
        "set",
        "slice",
        "sorted",
        "str",
        "sum",
        "tuple",
        "type",
        "zip",
        # Exception types useful for try/except inside custom code
        "ArithmeticError",
        "AttributeError",
        "Exception",
        "IndexError",
        "KeyError",
        "NameError",
        "NotImplementedError",
        "OverflowError",
        "RuntimeError",
        "StopIteration",
        "TypeError",
        "ValueError",
        "ZeroDivisionError",
    ]
)

_SAFE_BUILTINS_DICT: dict[str, Any] = {
    name: getattr(_builtins_module, name)
    for name in _SAFE_BUILTIN_NAMES
    if hasattr(_builtins_module, name)
}

# Names that must never appear in custom code (checked at AST level).
_BLOCKED_NAMES: frozenset[str] = frozenset(
    [
        "__import__",
        "__builtins__",
        "__loader__",
        "__spec__",
        "globals",
        "locals",
        "vars",
        "dir",
    ]
)

# Function calls that are never permitted in custom code.
_BLOCKED_CALLS: frozenset[str] = frozenset(
    [
        "eval",
        "exec",
        "compile",
        "open",
        "input",
        "breakpoint",
        "getattr",
        "setattr",
        "delattr",
    ]
)


def _check_code_safety(code: str) -> None:
    """Statically verify that *code* does not use dangerous constructs.

    Raises:
        ValueError: If the code contains any disallowed construct.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        raise ValueError(f"Syntax error in custom code: {exc}") from exc

    for node in ast.walk(tree):
        # Block all import statements
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            raise ValueError(
                "Import statements are not allowed in custom code. "
                "Use values already present in 'context'."
            )
        # Block dunder attribute access (e.g. obj.__class__, obj.__globals__)
        if isinstance(node, ast.Attribute):
            if node.attr.startswith("__") and node.attr.endswith("__"):
                raise ValueError(
                    f"Access to dunder attribute '{node.attr}' is not allowed "
                    "in custom code."
                )
        # Block dangerous name references
        if isinstance(node, ast.Name) and node.id in _BLOCKED_NAMES:
            raise ValueError(f"Use of '{node.id}' is not allowed in custom code.")
        # Block dangerous function calls by name
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in _BLOCKED_CALLS:
                raise ValueError(
                    f"Call to '{node.func.id}' is not allowed in custom code."
                )


class FlowNode:
    """Base class for flowchart nodes."""

    def __init__(
        self,
        extraction: Optional[dict[str, str]] = None,
        prompt_args: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize FlowNode.

        Args:
            extraction: Regex patterns to extract values from input messages.
            prompt_args: Arguments for prompt formatting.
            **kwargs: Additional keyword arguments including 'set', 'inputs', 'outputs'.
        """
        self.extraction = extraction or {}
        self.set = kwargs.get("set", {})
        self.prompt_args = prompt_args or {}

        # Message-based execution
        self.required_inputs: list[str] = kwargs.get("inputs", ["default"])
        self.output_keys: list[str] = kwargs.get("outputs", ["default"])

    def execute_with_messages(
        self, input_state: "NodeInputState", message_router: Optional[Any] = None
    ) -> list["Message"]:
        """Execute node with message-based inputs and produce message outputs.

        Args:
            input_state: The node input state containing all received messages.
            message_router: Optional message router for accessing shared context.

        Returns:
            List of output messages produced by this node.
        """
        # Build execution context from input messages
        context = self._build_context_from_messages(input_state, message_router)

        # Execute node logic
        result = self._execute(context)

        # Convert result to output messages
        messages = self._create_output_messages(result, context)

        return messages

    def _build_context_from_messages(
        self, input_state: "NodeInputState", message_router: Optional[Any] = None
    ) -> dict[str, Any]:
        """Build execution context from input messages.

        Args:
            input_state: Input state with received messages.
            message_router: Optional message router for accessing shared context.

        Returns:
            Context dict for execution.
        """
        context = {}

        # Include shared context if available
        if message_router and hasattr(message_router, "shared_context"):
            context.update(message_router.shared_context)

        # Extract data from all input messages
        for key, message in input_state.received_inputs.items():
            context[key] = message.data

        # Apply extractions if configured
        if self.extraction and "default" in context:
            extracted = self.parse_extractions(str(context["default"]))
            context.update(extracted)

        # Apply set values
        for var_name, value in self.set.items():
            context[var_name] = self._stateful_format(value, context)

        # Format prompt args
        context["prompt_args"] = {
            k: self._stateful_format(v, context) for k, v in self.prompt_args.items()
        }

        return context

    def _execute(self, context: dict[str, Any]) -> Any:
        """Execute the node's core logic. Override in subclasses.

        Args:
            context: Execution context built from input messages.

        Returns:
            Execution result (will be converted to messages).
        """
        raise NotImplementedError("Subclasses should implement _execute()!")

    def _create_output_messages(
        self, result: Any, context: dict[str, Any]
    ) -> list["Message"]:
        """Create output messages from execution result.

        Args:
            result: The result from _execute.
            context: Execution context.

        Returns:
            List of output messages.
        """
        from .message import Message

        messages = []

        # If result is a dict with state, extract relevant outputs
        if isinstance(result, dict):
            for output_key in self.output_keys:
                # Map common output keys
                if output_key == "default":
                    # Use formatted_prompt or current_input as default output
                    data = result.get("formatted_prompt") or result.get("current_input")
                else:
                    data = result.get(output_key)

                if data is not None:
                    messages.append(
                        Message(
                            data=data,
                            source_node=None,  # Will be set by flowchart
                            input_key=output_key,
                        )
                    )
        else:
            # Single value result
            messages.append(Message(data=result, source_node=None, input_key="default"))

        return messages

    def parse_extractions(self, input: str) -> dict[str, str]:
        """Extract values from input using regex patterns.

        Args:
            input: Input string to parse.

        Returns:
            Dictionary of extracted values.
        """
        extracted_values = {}
        for var_name, regex in self.extraction.items():
            match = re.search(regex, input)
            if match:
                # Use first capturing group if available, otherwise full match
                extracted_values[var_name] = (
                    match.group(1) if match.lastindex else match.group(0)
                )
        return extracted_values

    def _stateful_format(self, x: Any, state: dict[str, Any]) -> Any:
        """Format strings with values from state.

        Args:
            x: Value to format (if string).
            state: State dictionary with values.

        Returns:
            Formatted value.
        """
        if isinstance(x, str):
            needed_vars = re.findall(r"\{(.*?)\}", x)
            filtered_state = {k: v for k, v in state.items() if k in needed_vars}
            return x.format(**filtered_state) if filtered_state else x
        return x

    def set_values(self, state: dict[str, Any]) -> dict[str, Any]:
        """Set values in state based on the 'set' configuration.

        Args:
            state: Current state.

        Returns:
            Updated state.
        """
        for var_name, value in self.set.items():
            state[var_name] = self._stateful_format(value, state)
        state["prompt_args"] = {
            k: self._stateful_format(v, state) for k, v in self.prompt_args.items()
        }
        return state

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FlowNode":
        """Create a FlowNode from a dictionary.

        Args:
            data: Configuration dictionary.

        Returns:
            FlowNode instance.
        """
        if "type" in data:
            del data["type"]
        if "extraction" not in data:
            data["extraction"] = {}
        if "set" not in data:
            data["set"] = {}
        return cls(**data)

    def to_dict(self) -> dict[str, Any]:
        """Serialize node to dictionary.

        Returns:
            Dictionary representation of the node.
        """
        d = {
            k: v
            for k, v in self.__dict__.items()
            if not k.startswith("_") and not callable(v) and v
        }
        d["type"] = self.__class__.__name__.lower()[
            :-4
        ]  # lowercase class name without 'Node'
        return d


class InputNode(FlowNode):
    """Base class for nodes that bring data into a flowchart.

    InputNodes are sources of data - they might read from files, listen on ports,
    accept user input, or fetch data from external sources. Every flowchart must
    have at least one InputNode.
    """

    def __init__(
        self,
        extraction: Optional[dict[str, str]] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize InputNode.

        Args:
            extraction: Regex patterns for extracting values.
            **kwargs: Additional arguments passed to FlowNode.
        """
        super().__init__(extraction, **kwargs)

    def _execute(self, context: dict[str, Any]) -> Any:
        """Execute the node's input logic. Override in subclasses.

        Args:
            context: Execution context.

        Returns:
            The input data fetched by this node.
        """
        raise NotImplementedError("Subclasses must implement _execute()")


class OutputNode(FlowNode):
    """Base class for nodes that write data to external destinations.

    OutputNodes handle data output - they might save to files, display to users,
    send to network endpoints, or write to databases. Every flowchart must have
    at least one OutputNode.
    """

    def __init__(
        self,
        extraction: Optional[dict[str, str]] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize OutputNode.

        Args:
            extraction: Regex patterns for extracting values.
            **kwargs: Additional arguments passed to FlowNode.
        """
        super().__init__(extraction, **kwargs)

    def _execute(self, context: dict[str, Any]) -> Any:
        """Execute the node's output logic. Override in subclasses.

        Args:
            context: Execution context.

        Returns:
            Result of the output operation.
        """
        raise NotImplementedError("Subclasses must implement _execute()")


class PromptNode(FlowNode):
    """Node that formats a prompt and sends it to an LLM agent."""

    def __init__(self, extraction: dict[str, str], prompt: str, **kwargs: Any) -> None:
        """Initialize PromptNode.

        Args:
            extraction: Regex patterns for extracting values.
            prompt: Prompt template with placeholders.
            **kwargs: Additional arguments passed to FlowNode.
        """
        super().__init__(extraction, **kwargs)
        self.prompt = prompt

    def _execute(self, context: dict[str, Any]) -> Any:
        """Format the prompt and optionally send it to the agent.

        Args:
            context: Execution context (includes agent, context_name, model if available).

        Returns:
            Dict with formatted_prompt and optionally the agent's response.
        """
        formatted_prompt = self._stateful_format(self.prompt, context)

        result = {"formatted_prompt": formatted_prompt}

        # If agent is available in context, send the prompt and get response
        if "agent" in context:
            agent = context["agent"]
            ctx_name = context.get("context_name")
            model = context.get("model")
            verbose = context.get("verbose", False)
            kwargs = context.get("prompt_args", {})

            # Send prompt to agent
            response = agent.send(
                formatted_prompt, ctx_name, verbose=verbose, model=model, **kwargs
            )

            result["current_input"] = response
            result["agent_response"] = response
        else:
            # No agent available, just return formatted prompt
            result["current_input"] = formatted_prompt

        return result


class CustomNode(FlowNode):
    """Node that executes custom Python code in a sandboxed environment.

    The sandbox enforces the following restrictions:

    * Built-ins are limited to a safe whitelist — ``open``, ``eval``,
      ``exec``, ``compile``, ``__import__``, ``getattr``/``setattr``/
      ``delattr``, ``globals``, ``locals``, ``dir``, etc. are all excluded.
    * Import statements (both ``import x`` and ``from x import y``) are
      blocked at the AST level before the code is run.
    * Dunder attribute access (e.g. ``obj.__class__``, ``obj.__globals__``)
      is blocked at the AST level to prevent class-hierarchy escapes.
    * Execution time is bounded by a configurable *timeout* (default 30 s)
      enforced via a daemon thread.  If the code hangs the ``_execute``
      call raises ``TimeoutError``.

    The only external name exposed to the code is ``context`` — the
    flowchart state dict that the code is expected to read and modify.
    """

    _DEFAULT_TIMEOUT: float = 30.0

    def __init__(
        self,
        extraction: dict[str, str],
        custom_code: str,
        timeout: float = _DEFAULT_TIMEOUT,
        **kwargs: Any,
    ) -> None:
        """Initialize CustomNode.

        Args:
            extraction: Regex patterns for extracting values.
            custom_code: Python code to execute with 'context' available.
            timeout: Maximum allowed execution time in seconds (default 30).
            **kwargs: Additional arguments passed to FlowNode.

        Raises:
            ValueError: If *custom_code* contains disallowed constructs.
        """
        _check_code_safety(custom_code)
        super().__init__(extraction, **kwargs)
        self.custom_code = custom_code
        self.timeout = timeout

    def _execute(self, context: dict[str, Any]) -> Any:
        """Execute custom code inside the sandbox.

        Args:
            context: Execution context modified in-place by the custom code.

        Returns:
            The (updated) context dict.

        Raises:
            TimeoutError: If execution exceeds *self.timeout* seconds.
            RuntimeError: If the custom code raises an exception.
        """
        sandbox_globals: dict[str, Any] = {
            "__builtins__": _SAFE_BUILTINS_DICT,
            "context": context,
        }

        exc_holder: list[Optional[BaseException]] = [None]

        def _run() -> None:
            try:
                exec(self.custom_code, sandbox_globals)  # noqa: S102
            except Exception as exc:  # pragma: no cover
                exc_holder[0] = exc

        thread = threading.Thread(target=_run, daemon=True)
        thread.start()
        thread.join(timeout=self.timeout)

        if thread.is_alive():
            raise TimeoutError(f"Custom code execution timed out after {self.timeout}s")

        if exc_holder[0] is not None:
            raise RuntimeError(
                f"Custom code raised an exception: {exc_holder[0]}"
            ) from exc_holder[0]

        return context


class ToolCallNode(FlowNode):
    """Node that executes a command-line tool and saves results to state."""

    def __init__(
        self,
        command: str,
        save_to: str = "tool_result",
        error_handling: str = "continue",
        extraction: Optional[dict[str, str]] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize ToolCallNode.

        Args:
            command: Command template to execute (e.g., "git log --oneline -n {n}").
            save_to: State variable name to save results to.
            error_handling: How to handle errors: 'continue', 'stop', or 'retry'.
            extraction: Regex patterns for extracting values.
            **kwargs: Additional arguments passed to FlowNode.
        """
        super().__init__(extraction, **kwargs)
        self.command = command
        self.save_to = save_to
        self.error_handling = error_handling

    def _execute(self, context: dict[str, Any]) -> Any:
        """Execute the tool and save results to context.

        Args:
            context: Execution context.

        Returns:
            Dict with tool results.

        Raises:
            RuntimeError: If error_handling is 'stop' and execution fails.
        """
        # Format command with context variables
        cmd = self._stateful_format(self.command, context)

        # Get tool executor and registry from context
        tool_executor = context.get("tool_executor")
        tool_registry = context.get("tool_registry")

        tool_result = {
            "stdout": "",
            "stderr": "",
            "success": False,
            "exit_code": -1,
        }

        if not tool_executor or not tool_registry:
            # If tools not available, save error info
            tool_result["stderr"] = "Tool execution not available in flowchart context"
            if self.error_handling == "stop":
                raise RuntimeError("Tool execution not available")
        else:
            # Execute tool
            result = tool_executor.run(cmd, tool_registry)

            # Build result dict
            tool_result = {
                "stdout": result.stdout,
                "stderr": result.stderr,
                "success": result.success,
                "exit_code": result.exit_code,
                "command": result.command,
                "execution_time": result.execution_time,
            }

            # Handle errors based on policy
            if not result.success and self.error_handling == "stop":
                raise RuntimeError(f"Tool execution failed: {result.stderr}")

        # Return result with save_to key
        return {self.save_to: tool_result, "current_input": tool_result}


class TextParseNode(FlowNode):
    """Node that parses and extracts information from text input.

    This node is specifically designed for text parsing operations:
    - Extracting values using regex patterns
    - Setting variables based on current input
    - Transforming or filtering text

    Unlike PromptNode which generates prompts, TextParseNode focuses on
    data extraction and state management.
    """

    def __init__(
        self,
        extraction: Optional[dict[str, str]] = None,
        transform: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize TextParseNode.

        Args:
            extraction: Regex patterns for extracting values from input.
            transform: Optional transformation instruction for the text.
            **kwargs: Additional arguments passed to FlowNode (including 'set').
        """
        super().__init__(extraction, **kwargs)
        self.transform = transform

    def _execute(self, context: dict[str, Any]) -> Any:
        """Parse input and update context.

        The base FlowNode._build_context_from_messages already handles extraction and set operations.
        This just passes through or transforms the current_input if needed.

        Args:
            context: Execution context.

        Returns:
            Updated context with extracted/set values.
        """
        # The parent class _build_context_from_messages has already done extraction and set operations
        # Here we just optionally transform the current_input
        if self.transform:
            # Could implement transformation logic here
            # For now, just pass through
            pass

        return context


class AgentPromptNode(FlowNode):
    """Node that executes a prompt using a specific agent in team flowcharts.

    This node is designed for multi-agent team workflows where different
    agents (roles) collaborate. Each agent brings its own system prompt,
    temperature settings, and conversation history.
    """

    def __init__(
        self,
        agent: str,
        prompt: str,
        extraction: Optional[dict[str, str]] = None,
        model: Optional[str] = None,
        context_name: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize AgentPromptNode.

        Args:
            agent: Agent identifier (name of the agent to use).
            prompt: Prompt template with placeholders.
            extraction: Regex patterns for extracting values.
            model: Optional model override for this specific call.
            context_name: Optional context name for the agent to use.
            **kwargs: Additional arguments passed to FlowNode.
        """
        super().__init__(extraction, **kwargs)
        self.agent = agent
        self.prompt = prompt
        self.model = model
        self.context_name = context_name

    def _execute(self, context: dict[str, Any]) -> Any:
        """Execute the prompt using the specified agent.

        Args:
            context: Execution context (must include 'agents' dict).

        Returns:
            Dict with agent response.

        Raises:
            ValueError: If agent is not found in context['agents'].
        """
        # Get the agent from context
        agents = context.get("agents", {})
        if self.agent not in agents:
            raise ValueError(
                f"Agent '{self.agent}' not found in context. "
                "Team flowcharts must include agents in context['agents']."
            )

        agent_instance = agents[self.agent]

        # Format the prompt with context values
        formatted_prompt = self._stateful_format(self.prompt, context)

        # Execute using the agent with optional model override
        response = agent_instance.send(
            formatted_prompt,
            context_name=self.context_name,
            model=self.model,
        )

        # Return response
        return {
            "formatted_prompt": formatted_prompt,
            "current_input": response,
            f"{self.agent}_response": response,
        }


class GetHistoryNode(FlowNode):
    """Node that extracts conversation history from an agent.

    This allows team flowcharts to inspect and use an agent's conversation
    history for context-sharing or coordination purposes.
    """

    def __init__(
        self,
        agent: str,
        save_to: str = "agent_history",
        context_name: Optional[str] = None,
        extraction: Optional[dict[str, str]] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize GetHistoryNode.

        Args:
            agent: Agent identifier to get history from.
            save_to: State variable name to save history to.
            context_name: Optional specific context to get history from.
            extraction: Regex patterns for extracting values.
            **kwargs: Additional arguments passed to FlowNode.
        """
        super().__init__(extraction, **kwargs)
        self.agent = agent
        self.save_to = save_to
        self.context_name = context_name

    def _execute(self, context: dict[str, Any]) -> Any:
        """Extract agent history and return it.

        Args:
            context: Execution context (must include 'agents' dict).

        Returns:
            Dict with agent history.

        Raises:
            ValueError: If agent is not found in context['agents'].
        """
        agents = context.get("agents", {})
        if self.agent not in agents:
            raise ValueError(f"Agent '{self.agent}' not found in context.")

        agent_instance = agents[self.agent]

        # Determine which context to use
        ctx_name = self.context_name or agent_instance.current_context
        if not ctx_name or ctx_name not in agent_instance.contexts:
            return {self.save_to: [], "current_input": []}

        # Get the message history
        agent_context = agent_instance.contexts[ctx_name]
        history = agent_context.message_history.copy()

        return {self.save_to: history, "current_input": history}


class SetHistoryNode(FlowNode):
    """Node that sets/injects conversation history into an agent.

    This allows team flowcharts to share context between agents or
    restore previous conversation states.
    """

    def __init__(
        self,
        agent: str,
        history_from: str = "agent_history",
        context_name: Optional[str] = None,
        mode: str = "replace",
        extraction: Optional[dict[str, str]] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize SetHistoryNode.

        Args:
            agent: Agent identifier to set history for.
            history_from: State variable containing history to set.
            context_name: Optional specific context to set history in.
            mode: How to set history - 'replace' (default) or 'append'.
            extraction: Regex patterns for extracting values.
            **kwargs: Additional arguments passed to FlowNode.
        """
        super().__init__(extraction, **kwargs)
        self.agent = agent
        self.history_from = history_from
        self.context_name = context_name
        self.mode = mode

    def _execute(self, context: dict[str, Any]) -> Any:
        """Set agent history from context variable.

        Args:
            context: Execution context (must include 'agents' dict).

        Returns:
            Dict confirming operation (agent context is modified in place).

        Raises:
            ValueError: If agent is not found or history data is invalid.
        """
        agents = context.get("agents", {})
        if self.agent not in agents:
            raise ValueError(f"Agent '{self.agent}' not found in context.")

        agent_instance = agents[self.agent]

        # Get the history data from context
        history = context.get(self.history_from, [])
        if not isinstance(history, list):
            raise ValueError(
                f"History data in '{self.history_from}' must be a list, got {type(history)}"
            )

        # Determine which context to use
        ctx_name = self.context_name or agent_instance.current_context
        if not ctx_name:
            raise ValueError(f"No context available for agent '{self.agent}'")

        # Ensure context exists
        if ctx_name not in agent_instance.contexts:
            agent_instance.create_context(ctx_name)

        agent_context = agent_instance.contexts[ctx_name]

        # Set history based on mode
        if self.mode == "replace":
            agent_context.message_history = history.copy()
        elif self.mode == "append":
            agent_context.message_history.extend(history)
        else:
            raise ValueError(f"Unknown mode: {self.mode}. Use 'replace' or 'append'.")

        return {"current_input": f"History set for agent '{self.agent}'"}


class ChatInputNode(InputNode):
    """Input node that receives user input from the chat interface.

    This is the default input node for interactive flowcharts.
    """

    def __init__(
        self,
        prompt_message: str = "Enter your input:",
        save_to: str = "user_input",
        extraction: Optional[dict[str, str]] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize ChatInputNode.

        Args:
            prompt_message: Message to display when requesting input.
            save_to: State variable name to save input to.
            extraction: Regex patterns for extracting values.
            **kwargs: Additional arguments passed to InputNode.
        """
        super().__init__(extraction, **kwargs)
        self.prompt_message = prompt_message
        self.save_to = save_to

    def _execute(self, context: dict[str, Any]) -> Any:
        """Get user input from chat.

        Args:
            context: Execution context.

        Returns:
            Dict with user input.
        """
        # In actual implementation, this would interface with the chat system
        # For now, use default input if available, otherwise empty string
        user_input = context.get("default", "")
        return {self.save_to: user_input, "current_input": user_input}


class ChatOutputNode(OutputNode):
    """Output node that displays data in the chat interface.

    This is the default output node for interactive flowcharts.
    """

    def __init__(
        self,
        source: str = "current_input",
        format_template: Optional[str] = None,
        extraction: Optional[dict[str, str]] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize ChatOutputNode.

        Args:
            source: State variable to output (default: "current_input").
            format_template: Optional template for formatting output (e.g., "Result: {source}").
            extraction: Regex patterns for extracting values.
            **kwargs: Additional arguments passed to OutputNode.
        """
        super().__init__(extraction, **kwargs)
        self.source = source
        self.format_template = format_template

    def _execute(self, context: dict[str, Any]) -> Any:
        """Display output to chat.

        Args:
            context: Execution context.

        Returns:
            Dict with output stored.
        """
        # Get the data to output
        output_data = context.get(self.source, "")

        # Apply formatting if template provided
        if self.format_template:
            output_data = self._stateful_format(self.format_template, context)

        # Return for the chat system to display
        return {"current_input": str(output_data), "chat_output": str(output_data)}


class FileInputNode(InputNode):
    """Input node that reads data from a file."""

    def __init__(
        self,
        file_path: str,
        save_to: str = "file_content",
        encoding: str = "utf-8",
        extraction: Optional[dict[str, str]] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize FileInputNode.

        Args:
            file_path: Path to the file to read (can contain {placeholders}).
            save_to: State variable name to save file content to.
            encoding: File encoding (default: utf-8).
            extraction: Regex patterns for extracting values.
            **kwargs: Additional arguments passed to InputNode.
        """
        super().__init__(extraction, **kwargs)
        self.file_path = file_path
        self.save_to = save_to
        self.encoding = encoding

    def _execute(self, context: dict[str, Any]) -> Any:
        """Read file and return content.

        Args:
            context: Execution context.

        Returns:
            Dict with file content.

        Raises:
            FileNotFoundError: If file doesn't exist.
            IOError: If file can't be read.
        """
        # Format file path with context variables
        formatted_path = self._stateful_format(self.file_path, context)

        try:
            with open(formatted_path, "r", encoding=self.encoding) as f:
                content = f.read()

            return {self.save_to: content, "current_input": content}

        except FileNotFoundError:
            raise FileNotFoundError(f"File not found: {formatted_path}")
        except Exception as e:
            raise IOError(f"Error reading file {formatted_path}: {str(e)}")


class FileOutputNode(OutputNode):
    """Output node that writes data to a file."""

    def __init__(
        self,
        file_path: str,
        source: str = "current_input",
        mode: str = "w",
        encoding: str = "utf-8",
        extraction: Optional[dict[str, str]] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize FileOutputNode.

        Args:
            file_path: Path to write to (can contain {placeholders}).
            source: State variable to write to file.
            mode: File mode ('w' for write, 'a' for append).
            encoding: File encoding (default: utf-8).
            extraction: Regex patterns for extracting values.
            **kwargs: Additional arguments passed to OutputNode.
        """
        super().__init__(extraction, **kwargs)
        self.file_path = file_path
        self.source = source
        self.mode = mode
        self.encoding = encoding

    def _execute(self, context: dict[str, Any]) -> Any:
        """Write data to file.

        Args:
            context: Execution context.

        Returns:
            Dict with file path and data.

        Raises:
            IOError: If file can't be written.
        """
        # Format file path with context variables
        formatted_path = self._stateful_format(self.file_path, context)

        # Get data to write
        data = context.get(self.source, "")

        try:
            with open(formatted_path, self.mode, encoding=self.encoding) as f:
                f.write(str(data))

            return {"file_output_path": formatted_path, "current_input": data}

        except Exception as e:
            raise IOError(f"Error writing to file {formatted_path}: {str(e)}")


def create_node(node_type: str, data: dict[str, Any]) -> Optional[FlowNode]:
    """Factory function to create nodes from configuration.

    Args:
        node_type: Type of node to create.
        data: Configuration dictionary.

    Returns:
        FlowNode instance.

    Raises:
        ValueError: If node type is unknown.
    """
    node_classes = {
        "prompt": PromptNode,
        "promptnode": PromptNode,
        "custom": CustomNode,
        "customnode": CustomNode,
        "toolcall": ToolCallNode,
        "toolcallnode": ToolCallNode,
        "textparse": TextParseNode,
        "textparsenode": TextParseNode,
        "agentprompt": AgentPromptNode,
        "agentpromptnode": AgentPromptNode,
        "gethistory": GetHistoryNode,
        "gethistorynode": GetHistoryNode,
        "sethistory": SetHistoryNode,
        "sethistorynode": SetHistoryNode,
        "chatinput": ChatInputNode,
        "chatinputnode": ChatInputNode,
        "chatoutput": ChatOutputNode,
        "chatoutputnode": ChatOutputNode,
        "fileinput": FileInputNode,
        "fileinputnode": FileInputNode,
        "fileoutput": FileOutputNode,
        "fileoutputnode": FileOutputNode,
    }
    node_class = node_classes.get(node_type.replace("_", "").lower(), None)
    if not node_class:
        raise ValueError(f"Unknown node type: {node_type}")
    return node_class.from_dict(data)
