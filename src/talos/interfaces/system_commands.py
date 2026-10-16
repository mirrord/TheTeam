"""System commands shared across Talos interfaces.

A *system command* is a special user input that a Talos interface intercepts
and handles locally instead of forwarding it to the agent for inference.  The
canonical example is :data:`SYSTEM_RESET_COMMAND` (``"system reset"``), which
clears the agent's conversation history for the active context and begins a
fresh session.

Interfaces call :func:`handle_system_command` with each raw user input.  When
it returns a string the input was a recognised system command and the returned
text is the user-facing confirmation to display; when it returns ``None`` the
input is an ordinary prompt that should be sent to the agent as usual.
"""

from __future__ import annotations

from typing import Optional

from pithos.agent import Agent

#: Exact input (ignoring surrounding whitespace) that resets the conversation.
SYSTEM_RESET_COMMAND = "system reset"

#: Confirmation shown to the user after a successful reset.
SYSTEM_RESET_RESPONSE = "[system] History cleared — new session started."


def is_system_command(text: str) -> bool:
    """Return ``True`` when *text* is a recognised system command."""
    return text.strip() == SYSTEM_RESET_COMMAND


def handle_system_command(
    agent: Agent, text: str, context_name: Optional[str] = None
) -> Optional[str]:
    """Intercept and execute a system command contained in *text*.

    Args:
        agent: The agent whose state the command acts on.
        text: The raw user input.
        context_name: The context to act on.  Defaults to the agent's current
            context when omitted.

    Returns:
        A user-facing confirmation string when *text* is a recognised system
        command, otherwise ``None`` (the caller should forward *text* to the
        agent).
    """
    if text.strip() != SYSTEM_RESET_COMMAND:
        return None
    _reset_session(agent, context_name)
    return SYSTEM_RESET_RESPONSE


def _reset_session(agent: Agent, context_name: Optional[str] = None) -> None:
    """Clear the message history of *context_name*, starting a new session.

    The context's system prompt is preserved; only the conversation history is
    discarded.  Missing contexts are ignored so a reset is always safe.
    """
    ctx = context_name or agent.current_context
    if ctx and ctx in agent.contexts:
        agent.clear_context(ctx)


__all__ = [
    "SYSTEM_RESET_COMMAND",
    "SYSTEM_RESET_RESPONSE",
    "is_system_command",
    "handle_system_command",
]
