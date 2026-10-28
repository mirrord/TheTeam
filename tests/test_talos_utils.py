"""Tests for talos.utils.clean_agent_response."""

from __future__ import annotations

from talos.utils import clean_agent_response


def _call(command: str) -> str:
    """Render a JSON run tool call inside a fenced ``tool_call`` block."""
    return (
        '```tool_call\n'
        '{"name": "run", "arguments": {"command": "' + command + '"}}\n'
        '```'
    )


def test_passthrough_when_no_tool_calls() -> None:
    text = "Just a normal reply with no tool calls."
    assert clean_agent_response(text) == text


def test_empty_string_passthrough() -> None:
    assert clean_agent_response("") == ""


def test_strips_single_call() -> None:
    text = "Let me check.\n" + _call("git status") + "\nDone."
    cleaned = clean_agent_response(text)
    assert "tool_call" not in cleaned
    assert "git status" not in cleaned
    assert "Let me check." in cleaned
    assert "Done." in cleaned


def test_strips_fenced_block() -> None:
    text = "Result:\n" + _call("ls -la") + "\nThat's it."
    cleaned = clean_agent_response(text)
    assert "```" not in cleaned
    assert "ls -la" not in cleaned
    assert "Result:" in cleaned
    assert "That's it." in cleaned


def test_strips_inline_call() -> None:
    text = "Running " + _call("echo hi") + " for you."
    cleaned = clean_agent_response(text)
    assert "tool_call" not in cleaned
    assert "echo hi" not in cleaned
    assert "Running" in cleaned
    assert "for you." in cleaned


def test_strips_multiple_calls() -> None:
    text = "Step 1: " + _call("echo a") + "\nStep 2: " + _call("echo b") + "\nFinished."
    cleaned = clean_agent_response(text)
    assert "tool_call" not in cleaned
    assert "echo a" not in cleaned
    assert "echo b" not in cleaned
    assert "Step 1:" in cleaned
    assert "Step 2:" in cleaned
    assert "Finished." in cleaned


def test_collapses_extra_blank_lines() -> None:
    # Several tool-call blocks back-to-back would leave blank lines after removal.
    text = (
        "Start.\n"
        + _call("a") + "\n"
        + _call("b") + "\n"
        + _call("c") + "\nEnd."
    )
    cleaned = clean_agent_response(text)
    # No more than two consecutive newlines.
    assert "\n\n\n" not in cleaned
    assert cleaned.startswith("Start.")
    assert cleaned.endswith("End.")
