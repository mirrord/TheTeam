"""Tests for talos.utils.clean_agent_response."""

from __future__ import annotations

from talos.utils import clean_agent_response


def test_passthrough_when_no_tool_calls() -> None:
    text = "Just a normal reply with no tool calls."
    assert clean_agent_response(text) == text


def test_empty_string_passthrough() -> None:
    assert clean_agent_response("") == ""


def test_strips_run_prefix() -> None:
    text = "Let me check.\nRUN: git status\nDone."
    cleaned = clean_agent_response(text)
    assert "RUN:" not in cleaned
    assert "git status" not in cleaned
    assert "Let me check." in cleaned
    assert "Done." in cleaned


def test_strips_bracket_form() -> None:
    text = "Result:\n[RUN]ls -la[/RUN]\nThat's it."
    cleaned = clean_agent_response(text)
    assert "[RUN]" not in cleaned
    assert "[/RUN]" not in cleaned
    assert "ls -la" not in cleaned
    assert "Result:" in cleaned
    assert "That's it." in cleaned


def test_strips_function_form() -> None:
    text = "Running run(echo hi) for you."
    cleaned = clean_agent_response(text)
    assert "run(echo hi)" not in cleaned
    assert "Running" in cleaned
    assert "for you." in cleaned


def test_strips_multiple_calls() -> None:
    text = "Step 1: RUN: echo a\nStep 2: RUN: echo b\nFinished."
    cleaned = clean_agent_response(text)
    assert "RUN:" not in cleaned
    assert "echo a" not in cleaned
    assert "echo b" not in cleaned
    assert "Step 1:" in cleaned
    assert "Step 2:" in cleaned
    assert "Finished." in cleaned


def test_collapses_extra_blank_lines() -> None:
    # Three RUN: lines back-to-back would leave three blank lines after removal.
    text = "Start.\nRUN: a\nRUN: b\nRUN: c\nEnd."
    cleaned = clean_agent_response(text)
    # No more than two consecutive newlines.
    assert "\n\n\n" not in cleaned
    assert cleaned.startswith("Start.")
    assert cleaned.endswith("End.")
