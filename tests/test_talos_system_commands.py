"""Tests for Talos system commands (e.g. ``system reset``)."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from pithos.context import AgentMsg, UserMsg
from talos.config import TalosConfig, TelegramConfig, build_agent
from talos.interfaces.system_commands import (
    SYSTEM_RESET_COMMAND,
    SYSTEM_RESET_RESPONSE,
    handle_system_command,
    is_system_command,
)
from talos.interfaces.telegram import TelegramInterface
from talos.interfaces.voice import VOICE_CONTEXT_NAME


def _make_agent():
    return build_agent(TalosConfig())


def _seed_history(agent, context_name: str = "default") -> None:
    ctx = agent.contexts[context_name]
    ctx.add_message(UserMsg("hello"))
    ctx.add_message(AgentMsg("hi there"))


# ---------------------------------------------------------------------------
# is_system_command
# ---------------------------------------------------------------------------


def test_is_system_command_matches_exact_string() -> None:
    assert is_system_command("system reset")


def test_is_system_command_tolerates_surrounding_whitespace() -> None:
    assert is_system_command("  system reset  ")
    assert is_system_command("system reset\n")


def test_is_system_command_rejects_non_commands() -> None:
    assert not is_system_command("please system reset now")
    assert not is_system_command("System Reset")
    assert not is_system_command("reset")
    assert not is_system_command("")


# ---------------------------------------------------------------------------
# handle_system_command
# ---------------------------------------------------------------------------


def test_handle_returns_none_for_ordinary_prompt() -> None:
    agent = _make_agent()
    _seed_history(agent)
    assert handle_system_command(agent, "what's the weather?") is None
    # History must be untouched for a normal prompt.
    assert len(agent.contexts["default"].message_history) == 2


def test_handle_system_reset_clears_history_and_confirms() -> None:
    agent = _make_agent()
    _seed_history(agent)
    assert agent.contexts["default"].message_history

    result = handle_system_command(agent, "system reset")

    assert result == SYSTEM_RESET_RESPONSE
    assert agent.contexts["default"].message_history == []


def test_handle_system_reset_preserves_system_prompt() -> None:
    agent = _make_agent()
    agent.set_system_prompt("You are Talos.")
    _seed_history(agent)

    handle_system_command(agent, "system reset")

    assert agent.contexts["default"].get_system_prompt() == "You are Talos."


def test_handle_system_reset_targets_named_context() -> None:
    agent = _make_agent()
    agent.copy_context("default", "other")
    _seed_history(agent, "default")
    _seed_history(agent, "other")

    handle_system_command(agent, SYSTEM_RESET_COMMAND, context_name="other")

    assert agent.contexts["other"].message_history == []
    # The untargeted context is left alone.
    assert len(agent.contexts["default"].message_history) == 2


def test_handle_system_reset_missing_context_is_safe() -> None:
    agent = _make_agent()
    # Should not raise even though "ghost" does not exist.
    assert (
        handle_system_command(agent, "system reset", context_name="ghost")
        == SYSTEM_RESET_RESPONSE
    )


# ---------------------------------------------------------------------------
# Shell integration (via interactive_chat command_handler hook)
# ---------------------------------------------------------------------------


def test_interactive_chat_runs_command_handler(capsys) -> None:
    from pithos.agent.cli import interactive_chat

    agent = MagicMock()
    agent.stream.return_value = iter(["ignored"])

    # First input is a command, second ends the session.
    inputs = iter(["system reset", EOFError()])

    def fake_input(_prompt: str = "") -> str:
        value = next(inputs)
        if isinstance(value, BaseException):
            raise value
        return value

    handler = MagicMock(return_value="[system] done")

    with patch("builtins.input", side_effect=fake_input):
        try:
            interactive_chat(agent, command_handler=handler)
        except EOFError:
            pass

    handler.assert_called_once_with("system reset")
    # The agent is never prompted for a handled command.
    agent.stream.assert_not_called()
    assert "[system] done" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# Telegram integration
# ---------------------------------------------------------------------------


def _make_update(user_id: int, text: str) -> MagicMock:
    update = MagicMock()
    update.effective_user.id = user_id
    update.effective_user.first_name = f"User{user_id}"
    update.message.text = text
    update.message.reply_text = AsyncMock()
    return update


def test_telegram_system_reset_clears_context_without_prompting() -> None:
    agent = _make_agent()
    iface = TelegramInterface(agent, TelegramConfig(bot_token="t"))

    ctx_name = iface._ensure_context(7)
    _seed_history(agent, ctx_name)

    called = {"sent": False}

    def fake_send(*_args, **_kwargs):
        called["sent"] = True
        return "should not run"

    agent.send = fake_send  # type: ignore[assignment]

    update = _make_update(7, "system reset")
    asyncio.run(iface._handle_message(update, MagicMock()))

    assert called["sent"] is False
    assert agent.contexts[ctx_name].message_history == []
    update.message.reply_text.assert_awaited_once_with(SYSTEM_RESET_RESPONSE)
