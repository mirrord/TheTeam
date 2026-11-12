"""Tests for talos.interfaces — shell, voice (mocked), telegram (mocked)."""

from __future__ import annotations

import asyncio
from typing import Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from talos.config import (
    DEFAULT_WAKE_WORD,
    TalosConfig,
    TelegramConfig,
    VoiceConfig,
    build_agent,
)
from talos.interfaces.shell import ShellInterface
from talos.interfaces.telegram import TelegramInterface, _context_name_for_user

# ---------------------------------------------------------------------------
# Shell
# ---------------------------------------------------------------------------


def test_shell_delegates_to_interactive_chat() -> None:
    agent = MagicMock()
    iface = ShellInterface(agent, verbose=True)
    with patch("talos.interfaces.shell.interactive_chat") as chat:
        iface.run()
    chat.assert_called_once()
    args, kwargs = chat.call_args
    assert args == (agent,)
    assert kwargs["verbose"] is True
    assert callable(kwargs["command_handler"])


# ---------------------------------------------------------------------------
# Telegram
# ---------------------------------------------------------------------------


def _make_agent():
    return build_agent(TalosConfig())


def test_telegram_requires_token() -> None:
    with pytest.raises(ValueError, match="bot_token"):
        TelegramInterface(_make_agent(), TelegramConfig(bot_token=""))


def test_telegram_ensure_context_per_user() -> None:
    agent = _make_agent()
    iface = TelegramInterface(agent, TelegramConfig(bot_token="t"))

    ctx_a = iface._ensure_context(1)
    ctx_b = iface._ensure_context(2)
    assert ctx_a == _context_name_for_user(1)
    assert ctx_b == _context_name_for_user(2)
    assert ctx_a in agent.contexts
    assert ctx_b in agent.contexts
    # Idempotent.
    assert iface._ensure_context(1) == ctx_a


def _make_update(user_id: int, text: str) -> MagicMock:
    update = MagicMock()
    update.effective_user.id = user_id
    update.effective_user.first_name = f"User{user_id}"
    update.message.text = text
    update.message.reply_text = AsyncMock()
    return update


def test_telegram_handle_message_routes_to_user_context() -> None:
    agent = _make_agent()
    iface = TelegramInterface(agent, TelegramConfig(bot_token="t"))

    captured: dict[str, object] = {}

    def fake_send(text, context_name=None, **kwargs):
        captured["text"] = text
        captured["context"] = context_name
        return "reply text"

    agent.send = fake_send  # type: ignore[assignment]

    update = _make_update(42, "hello there")
    asyncio.run(iface._handle_message(update, MagicMock()))

    assert captured == {"text": "hello there", "context": "telegram_42"}
    update.message.reply_text.assert_awaited_once_with("reply text")


def test_telegram_show_streams_to_stdout(capsys) -> None:
    """With show=True, the reply is streamed to stdout and sent to Telegram."""
    agent = _make_agent()
    iface = TelegramInterface(agent, TelegramConfig(bot_token="t"), show=True)

    def fake_stream(text, context_name=None, **kwargs):
        assert context_name == "telegram_5"
        yield "Hello"
        yield ", "
        yield "world"

    agent.stream = fake_stream  # type: ignore[assignment]
    agent._pending_image_paths = []

    update = _make_update(5, "hi")
    asyncio.run(iface._handle_message(update, MagicMock()))

    out = capsys.readouterr().out
    assert "[User5 (5)] hi" in out
    assert "Hello, world" in out
    update.message.reply_text.assert_awaited_once_with("Hello, world")


def test_telegram_show_displays_tool_call_and_output(capsys) -> None:
    """With show=True, tool results are printed with call/output separation."""
    agent = _make_agent()
    iface = TelegramInterface(agent, TelegramConfig(bot_token="t"), show=True)

    def fake_stream(text, context_name=None, status_callback=None, **kwargs):
        yield "Let me check. "
        if status_callback is not None:
            status_callback("tool_call", "ls -la")
            status_callback("tool_result", "file_a.txt\nfile_b.txt")
        yield "Here are the files."

    agent.stream = fake_stream  # type: ignore[assignment]
    agent._pending_image_paths = []

    update = _make_update(9, "list files")
    asyncio.run(iface._handle_message(update, MagicMock()))

    out = capsys.readouterr().out
    # Tool call header and command are shown.
    assert "tool call: ls -la" in out
    # Tool output block is clearly separated from the call.
    assert "tool output:" in out
    assert "file_a.txt" in out
    assert "file_b.txt" in out
    assert "end tool output" in out
    # Both the pre- and post-tool agent text made it to Telegram.
    update.message.reply_text.assert_awaited_once_with(
        "Let me check. Here are the files."
    )


def test_telegram_show_false_uses_send(capsys) -> None:
    """With show=False (default), send() is used and nothing is printed."""
    agent = _make_agent()
    iface = TelegramInterface(agent, TelegramConfig(bot_token="t"))

    def fake_send(text, context_name=None, **kwargs):
        return "quiet reply"

    agent.send = fake_send  # type: ignore[assignment]
    agent._pending_image_paths = []

    update = _make_update(6, "hi")
    asyncio.run(iface._handle_message(update, MagicMock()))

    assert capsys.readouterr().out == ""
    update.message.reply_text.assert_awaited_once_with("quiet reply")


def test_telegram_start_creates_context_and_greets() -> None:
    agent = _make_agent()
    iface = TelegramInterface(agent, TelegramConfig(bot_token="t"))
    update = _make_update(7, "/start")
    asyncio.run(iface._start(update, MagicMock()))
    assert _context_name_for_user(7) in agent.contexts
    update.message.reply_text.assert_awaited_once()


def _make_update_with_photo(user_id: int, text: str) -> MagicMock:
    update = _make_update(user_id, text)
    update.message.reply_photo = AsyncMock()
    return update


def test_telegram_sends_image_when_agent_generates_one(tmp_path) -> None:
    """reply_photo is called for each image path returned by the agent."""
    img = tmp_path / "photo.png"
    img.write_bytes(b"\x89PNG fake")

    agent = _make_agent()
    iface = TelegramInterface(agent, TelegramConfig(bot_token="t"))

    def fake_send(text, context_name=None, **kwargs):
        return "here is your image"

    agent.send = fake_send  # type: ignore[assignment]
    agent._pending_image_paths = [str(img)]

    update = _make_update_with_photo(1, "generate a cat")
    asyncio.run(iface._handle_message(update, MagicMock()))

    update.message.reply_text.assert_awaited()
    assert update.message.reply_photo.await_count == 1


def test_telegram_no_image_sent_when_none_generated() -> None:
    """reply_photo is never called when the agent produces no images."""
    agent = _make_agent()
    iface = TelegramInterface(agent, TelegramConfig(bot_token="t"))

    def fake_send(text, context_name=None, **kwargs):
        return "just text"

    agent.send = fake_send  # type: ignore[assignment]
    agent._pending_image_paths = []

    update = _make_update_with_photo(2, "hello")
    asyncio.run(iface._handle_message(update, MagicMock()))

    update.message.reply_photo.assert_not_awaited()


def test_telegram_skips_missing_image_without_crash(tmp_path) -> None:
    """A non-existent image path is skipped gracefully; text reply still sent."""
    agent = _make_agent()
    iface = TelegramInterface(agent, TelegramConfig(bot_token="t"))

    def fake_send(text, context_name=None, **kwargs):
        return "response"

    agent.send = fake_send  # type: ignore[assignment]
    agent._pending_image_paths = [str(tmp_path / "ghost.png")]  # does not exist

    update = _make_update_with_photo(3, "make something")
    asyncio.run(iface._handle_message(update, MagicMock()))

    update.message.reply_text.assert_awaited()
    update.message.reply_photo.assert_not_awaited()


def test_telegram_sends_report_contents_when_agent_produces_one(tmp_path) -> None:
    """The full report file content is sent, in addition to the agent's reply."""
    report = tmp_path / "news_report.md"
    report.write_text("# Report\n\nSome findings.", encoding="utf-8")

    agent = _make_agent()
    iface = TelegramInterface(agent, TelegramConfig(bot_token="t"))

    def fake_send(text, context_name=None, **kwargs):
        return "here is a summary"

    agent.send = fake_send  # type: ignore[assignment]
    agent._pending_image_paths = []
    agent._pending_report_paths = [str(report)]

    update = _make_update(10, "research something")
    asyncio.run(iface._handle_message(update, MagicMock()))

    calls = [c.args[0] for c in update.message.reply_text.await_args_list]
    assert "here is a summary" in calls
    assert any("news_report.md" in c for c in calls)
    assert any("Some findings." in c for c in calls)


def test_telegram_report_split_across_multiple_messages(tmp_path) -> None:
    """Reports longer than the chunk size are split into multiple messages."""
    long_content = "x" * 9000
    report = tmp_path / "long_report.md"
    report.write_text(long_content, encoding="utf-8")

    agent = _make_agent()
    iface = TelegramInterface(agent, TelegramConfig(bot_token="t"))

    def fake_send(text, context_name=None, **kwargs):
        return "summary"

    agent.send = fake_send  # type: ignore[assignment]
    agent._pending_image_paths = []
    agent._pending_report_paths = [str(report)]

    update = _make_update(11, "research something long")
    asyncio.run(iface._handle_message(update, MagicMock()))

    calls = [c.args[0] for c in update.message.reply_text.await_args_list]
    # summary + header + 3 content chunks (9000 / 4000 -> 3 chunks)
    content_chunks = [c for c in calls if set(c) == {"x"}]
    assert len(content_chunks) == 3
    assert sum(len(c) for c in content_chunks) == len(long_content)


def test_telegram_skips_missing_report_without_crash() -> None:
    """A non-existent report path is skipped gracefully; text reply still sent."""
    agent = _make_agent()
    iface = TelegramInterface(agent, TelegramConfig(bot_token="t"))

    def fake_send(text, context_name=None, **kwargs):
        return "response"

    agent.send = fake_send  # type: ignore[assignment]
    agent._pending_image_paths = []
    agent._pending_report_paths = ["/nonexistent/ghost_report.md"]

    update = _make_update(12, "research something")
    asyncio.run(iface._handle_message(update, MagicMock()))

    calls = [c.args[0] for c in update.message.reply_text.await_args_list]
    assert calls == ["response"]


def test_telegram_sends_attachment_as_document_not_inline(tmp_path) -> None:
    """Attachment-only artifacts (e.g. coding reports) are sent as files, not inlined."""
    report = tmp_path / "code_task.json"
    report.write_text('{"summary": "did the thing", "diff": "secret"}', encoding="utf-8")

    agent = _make_agent()
    iface = TelegramInterface(agent, TelegramConfig(bot_token="t"))

    def fake_send(text, context_name=None, **kwargs):
        return "I made the change."

    agent.send = fake_send  # type: ignore[assignment]
    agent._pending_image_paths = []
    agent._pending_report_paths = []
    agent._pending_attachment_paths = [str(report)]

    update = _make_update(20, "code do the thing")
    update.message.reply_document = AsyncMock()
    asyncio.run(iface._handle_message(update, MagicMock()))

    # The agent's prose reply is sent as text.
    text_calls = [c.args[0] for c in update.message.reply_text.await_args_list]
    assert "I made the change." in text_calls
    # The report file is attached as a document.
    update.message.reply_document.assert_awaited_once()
    # Its contents are NOT dumped into any text message.
    assert not any("secret" in c for c in text_calls)


def test_telegram_skips_missing_attachment_without_crash(tmp_path) -> None:
    """A non-existent attachment path is skipped gracefully; text reply still sent."""
    agent = _make_agent()
    iface = TelegramInterface(agent, TelegramConfig(bot_token="t"))

    def fake_send(text, context_name=None, **kwargs):
        return "response"

    agent.send = fake_send  # type: ignore[assignment]
    agent._pending_image_paths = []
    agent._pending_report_paths = []
    agent._pending_attachment_paths = ["/nonexistent/ghost_trajectory.json"]

    update = _make_update(21, "code something")
    update.message.reply_document = AsyncMock()
    asyncio.run(iface._handle_message(update, MagicMock()))

    calls = [c.args[0] for c in update.message.reply_text.await_args_list]
    assert calls == ["response"]
    update.message.reply_document.assert_not_awaited()


def _make_prompt_message(chat_id: int, message_id: int) -> MagicMock:
    msg = MagicMock()
    msg.chat_id = chat_id
    msg.message_id = message_id
    return msg


def _make_reaction_update(
    chat_id: int, message_id: int, user_id: Optional[int], emojis: list[str]
) -> MagicMock:
    update = MagicMock()
    reaction = update.message_reaction
    reaction.chat.id = chat_id
    reaction.message_id = message_id
    reaction.user = None if user_id is None else MagicMock(id=user_id)
    reaction.new_reaction = [MagicMock(emoji=e) for e in emojis]
    return update


def test_telegram_resolve_reaction_only_matches_its_prompt() -> None:
    """A reaction resolves only the prompt it is on, and only for the asked user."""
    from talos.interfaces.telegram import _PendingConfirm

    iface = TelegramInterface(_make_agent(), TelegramConfig(bot_token="t"))

    # No pending prompt for that message → nothing resolved.
    assert iface._resolve_reaction(1, 10, 5, ["\U0001f44d"]) is False

    pending = _PendingConfirm(user_id=5)
    iface._pending_confirms[(1, 10)] = pending
    # Wrong user and wrong message are ignored.
    assert iface._resolve_reaction(1, 10, 6, ["\U0001f44d"]) is False
    assert iface._resolve_reaction(1, 11, 5, ["\U0001f44d"]) is False
    assert not pending.event.is_set()

    assert iface._resolve_reaction(1, 10, 5, ["\U0001f44d"]) is True
    assert pending.approved is True
    # A second reaction on an already-answered prompt changes nothing.
    assert iface._resolve_reaction(1, 10, 5, ["\U0001f44e"]) is False
    assert pending.approved is True


def test_telegram_any_other_reaction_denies() -> None:
    from talos.interfaces.telegram import _PendingConfirm

    iface = TelegramInterface(_make_agent(), TelegramConfig(bot_token="t"))
    pending = _PendingConfirm(user_id=5)
    iface._pending_confirms[(1, 10)] = pending

    assert iface._resolve_reaction(1, 10, 5, ["\u2764"]) is True
    assert pending.approved is False


def test_telegram_reaction_handler_ignores_removal_and_anonymous() -> None:
    from talos.interfaces.telegram import _PendingConfirm

    iface = TelegramInterface(_make_agent(), TelegramConfig(bot_token="t"))
    pending = _PendingConfirm(user_id=5)
    iface._pending_confirms[(1, 10)] = pending

    asyncio.run(iface._handle_reaction(_make_reaction_update(1, 10, 5, []), MagicMock()))
    asyncio.run(
        iface._handle_reaction(
            _make_reaction_update(1, 10, None, ["\U0001f44d"]), MagicMock()
        )
    )
    assert not pending.event.is_set()

    asyncio.run(
        iface._handle_reaction(
            _make_reaction_update(1, 10, 5, ["\U0001f44d"]), MagicMock()
        )
    )
    assert pending.event.is_set() and pending.approved is True


def _run_confirm_scenario(user_id: int, emoji: str) -> tuple[dict, MagicMock]:
    """Start a turn that needs approval, answer it with a reaction *emoji*."""
    captured: dict[str, object] = {}

    async def scenario() -> MagicMock:
        agent = _make_agent()
        iface = TelegramInterface(agent, TelegramConfig(bot_token="t"))

        def fake_send(text, context_name=None, **kwargs):
            captured["approved"] = agent.tool_confirm_callback(
                "powershell -Command whoami"
            )
            captured["sends"] = captured.get("sends", 0) + 1  # type: ignore[operator]
            return "all done"

        agent.send = fake_send  # type: ignore[assignment]
        agent._pending_image_paths = []
        agent._pending_report_paths = []
        agent._pending_attachment_paths = []

        update1 = _make_update(user_id, "code do the thing")
        update1.message.reply_document = AsyncMock()
        prompt_msg = _make_prompt_message(chat_id=777, message_id=55)
        update1.message.reply_text = AsyncMock(return_value=prompt_msg)

        task = asyncio.create_task(iface._handle_message(update1, MagicMock()))
        for _ in range(300):
            await asyncio.sleep(0.01)
            if (777, 55) in iface._pending_confirms:
                break
        assert (777, 55) in iface._pending_confirms  # user was prompted

        # Regression: a text "yes" while the prompt is pending must neither
        # answer the prompt nor start a new agent turn.
        update_yes = _make_update(user_id, "yes")
        await iface._handle_message(update_yes, MagicMock())
        assert not iface._pending_confirms[(777, 55)].event.is_set()
        captured["yes_reply"] = update_yes.message.reply_text.await_args.args[0]

        await iface._handle_reaction(
            _make_reaction_update(777, 55, user_id, [emoji]), MagicMock()
        )
        await task
        return update1

    update1 = asyncio.run(scenario())
    return captured, update1


def test_telegram_tool_confirmation_approved_by_thumbs_up() -> None:
    captured, update1 = _run_confirm_scenario(100, "\U0001f44d")

    assert captured["approved"] is True
    assert captured["sends"] == 1  # the text "yes" never became a turn
    assert "react" in str(captured["yes_reply"]).lower()
    prompts = [c.args[0] for c in update1.message.reply_text.await_args_list]
    assert any("powershell -Command whoami" in p for p in prompts)


def test_telegram_tool_confirmation_denied_by_other_reaction() -> None:
    captured, _ = _run_confirm_scenario(101, "\U0001f44e")

    assert captured["approved"] is False
    assert captured["sends"] == 1


def test_telegram_build_application_subscribes_to_reactions() -> None:
    from telegram.ext import MessageReactionHandler

    iface = TelegramInterface(_make_agent(), TelegramConfig(bot_token="123:abc"))
    app = iface.build_application()

    handlers = [h for group in app.handlers.values() for h in group]
    assert any(isinstance(h, MessageReactionHandler) for h in handlers)
    assert "message_reaction" in iface._allowed_updates
    # Updates must be processed concurrently so a reaction is never queued
    # behind the turn that is blocked waiting for it.
    assert app.concurrent_updates > 1


def test_telegram_refuses_concurrent_turn_while_busy() -> None:
    """A new message while a turn is in flight must NOT start a second turn.

    Regression: the agent (and its shared tool-confirm channel) runs one turn
    at a time. If a message arrives mid-turn — before any approval prompt is
    registered — it previously spawned a concurrent turn, so the agent kept
    generating and sent extra approval prompts while an earlier one was still
    pending. The second message must instead be refused with a "please wait".
    """
    import threading

    async def scenario() -> None:
        agent = _make_agent()
        iface = TelegramInterface(agent, TelegramConfig(bot_token="t"))

        started = threading.Event()
        release = threading.Event()
        calls: list[str] = []

        def fake_send(text, context_name=None, **kwargs):
            calls.append(text)
            started.set()
            # Simulate a long-running turn that has NOT yet asked for approval.
            release.wait(timeout=5)
            return "done"

        agent.send = fake_send  # type: ignore[assignment]
        agent._pending_image_paths = []
        agent._pending_report_paths = []
        agent._pending_attachment_paths = []

        update1 = _make_update(200, "code do the thing")
        update1.message.reply_document = AsyncMock()
        task = asyncio.create_task(iface._handle_message(update1, MagicMock()))

        # Wait until the first turn is actually running in its worker thread.
        for _ in range(500):
            await asyncio.sleep(0.01)
            if started.is_set():
                break
        assert started.is_set()
        assert 200 not in iface._pending_confirms  # no approval pending yet

        # A second, unrelated message arrives mid-turn.
        update2 = _make_update(200, "another request")
        await iface._handle_message(update2, MagicMock())

        # It must be refused, not run as a concurrent turn.
        assert len(calls) == 1
        prompts = [c.args[0] for c in update2.message.reply_text.await_args_list]
        assert any("still working" in p.lower() or "wait" in p.lower() for p in prompts)

        # Let the first turn finish.
        release.set()
        await task
        assert len(calls) == 1

    asyncio.run(scenario())


# ---------------------------------------------------------------------------
# Voice — only test the constructor wiring; full pipeline requires audio deps.
# ---------------------------------------------------------------------------


def test_voice_interface_sets_up_voice_context() -> None:
    from talos.interfaces.voice import (
        VOICE_CONTEXT_NAME,
        VOICE_PROMPT_PREFIX,
        VoiceInterface,
    )

    agent = build_agent(TalosConfig())
    iface = VoiceInterface(agent, VoiceConfig())
    assert VOICE_CONTEXT_NAME in agent.contexts
    sp = agent.contexts[VOICE_CONTEXT_NAME].get_system_prompt()
    assert sp.startswith(VOICE_PROMPT_PREFIX)
    assert iface.config.wake_word == DEFAULT_WAKE_WORD


# ---------------------------------------------------------------------------
# TTS — mock kokoro-onnx so we don't download/load a real model.
# ---------------------------------------------------------------------------


def _patch_kokoro(monkeypatch, **kokoro_attrs):
    """Install a fake ``kokoro_onnx`` module and patch download/cache logic."""
    import sys
    import types

    import numpy as np

    fake_module = types.ModuleType("kokoro_onnx")
    fake_kokoro = MagicMock()
    for k, v in kokoro_attrs.items():
        setattr(fake_kokoro, k, v)
    fake_kokoro.create.return_value = (np.zeros(2400, dtype=np.float32), 24_000)
    fake_kokoro.get_voices.return_value = ["af_heart", "am_adam"]
    fake_module.Kokoro = MagicMock(return_value=fake_kokoro)
    monkeypatch.setitem(sys.modules, "kokoro_onnx", fake_module)
    return fake_kokoro


def test_tts_synthesize_returns_audio(monkeypatch, tmp_path) -> None:
    import numpy as np

    fake_kokoro = _patch_kokoro(monkeypatch)
    from talos import tts as tts_mod

    monkeypatch.setattr(tts_mod, "_download", lambda url, dest: None)
    # Pre-create cache files so _download is never called.
    (tmp_path / tts_mod._MODEL_FILENAME).write_bytes(b"x")
    (tmp_path / tts_mod._VOICES_FILENAME).write_bytes(b"x")

    svc = tts_mod.TextToSpeechService(model_cache=tmp_path)
    sr, audio = svc.synthesize("hello world", voice_preset="af_heart")
    assert sr == 24_000
    assert isinstance(audio, np.ndarray)
    assert audio.dtype == np.float32
    fake_kokoro.create.assert_called_once()


def test_tts_long_form_delegates_to_synthesize(monkeypatch, tmp_path) -> None:
    _patch_kokoro(monkeypatch)
    from talos import tts as tts_mod

    monkeypatch.setattr(tts_mod, "_download", lambda url, dest: None)
    (tmp_path / tts_mod._MODEL_FILENAME).write_bytes(b"x")
    (tmp_path / tts_mod._VOICES_FILENAME).write_bytes(b"x")

    svc = tts_mod.TextToSpeechService(model_cache=tmp_path)
    sr, audio = svc.long_form_synthesize("one. two. three.")
    assert sr == 24_000
    assert audio.shape[0] > 0


def test_tts_unknown_voice_falls_back_to_default(monkeypatch, tmp_path) -> None:
    fake_kokoro = _patch_kokoro(monkeypatch)
    from talos import tts as tts_mod

    monkeypatch.setattr(tts_mod, "_download", lambda url, dest: None)
    (tmp_path / tts_mod._MODEL_FILENAME).write_bytes(b"x")
    (tmp_path / tts_mod._VOICES_FILENAME).write_bytes(b"x")

    svc = tts_mod.TextToSpeechService(model_cache=tmp_path)
    svc.synthesize("hi", voice_preset="v2/en_speaker_1")  # legacy Bark preset
    call_kwargs = fake_kokoro.create.call_args.kwargs
    assert call_kwargs["voice"] == "af_heart"


# ---------------------------------------------------------------------------
# Tool-call cleaning applied at the interface layer.
# ---------------------------------------------------------------------------


def test_telegram_strips_tool_calls_before_replying() -> None:
    agent = _make_agent()
    iface = TelegramInterface(agent, TelegramConfig(bot_token="t"))

    def fake_send(text, context_name=None, **kwargs):
        return (
            "Sure thing.\n```tool_call\n"
            '{"name": "run", "arguments": {"command": "ls -la"}}'
            "\n```\nDone."
        )

    agent.send = fake_send  # type: ignore[assignment]

    update = _make_update(99, "hi")
    asyncio.run(iface._handle_message(update, MagicMock()))

    sent = update.message.reply_text.await_args.args[0]
    assert "tool_call" not in sent
    assert "ls -la" not in sent
    assert "Sure thing." in sent
    assert "Done." in sent


def test_voice_strips_tool_calls_before_speaking() -> None:
    from talos.interfaces.voice import VoiceInterface

    agent = build_agent(TalosConfig())
    iface = VoiceInterface(agent, VoiceConfig())

    # Stub agent / TTS / audio so we can drive a single iteration of run().
    agent.send = MagicMock(
        return_value=(
            "Yes.\n```tool_call\n"
            '{"name": "run", "arguments": {"command": "echo hi"}}'
            "\n```\nOkay."
        )
    )  # type: ignore[assignment]
    iface._load_audio_libs = MagicMock()  # type: ignore[assignment]
    iface._load_whisper = MagicMock()  # type: ignore[assignment]
    iface._load_tts = MagicMock()  # type: ignore[assignment]
    iface._play_tone = MagicMock()  # type: ignore[assignment]
    iface._record_until_silence = MagicMock(return_value=b"audio")  # type: ignore[assignment]
    iface._transcribe = MagicMock(return_value="hello")  # type: ignore[assignment]
    iface._speak = MagicMock()  # type: ignore[assignment]

    call_count = {"n": 0}

    def fake_wait():
        call_count["n"] += 1
        if call_count["n"] > 1:
            iface._stop.set()
            return False
        return True

    iface._wait_for_wake_word = fake_wait  # type: ignore[assignment]
    iface.run()

    spoken = iface._speak.call_args.args[0]
    assert "tool_call" not in spoken
    assert "echo hi" not in spoken
    assert "Yes." in spoken
    assert "Okay." in spoken
