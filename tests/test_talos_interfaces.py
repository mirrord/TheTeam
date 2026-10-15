"""Tests for talos.interfaces — shell, voice (mocked), telegram (mocked)."""

from __future__ import annotations

import asyncio
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
    chat.assert_called_once_with(agent, verbose=True)


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


def test_telegram_resolve_pending_confirm() -> None:
    """A pending confirmation consumes the next message as its yes/no answer."""
    agent = _make_agent()
    iface = TelegramInterface(agent, TelegramConfig(bot_token="t"))

    # No pending → message is not consumed.
    assert iface._resolve_pending_confirm(5, "hello") is False

    from talos.interfaces.telegram import _PendingConfirm

    pending = _PendingConfirm()
    iface._pending_confirms[5] = pending
    assert iface._resolve_pending_confirm(5, "YES") is True
    assert pending.approved is True
    assert pending.event.is_set()

    pending2 = _PendingConfirm()
    iface._pending_confirms[6] = pending2
    assert iface._resolve_pending_confirm(6, "no thanks") is True
    assert pending2.approved is False


def test_telegram_tool_confirmation_round_trip() -> None:
    """A mid-task tool-confirm prompt is answered by the user's next message."""

    async def scenario() -> None:
        agent = _make_agent()
        iface = TelegramInterface(agent, TelegramConfig(bot_token="t"))
        captured: dict[str, object] = {}

        def fake_send(text, context_name=None, **kwargs):
            # The coding subagent would consult the caller's confirm channel.
            cb = agent.tool_confirm_callback
            captured["approved"] = cb("powershell -Command whoami")
            return "all done"

        agent.send = fake_send  # type: ignore[assignment]
        agent._pending_image_paths = []
        agent._pending_report_paths = []
        agent._pending_attachment_paths = []

        update1 = _make_update(100, "code do the thing")
        update1.message.reply_document = AsyncMock()

        # Start the agent turn; it blocks in a worker thread awaiting confirm.
        task = asyncio.create_task(iface._handle_message(update1, MagicMock()))
        for _ in range(200):
            await asyncio.sleep(0.01)
            if 100 in iface._pending_confirms:
                break
        assert 100 in iface._pending_confirms  # user was prompted

        # Deliver the approval as the user's next message.
        update2 = _make_update(100, "yes")
        await iface._handle_message(update2, MagicMock())
        await task

        assert captured["approved"] is True
        # The confirm prompt included the exact command.
        prompts = [c.args[0] for c in update1.message.reply_text.await_args_list]
        assert any("powershell -Command whoami" in p for p in prompts)

    asyncio.run(scenario())


def test_telegram_tool_confirmation_denied() -> None:
    """Replying with anything other than yes denies the tool call."""

    async def scenario() -> None:
        agent = _make_agent()
        iface = TelegramInterface(agent, TelegramConfig(bot_token="t"))
        captured: dict[str, object] = {}

        def fake_send(text, context_name=None, **kwargs):
            captured["approved"] = agent.tool_confirm_callback("bash -c rm")
            return "ok"

        agent.send = fake_send  # type: ignore[assignment]
        agent._pending_image_paths = []
        agent._pending_report_paths = []
        agent._pending_attachment_paths = []

        update1 = _make_update(101, "code do the thing")
        update1.message.reply_document = AsyncMock()
        task = asyncio.create_task(iface._handle_message(update1, MagicMock()))
        for _ in range(200):
            await asyncio.sleep(0.01)
            if 101 in iface._pending_confirms:
                break
        assert 101 in iface._pending_confirms

        update2 = _make_update(101, "no")
        await iface._handle_message(update2, MagicMock())
        await task

        assert captured["approved"] is False

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
        return "Sure thing.\n[RUN]ls -la[/RUN]\nDone."

    agent.send = fake_send  # type: ignore[assignment]

    update = _make_update(99, "hi")
    asyncio.run(iface._handle_message(update, MagicMock()))

    sent = update.message.reply_text.await_args.args[0]
    assert "[RUN]" not in sent
    assert "ls -la" not in sent
    assert "Sure thing." in sent
    assert "Done." in sent


def test_voice_strips_tool_calls_before_speaking() -> None:
    from talos.interfaces.voice import VoiceInterface

    agent = build_agent(TalosConfig())
    iface = VoiceInterface(agent, VoiceConfig())

    # Stub agent / TTS / audio so we can drive a single iteration of run().
    agent.send = MagicMock(return_value="Yes.\n[RUN]echo hi[/RUN]\nOkay.")  # type: ignore[assignment]
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
    assert "[RUN]" not in spoken
    assert "echo hi" not in spoken
    assert "Yes." in spoken
    assert "Okay." in spoken
