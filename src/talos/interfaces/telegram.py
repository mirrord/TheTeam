"""Telegram interface — per-user persistent contexts on a pithos agent."""

from __future__ import annotations

import asyncio
import logging
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from pithos.agent import Agent

from ..config import TelegramConfig
from ..utils import clean_agent_response
from .system_commands import handle_system_command

logger = logging.getLogger(__name__)

#: How long (seconds) a mid-task tool-confirmation prompt waits for the user's
#: reaction before it is treated as a denial.
_CONFIRM_TIMEOUT_SECONDS = 300.0

#: The only reaction that approves a tool call; any other reaction denies it.
#: Approvals are given exclusively via reactions on the prompt message so a
#: permission answer can never be mistaken for (or double-counted as) a chat
#: message to the agent.
_APPROVE_REACTION = "\U0001f44d"  # 👍


@dataclass
class _PendingConfirm:
    """State for an outstanding tool-confirmation prompt.

    Keyed in :attr:`TelegramInterface._pending_confirms` by the
    ``(chat_id, message_id)`` of the prompt message, so each reaction resolves
    exactly the prompt it was placed on.
    """

    user_id: int
    event: threading.Event = field(default_factory=threading.Event)
    approved: bool = False


def _context_name_for_user(user_id: int) -> str:
    return f"telegram_{user_id}"


class TelegramInterface:
    """Telegram bot that gives each Telegram user their own agent context."""

    def __init__(
        self, agent: Agent, config: TelegramConfig, show: bool = False
    ) -> None:
        if not config.bot_token:
            raise ValueError(
                "Telegram bot_token is not set. Run 'talos --reconfigure' "
                "or edit ~/.talos/config.yaml to add a token from @BotFather."
            )
        self.agent = agent
        self.config = config
        # When True, mirror the conversation to stdout and stream the agent's
        # response tokens as they are produced.
        self.show = show
        # Outstanding tool-confirmation prompts, keyed by the prompt message's
        # ``(chat_id, message_id)``. They are resolved only by a reaction on that
        # exact message — never by a text message.
        self._pending_confirms: dict[tuple[int, int], _PendingConfirm] = {}
        self._pending_lock = threading.Lock()
        # Serializes agent turns. The agent (and its single, shared
        # ``tool_confirm_callback``) can only safely run one turn at a time, so
        # a new message must never start a second concurrent turn while another
        # is in flight — otherwise the agent keeps generating and emits more
        # approval prompts while an earlier one is still awaiting the user's
        # reply. Held (non-blocking) for the duration of each turn.
        self._turn_lock = threading.Lock()
        # Update types requested from Telegram when polling (set by
        # :meth:`build_application`; must include ``message_reaction``).
        self._allowed_updates: Optional[list[str]] = None
    # ------------------------------------------------------------------

    def _ensure_context(self, user_id: int) -> str:
        ctx_name = _context_name_for_user(user_id)
        if ctx_name not in self.agent.contexts:
            self.agent.copy_context(self.agent.current_context, ctx_name)
        return ctx_name

    async def _start(self, update: Any, _context: Any) -> None:
        user = update.effective_user
        self._ensure_context(user.id)
        await update.message.reply_text(
            f"Hi {user.first_name}! I'm Talos. Send me a message to chat."
        )

    async def _handle_message(self, update: Any, _context: Any) -> None:
        user = update.effective_user
        text = update.message.text or ""
        if not text.strip():
            return
        # Tool-call approvals are given only via reactions on the prompt
        # message. A text message while an approval is pending is never treated
        # as the answer (nor as a new turn) — it would otherwise be ambiguous.
        if self._has_pending_confirm(user.id):
            await update.message.reply_text(
                "\u23f3 I'm waiting for your permission on a tool call. React "
                f"{_APPROVE_REACTION} to that prompt to allow it, or with any "
                "other reaction to deny it."
            )
            return
        # Serialize agent turns. While a turn is in flight the agent may be
        # paused waiting for a tool-call approval; starting a second turn (or
        # running a system command such as "system reset" against the context
        # mid-turn) would touch the shared agent concurrently. Refuse instead.
        if not self._turn_lock.acquire(blocking=False):
            await update.message.reply_text(
                "\u23f3 I'm still working on your previous message — please wait "
                "for it to finish."
            )
            return
        try:
            await self._run_turn(update, user, text)
        finally:
            self._turn_lock.release()

    async def _run_turn(self, update: Any, user: Any, text: str) -> None:
        """Handle one message while holding :attr:`_turn_lock`."""
        ctx_name = self._ensure_context(user.id)
        # Interface-level commands (e.g. "system reset") are handled locally
        # against this user's context and never forwarded to the agent.
        system_response = handle_system_command(
            self.agent, text, context_name=ctx_name
        )
        if system_response is not None:
            if self.show:
                print(system_response, flush=True)
            await update.message.reply_text(system_response)
            return
        if self.show:
            print(f"\n[{user.first_name} ({user.id})] {text}", flush=True)
        loop = asyncio.get_running_loop()
        confirm_cb = self._make_confirm_callback(user.id, update, loop)
        try:
            # Run the (blocking) agent turn off the event loop so the loop stays
            # free to receive the user's approval reactions mid-task.
            response = await asyncio.to_thread(
                self._generate_response_with_confirm, text, ctx_name, confirm_cb
            )
        except Exception as exc:  # surface backend errors to the user
            logger.exception("Agent send failed for user %s", user.id)
            if self.show:
                print(f"[error] {exc}", flush=True)
            await update.message.reply_text(f"[error] {exc}")
            return
        # Strip tool-call syntax so users see a clean reply.
        response = clean_agent_response(response)
        # Telegram caps messages at 4096 characters.
        await self._send_chunked(update, response)
        # Send any images generated during the agent's response.
        for image_path in self.agent.last_image_paths:
            p = Path(image_path)
            if not p.exists():
                logger.warning("Generated image not found, skipping: %s", image_path)
                continue
            try:
                with p.open("rb") as fh:
                    await update.message.reply_photo(photo=fh)
            except Exception as exc:
                logger.warning("Failed to send image %s: %s", image_path, exc)
        # Send the full contents of any reports generated during the
        # response (e.g. news-research, web-research, craft-write), since
        # the model's prose reply is only a summary of these documents.
        for report_path in self.agent.last_report_paths:
            p = Path(report_path)
            if not p.exists():
                logger.warning("Report file not found, skipping: %s", report_path)
                continue
            try:
                content = p.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError) as exc:
                logger.warning("Failed to read report %s: %s", report_path, exc)
                continue
            await update.message.reply_text(f"\U0001f4c4 Report: {p.name}")
            await self._send_chunked(update, content)
        # Surface attachment-only artifacts (e.g. the coding harness report) as
        # file attachments WITHOUT dumping their contents — the user gets the
        # report file plus the agent's own prose reply, not the raw report body.
        for attachment_path in self.agent.last_attachment_paths:
            p = Path(attachment_path)
            if not p.exists():
                logger.warning(
                    "Attachment file not found, skipping: %s", attachment_path
                )
                continue
            try:
                with p.open("rb") as fh:
                    await update.message.reply_document(document=fh, filename=p.name)
            except Exception as exc:
                logger.warning("Failed to send attachment %s: %s", attachment_path, exc)

    @staticmethod
    async def _send_chunked(update: Any, text: str, chunk_size: int = 4000) -> None:
        """Send ``text`` as one or more Telegram messages under the 4096-char cap."""
        for i in range(0, len(text) or 1, chunk_size):
            chunk = text[i : i + chunk_size] or "(no response)"
            await update.message.reply_text(chunk)

    # ------------------------------------------------------------------
    # Tool-confirmation bridge
    # ------------------------------------------------------------------

    def _has_pending_confirm(self, user_id: int) -> bool:
        """Return True if *user_id* has an outstanding tool-confirmation prompt."""
        with self._pending_lock:
            return any(p.user_id == user_id for p in self._pending_confirms.values())

    async def _handle_reaction(self, update: Any, _context: Any) -> None:
        """Resolve a pending tool-confirmation prompt from a message reaction.

        Only reactions placed on a prompt message, by the user who was asked,
        count: :data:`_APPROVE_REACTION` approves, any other reaction denies.
        Reactions on other messages, and reaction removals, are ignored.
        """
        reaction = getattr(update, "message_reaction", None)
        if reaction is None:
            return
        user = getattr(reaction, "user", None)
        if user is None:  # anonymous reactions can't be attributed
            return
        new = list(getattr(reaction, "new_reaction", None) or [])
        if not new:  # reaction removed — not an answer
            return
        emojis = [getattr(r, "emoji", None) for r in new]
        self._resolve_reaction(
            reaction.chat.id, reaction.message_id, user.id, emojis
        )

    def _resolve_reaction(
        self,
        chat_id: int,
        message_id: int,
        user_id: int,
        emojis: list[Optional[str]],
    ) -> bool:
        """Resolve the prompt at ``(chat_id, message_id)`` from *emojis*.

        Returns True when a pending prompt was resolved.
        """
        with self._pending_lock:
            pending = self._pending_confirms.get((chat_id, message_id))
            if pending is None or pending.user_id != user_id or pending.event.is_set():
                return False
            pending.approved = _APPROVE_REACTION in emojis
            pending.event.set()
        return True

    def _make_confirm_callback(self, user_id: int, update: Any, loop: Any):
        """Build a (blocking) confirm callback for one agent turn.

        The callback runs in the worker thread executing the agent turn. It
        posts the exact tool call to the user, then blocks until the user
        reacts to that prompt (handled on the event loop) or it times out.
        """

        async def _send_and_register(prompt: str) -> tuple[tuple[int, int], _PendingConfirm]:
            msg = await update.message.reply_text(prompt)
            key = (msg.chat_id, msg.message_id)
            pending = _PendingConfirm(user_id=user_id)
            # Registered on the event loop, which also runs the reaction
            # handler, so a reaction can never be processed before this.
            with self._pending_lock:
                self._pending_confirms[key] = pending
            return key, pending

        def confirm(command: str) -> bool:
            prompt = (
                "\u26a0\ufe0f The coding agent wants to run this tool call:\n\n"
                f"{command}\n\n"
                f"React {_APPROVE_REACTION} to this message to allow it, or with "
                "any other reaction to deny it."
            )
            try:
                future = asyncio.run_coroutine_threadsafe(
                    _send_and_register(prompt), loop
                )
                key, pending = future.result(timeout=30)
            except Exception as exc:  # pragma: no cover - network guard
                logger.warning("Failed to send confirmation prompt: %s", exc)
                return False
            got_reply = pending.event.wait(timeout=_CONFIRM_TIMEOUT_SECONDS)
            with self._pending_lock:
                self._pending_confirms.pop(key, None)
            if not got_reply:
                return False
            return pending.approved

        return confirm

    def _generate_response_with_confirm(
        self, text: str, ctx_name: str, confirm_cb
    ) -> str:
        """Run one agent turn with *confirm_cb* installed as the tool gate."""
        self.agent.set_tool_confirm_callback(confirm_cb)
        try:
            return self._generate_response(text, ctx_name)
        finally:
            self.agent.set_tool_confirm_callback(None)

    def _generate_response(self, text: str, ctx_name: str) -> str:
        """Produce the agent's reply, streaming to stdout when ``show`` is set."""
        if not self.show:
            return self.agent.send(text, context_name=ctx_name)
        # Mirror the streamed response to stdout token-by-token while
        # accumulating the full reply for Telegram delivery.
        self.agent._pending_image_paths = []
        self.agent._pending_report_paths = []
        self.agent._pending_attachment_paths = []
        print("[Talos] ", end="", flush=True)
        parts: list[str] = []
        for chunk in self.agent.stream(
            text, context_name=ctx_name, status_callback=self._on_status
        ):
            print(chunk, end="", flush=True)
            parts.append(chunk)
        print(flush=True)
        return "".join(parts)

    def _on_status(self, status: str, detail: Any) -> None:
        """Render tool activity to stdout with clear call/output separation."""
        if not self.show:
            return
        if status == "tool_call":
            command = detail or "(unknown)"
            print(f"\n\n\u250c\u2500 tool call: {command}", flush=True)
        elif status == "tool_result":
            output = (detail or "").rstrip() or "(no output)"
            indented = "\n".join(f"\u2502 {line}" for line in output.splitlines())
            print("\u251c\u2500 tool output:", flush=True)
            print(indented, flush=True)
            print("\u2514\u2500 end tool output\n", flush=True)
            print("[Talos] ", end="", flush=True)

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def build_application(self) -> Any:
        """Construct the python-telegram-bot :class:`Application` instance."""
        try:
            from telegram import Update  # type: ignore
            from telegram.ext import (  # type: ignore
                ApplicationBuilder,
                CommandHandler,
                MessageHandler,
                MessageReactionHandler,
                filters,
            )
        except ImportError as exc:
            raise ImportError(
                "Talos telegram interface requires 'python-telegram-bot'. "
                "Install with: pip install -e .[talos]"
            ) from exc

        # Updates must be processed concurrently: an agent turn blocks its
        # handler while it waits for tool-call approval, and with sequential
        # processing the approving reaction would queue behind that very turn
        # (until the prompt timed out). Concurrent turns are prevented
        # separately by ``_turn_lock``.
        app = (
            ApplicationBuilder()
            .token(self.config.bot_token)
            .concurrent_updates(True)
            .build()
        )
        app.add_handler(CommandHandler("start", self._start))
        app.add_handler(
            MessageHandler(filters.TEXT & ~filters.COMMAND, self._handle_message)
        )
        app.add_handler(MessageReactionHandler(self._handle_reaction))
        # Telegram only delivers ``message_reaction`` updates when explicitly
        # requested via ``allowed_updates``.
        self._allowed_updates = Update.ALL_TYPES
        return app

    def run(self) -> None:
        """Start polling. Blocks until Ctrl+C."""
        app = self.build_application()
        print("Talos telegram bot started. Press Ctrl+C to exit.")
        app.run_polling(allowed_updates=self._allowed_updates)


__all__ = ["TelegramInterface"]
