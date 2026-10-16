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
#: reply before it is treated as a denial.
_CONFIRM_TIMEOUT_SECONDS = 300.0

#: Replies (case-insensitive, stripped) that approve a confirmation prompt.
_AFFIRMATIVE = {"yes", "y", "yeah", "yep", "ok", "okay", "sure", "allow", "approve"}


@dataclass
class _PendingConfirm:
    """State for an outstanding tool-confirmation prompt for one user."""

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
        # Outstanding tool-confirmation prompts, keyed by Telegram user id. A
        # user's next message is interpreted as the answer to their pending
        # prompt rather than as a new chat turn.
        self._pending_confirms: dict[int, _PendingConfirm] = {}
        self._pending_lock = threading.Lock()

    # ------------------------------------------------------------------
    # Handlers
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
        # If this user has an outstanding tool-confirmation prompt, treat their
        # message as the answer rather than a new chat turn.
        if self._resolve_pending_confirm(user.id, text):
            return
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
            # free to receive the user's confirmation reply mid-task.
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

    def _resolve_pending_confirm(self, user_id: int, text: str) -> bool:
        """If *user_id* has a pending confirmation, resolve it with *text*.

        Returns True when the message was consumed as a confirmation answer.
        """
        with self._pending_lock:
            pending = self._pending_confirms.get(user_id)
        if pending is None:
            return False
        pending.approved = text.strip().lower() in _AFFIRMATIVE
        pending.event.set()
        return True

    def _make_confirm_callback(self, user_id: int, update: Any, loop: Any):
        """Build a (blocking) confirm callback for one agent turn.

        The callback runs in the worker thread executing the agent turn. It
        posts the exact tool call to the user, then blocks until the user's
        reply arrives (handled on the event loop) or the prompt times out.
        """

        def confirm(command: str) -> bool:
            pending = _PendingConfirm()
            # Register before sending so a fast reply is never missed.
            with self._pending_lock:
                self._pending_confirms[user_id] = pending
            prompt = (
                "\u26a0\ufe0f The coding agent wants to run this tool call:\n\n"
                f"{command}\n\n"
                "Reply 'yes' to allow, anything else to deny."
            )
            try:
                future = asyncio.run_coroutine_threadsafe(
                    update.message.reply_text(prompt), loop
                )
                future.result(timeout=30)
            except Exception as exc:  # pragma: no cover - network guard
                logger.warning("Failed to send confirmation prompt: %s", exc)
                with self._pending_lock:
                    self._pending_confirms.pop(user_id, None)
                return False
            got_reply = pending.event.wait(timeout=_CONFIRM_TIMEOUT_SECONDS)
            with self._pending_lock:
                self._pending_confirms.pop(user_id, None)
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
            from telegram.ext import (  # type: ignore
                ApplicationBuilder,
                CommandHandler,
                MessageHandler,
                filters,
            )
        except ImportError as exc:
            raise ImportError(
                "Talos telegram interface requires 'python-telegram-bot'. "
                "Install with: pip install -e .[talos]"
            ) from exc

        app = ApplicationBuilder().token(self.config.bot_token).build()
        app.add_handler(CommandHandler("start", self._start))
        app.add_handler(
            MessageHandler(filters.TEXT & ~filters.COMMAND, self._handle_message)
        )
        return app

    def run(self) -> None:
        """Start polling. Blocks until Ctrl+C."""
        app = self.build_application()
        print("Talos telegram bot started. Press Ctrl+C to exit.")
        app.run_polling()


__all__ = ["TelegramInterface"]
