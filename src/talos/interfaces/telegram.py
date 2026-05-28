"""Telegram interface — per-user persistent contexts on a pithos agent."""

from __future__ import annotations

import logging
from typing import Any

from pithos.agent import Agent

from ..config import TelegramConfig
from ..utils import clean_agent_response

logger = logging.getLogger(__name__)


def _context_name_for_user(user_id: int) -> str:
    return f"telegram_{user_id}"


class TelegramInterface:
    """Telegram bot that gives each Telegram user their own agent context."""

    def __init__(self, agent: Agent, config: TelegramConfig) -> None:
        if not config.bot_token:
            raise ValueError(
                "Telegram bot_token is not set. Run 'talos --reconfigure' "
                "or edit ~/.talos/config.yaml to add a token from @BotFather."
            )
        self.agent = agent
        self.config = config

    # ------------------------------------------------------------------
    # Handlers
    # ------------------------------------------------------------------

    def _ensure_context(self, user_id: int) -> str:
        ctx_name = _context_name_for_user(user_id)
        if ctx_name not in self.agent.contexts:
            self.agent.create_context(ctx_name)
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
        ctx_name = self._ensure_context(user.id)
        try:
            response = self.agent.send(text, context_name=ctx_name)
        except Exception as exc:  # surface backend errors to the user
            logger.exception("Agent send failed for user %s", user.id)
            await update.message.reply_text(f"[error] {exc}")
            return
        # Strip tool-call syntax so users see a clean reply.
        response = clean_agent_response(response)
        # Telegram caps messages at 4096 characters.
        for i in range(0, len(response) or 1, 4000):
            chunk = response[i : i + 4000] or "(no response)"
            await update.message.reply_text(chunk)

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
