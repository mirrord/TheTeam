"""Telegram interface — per-user persistent contexts on a pithos agent."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from pithos.agent import Agent

from ..config import TelegramConfig
from ..utils import clean_agent_response

logger = logging.getLogger(__name__)


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
        ctx_name = self._ensure_context(user.id)
        if self.show:
            print(f"\n[{user.first_name} ({user.id})] {text}", flush=True)
        try:
            response = self._generate_response(text, ctx_name)
        except Exception as exc:  # surface backend errors to the user
            logger.exception("Agent send failed for user %s", user.id)
            if self.show:
                print(f"[error] {exc}", flush=True)
            await update.message.reply_text(f"[error] {exc}")
            return
        # Strip tool-call syntax so users see a clean reply.
        response = clean_agent_response(response)
        # Telegram caps messages at 4096 characters.
        for i in range(0, len(response) or 1, 4000):
            chunk = response[i : i + 4000] or "(no response)"
            await update.message.reply_text(chunk)
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

    def _generate_response(self, text: str, ctx_name: str) -> str:
        """Produce the agent's reply, streaming to stdout when ``show`` is set."""
        if not self.show:
            return self.agent.send(text, context_name=ctx_name)
        # Mirror the streamed response to stdout token-by-token while
        # accumulating the full reply for Telegram delivery.
        self.agent._pending_image_paths = []
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
