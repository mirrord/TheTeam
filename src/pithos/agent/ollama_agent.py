"""OllamaAgent — streaming-first LLM agent backed by Ollama.

The bulk of the streaming orchestration (tool detection, recall,
history, compaction, metrics) lives in :class:`pithos.agent.agent.Agent`.
This module only implements the backend-specific
:meth:`Agent._raw_stream` hook plus error translation.
"""

from typing import Any, Iterator
import logging

from ollama import chat
from ollama._types import ResponseError as OllamaResponseError

from .agent import Agent

logger = logging.getLogger(__name__)


class OllamaAgent(Agent):
    """LLM agent backed by Ollama.

    Streaming is the primary execution path; :meth:`Agent.stream` drives
    the conversation while this subclass produces raw tokens via the
    Ollama Python client.
    """

    def _raw_stream(
        self,
        messages: list[dict[str, Any]],
        model: str,
        options: dict[str, Any],
    ) -> Iterator[tuple[str, Any]]:
        # Ollama expects ``num_predict``; strip the generic ``max_tokens``
        # alias also set by the base class for cross-backend uniformity.
        ollama_options: dict[str, Any] = {
            k: v for k, v in options.items() if k != "max_tokens"
        }
        raw_stream = chat(
            model=model,
            messages=messages,
            options=ollama_options,
            stream=True,
        )
        for chunk in raw_stream:
            token = chunk.message.content or ""
            yield token, chunk

    def _wrap_backend_error(self, exc: BaseException) -> BaseException:
        # Preserve Ollama-native errors so existing callers can catch them.
        if isinstance(exc, OllamaResponseError):
            return exc
        return RuntimeError(
            f"Failed to communicate with Ollama: {exc}. "
            "Ensure Ollama is running. "
            "If using localhost, try setting "
            "OLLAMA_HOST=http://127.0.0.1:11434 to avoid IPv6 resolution issues."
        )
