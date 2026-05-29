"""LlamacppAgent — LLM agent backed by ``llama-cpp-python``.

The backend package is an optional dependency.  Importing this module
always succeeds, but constructing an agent without ``llama-cpp-python``
installed raises :class:`ImportError` with installation guidance.

Streaming orchestration (tool calls, recall, history, compaction,
metrics) lives in :class:`pithos.agent.agent.Agent`; this subclass only
implements :meth:`Agent._raw_stream` plus lazy model loading.
"""

from __future__ import annotations

from typing import Any, Iterator, Optional
import logging

from .agent import Agent

logger = logging.getLogger(__name__)

try:  # pragma: no cover - exercised via monkeypatch in tests
    from llama_cpp import Llama  # type: ignore[import-not-found]

    _LLAMACPP_AVAILABLE = True
except ImportError:  # pragma: no cover
    Llama = None  # type: ignore[assignment,misc]
    _LLAMACPP_AVAILABLE = False


_INSTALL_HINT = (
    "llama-cpp-python is not installed. "
    "Install with: pip install llama-cpp-python "
    "(or: pip install theteam[llamacpp])"
)


class LlamacppAgent(Agent):
    """LLM agent backed by ``llama-cpp-python`` (the GGUF runtime).

    The ``default_model`` argument must be a filesystem path to a GGUF
    model file.  Backend-specific load options (``n_ctx``,
    ``n_gpu_layers``, ``chat_format``, …) are passed through
    ``backend_options``.

    Example::

        agent = LlamacppAgent(
            default_model="/models/llama-3.2-8b.gguf",
            backend_options={"n_ctx": 8192, "n_gpu_layers": -1},
        )
        for tok in agent.stream("Hello"):
            print(tok, end="")
    """

    def __init__(
        self,
        default_model: str,
        agent_name: Optional[str] = None,
        system_prompt: str = "",
        temperature: Optional[float] = None,
        backend_options: Optional[dict[str, Any]] = None,
    ) -> None:
        if not _LLAMACPP_AVAILABLE or Llama is None:
            raise ImportError(_INSTALL_HINT)
        super().__init__(default_model, agent_name, system_prompt, temperature)
        self.backend_options: dict[str, Any] = dict(backend_options or {})
        # Sensible defaults applied only when not overridden by the caller.
        self.backend_options.setdefault("n_ctx", 4096)
        self.backend_options.setdefault("verbose", False)
        # Cached llama.cpp model + the path it was loaded from.
        self._llama: Any = None
        self._loaded_model_path: Optional[str] = None

    # ------------------------------------------------------------------
    # Model lifecycle
    # ------------------------------------------------------------------

    def _load_model(self, model_path: str) -> Any:
        """Lazily load (and cache) the underlying ``Llama`` instance."""
        if self._llama is not None and self._loaded_model_path == model_path:
            return self._llama
        if self._llama is not None:
            logger.info(
                "LlamacppAgent: reloading model (was %r, now %r)",
                self._loaded_model_path,
                model_path,
            )
        self._llama = Llama(model_path=model_path, **self.backend_options)
        self._loaded_model_path = model_path
        return self._llama

    def close(self) -> None:  # type: ignore[override]
        # Release the GGUF mapping promptly on close.
        super().close()
        self._llama = None
        self._loaded_model_path = None

    # ------------------------------------------------------------------
    # Backend hooks
    # ------------------------------------------------------------------

    def _raw_stream(
        self,
        messages: list[dict[str, Any]],
        model: str,
        options: dict[str, Any],
    ) -> Iterator[tuple[str, Any]]:
        llama = self._load_model(model)

        kwargs: dict[str, Any] = {
            "messages": messages,
            "stream": True,
            "temperature": options.get("temperature", self.temperature),
        }
        max_tokens = options.get("max_tokens", options.get("num_predict"))
        if max_tokens is not None and max_tokens != -1:
            kwargs["max_tokens"] = max_tokens

        last_chunk: Any = None
        for chunk in llama.create_chat_completion(**kwargs):
            last_chunk = chunk
            try:
                delta = chunk["choices"][0].get("delta", {})
                token = delta.get("content") or ""
            except (KeyError, IndexError, TypeError):
                token = ""
            yield token, chunk

        # Surface usage on the very last yielded chunk if the backend
        # didn't already attach it.
        if last_chunk is not None and isinstance(last_chunk, dict):
            usage = last_chunk.get("usage")
            if usage:
                # Stash for _extract_token_usage.
                last_chunk.setdefault("_usage_for_metrics", usage)

    def _extract_token_usage(self, last_chunk: Any) -> tuple[int, int]:
        if not last_chunk:
            return 0, 0
        try:
            if isinstance(last_chunk, dict):
                usage = last_chunk.get("usage") or last_chunk.get(
                    "_usage_for_metrics"
                )
                if usage:
                    return (
                        int(usage.get("prompt_tokens", 0) or 0),
                        int(usage.get("completion_tokens", 0) or 0),
                    )
        except Exception:
            pass
        return 0, 0

    def _wrap_backend_error(self, exc: BaseException) -> BaseException:
        return RuntimeError(
            f"Failed to communicate with llama.cpp backend: {exc}. "
            f"Verify the model path '{self._loaded_model_path or self.default_model}' "
            "is a valid GGUF file."
        )
