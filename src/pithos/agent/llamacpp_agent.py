"""LlamacppAgent — LLM agent backed by llama.cpp.

Two interchangeable backends are supported:

* **Server mode** (recommended for a self-hosted ``llama-server``): the
  agent talks to a running llama.cpp HTTP server over its
  OpenAI-compatible ``/v1/chat/completions`` streaming endpoint.  This
  mode needs **no** extra Python packages — only the standard library —
  and is selected by passing ``base_url`` (or putting ``base_url`` in
  ``backend_options`` / an agent config).

* **In-process mode**: GGUF inference inside this process via the
  optional ``llama-cpp-python`` package.  Selected when no ``base_url``
  is given; ``default_model`` must then be a path to a ``.gguf`` file.

Streaming orchestration (tool calls, recall, history, compaction,
metrics) lives in :class:`pithos.agent.agent.Agent`; this subclass only
implements :meth:`Agent._raw_stream` plus backend lifecycle.
"""

from __future__ import annotations

from typing import Any, Iterator, Optional
import json
import logging
import urllib.error
import urllib.request

from .agent import Agent
from ..config_manager import ConfigManager

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
    "(or: pip install theteam[llamacpp]). "
    "Alternatively, run a llama.cpp server and point the agent at it with "
    "base_url (e.g. base_url='http://127.0.0.1:8080') — that mode needs no "
    "extra packages."
)

# Options that configure how the *client* behaves rather than parameters
# forwarded to the server in the request body.
_SERVER_CLIENT_OPTIONS = frozenset(
    {"base_url", "server_url", "api_key", "timeout", "request_timeout",
     "include_reasoning"}
)
# Model-load options meaningful only to in-process ``Llama(...)``; if a user
# leaves them in ``backend_options`` while switching to server mode they must
# not be forwarded to the server (it configures these at launch time).
_INPROCESS_ONLY_OPTIONS = frozenset(
    {"n_ctx", "n_gpu_layers", "n_batch", "n_threads", "chat_format",
     "verbose", "use_mmap", "use_mlock", "seed", "n_threads_batch",
     "main_gpu", "tensor_split", "rope_freq_base", "rope_freq_scale",
     "model_path"}
)


class LlamacppAgent(Agent):
    """LLM agent backed by llama.cpp (server HTTP API or in-process GGUF).

    Server mode (no backend package required)::

        agent = LlamacppAgent(
            default_model="qwen3",             # name is informational
            base_url="http://127.0.0.1:8080",  # a running llama.cpp server
        )
        for tok in agent.stream("Hello"):
            print(tok, end="")

    In-process mode (requires ``pip install theteam[llamacpp]``)::

        agent = LlamacppAgent(
            default_model="/models/llama-3.2-8b.gguf",
            backend_options={"n_ctx": 8192, "n_gpu_layers": -1},
        )
    """

    def __init__(
        self,
        default_model: str,
        agent_name: Optional[str] = None,
        system_prompt: str = "",
        temperature: Optional[float] = None,
        backend_options: Optional[dict[str, Any]] = None,
        base_url: Optional[str] = None,
    ) -> None:
        super().__init__(default_model, agent_name, system_prompt, temperature)

        opts: dict[str, Any] = dict(backend_options or {})

        # Resolve server mode: explicit ``base_url`` arg wins, otherwise fall
        # back to ``base_url`` / ``server_url`` inside ``backend_options``.
        resolved_base = (
            base_url
            or opts.pop("base_url", None)
            or opts.pop("server_url", None)
        )
        # Drop the alias even when the primary key was used.
        opts.pop("base_url", None)
        opts.pop("server_url", None)

        self.server_mode: bool = resolved_base is not None
        self.api_key: Optional[str] = opts.pop("api_key", None)
        # Per-request socket timeout (seconds). Generous default so long
        # generations on large models are not cut off; streaming keeps the
        # socket active between tokens so this is effectively a stall timeout.
        self.request_timeout: Optional[float] = opts.pop(
            "timeout", opts.pop("request_timeout", 600)
        )
        # Qwen3 (and other reasoning models) stream chain-of-thought as
        # ``reasoning_content`` deltas. By default that is NOT surfaced in the
        # token stream (keeps history/tool-call extraction clean); opt in to
        # have it wrapped in ``<think>…</think>`` tags.
        self.include_reasoning: bool = bool(opts.pop("include_reasoning", False))

        if self.server_mode:
            self.base_url: Optional[str] = str(resolved_base)
            self._chat_endpoint: Optional[str] = self._build_chat_endpoint(
                self.base_url
            )
            # Forward only genuine sampling/request params to the server.
            self.backend_options = {
                k: v
                for k, v in opts.items()
                if k not in _INPROCESS_ONLY_OPTIONS
            }
        else:
            if not _LLAMACPP_AVAILABLE or Llama is None:
                raise ImportError(_INSTALL_HINT)
            self.base_url = None
            self._chat_endpoint = None
            self.backend_options = opts
            # Sensible defaults applied only when not overridden by the caller.
            self.backend_options.setdefault("n_ctx", 4096)
            self.backend_options.setdefault("verbose", False)

        # Cached in-process llama.cpp model + the path it was loaded from.
        self._llama: Any = None
        self._loaded_model_path: Optional[str] = None

    # ------------------------------------------------------------------
    # Config construction
    # ------------------------------------------------------------------

    @classmethod
    def from_dict(
        cls, config: dict[str, Any], config_manager: ConfigManager
    ) -> "LlamacppAgent":
        """Create a :class:`LlamacppAgent` from a config dict.

        Extends the base implementation to honour ``base_url`` /
        ``server_url`` and ``backend_options`` keys so server-backed agents
        can be defined entirely in YAML.
        """
        model = config.get("model")
        if not model:
            raise ValueError("Agent config must specify 'model'")
        agent = cls(
            model,
            config.get("name"),
            config.get("system_prompt", ""),
            config.get("temperature"),
            backend_options=config.get("backend_options"),
            base_url=config.get("base_url") or config.get("server_url"),
        )
        agent._apply_config_from_dict(config, config_manager)
        return agent

    @staticmethod
    def _build_chat_endpoint(base_url: str) -> str:
        """Normalise a server base URL to its chat-completions endpoint."""
        url = base_url.strip().rstrip("/")
        if url.endswith("/chat/completions"):
            return url
        if url.endswith("/v1"):
            return url + "/chat/completions"
        return url + "/v1/chat/completions"

    # ------------------------------------------------------------------
    # Model lifecycle (in-process mode only)
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
        if self.server_mode:
            yield from self._raw_stream_server(messages, model, options)
        else:
            yield from self._raw_stream_inprocess(messages, model, options)

    # -- in-process -----------------------------------------------------

    def _raw_stream_inprocess(
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

    # -- server (OpenAI-compatible HTTP) --------------------------------

    def _raw_stream_server(
        self,
        messages: list[dict[str, Any]],
        model: str,
        options: dict[str, Any],
    ) -> Iterator[tuple[str, Any]]:
        body: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": True,
            # Ask the server to emit a trailing usage chunk for metrics.
            "stream_options": {"include_usage": True},
            "temperature": options.get("temperature", self.temperature),
        }
        max_tokens = options.get("max_tokens", options.get("num_predict"))
        if max_tokens is not None and max_tokens != -1:
            body["max_tokens"] = max_tokens
        # Forward any extra sampling params (top_p, top_k, repeat_penalty, …)
        # without clobbering the keys we set explicitly above.
        for key, value in self.backend_options.items():
            body.setdefault(key, value)

        headers = {
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        data = json.dumps(body).encode("utf-8")
        endpoint = self._chat_endpoint
        assert endpoint is not None  # always set in server mode
        request = urllib.request.Request(
            endpoint, data=data, headers=headers, method="POST"
        )

        reasoning_open = False
        try:
            response = urllib.request.urlopen(
                request, timeout=self.request_timeout
            )
        except urllib.error.HTTPError as exc:
            detail = ""
            try:
                detail = exc.read().decode("utf-8", "replace")[:500]
            except Exception:
                pass
            raise RuntimeError(
                f"llama.cpp server returned HTTP {exc.code} for "
                f"{self._chat_endpoint}: {detail or exc.reason}"
            ) from exc

        with response:
            for raw_line in response:
                line = raw_line.decode("utf-8", "replace").strip()
                if not line or not line.startswith("data:"):
                    continue
                payload = line[len("data:"):].strip()
                if payload == "[DONE]":
                    break
                try:
                    chunk = json.loads(payload)
                except json.JSONDecodeError:
                    continue

                choices = chunk.get("choices") or []
                delta = choices[0].get("delta", {}) if choices else {}
                content = delta.get("content")
                reasoning = delta.get("reasoning_content")

                emit = ""
                if reasoning and self.include_reasoning:
                    if not reasoning_open:
                        emit += "<think>"
                        reasoning_open = True
                    emit += reasoning
                    yield emit, chunk
                    continue
                if content:
                    if reasoning_open:
                        emit += "</think>\n"
                        reasoning_open = False
                    emit += content
                yield emit, chunk

            if reasoning_open:
                yield "</think>\n", None

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
        if self.server_mode:
            return RuntimeError(
                f"Failed to communicate with llama.cpp server at "
                f"'{self.base_url}': {exc}. Ensure the server is running "
                f"(e.g. llama-server -m model.gguf --host 127.0.0.1 --port "
                f"8080) and reachable at the configured base_url."
            )
        return RuntimeError(
            f"Failed to communicate with llama.cpp backend: {exc}. "
            f"Verify the model path '{self._loaded_model_path or self.default_model}' "
            "is a valid GGUF file."
        )
