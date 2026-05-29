"""EXLAgent — LLM agent backed by ExLlamaV2.

The backend packages (``exllamav2`` and ``transformers`` for chat
templating) are optional dependencies.  Importing this module always
succeeds, but constructing an agent without them installed raises
:class:`ImportError` with installation guidance.

Streaming orchestration lives in :class:`pithos.agent.agent.Agent`;
this subclass implements :meth:`Agent._raw_stream` plus lazy model
loading and chat-template rendering.
"""

from __future__ import annotations

from typing import Any, Iterator, Optional
import logging

from .agent import Agent

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------
# Optional backend imports
# ---------------------------------------------------------------------

try:  # pragma: no cover - exercised via monkeypatch in tests
    from exllamav2 import (  # type: ignore[import-not-found]
        ExLlamaV2,
        ExLlamaV2Cache,
        ExLlamaV2Config,
        ExLlamaV2Tokenizer,
    )
    from exllamav2.generator import (  # type: ignore[import-not-found]
        ExLlamaV2DynamicGenerator,
        ExLlamaV2Sampler,
    )

    _EXL_AVAILABLE = True
except ImportError:  # pragma: no cover
    ExLlamaV2 = None  # type: ignore[assignment,misc]
    ExLlamaV2Cache = None  # type: ignore[assignment,misc]
    ExLlamaV2Config = None  # type: ignore[assignment,misc]
    ExLlamaV2Tokenizer = None  # type: ignore[assignment,misc]
    ExLlamaV2DynamicGenerator = None  # type: ignore[assignment,misc]
    ExLlamaV2Sampler = None  # type: ignore[assignment,misc]
    _EXL_AVAILABLE = False

try:  # pragma: no cover
    from transformers import AutoTokenizer  # type: ignore[import-not-found]

    _HF_AVAILABLE = True
except ImportError:  # pragma: no cover
    AutoTokenizer = None  # type: ignore[assignment,misc]
    _HF_AVAILABLE = False


_INSTALL_HINT = (
    "ExLlamaV2 backend dependencies are not installed. "
    "Install with: pip install exllamav2 transformers "
    "(or: pip install theteam[exllamav2])"
)

# Generic ChatML fallback when no HF tokenizer is available and the
# caller did not supply a custom ``chat_template``.
_CHATML_FALLBACK = (
    "{% for m in messages %}<|im_start|>{{ m['role'] }}\n"
    "{{ m['content'] }}<|im_end|>\n{% endfor %}"
    "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
)


class EXLAgent(Agent):
    """LLM agent backed by ExLlamaV2 (fast quantized GPU inference).

    ``default_model`` must be a path to an ExLlamaV2 model directory
    (containing ``config.json``, weights, and a tokenizer).
    Backend-specific options go in ``backend_options``:

    * ``max_seq_len`` — context window override.
    * ``cache_size`` — KV-cache token budget.
    * ``cache_bits`` — KV-cache quantization (8 / 6 / 4).
    * ``chat_template`` — Jinja chat template override (otherwise the
      HF ``AutoTokenizer`` template embedded in the model dir is used).

    Example::

        agent = EXLAgent(
            default_model="/models/Llama-3.2-8B-exl2-6bpw",
            backend_options={"max_seq_len": 16384, "cache_bits": 8},
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
        if not _EXL_AVAILABLE:
            raise ImportError(_INSTALL_HINT)
        super().__init__(default_model, agent_name, system_prompt, temperature)
        self.backend_options: dict[str, Any] = dict(backend_options or {})
        # Cached model state.
        self._exl_model: Any = None
        self._exl_cache: Any = None
        self._exl_tokenizer: Any = None
        self._exl_generator: Any = None
        self._hf_tokenizer: Any = None
        self._loaded_model_path: Optional[str] = None
        # Per-call token counters surfaced to _extract_token_usage.
        self._last_prompt_tokens: int = 0
        self._last_completion_tokens: int = 0

    # ------------------------------------------------------------------
    # Model lifecycle
    # ------------------------------------------------------------------

    def _load_model(self, model_dir: str) -> None:
        """Lazily load (and cache) the ExLlamaV2 stack."""
        if self._exl_generator is not None and self._loaded_model_path == model_dir:
            return
        if self._exl_generator is not None:
            logger.info(
                "EXLAgent: reloading model (was %r, now %r)",
                self._loaded_model_path,
                model_dir,
            )

        cfg = ExLlamaV2Config(model_dir)
        max_seq_len = self.backend_options.get("max_seq_len")
        if max_seq_len is not None:
            cfg.max_seq_len = max_seq_len

        model = ExLlamaV2(cfg)
        cache_size = self.backend_options.get("cache_size", cfg.max_seq_len)
        cache_bits = self.backend_options.get("cache_bits")
        cache_kwargs: dict[str, Any] = {"max_seq_len": cache_size, "lazy": True}
        if cache_bits is not None:
            cache_kwargs["base"] = cache_bits  # interpreted by ExLlamaV2Cache
        cache = ExLlamaV2Cache(model, **cache_kwargs)
        model.load_autosplit(cache)

        tokenizer = ExLlamaV2Tokenizer(cfg)
        generator = ExLlamaV2DynamicGenerator(
            model=model, cache=cache, tokenizer=tokenizer
        )

        # HF tokenizer for chat templating; non-fatal if unavailable.
        hf_tok = None
        if _HF_AVAILABLE and AutoTokenizer is not None:
            try:
                hf_tok = AutoTokenizer.from_pretrained(model_dir)
            except Exception as exc:
                logger.warning(
                    "EXLAgent: falling back to ChatML — could not load HF "
                    "tokenizer from %r: %s",
                    model_dir,
                    exc,
                )

        self._exl_model = model
        self._exl_cache = cache
        self._exl_tokenizer = tokenizer
        self._exl_generator = generator
        self._hf_tokenizer = hf_tok
        self._loaded_model_path = model_dir

    def close(self) -> None:  # type: ignore[override]
        super().close()
        self._exl_model = None
        self._exl_cache = None
        self._exl_tokenizer = None
        self._exl_generator = None
        self._hf_tokenizer = None
        self._loaded_model_path = None

    # ------------------------------------------------------------------
    # Chat templating
    # ------------------------------------------------------------------

    def _render_prompt(self, messages: list[dict[str, Any]]) -> str:
        """Render a list of chat messages to a single prompt string."""
        # 1. HF tokenizer's apply_chat_template (preferred).
        if self._hf_tokenizer is not None:
            try:
                return self._hf_tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            except Exception as exc:
                logger.warning(
                    "EXLAgent: apply_chat_template failed, falling back: %s",
                    exc,
                )

        # 2. Caller-supplied template (Jinja).
        template = self.backend_options.get("chat_template") or _CHATML_FALLBACK
        try:
            from jinja2 import Template  # type: ignore[import-not-found]

            return Template(template).render(
                messages=messages, add_generation_prompt=True
            )
        except ImportError:
            # 3. Final fallback: simple role-prefixed concatenation.
            parts = [f"{m.get('role', 'user')}: {m.get('content', '')}" for m in messages]
            parts.append("assistant:")
            return "\n".join(parts)

    # ------------------------------------------------------------------
    # Backend hooks
    # ------------------------------------------------------------------

    def _raw_stream(
        self,
        messages: list[dict[str, Any]],
        model: str,
        options: dict[str, Any],
    ) -> Iterator[tuple[str, Any]]:
        self._load_model(model)
        gen = self._exl_generator
        assert gen is not None  # narrow for type checker

        prompt = self._render_prompt(messages)

        sampler_settings = ExLlamaV2Sampler.Settings()
        sampler_settings.temperature = float(
            options.get("temperature", self.temperature)
        )

        max_new_tokens = options.get("max_tokens", options.get("num_predict"))
        if max_new_tokens is None or max_new_tokens == -1:
            # Reasonable default; ExLlamaV2 requires a positive value.
            max_new_tokens = 512

        # Count prompt tokens up-front for metrics.
        try:
            self._last_prompt_tokens = int(
                self._exl_tokenizer.encode(prompt).shape[-1]
            )
        except Exception:
            self._last_prompt_tokens = 0

        # ExLlamaV2DynamicGenerator exposes a ``stream(...)`` method that
        # yields incremental text deltas.  We delegate to it.
        stream_fn = getattr(gen, "stream", None)
        if stream_fn is None:
            # Fallback: use generate() and yield once at the end.
            text = gen.generate(
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                gen_settings=sampler_settings,
            )
            self._last_completion_tokens = int(
                self._exl_tokenizer.encode(text).shape[-1]
            ) if text else 0
            yield text or "", {"final": True}
            return

        completion_tokens = 0
        last_chunk: Any = None
        for chunk in stream_fn(
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            gen_settings=sampler_settings,
        ):
            # Chunks may be plain strings or dict-like objects with a
            # ``text`` field; handle both shapes.
            if isinstance(chunk, str):
                token = chunk
                meta: Any = None
            elif isinstance(chunk, dict):
                token = chunk.get("text") or chunk.get("delta") or ""
                meta = chunk
            else:
                token = getattr(chunk, "text", "") or ""
                meta = chunk
            if token:
                completion_tokens += 1
            last_chunk = meta if meta is not None else last_chunk
            yield token, meta

        self._last_completion_tokens = completion_tokens
        # Always emit a final marker chunk so _extract_token_usage runs
        # against something even when stream_fn yielded nothing.
        if last_chunk is None:
            yield "", {"final": True}

    def _extract_token_usage(self, last_chunk: Any) -> tuple[int, int]:
        return self._last_prompt_tokens, self._last_completion_tokens

    def _wrap_backend_error(self, exc: BaseException) -> BaseException:
        return RuntimeError(
            f"Failed to communicate with ExLlamaV2 backend: {exc}. "
            f"Verify the model directory '{self._loaded_model_path or self.default_model}' "
            "contains a valid ExLlamaV2 model."
        )
