"""Agent provider factory.

TheTeam ships three model-provider backends:

* ``ollama``   — :class:`pithos.agent.ollama_agent.OllamaAgent`
* ``llamacpp`` — :class:`pithos.agent.llamacpp_agent.LlamacppAgent`
* ``exl``      — :class:`pithos.agent.exl_agent.EXLAgent`

Historically every configured model was instantiated as an
:class:`OllamaAgent`.  This module centralises provider selection so that
*wherever a model can be configured for use* the caller may also declare
which provider should serve it (via a ``provider`` config key or an
explicit argument) and get the matching agent subclass back.

The llama.cpp and ExLlamaV2 backends depend on optional heavyweight
packages.  Their classes are imported lazily (only when their provider is
actually requested) so merely importing this module — or using the
default Ollama provider — never drags in those dependencies.
"""

from __future__ import annotations

from typing import Any, Optional, Type

from .agent import Agent

__all__ = [
    "DEFAULT_PROVIDER",
    "SUPPORTED_PROVIDERS",
    "canonical_provider",
    "agent_class_for_provider",
    "build_agent",
    "build_agent_from_config",
]

#: Provider used when a config/caller does not specify one.
DEFAULT_PROVIDER = "ollama"

# Maps every accepted spelling/alias to a canonical provider key.
_PROVIDER_ALIASES: dict[str, str] = {
    "ollama": "ollama",
    "llamacpp": "llamacpp",
    "llama.cpp": "llamacpp",
    "llama_cpp": "llamacpp",
    "llama-cpp": "llamacpp",
    "llamacpp_server": "llamacpp",
    "exl": "exl",
    "exl2": "exl",
    "exllama": "exl",
    "exllamav2": "exl",
}

#: Canonical provider keys understood by the factory.
SUPPORTED_PROVIDERS = ("ollama", "llamacpp", "exl")


def canonical_provider(provider: Optional[str]) -> str:
    """Normalise a provider name to its canonical key.

    ``None``/empty resolves to :data:`DEFAULT_PROVIDER`.  Matching is
    case-insensitive and tolerant of common aliases (``llama.cpp``,
    ``exllamav2`` …).

    Raises:
        ValueError: If ``provider`` is not a recognised backend.
    """
    if provider is None:
        return DEFAULT_PROVIDER
    key = str(provider).strip().lower()
    if not key:
        return DEFAULT_PROVIDER
    resolved = _PROVIDER_ALIASES.get(key)
    if resolved is None:
        raise ValueError(
            f"Unknown model provider {provider!r}. "
            f"Supported providers: {', '.join(SUPPORTED_PROVIDERS)}."
        )
    return resolved


def agent_class_for_provider(provider: Optional[str]) -> Type[Agent]:
    """Return the :class:`Agent` subclass implementing ``provider``.

    The backend module is imported lazily so optional dependencies are
    only touched when their provider is requested.
    """
    key = canonical_provider(provider)
    if key == "ollama":
        from .ollama_agent import OllamaAgent

        return OllamaAgent
    if key == "llamacpp":
        from .llamacpp_agent import LlamacppAgent

        return LlamacppAgent
    if key == "exl":
        from .exl_agent import EXLAgent

        return EXLAgent
    # canonical_provider already guards unknown values.
    raise ValueError(f"Unsupported provider {provider!r}")  # pragma: no cover


def build_agent(
    model: str,
    provider: Optional[str] = None,
    agent_name: Optional[str] = None,
    system_prompt: str = "",
    temperature: Optional[float] = None,
    backend_options: Optional[dict[str, Any]] = None,
    base_url: Optional[str] = None,
) -> Agent:
    """Construct an agent for ``model`` served by ``provider``.

    ``backend_options`` / ``base_url`` are forwarded only to providers
    that accept them (llama.cpp and ExLlamaV2); they are ignored for the
    Ollama provider so callers can pass them uniformly.
    """
    key = canonical_provider(provider)
    cls = agent_class_for_provider(key)
    if key == "ollama":
        return cls(model, agent_name, system_prompt, temperature)
    if key == "llamacpp":
        return cls(
            model,
            agent_name,
            system_prompt,
            temperature,
            backend_options=backend_options,
            base_url=base_url,
        )
    # exl
    return cls(
        model,
        agent_name,
        system_prompt,
        temperature,
        backend_options=backend_options,
    )


def build_agent_from_config(
    config: dict[str, Any], config_manager: Any
) -> Agent:
    """Build an agent from a config dict, honouring its ``provider`` key.

    Dispatches to the selected provider class's own ``from_dict`` so each
    backend applies its specific construction logic (e.g. llama.cpp's
    ``base_url`` / ``backend_options``) while sharing the common
    context/compaction/recall/inference loading.
    """
    cls = agent_class_for_provider(config.get("provider"))
    return cls.from_dict(config, config_manager)
