"""LlamacppAgent — stub LLM agent for llama.cpp backend (not yet implemented)."""

from typing import Optional, Iterator

from .agent import Agent


class LlamacppAgent(Agent):
    """LLM agent backed by llama.cpp.

    This is a stub — llama.cpp backend support is planned for a future release.
    """

    def stream(
        self,
        content: str,
        context_name: Optional[str] = None,
        workspace: Optional[str] = None,
        verbose: bool = False,
        model: Optional[str] = None,
    ) -> Iterator[str]:
        raise NotImplementedError(
            "LlamacppAgent.stream() is not yet implemented. "
            "llama.cpp backend support is planned for a future release."
        )
        # Required to make this a generator in the type system.
        yield  # type: ignore[misc]
