"""EXLAgent — stub LLM agent for ExLlamaV2 backend (not yet implemented)."""

from typing import Optional, Iterator

from .agent import Agent


class EXLAgent(Agent):
    """LLM agent backed by ExLlamaV2.

    This is a stub — ExLlamaV2 backend support is planned for a future release.
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
            "EXLAgent.stream() is not yet implemented. "
            "ExLlamaV2 backend support is planned for a future release."
        )
        # Required to make this a generator in the type system.
        yield  # type: ignore[misc]
