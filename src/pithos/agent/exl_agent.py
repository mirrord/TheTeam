"""EXLAgent — stub LLM agent for ExLlamaV2 backend (not yet implemented)."""

from typing import Any, Iterator, Optional

from .agent import Agent

_NOT_IMPLEMENTED_MSG = (
    "EXLAgent backend is planned but not yet implemented. "
    "See docs/ARCHITECTURE.md → Roadmap. "
    "Use OllamaAgent for now."
)


class EXLAgent(Agent):
    """LLM agent backed by ExLlamaV2.

    This is a stub — ExLlamaV2 backend support is planned for a future release.
    Constructing an instance raises :class:`NotImplementedError` so callers fail
    fast rather than receive a half-working agent.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:  # noqa: D401
        raise NotImplementedError(_NOT_IMPLEMENTED_MSG)

    def stream(
        self,
        content: str,
        context_name: Optional[str] = None,
        workspace: Optional[str] = None,
        verbose: bool = False,
        model: Optional[str] = None,
    ) -> Iterator[str]:
        raise NotImplementedError(_NOT_IMPLEMENTED_MSG)
