"""Talos utility helpers."""

from __future__ import annotations

import re


def clean_agent_response(text: str) -> str:
    """Strip tool-call syntax from an agent response.

    The pithos tool-calling system parses the inline bracket syntax
    (``[RUN]...[/RUN]``, ``<RUN>...</RUN>``, ``[EXEC]...[/EXEC]``) directly
    out of the agent's prose so the model can chain actions naturally.
    Those snippets are noise for any downstream consumer that just wants
    the human-readable reply — most importantly the voice interface,
    which should not synthesise speech for ``[RUN]ls -la[/RUN]``.

    This helper removes every detected tool-call span (matched by
    :class:`pithos.tools.ToolCallExtractor`) and collapses the resulting
    run of blank lines so the cleaned text reads naturally.

    Args:
        text: The raw agent response.

    Returns:
        The response with all tool-call syntax removed.
    """
    if not text:
        return text

    from pithos.tools import ToolCallExtractor

    extractor = ToolCallExtractor()
    for call in extractor.extract(text):
        text = text.replace(call.raw_text, "")

    # Collapse 3+ consecutive newlines (left behind by the removals) to 2.
    return re.sub(r"\n{3,}", "\n\n", text).strip()


__all__ = ["clean_agent_response"]
