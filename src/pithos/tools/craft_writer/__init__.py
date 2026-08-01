"""CraftWriter tool - subagent-driven story writing guided by craft notes.

Given user direction (freeform guidance on what to write) this tool retrieves
prescriptive craft notes previously produced by the ``craft-notes`` tool (see
:mod:`pithos.tools.craft_analyzer`) from the shared knowledge base and drives
a subagent through a 3-stage pipeline:

1. **Outline** - plan a title, premise, and section breakdown.
2. **Draft** - write each section in turn, applying relevant craft notes and
   maintaining continuity with the story so far.
3. **Revise** - a final pass over the full draft for prose style and
   consistency (optional, ``revision_passes`` may be 0 to skip).

The generated story is persisted to a ``craft_stories`` MemoryStore category
and written to a Markdown document.

This module has no optional third-party dependencies beyond what pithos
already requires; :data:`CRAFT_WRITING_AVAILABLE` is always ``True`` and
exists purely for interface parity with other virtual tools that gate on
optional extras.
"""

from .models import (
    CraftStory,
    CraftWriteConfig,
    CraftWriteRequest,
    StoryOutline,
    StorySection,
)

CRAFT_WRITING_AVAILABLE = True


def __getattr__(name):  # pragma: no cover - thin lazy-import shim
    if name == "CraftWriter":
        from .writer import CraftWriter

        return CraftWriter
    if name == "CraftWriterToolExecutor":
        from .writer import CraftWriterToolExecutor

        return CraftWriterToolExecutor
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "CraftStory",
    "CraftWriteConfig",
    "CraftWriteRequest",
    "StoryOutline",
    "StorySection",
    "CraftWriter",
    "CraftWriterToolExecutor",
    "CRAFT_WRITING_AVAILABLE",
]
