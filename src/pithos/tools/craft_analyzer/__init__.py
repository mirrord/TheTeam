"""CraftAnalyzer tool - creative-writing craft notes from stories.

Given a story (raw text, a single file, or a directory/glob collection of
text files), a subagent analyzes the source across a configurable set of
craft dimensions (characterization, scene construction, themes, prose style
and voice, dialogue, plot structure and pacing) and produces prescriptive
"how-to" notes - reusable writing advice grounded in short evidence quotes
from the source - rather than a plain description of the story.

Notes and source provenance are persisted to a ChromaDB-backed
:class:`~pithos.tools.memory_tool.MemoryStore` so they can be semantically
retrieved later (e.g. by an agent drafting a similar story). A Markdown
report grouping notes by dimension is also produced.

This module has no optional third-party dependencies beyond what pithos
already requires (ollama, chromadb for storage); :data:`CRAFT_ANALYSIS_AVAILABLE`
is always ``True`` and exists purely for interface parity with other virtual
tools (e.g. news_researcher) that gate on optional extras.
"""

from .models import (
    DIMENSIONS,
    CraftAnalysisConfig,
    CraftAnalysisRequest,
    CraftNote,
    CraftReport,
)

CRAFT_ANALYSIS_AVAILABLE = True


def __getattr__(name):  # pragma: no cover - thin lazy-import shim
    if name == "CraftAnalyzer":
        from .analyzer import CraftAnalyzer

        return CraftAnalyzer
    if name == "CraftAnalyzerToolExecutor":
        from .analyzer import CraftAnalyzerToolExecutor

        return CraftAnalyzerToolExecutor
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "DIMENSIONS",
    "CraftAnalysisConfig",
    "CraftAnalysisRequest",
    "CraftNote",
    "CraftReport",
    "CraftAnalyzer",
    "CraftAnalyzerToolExecutor",
    "CRAFT_ANALYSIS_AVAILABLE",
]
