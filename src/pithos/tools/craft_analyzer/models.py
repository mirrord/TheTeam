"""Data models for the CraftAnalyzer tool."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Any, Optional

#: The six creative-writing craft dimensions analyzed by this tool.
DIMENSIONS: tuple[str, ...] = (
    "characterization",
    "scene_construction",
    "themes",
    "prose_style_and_voice",
    "dialogue",
    "plot_structure_and_pacing",
)


def _content_hash(text: str) -> str:
    """Return a stable SHA-1 hash of normalised text.

    Whitespace is collapsed and case-folded before hashing so trivially
    different copies of the same source collide.
    """
    normalised = re.sub(r"\s+", " ", text or "").strip().lower()
    return hashlib.sha1(normalised.encode("utf-8"), usedforsecurity=False).hexdigest()


@dataclass
class CraftNote:
    """A single prescriptive how-to note produced for one craft dimension.

    Attributes:
        dimension: One of :data:`DIMENSIONS`.
        note: Prescriptive writing guidance (how to write similarly), e.g.
            "Introduce a character's core flaw through action before naming it."
        evidence: Short quote/paraphrase from the source grounding the note.
        source_title: Title of the analyzed story (best-effort).
        tags: Free-form keyword tags for the note.
        metadata: Free-form metadata stored alongside the note.
    """

    dimension: str
    note: str
    evidence: str = ""
    source_title: str = ""
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def content_hash(self) -> str:
        """Return a stable hash of ``note`` for dedup purposes."""
        return _content_hash(self.note)


@dataclass
class CraftAnalysisConfig:
    """Runtime configuration for a craft analysis run."""

    # --- Dimensions ----------------------------------------------------
    dimensions: list[str] = field(default_factory=lambda: list(DIMENSIONS))

    # --- Ingestion -------------------------------------------------------
    include: list[str] = field(default_factory=lambda: ["**/*.txt", "**/*.md"])
    exclude: list[str] = field(default_factory=list)
    max_files: int = 200
    chunk_char_cap: int = 6000  # max chars sent to the model per chunk
    chunk_overlap: int = 300  # chars of overlap between consecutive chunks
    max_chunks: int = 20  # hard cap on chunks analyzed per run

    # --- Subagent ----------------------------------------------------------
    subagent_config_name: str = "craft_analyst"
    subagent_model: Optional[str] = None
    max_notes_per_dimension: int = 8
    dedup_notes: bool = True

    # --- Knowledge base (persistent MemoryStore) ----------------------------
    note_category: str = "craft_notes"
    source_category: str = "craft_sources"
    memory_persist_directory: Optional[str] = None  # None = MemoryStore default

    # --- Output --------------------------------------------------------
    output_dir: str = "./data/research/craft"
    write_document: bool = True

    enabled: bool = True

    @classmethod
    def from_dict(cls, data: Optional[dict[str, Any]]) -> "CraftAnalysisConfig":
        """Build a config from a (possibly partial) dict, applying defaults."""
        if not data:
            return cls()
        known = {f for f in cls.__dataclass_fields__}
        kwargs = {k: v for k, v in data.items() if k in known}
        return cls(**kwargs)


@dataclass
class CraftAnalysisRequest:
    """A single craft analysis request.

    Exactly one of ``text``, ``file_path``, or ``roots`` should be set;
    :func:`~pithos.tools.craft_analyzer.ingest.resolve_source` validates this.
    """

    text: Optional[str] = None
    file_path: Optional[str] = None
    roots: Optional[list[str]] = None
    title: Optional[str] = None
    dimensions_override: Optional[list[str]] = None


@dataclass
class CraftReport:
    """Output of a craft analysis run."""

    title: str
    notes: list[CraftNote]
    document_path: Optional[str] = None
    stats: dict[str, Any] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)

    def notes_by_dimension(self, dimension: str) -> list[CraftNote]:
        """Return only the notes for a given dimension."""
        return [n for n in self.notes if n.dimension == dimension]

    def to_markdown(self) -> str:
        """Render the report as a Markdown document.

        Groups notes by dimension, in the canonical :data:`DIMENSIONS` order
        (falling back to encounter order for unknown dimension names).
        """
        parts = [f"# Craft notes: {self.title}", ""]
        parts.append(f"**Total notes:** {len(self.notes)}")
        parts.append("")

        seen_order: list[str] = []
        for n in self.notes:
            if n.dimension not in seen_order:
                seen_order.append(n.dimension)
        ordered_dims = [d for d in DIMENSIONS if d in seen_order]
        ordered_dims += [d for d in seen_order if d not in ordered_dims]

        if ordered_dims:
            for dim in ordered_dims:
                dim_notes = self.notes_by_dimension(dim)
                heading = dim.replace("_", " ").title()
                parts.append(f"## {heading} ({len(dim_notes)})")
                parts.append("")
                for note in dim_notes:
                    parts.append(f"- {note.note.strip()}")
                    if note.evidence.strip():
                        parts.append(f"  > {note.evidence.strip()}")
                parts.append("")
        else:
            parts.append("_No craft notes were produced._")
            parts.append("")

        if self.errors:
            parts.append("## Errors")
            parts.append("")
            for err in self.errors:
                parts.append(f"- {err}")
            parts.append("")

        if self.stats:
            parts.append("## Stats")
            parts.append("")
            for k, v in self.stats.items():
                parts.append(f"- **{k}**: {v}")
        return "\n".join(parts)
