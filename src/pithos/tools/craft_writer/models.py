"""Data models for the CraftWriter tool."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from ..craft_analyzer.models import DIMENSIONS


@dataclass
class StorySection:
    """A single section (scene/chapter) of a story, planned or drafted.

    Attributes:
        heading: Short section heading (e.g. "Opening", "The Heist").
        summary: One-line plan for what the section should accomplish
            (produced by the outline stage).
        text: Drafted prose for the section (filled in by the draft stage).
    """

    heading: str
    summary: str = ""
    text: str = ""


@dataclass
class StoryOutline:
    """Planning-stage output: title, premise, and section breakdown."""

    title: str
    premise: str = ""
    sections: list[StorySection] = field(default_factory=list)


@dataclass
class CraftWriteConfig:
    """Runtime configuration for a craft-guided story-writing run."""

    # --- Craft note retrieval --------------------------------------------
    dimensions: list[str] = field(default_factory=lambda: list(DIMENSIONS))
    notes_per_dimension: int = 5
    min_relevance: Optional[float] = None

    # --- Knowledge base (persistent MemoryStore) -------------------------
    note_category: str = "craft_notes"
    story_category: str = "craft_stories"
    memory_persist_directory: Optional[str] = None  # None = MemoryStore default

    # --- Subagent ----------------------------------------------------------
    subagent_config_name: str = "craft_writer"
    subagent_model: Optional[str] = None

    # --- Writing pipeline -------------------------------------------------
    target_word_count: int = 2000
    num_sections: Optional[int] = None
    revision_passes: int = 1
    story_context_char_cap: int = 8000  # cap on story-so-far injected per section

    # --- Output --------------------------------------------------------
    output_dir: str = "./data/research/stories"
    write_document: bool = True
    store_story: bool = True

    enabled: bool = True

    @classmethod
    def from_dict(cls, data: Optional[dict[str, Any]]) -> "CraftWriteConfig":
        """Build a config from a (possibly partial) dict, applying defaults."""
        if not data:
            return cls()
        known = {f for f in cls.__dataclass_fields__}
        kwargs = {k: v for k, v in data.items() if k in known}
        return cls(**kwargs)


@dataclass
class CraftWriteRequest:
    """A single story-writing request.

    ``direction`` (freeform user guidance on what to write) is required;
    everything else refines the request or defaults to config values.
    """

    direction: str
    title: Optional[str] = None
    genre: Optional[str] = None
    tone: Optional[str] = None
    target_word_count: Optional[int] = None
    num_sections: Optional[int] = None
    source_title: Optional[str] = None
    dimensions_override: Optional[list[str]] = None
    revise: Optional[bool] = None


@dataclass
class CraftStory:
    """Output of a craft-guided story-writing run."""

    title: str
    premise: str = ""
    outline: Optional[StoryOutline] = None
    sections: list[StorySection] = field(default_factory=list)
    full_text: str = ""
    notes_used: dict[str, list[Any]] = field(default_factory=dict)
    document_path: Optional[str] = None
    stats: dict[str, Any] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)

    def to_markdown(self) -> str:
        """Render the story as a Markdown document."""
        parts = [f"# {self.title}", ""]
        if self.premise.strip():
            parts.append(f"_{self.premise.strip()}_")
            parts.append("")

        if self.full_text.strip():
            parts.append(self.full_text.strip())
            parts.append("")
        elif self.sections:
            for section in self.sections:
                if section.heading:
                    parts.append(f"## {section.heading}")
                    parts.append("")
                if section.text.strip():
                    parts.append(section.text.strip())
                    parts.append("")
        else:
            parts.append("_No story text was produced._")
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
