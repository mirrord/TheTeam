"""Per-dimension prompt templates and note parsing for the CraftAnalyzer tool.

For each craft dimension the subagent is asked to produce prescriptive
"how-to" notes (reusable writing advice) grounded in short evidence quotes
from the source text, rather than a plain description of what the story
does.
"""

from __future__ import annotations

import re
from typing import Optional

from .models import DIMENSIONS, CraftNote

#: Human-readable label for each dimension, used in prompts and headings.
DIMENSION_LABELS: dict[str, str] = {
    "characterization": "characterization",
    "scene_construction": "scene construction",
    "themes": "thematic development",
    "prose_style_and_voice": "prose style and narrative voice",
    "dialogue": "dialogue craft",
    "plot_structure_and_pacing": "plot structure and pacing",
}

_NOTE_BLOCK_RE = re.compile(
    r"NOTE:\s*(?P<note>.+?)\s*(?:EVIDENCE:\s*(?P<evidence>.*?))?\s*(?=(?:^NOTE:)|\Z)",
    re.DOTALL | re.MULTILINE | re.IGNORECASE,
)


def dimension_system_prompt(dimension: str, max_notes: int) -> str:
    """Return the subagent system prompt for analyzing ``dimension``.

    Raises:
        ValueError: If ``dimension`` is not one of :data:`DIMENSIONS`.
    """
    if dimension not in DIMENSIONS:
        raise ValueError(f"unknown dimension: {dimension}")
    label = DIMENSION_LABELS[dimension]
    return (
        f"You are a creative writing craft analyst specializing in {label}.\n\n"
        "Given a passage from a story, extract PRESCRIPTIVE how-to notes that "
        "a writer could apply to write similar stories. Each note must be "
        "actionable writing advice (a technique, pattern, or principle) - NOT "
        "a plain description or summary of what happens in the passage.\n\n"
        f"Produce at most {max_notes} notes, ordered by importance. Reply with "
        "EXACTLY this format, repeated once per note, and output nothing else:\n\n"
        "NOTE: <one sentence of actionable writing advice>\n"
        "EVIDENCE: <a short quote or close paraphrase from the passage supporting the note>\n"
        "---\n"
    )


def build_user_prompt(dimension: str, source_text: str, title: str) -> str:
    """Build the user-turn prompt asking the subagent to analyze ``dimension``."""
    label = DIMENSION_LABELS.get(dimension, dimension)
    return (
        f"Story title: {title}\n\n"
        f"Analyze the following passage for {label}.\n\n"
        f"Passage:\n{source_text}\n\n"
        "Produce the NOTE/EVIDENCE blocks now."
    )


def parse_notes(
    reply: str,
    dimension: str,
    source_title: str = "",
    max_notes: Optional[int] = None,
) -> list[CraftNote]:
    """Parse a subagent reply into a list of :class:`CraftNote`.

    Tolerant of missing ``EVIDENCE`` lines and trailing/leading whitespace.
    Blocks with an empty ``note`` are dropped.
    """
    if not reply or not reply.strip():
        return []

    notes: list[CraftNote] = []
    for match in _NOTE_BLOCK_RE.finditer(reply):
        note_text = (match.group("note") or "").strip()
        # Strip a trailing "---" or stray "EVIDENCE:" leakage from the note
        # capture when the regex couldn't cleanly separate blocks.
        note_text = re.sub(r"-{2,}\s*$", "", note_text).strip()
        if not note_text:
            continue
        evidence = (match.group("evidence") or "").strip()
        evidence = re.sub(r"-{2,}\s*$", "", evidence).strip()
        notes.append(
            CraftNote(
                dimension=dimension,
                note=note_text,
                evidence=evidence,
                source_title=source_title,
            )
        )
        if max_notes is not None and len(notes) >= max_notes:
            break
    return notes


def dedup_notes(notes: list[CraftNote]) -> list[CraftNote]:
    """Remove notes with duplicate content hashes, preserving first occurrence."""
    seen: set[str] = set()
    result: list[CraftNote] = []
    for note in notes:
        h = note.content_hash()
        if h in seen:
            continue
        seen.add(h)
        result.append(note)
    return result
