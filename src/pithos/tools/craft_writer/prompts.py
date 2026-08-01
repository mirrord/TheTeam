"""Prompt templates and reply parsing for the CraftWriter tool's 3-stage
outline -> per-section draft -> revision pipeline.

Each stage sends one-shot, stateless prompts to the subagent (the running
story-so-far is passed explicitly in the prompt rather than relying on
conversational memory), mirroring the news_researcher assessor pattern.
"""

from __future__ import annotations

import re

from .models import StoryOutline, StorySection

_TITLE_RE = re.compile(r"^TITLE:\s*(.+)$", re.MULTILINE | re.IGNORECASE)
_PREMISE_RE = re.compile(
    r"^PREMISE:\s*(.+?)(?=^SECTION:|\Z)", re.MULTILINE | re.IGNORECASE | re.DOTALL
)
_SECTION_RE = re.compile(
    r"^SECTION:\s*(?P<heading>[^-\n]+?)\s*-\s*(?P<summary>.+)$",
    re.MULTILINE | re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Stage 1: outline
# ---------------------------------------------------------------------------


def outline_system_prompt(num_sections: int) -> str:
    """Return the system prompt for the outline-planning stage."""
    return (
        "You are a story planner. Given creative direction and craft notes "
        "distilled from other stories, produce a tight outline for a new "
        f"short story with about {num_sections} sections.\n\n"
        "Reply with EXACTLY this format and nothing else:\n\n"
        "TITLE: <a story title>\n"
        "PREMISE: <one or two sentence premise>\n"
        "SECTION: <section heading> - <one sentence of what this section accomplishes>\n"
        "(repeat the SECTION line once per section)\n"
    )


def build_outline_user_prompt(
    direction: str,
    genre: str = "",
    tone: str = "",
    notes_text: str = "",
) -> str:
    """Build the user-turn prompt for the outline stage."""
    parts = [f"Direction: {direction}"]
    if genre:
        parts.append(f"Genre: {genre}")
    if tone:
        parts.append(f"Tone: {tone}")
    parts.append("")
    parts.append("Craft notes to apply:")
    parts.append(notes_text or "None available.")
    parts.append("")
    parts.append("Produce the TITLE/PREMISE/SECTION lines now.")
    return "\n".join(parts)


def parse_outline(reply: str) -> StoryOutline:
    """Parse a subagent reply into a :class:`StoryOutline`.

    Tolerant of a missing TITLE (defaults to ``"Untitled"``) or PREMISE
    (defaults to an empty string). Returns an outline with no sections for
    a blank/unparseable reply.
    """
    if not reply or not reply.strip():
        return StoryOutline(title="Untitled", premise="", sections=[])

    title_match = _TITLE_RE.search(reply)
    title = title_match.group(1).strip() if title_match else ""

    premise_match = _PREMISE_RE.search(reply)
    premise = premise_match.group(1).strip() if premise_match else ""

    sections: list[StorySection] = []
    for match in _SECTION_RE.finditer(reply):
        heading = match.group("heading").strip()
        summary = match.group("summary").strip()
        if heading:
            sections.append(StorySection(heading=heading, summary=summary))

    return StoryOutline(title=title or "Untitled", premise=premise, sections=sections)


# ---------------------------------------------------------------------------
# Stage 2: per-section draft
# ---------------------------------------------------------------------------


def section_system_prompt(target_words: int) -> str:
    """Return the system prompt for the per-section drafting stage."""
    return (
        "You are a fiction writer drafting one section of a short story at a "
        "time. Apply the given craft notes naturally - do not mention them "
        "explicitly. Maintain continuity with the story so far. Write prose "
        f"only (no headings, no meta commentary), aiming for about "
        f"{target_words} words for this section."
    )


def build_section_user_prompt(
    title: str,
    premise: str,
    section: StorySection,
    story_so_far: str = "",
    notes_text: str = "",
) -> str:
    """Build the user-turn prompt asking the subagent to draft one section."""
    parts = [f"Story title: {title}", f"Premise: {premise}", ""]
    parts.append(f"This section: {section.heading} - {section.summary}")
    parts.append("")
    if story_so_far.strip():
        parts.append("Story so far:")
        parts.append(story_so_far.strip())
        parts.append("")
    parts.append("Craft notes to apply:")
    parts.append(notes_text or "None available.")
    parts.append("")
    parts.append("Write this section's prose now.")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Stage 3: revision
# ---------------------------------------------------------------------------


def revision_system_prompt() -> str:
    """Return the system prompt for the whole-draft revision stage."""
    return (
        "You are a fiction editor. Revise the full draft for prose style, "
        "voice, and consistency (character details, timeline, tone) while "
        "preserving the plot and structure. Apply the given craft notes. "
        "Reply with the complete revised story text only - no commentary, "
        "no headings, no explanations."
    )


def build_revision_user_prompt(
    title: str, draft_text: str, notes_text: str = ""
) -> str:
    """Build the user-turn prompt for the revision stage."""
    parts = [
        f"Story title: {title}",
        "",
        "Full draft:",
        draft_text,
        "",
        "Craft notes to apply:",
        notes_text or "None available.",
        "",
        "Produce the complete revised story text now.",
    ]
    return "\n".join(parts)
