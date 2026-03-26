"""LLM-backed category tag suggestion for the pithos memory system."""

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Optional

try:
    from ollama import chat as ollama_chat
except ImportError:  # pragma: no cover
    ollama_chat = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

# Maximum characters of content sent to the LLM. Long content is truncated to
# avoid blowing out the context window with a prompt that is purely auxiliary.
_CONTENT_TRUNCATE = 800

# Prompt template used to ask the LLM for category tag suggestions.
_SUGGESTION_PROMPT = """\
You are a knowledge-management assistant. Your only job is to suggest short \
category tags that would be used to organise a piece of content in a memory store.

Rules:
- Return ONLY a valid JSON array — no prose, no markdown fences.
- Each element: {{"category": "<tag>", "confidence": <0.0-1.0>, "rationale": "<one sentence>"}}
- Tags must be lowercase, use underscores instead of spaces, max 30 chars.
- Suggest {max_suggestions} tag(s) at most. Fewer is fine when the content is narrow.
- Prefer reusing existing tags when they fit well.

Existing categories: {existing}

Content to categorise:
\"\"\"{content}\"\"\"

JSON array:"""


@dataclass(order=True)
class TagSuggestion:
    """A single LLM-generated category tag suggestion.

    Attributes:
        category: Normalised tag name (lowercase, underscores).
        confidence: Model confidence in the range [0.0, 1.0].
        rationale: One-sentence justification from the model.
    """

    # sort_index is excluded from eq/repr to keep the dataclass clean.
    sort_index: float = field(init=False, repr=False, compare=True)
    confidence: float
    category: str
    rationale: str = ""

    def __post_init__(self) -> None:
        # Clamp confidence and set descending sort key.
        self.confidence = max(0.0, min(1.0, float(self.confidence)))
        self.sort_index = -self.confidence  # negative so sorted() → descending
        # Normalise category name defensively.
        self.category = _normalise_tag(self.category)


def _normalise_tag(tag: str) -> str:
    """Return a safe, normalised category tag.

    Converts to lowercase, replaces runs of non-alphanumeric characters with
    underscores, and trims to 30 characters.
    """
    tag = tag.strip().lower()
    tag = re.sub(r"[^a-z0-9]+", "_", tag)
    tag = tag.strip("_")
    return tag[:30] if tag else "general"


def _parse_suggestions(raw: str, max_suggestions: int) -> list[TagSuggestion]:
    """Parse a JSON array from the model's raw output.

    Returns an empty list rather than raising if the output is malformed.
    """
    # Strip markdown code fences if the model wrapped the JSON.
    raw = re.sub(r"```(?:json)?", "", raw).strip()

    # Try to find the first JSON array in the text.
    match = re.search(r"\[.*\]", raw, re.DOTALL)
    if not match:
        logger.debug("tag_suggester: no JSON array found in model output: %r", raw)
        return []

    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError as exc:
        logger.debug("tag_suggester: JSON parse error: %s — raw: %r", exc, raw)
        return []

    if not isinstance(data, list):
        return []

    suggestions: list[TagSuggestion] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        category = item.get("category", "")
        if not category:
            continue
        try:
            suggestion = TagSuggestion(
                category=str(category),
                confidence=float(item.get("confidence", 0.5)),
                rationale=str(item.get("rationale", "")),
            )
        except (TypeError, ValueError):
            continue
        suggestions.append(suggestion)

    # Sort descending by confidence and cap at max_suggestions.
    return sorted(suggestions)[:max_suggestions]


class CategoryTagSuggester:
    """Uses a local LLM (via Ollama) to suggest category tags for content.

    Example::

        suggester = CategoryTagSuggester(model="llama3")
        tags = suggester.suggest(
            "Python uses indentation to delimit code blocks.",
            existing_categories=["python", "javascript"],
        )
        for tag in tags:
            print(tag.category, tag.confidence)
    """

    def __init__(
        self,
        model: str,
        max_suggestions: int = 3,
        temperature: float = 0.2,
        timeout: int = 30,
    ) -> None:
        """Initialise the suggester.

        Args:
            model: Ollama model name to use for suggestions.
            max_suggestions: Maximum number of tags to return (1-10).
            temperature: LLM sampling temperature. Low values produce more
                deterministic (and therefore more predictable) tag names.
            timeout: HTTP request timeout in seconds.
        """
        if not model:
            raise ValueError("model must be a non-empty string")
        self.model = model
        self.max_suggestions = max(1, min(10, int(max_suggestions)))
        self.temperature = float(temperature)
        self.timeout = int(timeout)

    def suggest(
        self,
        content: str,
        existing_categories: Optional[list[str]] = None,
    ) -> list[TagSuggestion]:
        """Ask the LLM to suggest category tags for *content*.

        Args:
            content: The text to categorise.
            existing_categories: Optional list of already-known categories.
                The LLM will prefer reusing these when appropriate.

        Returns:
            List of :class:`TagSuggestion` objects sorted by descending
            confidence.  Returns an empty list on any error so callers can
            treat suggestions as purely advisory.
        """
        if not content or not content.strip():
            return []

        truncated = content.strip()[:_CONTENT_TRUNCATE]
        existing_text = (
            ", ".join(existing_categories) if existing_categories else "none yet"
        )

        prompt = _SUGGESTION_PROMPT.format(
            max_suggestions=self.max_suggestions,
            existing=existing_text,
            content=truncated,
        )

        if ollama_chat is None:  # pragma: no cover
            logger.warning("tag_suggester: ollama is not installed")
            return []

        try:
            response = ollama_chat(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": self.temperature},
            )
            raw = response.message.content or ""
        except Exception as exc:
            logger.warning("tag_suggester: LLM call failed: %s", exc)
            return []

        return _parse_suggestions(raw, self.max_suggestions)
