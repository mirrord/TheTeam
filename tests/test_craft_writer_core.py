"""Unit tests for craft_writer models, note retrieval, and prompt building."""

from __future__ import annotations

import pytest

from pithos.tools.craft_writer.models import (
    CraftStory,
    CraftWriteConfig,
    CraftWriteRequest,
    StoryOutline,
    StorySection,
)
from pithos.tools.craft_writer.notes import (
    format_notes_for_prompt,
    retrieve_craft_notes,
)
from pithos.tools.craft_writer.prompts import (
    build_outline_user_prompt,
    build_revision_user_prompt,
    build_section_user_prompt,
    outline_system_prompt,
    parse_outline,
    revision_system_prompt,
    section_system_prompt,
)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class TestCraftWriteConfig:
    def test_defaults(self) -> None:
        cfg = CraftWriteConfig()
        assert len(cfg.dimensions) == 6
        assert cfg.notes_per_dimension == 5
        assert cfg.note_category == "craft_notes"
        assert cfg.story_category == "craft_stories"
        assert cfg.subagent_config_name == "craft_writer"
        assert cfg.target_word_count == 2000
        assert cfg.revision_passes == 1
        assert cfg.output_dir == "./data/research/stories"
        assert cfg.write_document is True
        assert cfg.store_story is True
        assert cfg.enabled is True

    def test_from_dict_none_uses_defaults(self) -> None:
        cfg = CraftWriteConfig.from_dict(None)
        assert cfg == CraftWriteConfig()

    def test_from_dict_partial_overrides(self) -> None:
        cfg = CraftWriteConfig.from_dict(
            {"notes_per_dimension": 3, "revision_passes": 0}
        )
        assert cfg.notes_per_dimension == 3
        assert cfg.revision_passes == 0
        assert cfg.target_word_count == 2000  # unspecified keeps default

    def test_from_dict_ignores_unknown_keys(self) -> None:
        cfg = CraftWriteConfig.from_dict({"bogus_key": "x", "notes_per_dimension": 2})
        assert cfg.notes_per_dimension == 2
        assert not hasattr(cfg, "bogus_key")


class TestCraftWriteRequest:
    def test_requires_direction(self) -> None:
        req = CraftWriteRequest(direction="a heist gone wrong")
        assert req.direction == "a heist gone wrong"
        assert req.title is None
        assert req.dimensions_override is None


class TestStoryModels:
    def test_story_section_defaults(self) -> None:
        section = StorySection(heading="Opening")
        assert section.summary == ""
        assert section.text == ""

    def test_story_outline_defaults(self) -> None:
        outline = StoryOutline(title="The Last Job")
        assert outline.premise == ""
        assert outline.sections == []

    def test_craft_story_to_markdown_includes_sections(self) -> None:
        story = CraftStory(
            title="The Last Job",
            premise="A crew's final heist goes wrong.",
            sections=[
                StorySection(heading="Opening", text="It was raining."),
                StorySection(heading="The Heist", text="Everything fell apart."),
            ],
            full_text="It was raining.\n\nEverything fell apart.",
        )
        md = story.to_markdown()
        assert "# The Last Job" in md
        assert "It was raining." in md
        assert "Everything fell apart." in md

    def test_craft_story_to_markdown_reports_errors(self) -> None:
        story = CraftStory(title="Untitled", errors=["outline generation failed"])
        md = story.to_markdown()
        assert "outline generation failed" in md


# ---------------------------------------------------------------------------
# Note retrieval
# ---------------------------------------------------------------------------


class FakeSearchResult:
    def __init__(self, content, metadata, relevance_score=0.9) -> None:
        self.content = content
        self.metadata = metadata
        self.relevance_score = relevance_score


class FakeMemoryStore:
    def __init__(self, hits_by_dimension=None) -> None:
        self.hits_by_dimension = hits_by_dimension or {}
        self.calls: list[dict] = []

    def retrieve(self, category, query, n_results=None, where=None, min_relevance=None):
        self.calls.append(
            {
                "category": category,
                "query": query,
                "n_results": n_results,
                "where": where,
                "min_relevance": min_relevance,
            }
        )
        dim = None
        if where:
            if "dimension" in where:
                dim = where["dimension"]
            elif "$and" in where:
                for part in where["$and"]:
                    if "dimension" in part:
                        dim = part["dimension"]
        return self.hits_by_dimension.get(dim, [])


class TestRetrieveCraftNotes:
    def test_returns_empty_dict_entries_when_store_is_none(self) -> None:
        result = retrieve_craft_notes(
            None, dimensions=["characterization", "themes"], query="a heist"
        )
        assert result == {"characterization": [], "themes": []}

    def test_returns_empty_when_query_blank(self) -> None:
        store = FakeMemoryStore()
        result = retrieve_craft_notes(store, dimensions=["themes"], query="   ")
        assert result == {"themes": []}
        assert store.calls == []

    def test_retrieves_per_dimension(self) -> None:
        hit = FakeSearchResult(
            "Ground flaws in action.", {"dimension": "characterization"}
        )
        store = FakeMemoryStore(hits_by_dimension={"characterization": [hit]})
        result = retrieve_craft_notes(
            store,
            dimensions=["characterization", "themes"],
            query="a heist",
            per_dimension=5,
        )
        assert result["characterization"] == [hit]
        assert result["themes"] == []
        assert len(store.calls) == 2
        assert store.calls[0]["category"] == "craft_notes"
        assert store.calls[0]["n_results"] == 5

    def test_uses_and_filter_when_source_title_given(self) -> None:
        store = FakeMemoryStore()
        retrieve_craft_notes(
            store, dimensions=["themes"], query="a heist", source_title="Some Story"
        )
        where = store.calls[0]["where"]
        assert "$and" in where
        keys = {k for part in where["$and"] for k in part}
        assert keys == {"dimension", "source_title"}

    def test_handles_retrieve_exception_gracefully(self) -> None:
        class ExplodingStore:
            def retrieve(self, **kwargs):
                raise RuntimeError("chromadb offline")

        result = retrieve_craft_notes(
            ExplodingStore(), dimensions=["themes"], query="a heist"
        )
        assert result == {"themes": []}


class TestFormatNotesForPrompt:
    def test_empty_returns_placeholder(self) -> None:
        text = format_notes_for_prompt({"themes": []})
        assert "themes" in text.lower() or text.strip()

    def test_formats_notes_with_evidence(self) -> None:
        hit = FakeSearchResult(
            "Ground flaws in action.",
            {"dimension": "characterization", "evidence": "he flinched"},
        )
        text = format_notes_for_prompt({"characterization": [hit]})
        assert "Ground flaws in action." in text
        assert "he flinched" in text


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------


class TestOutlinePrompts:
    def test_outline_system_prompt_mentions_format(self) -> None:
        prompt = outline_system_prompt(num_sections=3)
        assert "TITLE:" in prompt
        assert "PREMISE:" in prompt
        assert "SECTION:" in prompt

    def test_build_outline_user_prompt_includes_direction_and_notes(self) -> None:
        prompt = build_outline_user_prompt(
            direction="a heist gone wrong",
            genre="thriller",
            tone="tense",
            notes_text="- Ground flaws in action.",
        )
        assert "a heist gone wrong" in prompt
        assert "thriller" in prompt
        assert "tense" in prompt
        assert "Ground flaws in action." in prompt

    def test_parse_outline_extracts_title_premise_sections(self) -> None:
        reply = (
            "TITLE: The Last Job\n"
            "PREMISE: A crew's final heist goes wrong.\n"
            "SECTION: Opening - Establish the crew and the stakes.\n"
            "SECTION: The Heist - The plan unravels.\n"
            "SECTION: Aftermath - Reckoning with the fallout.\n"
        )
        outline = parse_outline(reply)
        assert outline.title == "The Last Job"
        assert "final heist" in outline.premise
        assert len(outline.sections) == 3
        assert outline.sections[0].heading == "Opening"
        assert "stakes" in outline.sections[0].summary

    def test_parse_outline_handles_missing_title(self) -> None:
        reply = "PREMISE: Something happens.\nSECTION: Only - It happens.\n"
        outline = parse_outline(reply)
        assert outline.title == "Untitled"
        assert len(outline.sections) == 1

    def test_parse_outline_empty_reply(self) -> None:
        outline = parse_outline("")
        assert outline.title == "Untitled"
        assert outline.sections == []


class TestSectionPrompts:
    def test_section_system_prompt_mentions_target_words(self) -> None:
        prompt = section_system_prompt(target_words=400)
        assert "400" in prompt

    def test_build_section_user_prompt_includes_context(self) -> None:
        prompt = build_section_user_prompt(
            title="The Last Job",
            premise="A crew's final heist goes wrong.",
            section=StorySection(heading="Opening", summary="Establish the crew."),
            story_so_far="",
            notes_text="- Ground flaws in action.",
        )
        assert "The Last Job" in prompt
        assert "Opening" in prompt
        assert "Establish the crew." in prompt
        assert "Ground flaws in action." in prompt

    def test_build_section_user_prompt_includes_story_so_far_when_present(self) -> None:
        prompt = build_section_user_prompt(
            title="T",
            premise="P",
            section=StorySection(heading="Next", summary="S"),
            story_so_far="Previously, the crew met.",
            notes_text="",
        )
        assert "Previously, the crew met." in prompt


class TestRevisionPrompts:
    def test_revision_system_prompt_mentions_consistency(self) -> None:
        prompt = revision_system_prompt()
        assert "consisten" in prompt.lower()

    def test_build_revision_user_prompt_includes_draft_and_notes(self) -> None:
        prompt = build_revision_user_prompt(
            title="T",
            draft_text="Once upon a time.",
            notes_text="- Vary sentence length.",
        )
        assert "Once upon a time." in prompt
        assert "Vary sentence length." in prompt
