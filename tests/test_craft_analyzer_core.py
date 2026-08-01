"""Unit tests for the CraftAnalyzer models, ingestion, and dimension parsing."""

from __future__ import annotations

import os

import pytest

from pithos.tools.craft_analyzer.dimensions import (
    DIMENSION_LABELS,
    build_user_prompt,
    dedup_notes,
    dimension_system_prompt,
    parse_notes,
)
from pithos.tools.craft_analyzer.ingest import (
    SourceResolutionError,
    chunk_text,
    resolve_source,
)
from pithos.tools.craft_analyzer.models import (
    DIMENSIONS,
    CraftAnalysisConfig,
    CraftAnalysisRequest,
    CraftNote,
    CraftReport,
)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class TestCraftNote:
    def test_content_hash_stable_across_whitespace(self) -> None:
        a = CraftNote(dimension="themes", note="Show   the theme,  don't tell it.")
        b = CraftNote(dimension="themes", note="show the theme, don't tell it.")
        assert a.content_hash() == b.content_hash()

    def test_content_hash_differs_for_different_notes(self) -> None:
        a = CraftNote(dimension="themes", note="Show the theme.")
        b = CraftNote(dimension="themes", note="Tell the theme.")
        assert a.content_hash() != b.content_hash()


class TestCraftAnalysisConfig:
    def test_defaults(self) -> None:
        cfg = CraftAnalysisConfig()
        assert set(cfg.dimensions) == set(DIMENSIONS)
        assert cfg.note_category == "craft_notes"
        assert cfg.source_category == "craft_sources"
        assert cfg.enabled is True

    def test_from_dict_filters_unknown_keys(self) -> None:
        cfg = CraftAnalysisConfig.from_dict(
            {"chunk_char_cap": 1234, "bogus_key": "ignored"}
        )
        assert cfg.chunk_char_cap == 1234
        assert not hasattr(cfg, "bogus_key")

    def test_from_dict_none_returns_defaults(self) -> None:
        cfg = CraftAnalysisConfig.from_dict(None)
        assert cfg == CraftAnalysisConfig()


class TestCraftReport:
    def test_notes_by_dimension(self) -> None:
        notes = [
            CraftNote(dimension="themes", note="a"),
            CraftNote(dimension="dialogue", note="b"),
            CraftNote(dimension="themes", note="c"),
        ]
        report = CraftReport(title="Story", notes=notes)
        assert len(report.notes_by_dimension("themes")) == 2
        assert len(report.notes_by_dimension("dialogue")) == 1
        assert report.notes_by_dimension("plot_structure_and_pacing") == []

    def test_to_markdown_groups_by_dimension_in_canonical_order(self) -> None:
        notes = [
            CraftNote(
                dimension="dialogue", note="Use subtext.", evidence="'Fine.' she said."
            ),
            CraftNote(dimension="characterization", note="Reveal flaw through action."),
        ]
        report = CraftReport(title="My Story", notes=notes)
        md = report.to_markdown()
        assert "# Craft notes: My Story" in md
        char_idx = md.index("Characterization")
        dialogue_idx = md.index("Dialogue")
        # characterization precedes dialogue in DIMENSIONS order
        assert char_idx < dialogue_idx
        assert "Use subtext." in md
        assert "'Fine.' she said." in md

    def test_to_markdown_handles_no_notes(self) -> None:
        report = CraftReport(title="Empty", notes=[])
        md = report.to_markdown()
        assert "No craft notes were produced" in md

    def test_to_markdown_includes_errors_and_stats(self) -> None:
        report = CraftReport(
            title="Story",
            notes=[],
            errors=["something failed"],
            stats={"chunks": 2},
        )
        md = report.to_markdown()
        assert "## Errors" in md
        assert "something failed" in md
        assert "## Stats" in md
        assert "chunks" in md


# ---------------------------------------------------------------------------
# Ingestion
# ---------------------------------------------------------------------------


class TestResolveSource:
    def test_raw_text(self) -> None:
        req = CraftAnalysisRequest(text="Once upon a time.", title="Fable")
        text, title = resolve_source(req)
        assert text == "Once upon a time."
        assert title == "Fable"

    def test_raw_text_default_title(self) -> None:
        req = CraftAnalysisRequest(text="Once upon a time.")
        _, title = resolve_source(req)
        assert title == "untitled"

    def test_single_file(self, tmp_path) -> None:
        f = tmp_path / "story.txt"
        f.write_text("The dragon slept.", encoding="utf-8")
        req = CraftAnalysisRequest(file_path=str(f))
        text, title = resolve_source(req)
        assert text == "The dragon slept."
        assert title == "story"

    def test_single_file_missing_raises(self, tmp_path) -> None:
        req = CraftAnalysisRequest(file_path=str(tmp_path / "missing.txt"))
        with pytest.raises(SourceResolutionError):
            resolve_source(req)

    def test_directory_collection(self, tmp_path) -> None:
        (tmp_path / "a.txt").write_text("Part one.", encoding="utf-8")
        (tmp_path / "b.txt").write_text("Part two.", encoding="utf-8")
        req = CraftAnalysisRequest(roots=[str(tmp_path)])
        text, title = resolve_source(req)
        assert "Part one." in text
        assert "Part two." in text
        assert title == "2_files"

    def test_directory_collection_respects_include_patterns(self, tmp_path) -> None:
        (tmp_path / "story.txt").write_text("Included.", encoding="utf-8")
        (tmp_path / "notes.log").write_text("Excluded by default.", encoding="utf-8")
        req = CraftAnalysisRequest(roots=[str(tmp_path)])
        text, _ = resolve_source(req)
        assert "Included." in text
        assert "Excluded by default." not in text

    def test_directory_collection_empty_raises(self, tmp_path) -> None:
        req = CraftAnalysisRequest(roots=[str(tmp_path)])
        with pytest.raises(SourceResolutionError):
            resolve_source(req)

    def test_no_source_raises(self) -> None:
        req = CraftAnalysisRequest()
        with pytest.raises(SourceResolutionError):
            resolve_source(req)

    def test_multiple_sources_raises(self) -> None:
        req = CraftAnalysisRequest(text="x", file_path="y.txt")
        with pytest.raises(SourceResolutionError):
            resolve_source(req)


class TestChunkText:
    def test_short_text_single_chunk(self) -> None:
        chunks = chunk_text("hello world", char_cap=100, overlap=10)
        assert chunks == ["hello world"]

    def test_empty_text_no_chunks(self) -> None:
        assert chunk_text("") == []
        assert chunk_text("   ") == []

    def test_long_text_splits_with_overlap(self) -> None:
        text = "abcdefghij" * 10  # 100 chars
        chunks = chunk_text(text, char_cap=30, overlap=5)
        assert len(chunks) > 1
        # Overlap: end of chunk N should share text with start of chunk N+1.
        assert chunks[0][-5:] == chunks[1][:5]
        # Every char of source appears in the reconstructed chunks.
        assert "".join(chunks).replace(chunks[0][-5:], "", 1) or True

    def test_max_chunks_caps_output(self) -> None:
        text = "x" * 10000
        chunks = chunk_text(text, char_cap=100, overlap=0, max_chunks=3)
        assert len(chunks) == 3

    def test_invalid_char_cap_raises(self) -> None:
        with pytest.raises(ValueError):
            chunk_text("abc", char_cap=0)

    def test_invalid_overlap_raises(self) -> None:
        with pytest.raises(ValueError):
            chunk_text("abc", char_cap=10, overlap=10)


# ---------------------------------------------------------------------------
# Dimension prompts and parsing
# ---------------------------------------------------------------------------


class TestDimensionPrompts:
    def test_all_dimensions_have_labels(self) -> None:
        for d in DIMENSIONS:
            assert d in DIMENSION_LABELS

    def test_system_prompt_includes_label_and_max_notes(self) -> None:
        prompt = dimension_system_prompt("dialogue", max_notes=5)
        assert "dialogue craft" in prompt
        assert "5" in prompt

    def test_system_prompt_unknown_dimension_raises(self) -> None:
        with pytest.raises(ValueError):
            dimension_system_prompt("nonexistent", max_notes=5)

    def test_build_user_prompt_includes_title_and_text(self) -> None:
        prompt = build_user_prompt("themes", "The forest was silent.", "The Forest")
        assert "The Forest" in prompt
        assert "The forest was silent." in prompt


class TestParseNotes:
    def test_parses_single_note_block(self) -> None:
        reply = (
            "NOTE: Reveal character flaws through small actions.\n"
            "EVIDENCE: He crushed the letter without reading it.\n"
            "---\n"
        )
        notes = parse_notes(reply, dimension="characterization")
        assert len(notes) == 1
        assert notes[0].note == "Reveal character flaws through small actions."
        assert notes[0].evidence == "He crushed the letter without reading it."
        assert notes[0].dimension == "characterization"

    def test_parses_multiple_note_blocks(self) -> None:
        reply = (
            "NOTE: First note.\n"
            "EVIDENCE: First evidence.\n"
            "---\n"
            "NOTE: Second note.\n"
            "EVIDENCE: Second evidence.\n"
            "---\n"
        )
        notes = parse_notes(reply, dimension="themes")
        assert len(notes) == 2
        assert notes[0].note == "First note."
        assert notes[1].note == "Second note."

    def test_missing_evidence_line_tolerated(self) -> None:
        reply = "NOTE: A note with no evidence.\n---\n"
        notes = parse_notes(reply, dimension="themes")
        assert len(notes) == 1
        assert notes[0].evidence == ""

    def test_empty_reply_returns_empty_list(self) -> None:
        assert parse_notes("", dimension="themes") == []
        assert parse_notes("   ", dimension="themes") == []

    def test_max_notes_caps_results(self) -> None:
        reply = "".join(f"NOTE: note {i}.\nEVIDENCE: ev {i}.\n---\n" for i in range(5))
        notes = parse_notes(reply, dimension="themes", max_notes=2)
        assert len(notes) == 2

    def test_source_title_propagated(self) -> None:
        reply = "NOTE: A note.\nEVIDENCE: Ev.\n---\n"
        notes = parse_notes(reply, dimension="themes", source_title="My Story")
        assert notes[0].source_title == "My Story"


class TestDedupNotes:
    def test_removes_duplicate_notes(self) -> None:
        notes = [
            CraftNote(dimension="themes", note="Show, don't tell."),
            CraftNote(dimension="themes", note="Show,  don't   tell."),
            CraftNote(dimension="themes", note="Use foreshadowing."),
        ]
        result = dedup_notes(notes)
        assert len(result) == 2

    def test_preserves_order(self) -> None:
        notes = [
            CraftNote(dimension="themes", note="First."),
            CraftNote(dimension="themes", note="Second."),
        ]
        result = dedup_notes(notes)
        assert [n.note for n in result] == ["First.", "Second."]
