"""Integration tests for the CraftAnalyzer facade and virtual tool executor."""

from __future__ import annotations

import os

import pytest

from pithos.tools.craft_analyzer.analyzer import (
    CraftAnalyzer,
    CraftAnalyzerToolExecutor,
)
from pithos.tools.craft_analyzer.models import CraftAnalysisConfig, CraftAnalysisRequest


class FakeMemoryStore:
    """Records store() calls; mirrors news_researcher tests' fake."""

    def __init__(self) -> None:
        self.calls: list[tuple] = []
        self._n = 0

    def store(self, category, content, metadata=None) -> str:
        self._n += 1
        self.calls.append((category, content, metadata or {}))
        return f"id_{self._n}"


class FakeAgent:
    """Agent whose replies depend on the currently-set system prompt."""

    def __init__(self) -> None:
        self.system = ""
        self.calls = 0

    def set_system_prompt(self, p: str) -> None:
        self.system = p

    def send(self, prompt: str, model=None) -> str:
        self.calls += 1
        label = "themes"
        for dim_label in (
            "characterization",
            "scene construction",
            "thematic development",
            "prose style",
            "dialogue craft",
            "plot structure",
        ):
            if dim_label in self.system:
                label = dim_label
                break
        return (
            f"NOTE: Apply a {label} technique here.\n"
            f"EVIDENCE: relevant quote for {label}.\n"
            "---\n"
        )


class FailingAgent:
    def set_system_prompt(self, p: str) -> None:
        pass

    def send(self, prompt: str, model=None) -> str:
        raise RuntimeError("model backend unavailable")


def _config(**overrides) -> CraftAnalysisConfig:
    base = dict(
        dimensions=["characterization", "themes"],
        chunk_char_cap=1000,
        chunk_overlap=0,
        write_document=False,
    )
    base.update(overrides)
    return CraftAnalysisConfig(**base)


class TestCraftAnalyzerAnalyze:
    def test_analyzes_raw_text_and_produces_notes(self) -> None:
        analyzer = CraftAnalyzer(
            config_manager=None,
            config=_config(),
            agent_factory=lambda: FakeAgent(),
            memory_store=FakeMemoryStore(),
        )
        report = analyzer.analyze("Once upon a time, a knight faced a dragon.")
        assert report.title == "untitled"
        assert len(report.notes) == 2  # one per dimension
        dims = {n.dimension for n in report.notes}
        assert dims == {"characterization", "themes"}
        assert report.errors == []

    def test_stores_notes_and_source_in_memory(self) -> None:
        store = FakeMemoryStore()
        analyzer = CraftAnalyzer(
            config_manager=None,
            config=_config(),
            agent_factory=lambda: FakeAgent(),
            memory_store=store,
        )
        analyzer.analyze(CraftAnalysisRequest(text="A story about loss.", title="Loss"))
        categories = [c[0] for c in store.calls]
        assert "craft_sources" in categories
        assert categories.count("craft_notes") == 2
        note_call = next(c for c in store.calls if c[0] == "craft_notes")
        assert note_call[2]["source_title"] == "Loss"
        assert note_call[2]["dimension"] in {"characterization", "themes"}

    def test_analyzes_file_source(self, tmp_path) -> None:
        f = tmp_path / "story.txt"
        f.write_text("The dragon slept beneath the mountain.", encoding="utf-8")
        analyzer = CraftAnalyzer(
            config_manager=None,
            config=_config(),
            agent_factory=lambda: FakeAgent(),
            memory_store=FakeMemoryStore(),
        )
        report = analyzer.analyze(CraftAnalysisRequest(file_path=str(f)))
        assert report.title == "story"
        assert len(report.notes) == 2

    def test_multiple_chunks_each_analyzed_per_dimension(self) -> None:
        long_text = "A. " * 2000  # forces multiple chunks at char_cap=1000
        analyzer = CraftAnalyzer(
            config_manager=None,
            config=_config(max_notes_per_dimension=50, dedup_notes=False),
            agent_factory=lambda: FakeAgent(),
            memory_store=FakeMemoryStore(),
        )
        report = analyzer.analyze(long_text)
        # More than one chunk should mean more than one note per dimension.
        assert len(report.notes_by_dimension("themes")) > 1

    def test_agent_failure_recorded_as_error_and_continues(self) -> None:
        analyzer = CraftAnalyzer(
            config_manager=None,
            config=_config(),
            agent_factory=lambda: FailingAgent(),
            memory_store=FakeMemoryStore(),
        )
        report = analyzer.analyze("Some story text.")
        assert report.notes == []
        assert len(report.errors) == 2  # one per dimension

    def test_empty_text_returns_report_with_error(self) -> None:
        analyzer = CraftAnalyzer(
            config_manager=None,
            config=_config(),
            agent_factory=lambda: FakeAgent(),
            memory_store=FakeMemoryStore(),
        )
        report = analyzer.analyze("   ")
        assert report.notes == []
        assert report.errors

    def test_dedup_caps_notes_per_dimension(self) -> None:
        analyzer = CraftAnalyzer(
            config_manager=None,
            config=_config(max_notes_per_dimension=1, dedup_notes=True),
            agent_factory=lambda: FakeAgent(),
            memory_store=FakeMemoryStore(),
        )
        long_text = "A. " * 2000
        report = analyzer.analyze(long_text)
        assert len(report.notes_by_dimension("themes")) <= 1

    def test_writes_document_when_enabled(self, tmp_path) -> None:
        out_dir = tmp_path / "craft_reports"
        analyzer = CraftAnalyzer(
            config_manager=None,
            config=_config(write_document=True, output_dir=str(out_dir)),
            agent_factory=lambda: FakeAgent(),
            memory_store=FakeMemoryStore(),
        )
        report = analyzer.analyze(CraftAnalysisRequest(text="Story text.", title="T"))
        assert report.document_path is not None
        assert os.path.isfile(report.document_path)


class TestCraftAnalyzerToolExecutor:
    def test_discover_returns_tool_metadata(self) -> None:
        executor = CraftAnalyzerToolExecutor(config_manager=None)
        tools = executor.discover()
        assert "craft-notes" in tools
        assert tools["craft-notes"].tool_type == "craft_analysis"

    def test_can_execute_matches_tool_name(self) -> None:
        executor = CraftAnalyzerToolExecutor(config_manager=None)
        assert executor.can_execute("craft-notes") is True
        assert executor.can_execute("other-tool") is False

    def test_execute_treats_non_file_arg_as_raw_text(self) -> None:
        analyzer = CraftAnalyzer(
            config_manager=None,
            config=_config(),
            agent_factory=lambda: FakeAgent(),
            memory_store=FakeMemoryStore(),
        )
        executor = CraftAnalyzerToolExecutor(config_manager=None, analyzer=analyzer)
        result = executor.execute("craft-notes A story about a hero.")
        assert result.success is True
        assert "Craft notes" in result.stdout

    def test_execute_treats_existing_file_path_as_file(self, tmp_path) -> None:
        f = tmp_path / "story.txt"
        f.write_text("A hero's journey begins.", encoding="utf-8")
        analyzer = CraftAnalyzer(
            config_manager=None,
            config=_config(),
            agent_factory=lambda: FakeAgent(),
            memory_store=FakeMemoryStore(),
        )
        executor = CraftAnalyzerToolExecutor(config_manager=None, analyzer=analyzer)
        result = executor.execute(f"craft-notes {f}")
        assert result.success is True

    def test_run_failure_returns_unsuccessful_result(self) -> None:
        analyzer = CraftAnalyzer(
            config_manager=None,
            config=_config(),
            agent_factory=lambda: FakeAgent(),
            memory_store=FakeMemoryStore(),
        )
        executor = CraftAnalyzerToolExecutor(config_manager=None, analyzer=analyzer)
        result = executor.run(CraftAnalysisRequest(file_path="/no/such/file.txt"))
        assert result.success is False
        assert result.error_hint
