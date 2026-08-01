"""Integration tests for the CraftWriter facade and virtual tool executor."""

from __future__ import annotations

import pytest

from pithos.tools.craft_writer.models import CraftWriteConfig, CraftWriteRequest
from pithos.tools.craft_writer.writer import CraftWriter, CraftWriterToolExecutor


class FakeMemoryStore:
    """Records store() calls and serves canned retrieve() hits."""

    def __init__(self, hits_by_dimension=None) -> None:
        self.hits_by_dimension = hits_by_dimension or {}
        self.store_calls: list[tuple] = []
        self.retrieve_calls: list[dict] = []
        self._n = 0

    def store(self, category, content, metadata=None) -> str:
        self._n += 1
        self.store_calls.append((category, content, metadata or {}))
        return f"id_{self._n}"

    def retrieve(self, category, query, n_results=None, where=None, min_relevance=None):
        self.retrieve_calls.append({"category": category, "where": where})
        dim = (where or {}).get("dimension")
        if dim is None and where and "$and" in where:
            for part in where["$and"]:
                if "dimension" in part:
                    dim = part["dimension"]
        return self.hits_by_dimension.get(dim, [])


class ScriptedAgent:
    """Agent whose reply depends on the currently-set system prompt stage."""

    def __init__(self) -> None:
        self.system = ""
        self.calls = 0
        self.section_call_count = 0

    def set_system_prompt(self, p: str) -> None:
        self.system = p

    def send(self, prompt: str, model=None) -> str:
        self.calls += 1
        if "story planner" in self.system:
            return (
                "TITLE: The Last Job\n"
                "PREMISE: A crew's final heist goes wrong.\n"
                "SECTION: Opening - Establish the crew and the stakes.\n"
                "SECTION: The Heist - The plan unravels.\n"
            )
        if "fiction writer drafting" in self.system:
            self.section_call_count += 1
            return f"Section prose number {self.section_call_count}."
        if "fiction editor" in self.system:
            return (
                "Revised: "
                + prompt.split("Full draft:\n", 1)[-1].split("\n\nCraft")[0].strip()
            )
        return ""


class FailingOutlineAgent:
    def set_system_prompt(self, p: str) -> None:
        pass

    def send(self, prompt: str, model=None) -> str:
        raise RuntimeError("model backend unavailable")


class FakeSearchResult:
    def __init__(self, content, metadata) -> None:
        self.content = content
        self.metadata = metadata
        self.relevance_score = 0.9


def _config(**overrides) -> CraftWriteConfig:
    base = dict(write_document=False, store_story=True, revision_passes=1)
    base.update(overrides)
    return CraftWriteConfig(**base)


class TestCraftWriterWrite:
    def test_writes_story_with_outline_draft_and_revision(self) -> None:
        writer = CraftWriter(
            config_manager=None,
            config=_config(),
            agent_factory=lambda: ScriptedAgent(),
            memory_store=FakeMemoryStore(),
        )
        story = writer.write("a heist gone wrong")
        assert story.title == "The Last Job"
        assert "final heist" in story.premise
        assert len(story.sections) == 2
        assert story.full_text.startswith("Revised:")
        assert story.errors == []

    def test_accepts_request_object(self) -> None:
        writer = CraftWriter(
            config_manager=None,
            config=_config(),
            agent_factory=lambda: ScriptedAgent(),
            memory_store=FakeMemoryStore(),
        )
        request = CraftWriteRequest(
            direction="a heist gone wrong", title="Custom Title"
        )
        story = writer.write(request)
        assert story.title == "Custom Title"

    def test_raises_on_empty_direction(self) -> None:
        writer = CraftWriter(
            config_manager=None,
            config=_config(),
            agent_factory=lambda: ScriptedAgent(),
            memory_store=FakeMemoryStore(),
        )
        with pytest.raises(ValueError):
            writer.write("   ")

    def test_skips_revision_when_disabled(self) -> None:
        writer = CraftWriter(
            config_manager=None,
            config=_config(revision_passes=0),
            agent_factory=lambda: ScriptedAgent(),
            memory_store=FakeMemoryStore(),
        )
        story = writer.write("a heist gone wrong")
        assert not story.full_text.startswith("Revised:")
        assert "Section prose" in story.full_text

    def test_revise_request_override_forces_revision(self) -> None:
        writer = CraftWriter(
            config_manager=None,
            config=_config(revision_passes=0),
            agent_factory=lambda: ScriptedAgent(),
            memory_store=FakeMemoryStore(),
        )
        story = writer.write(CraftWriteRequest(direction="a heist", revise=True))
        assert story.full_text.startswith("Revised:")

    def test_retrieves_notes_per_dimension_using_direction_as_query(self) -> None:
        store = FakeMemoryStore(
            hits_by_dimension={
                "characterization": [
                    FakeSearchResult(
                        "Ground flaws in action.", {"evidence": "he flinched"}
                    )
                ]
            }
        )
        writer = CraftWriter(
            config_manager=None,
            config=_config(dimensions=["characterization", "themes"]),
            agent_factory=lambda: ScriptedAgent(),
            memory_store=store,
        )
        story = writer.write("a heist gone wrong")
        assert "characterization" in story.notes_used
        assert story.notes_used["characterization"] == ["Ground flaws in action."]
        assert len(store.retrieve_calls) == 2

    def test_stores_story_in_memory(self) -> None:
        store = FakeMemoryStore()
        writer = CraftWriter(
            config_manager=None,
            config=_config(),
            agent_factory=lambda: ScriptedAgent(),
            memory_store=store,
        )
        writer.write(
            CraftWriteRequest(direction="a heist gone wrong", title="The Last Job")
        )
        categories = [c[0] for c in store.store_calls]
        assert "craft_stories" in categories
        story_call = next(c for c in store.store_calls if c[0] == "craft_stories")
        assert story_call[2]["title"] == "The Last Job"

    def test_does_not_store_story_when_disabled(self) -> None:
        store = FakeMemoryStore()
        writer = CraftWriter(
            config_manager=None,
            config=_config(store_story=False),
            agent_factory=lambda: ScriptedAgent(),
            memory_store=store,
        )
        writer.write("a heist gone wrong")
        assert store.store_calls == []

    def test_outline_failure_falls_back_gracefully(self) -> None:
        writer = CraftWriter(
            config_manager=None,
            config=_config(),
            agent_factory=lambda: FailingOutlineAgent(),
            memory_store=FakeMemoryStore(),
        )
        story = writer.write("a heist gone wrong")
        assert story.title == "Untitled"
        assert any("outline generation failed" in e for e in story.errors)

    def test_works_with_no_memory_store(self) -> None:
        writer = CraftWriter(
            config_manager=None,
            config=_config(),
            agent_factory=lambda: ScriptedAgent(),
            memory_store=None,
        )
        story = writer.write("a heist gone wrong")
        assert story.title == "The Last Job"
        assert story.errors == []


class TestCraftWriterToolExecutor:
    def test_discover_returns_metadata(self) -> None:
        executor = CraftWriterToolExecutor(config_manager=None, writer=None)
        # Avoid lazily constructing a real CraftWriter (needs config_manager);
        # discover() must not touch self.writer.
        meta = executor.discover()
        assert "craft-write" in meta
        assert meta["craft-write"].source == "virtual"

    def test_can_execute(self) -> None:
        executor = CraftWriterToolExecutor(config_manager=None, writer=None)
        assert executor.can_execute("craft-write") is True
        assert executor.can_execute("craft-notes") is False

    def test_execute_parses_direction_and_runs(self) -> None:
        fake_writer = CraftWriter(
            config_manager=None,
            config=_config(),
            agent_factory=lambda: ScriptedAgent(),
            memory_store=FakeMemoryStore(),
        )
        executor = CraftWriterToolExecutor(config_manager=None, writer=fake_writer)
        result = executor.execute("craft-write a heist gone wrong")
        assert result.success is True
        assert "The Last Job" in result.stdout

    def test_execute_failure_wraps_error(self) -> None:
        class ExplodingWriter:
            def write(self, request):
                raise RuntimeError("boom")

        executor = CraftWriterToolExecutor(
            config_manager=None, writer=ExplodingWriter()
        )
        result = executor.execute("craft-write ")
        assert result.success is False
        assert "direction" in result.stderr.lower() or "boom" in result.stderr.lower()
