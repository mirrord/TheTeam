"""Tests for the CraftNotesNode flowchart node."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from pithos.flownode import CraftNotesNode, create_node
from pithos.tools.craft_analyzer.models import CraftNote, CraftReport


def _fake_analyzer(report: CraftReport):
    a = MagicMock()
    a.analyze.return_value = report
    return a


def _report() -> CraftReport:
    return CraftReport(
        title="My Story",
        notes=[
            CraftNote(
                dimension="characterization",
                note="Reveal flaws through action.",
                evidence="He crushed the letter.",
            )
        ],
        document_path="/tmp/craft.md",
    )


class TestCraftNotesNode:
    def test_create_via_factory(self) -> None:
        node = create_node(
            "craftnotes",
            {"source": "{current_input}", "save_to": "out", "title": "T"},
        )
        assert isinstance(node, CraftNotesNode)
        assert node.save_to == "out"
        assert node.title == "T"

    def test_execute_populates_save_to(self) -> None:
        node = CraftNotesNode(source="{current_input}", save_to="r")
        out = node._execute(
            {"current_input": "A story.", "craft_analyzer": _fake_analyzer(_report())}
        )
        assert "r" in out
        assert out["r"]["title"] == "My Story"
        assert out["r"]["notes"][0]["dimension"] == "characterization"
        assert out["r"]["document_path"] == "/tmp/craft.md"
        assert "Reveal flaws through action." in out["current_input"]

    def test_execute_handles_missing_analyzer(self) -> None:
        node = CraftNotesNode(source="A story.", save_to="r")
        out = node._execute({})
        assert out["r"]["errors"]

    def test_execute_propagates_when_stop_on_error(self) -> None:
        node = CraftNotesNode(source="A story.", save_to="r", error_handling="stop")
        with pytest.raises(RuntimeError):
            node._execute({})

    def test_execute_treats_existing_file_path_as_file(self, tmp_path) -> None:
        f = tmp_path / "story.txt"
        f.write_text("A hero's journey.", encoding="utf-8")
        analyzer = _fake_analyzer(_report())
        node = CraftNotesNode(source=str(f), save_to="r")
        node._execute({"craft_analyzer": analyzer})
        request = analyzer.analyze.call_args[0][0]
        assert request.file_path == str(f)
