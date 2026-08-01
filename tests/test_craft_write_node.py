"""Tests for the CraftWriteNode flowchart node."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from pithos.flownode import CraftWriteNode, create_node
from pithos.tools.craft_writer.models import CraftStory, StorySection


def _fake_writer(story: CraftStory):
    w = MagicMock()
    w.write.return_value = story
    return w


def _story() -> CraftStory:
    return CraftStory(
        title="The Last Job",
        premise="A crew's final heist goes wrong.",
        sections=[
            StorySection(
                heading="Opening", summary="Set the stakes.", text="It rained."
            )
        ],
        full_text="It rained.",
        document_path="/tmp/story.md",
    )


class TestCraftWriteNode:
    def test_create_via_factory(self) -> None:
        node = create_node(
            "craftwrite",
            {"direction": "{current_input}", "save_to": "out", "title": "T"},
        )
        assert isinstance(node, CraftWriteNode)
        assert node.save_to == "out"
        assert node.title == "T"

    def test_execute_populates_save_to(self) -> None:
        node = CraftWriteNode(direction="{current_input}", save_to="s")
        out = node._execute(
            {
                "current_input": "a heist gone wrong",
                "craft_writer": _fake_writer(_story()),
            }
        )
        assert "s" in out
        assert out["s"]["title"] == "The Last Job"
        assert out["s"]["sections"][0]["heading"] == "Opening"
        assert out["s"]["document_path"] == "/tmp/story.md"
        assert out["current_input"] == "It rained."

    def test_execute_handles_missing_writer(self) -> None:
        node = CraftWriteNode(direction="a heist gone wrong", save_to="s")
        out = node._execute({})
        assert out["s"]["errors"]

    def test_execute_propagates_when_stop_on_error(self) -> None:
        node = CraftWriteNode(
            direction="a heist gone wrong", save_to="s", error_handling="stop"
        )
        with pytest.raises(RuntimeError):
            node._execute({})

    def test_execute_wraps_writer_exception(self) -> None:
        writer = MagicMock()
        writer.write.side_effect = RuntimeError("model backend unavailable")
        node = CraftWriteNode(direction="a heist gone wrong", save_to="s")
        out = node._execute({"craft_writer": writer})
        assert out["s"]["errors"] == ["model backend unavailable"]

    def test_execute_passes_request_fields(self) -> None:
        writer = _fake_writer(_story())
        node = CraftWriteNode(
            direction="{current_input}",
            save_to="s",
            title="T",
            genre="thriller",
            tone="tense",
            source_title="Some Story",
            dimensions=["dialogue"],
        )
        node._execute({"current_input": "a heist gone wrong", "craft_writer": writer})
        request = writer.write.call_args[0][0]
        assert request.direction == "a heist gone wrong"
        assert request.title == "T"
        assert request.genre == "thriller"
        assert request.tone == "tense"
        assert request.source_title == "Some Story"
        assert request.dimensions_override == ["dialogue"]
