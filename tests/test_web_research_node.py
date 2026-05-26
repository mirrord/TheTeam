"""Tests for the WebResearchNode flowchart node."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from pithos.flownode import WebResearchNode, create_node
from pithos.tools.web_researcher.models import ResearchReport


def _fake_researcher(report: ResearchReport):
    r = MagicMock()
    r.research.return_value = report
    return r


class TestWebResearchNode:
    def test_create_via_factory(self) -> None:
        node = create_node(
            "webresearch",
            {
                "inquiry": "{current_input}",
                "save_to": "out",
            },
        )
        assert isinstance(node, WebResearchNode)
        assert node.save_to == "out"

    def test_execute_populates_save_to(self) -> None:
        report = ResearchReport(
            inquiry="topic",
            summary="findings",
            excerpts=[],
            sources=["https://a/b"],
        )
        node = WebResearchNode(inquiry="{current_input}", save_to="r")
        context = {
            "current_input": "topic",
            "web_researcher": _fake_researcher(report),
        }
        out = node._execute(context)
        assert "r" in out
        assert out["r"]["summary"] == "findings"
        assert out["r"]["sources"] == ["https://a/b"]
        assert "topic" in out["current_input"] or "findings" in out["current_input"]

    def test_execute_handles_missing_researcher(self) -> None:
        node = WebResearchNode(inquiry="topic", save_to="r")
        out = node._execute({})  # no researcher, no config_manager
        assert out["r"]["errors"]

    def test_execute_propagates_when_stop_on_error(self) -> None:
        node = WebResearchNode(inquiry="topic", save_to="r", error_handling="stop")
        with pytest.raises(RuntimeError):
            node._execute({})
