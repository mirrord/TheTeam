"""Tests for the NewsResearchNode flowchart node."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from pithos.flownode import NewsResearchNode, create_node
from pithos.tools.news_researcher.models import ArticleAssessment, NewsReport


def _fake_researcher(report: NewsReport):
    r = MagicMock()
    r.research.return_value = report
    return r


def _report() -> NewsReport:
    return NewsReport(
        inquiry="topic",
        terms=["a", "b"],
        assessments=[
            ArticleAssessment(
                url="https://a/1",
                title="Relevant one",
                summary="a summary",
                relevant=True,
            )
        ],
        document_path="/tmp/news.md",
    )


class TestNewsResearchNode:
    def test_create_via_factory(self) -> None:
        node = create_node(
            "researchnews",
            {"inquiry": "{current_input}", "save_to": "out", "recency_days": 7},
        )
        assert isinstance(node, NewsResearchNode)
        assert node.save_to == "out"
        assert node.recency_days == 7

    def test_execute_populates_save_to(self) -> None:
        node = NewsResearchNode(inquiry="{current_input}", save_to="r")
        out = node._execute(
            {"current_input": "topic", "news_researcher": _fake_researcher(_report())}
        )
        assert "r" in out
        assert out["r"]["terms"] == ["a", "b"]
        assert out["r"]["relevant"][0]["url"] == "https://a/1"
        assert out["r"]["document_path"] == "/tmp/news.md"
        assert "Relevant one" in out["current_input"]

    def test_execute_handles_missing_researcher(self) -> None:
        node = NewsResearchNode(inquiry="topic", save_to="r")
        out = node._execute({})
        assert out["r"]["errors"]

    def test_execute_propagates_when_stop_on_error(self) -> None:
        node = NewsResearchNode(inquiry="topic", save_to="r", error_handling="stop")
        with pytest.raises(RuntimeError):
            node._execute({})
