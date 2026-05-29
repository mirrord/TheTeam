"""Tests for the bundled built-in datasets under
``src/pithos/eval/datasets/builtins/``.
"""

from __future__ import annotations

import pytest

from pithos.eval.datasets import (
    FreeFormDataset,
    MemoryRecallDataset,
    MultipleChoiceDataset,
    SelfReflectionDataset,
    ToolUseDataset,
    build_dataset,
)


def test_linguistic_basic_via_multiple_choice():
    ds = build_dataset({"type": "multiple_choice", "builtin": "linguistic_basic"})
    assert isinstance(ds, MultipleChoiceDataset)
    cases = list(ds.cases())
    assert len(cases) == 30
    sample = cases[0]
    assert sample.task_type == "multiple_choice"
    assert sample.expected
    assert "choices" in sample.metadata


def test_linguistic_basic_via_free_form():
    ds = build_dataset({"type": "free_form", "builtin": "linguistic_basic"})
    assert isinstance(ds, FreeFormDataset)
    assert len(list(ds.cases())) == 30


@pytest.mark.parametrize(
    "type_name,builtin,cls",
    [
        ("tool_use", "tool_use_basic", ToolUseDataset),
        ("memory_recall", "memory_recall_basic", MemoryRecallDataset),
        ("self_reflection", "self_reflection_basic", SelfReflectionDataset),
    ],
)
def test_capability_suite_builtins(type_name, builtin, cls):
    ds = build_dataset({"type": type_name, "builtin": builtin})
    assert isinstance(ds, cls)
    cases = list(ds.cases())
    assert cases, f"{builtin} should have at least one case"


def test_path_route_still_works(tmp_path):
    p = tmp_path / "mc.json"
    p.write_text(
        '[{"question": "Q", "correct_answer": "A", ' '"multiple_choice": ["A", "B"]}]',
        encoding="utf-8",
    )
    ds = build_dataset({"type": "multiple_choice", "path": str(p)})
    assert len(list(ds.cases())) == 1


def test_spec_requires_path_or_builtin():
    with pytest.raises(ValueError, match="path|builtin"):
        build_dataset({"type": "multiple_choice"})


def test_unknown_builtin_raises():
    with pytest.raises(FileNotFoundError):
        build_dataset({"type": "multiple_choice", "builtin": "does_not_exist"})


def test_unknown_dataset_type_raises():
    with pytest.raises(ValueError, match="Unknown dataset type"):
        build_dataset({"type": "nope", "builtin": "linguistic_basic"})
