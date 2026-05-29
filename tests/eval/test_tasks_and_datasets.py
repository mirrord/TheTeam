"""Tests for datasets and tasks."""

import json
import random

import pytest

from pithos.eval import GraderSpec, TaskSpec
from pithos.eval.datasets import (
    FreeFormDataset,
    MultipleChoiceDataset,
    load_dataset,
)
from pithos.eval.datasets.base import build_dataset
from pithos.eval.tasks import FreeFormTask, MultipleChoiceTask, build_task

# --------------------------------------------------------------------------
# Datasets
# --------------------------------------------------------------------------


def _write_mc(tmp_path):
    data = [
        {
            "question": "Capital of France?",
            "correct_answer": "Paris",
            "multiple_choice": ["Paris", "Berlin", "Rome", "Madrid"],
        },
        {
            "question": "2 + 2?",
            "correct_answer": "4",
            "multiple_choice": ["3", "4", "5", "6"],
            "category": "math",
        },
    ]
    p = tmp_path / "mc.json"
    p.write_text(json.dumps(data), encoding="utf-8")
    return p


def test_multiple_choice_dataset_loads_cases(tmp_path):
    random.seed(0)
    p = _write_mc(tmp_path)
    ds = MultipleChoiceDataset(str(p), shuffle_choices=False)
    cases = list(ds.cases())
    assert len(cases) == 2
    c0 = cases[0]
    assert c0.case_id == "mc_0"
    assert c0.task_type == "multiple_choice"
    assert "Capital of France?" in c0.prompt
    assert c0.expected == "A"  # not shuffled — Paris is first
    assert c0.metadata["correct_answer"] == "Paris"
    assert cases[1].metadata.get("category") == "math"


def test_multiple_choice_dataset_skips_record_with_missing_correct(tmp_path):
    bad = [
        {
            "question": "Q",
            "correct_answer": "Z",
            "multiple_choice": ["A", "B", "C", "D"],
        }
    ]
    p = tmp_path / "bad.json"
    p.write_text(json.dumps(bad), encoding="utf-8")
    ds = MultipleChoiceDataset(str(p), shuffle_choices=False)
    assert list(ds.cases()) == []


def test_multiple_choice_dataset_shuffle(tmp_path):
    p = _write_mc(tmp_path)
    random.seed(42)
    ds = MultipleChoiceDataset(str(p), shuffle_choices=True)
    cases = list(ds.cases())
    # The expected letter should still correspond to Paris.
    c = cases[0]
    idx = "ABCD".index(c.expected)
    assert c.metadata["choices"][idx] == "Paris"


def test_free_form_dataset(tmp_path):
    p = tmp_path / "ff.json"
    p.write_text(
        json.dumps(
            [
                {"question": "Hi?", "correct_answer": "Hello", "tag": "greet"},
            ]
        ),
        encoding="utf-8",
    )
    cases = list(FreeFormDataset(str(p)).cases())
    assert len(cases) == 1
    assert cases[0].prompt == "Hi?"
    assert cases[0].expected == "Hello"
    assert cases[0].metadata["tag"] == "greet"
    assert cases[0].case_id == "ff_0"


def test_load_dataset_dispatches():
    with pytest.raises(ValueError, match="Unknown dataset type"):
        load_dataset("bogus", "x.json")


def test_build_dataset_unknown():
    with pytest.raises(ValueError, match="Unknown dataset type"):
        build_dataset({"type": "bogus", "path": "x.json"})


def test_dataset_missing_file(tmp_path):
    ds = MultipleChoiceDataset(str(tmp_path / "missing.json"))
    with pytest.raises(FileNotFoundError):
        list(ds.cases())


# --------------------------------------------------------------------------
# Tasks
# --------------------------------------------------------------------------


def test_build_task_multiple_choice(tmp_path):
    p = _write_mc(tmp_path)
    spec = TaskSpec(
        name="mc",
        type="multiple_choice",
        dataset={"type": "multiple_choice", "path": str(p), "shuffle_choices": False},
        grader=GraderSpec(type="letter_match"),
    )
    task = build_task(spec)
    assert isinstance(task, MultipleChoiceTask)
    cases = list(task.cases())
    assert len(cases) == 2
    # Grade a correct candidate.
    result = task.grade(cases[0], '{"ANSWER": "A"}')
    assert result.passed is True


def test_build_task_free_form(tmp_path):
    p = tmp_path / "ff.json"
    p.write_text(
        json.dumps([{"question": "Q", "correct_answer": "yes"}]),
        encoding="utf-8",
    )
    spec = TaskSpec(
        name="ff",
        type="free_form",
        dataset={"type": "free_form", "path": str(p)},
        grader=GraderSpec(type="exact_match"),
    )
    task = build_task(spec)
    assert isinstance(task, FreeFormTask)
    cases = list(task.cases())
    r = task.grade(cases[0], "yes")
    assert r.passed is True


def test_build_task_rejects_unknown_type():
    spec = TaskSpec(
        name="x",
        type="nope",
        dataset={"type": "free_form", "path": "x"},
        grader=GraderSpec(type="exact_match"),
    )
    with pytest.raises(ValueError, match="Unknown task type"):
        build_task(spec)
