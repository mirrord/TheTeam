"""Tests for benchmark dataset abstraction."""

import json
import os
import pytest
import tempfile

from benchmarks.easy_problems.dataset import (
    Question,
    MultipleChoiceQuestion,
    MultipleChoiceDataset,
    FreeFormDataset,
    load_dataset,
)


def test_question_creation():
    """Test basic Question creation."""
    q = Question(
        question_id=1,
        question="What is 2+2?",
        correct_answer="4",
        metadata={"difficulty": "easy"},
    )

    assert q.question_id == 1
    assert q.question == "What is 2+2?"
    assert q.correct_answer == "4"
    assert q.metadata["difficulty"] == "easy"


def test_question_to_dict():
    """Test Question conversion to dictionary."""
    q = Question(
        question_id=0,
        question="Test question",
        correct_answer="Test answer",
        metadata={"key": "value"},
    )

    d = q.to_dict()

    assert d["question_id"] == 0
    assert d["question"] == "Test question"
    assert d["correct_answer"] == "Test answer"
    assert d["key"] == "value"


def test_multiple_choice_question():
    """Test MultipleChoiceQuestion with choices."""
    q = MultipleChoiceQuestion(
        question_id=2,
        question="What is the capital of France?",
        correct_answer="Paris",
        choices=["London", "Paris", "Berlin", "Madrid"],
        metadata={},
    )

    assert len(q.choices) == 4
    assert "Paris" in q.choices

    d = q.to_dict()
    assert "multiple_choice" in d
    assert len(d["multiple_choice"]) == 4


def test_multiple_choice_dataset_load():
    """Test loading MultipleChoiceDataset from JSON."""
    data = [
        {
            "question": "Question 1",
            "correct_answer": "Answer 1",
            "multiple_choice": ["A", "B", "C", "D"],
            "extra_field": "value1",
        },
        {
            "question": "Question 2",
            "correct_answer": "Answer 2",
            "multiple_choice": ["W", "X", "Y", "Z"],
        },
    ]

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(data, f)
        temp_path = f.name

    try:
        dataset = MultipleChoiceDataset(temp_path)
        questions = dataset.load()

        assert len(questions) == 2
        assert questions[0].question == "Question 1"
        assert questions[0].correct_answer == "Answer 1"
        assert len(questions[0].choices) == 4
        assert "extra_field" in questions[0].metadata
        assert questions[0].metadata["extra_field"] == "value1"

        assert questions[1].question == "Question 2"
        assert len(questions[1].choices) == 4
    finally:
        os.unlink(temp_path)


def test_free_form_dataset_load():
    """Test loading FreeFormDataset from JSON."""
    data = [
        {
            "question": "Explain recursion",
            "correct_answer": "A function that calls itself",
            "evaluation_criteria": "Should mention self-reference",
        },
        {"question": "What is OOP?", "correct_answer": "Object-oriented programming"},
    ]

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(data, f)
        temp_path = f.name

    try:
        dataset = FreeFormDataset(temp_path)
        questions = dataset.load()

        assert len(questions) == 2
        assert questions[0].question == "Explain recursion"
        assert "evaluation_criteria" in questions[0].metadata
        assert questions[1].question == "What is OOP?"
    finally:
        os.unlink(temp_path)


def test_dataset_get_questions():
    """Test get_questions caching behavior."""
    data = [
        {
            "question": "Test",
            "correct_answer": "Test",
            "multiple_choice": ["A", "B", "C", "D"],
        }
    ]

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(data, f)
        temp_path = f.name

    try:
        dataset = MultipleChoiceDataset(temp_path)

        assert dataset._loaded is False

        # First call should load
        questions1 = dataset.get_questions()
        assert dataset._loaded is True
        assert len(questions1) == 1

        # Second call should use cache
        questions2 = dataset.get_questions()
        assert questions1 is questions2  # Same object
    finally:
        os.unlink(temp_path)


def test_dataset_len():
    """Test __len__ method."""
    data = [
        {
            "question": f"Question {i}",
            "correct_answer": f"Answer {i}",
            "multiple_choice": ["A", "B", "C", "D"],
        }
        for i in range(5)
    ]

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(data, f)
        temp_path = f.name

    try:
        dataset = MultipleChoiceDataset(temp_path)
        assert len(dataset) == 5
    finally:
        os.unlink(temp_path)


def test_dataset_getitem():
    """Test __getitem__ indexing."""
    data = [
        {
            "question": "First",
            "correct_answer": "A1",
            "multiple_choice": ["A", "B", "C", "D"],
        },
        {
            "question": "Second",
            "correct_answer": "A2",
            "multiple_choice": ["A", "B", "C", "D"],
        },
    ]

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(data, f)
        temp_path = f.name

    try:
        dataset = MultipleChoiceDataset(temp_path)

        assert dataset[0].question == "First"
        assert dataset[1].question == "Second"
    finally:
        os.unlink(temp_path)


def test_dataset_to_json():
    """Test exporting dataset to JSON."""
    data = [
        {
            "question": "Test Q",
            "correct_answer": "Test A",
            "multiple_choice": ["A", "B", "C", "D"],
        }
    ]

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(data, f)
        input_path = f.name

    output_path = tempfile.mktemp(suffix=".json")

    try:
        dataset = MultipleChoiceDataset(input_path)
        dataset.to_json(output_path)

        with open(output_path, "r") as f:
            exported_data = json.load(f)

        assert len(exported_data) == 1
        assert exported_data[0]["question"] == "Test Q"
    finally:
        os.unlink(input_path)
        if os.path.exists(output_path):
            os.unlink(output_path)


def test_load_dataset_factory():
    """Test load_dataset factory function."""
    data = [
        {
            "question": "Test",
            "correct_answer": "Test",
            "multiple_choice": ["A", "B", "C", "D"],
        }
    ]

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(data, f)
        temp_path = f.name

    try:
        # Test multiple_choice
        dataset1 = load_dataset("multiple_choice", temp_path)
        assert isinstance(dataset1, MultipleChoiceDataset)

        # Test free_form
        dataset2 = load_dataset("free_form", temp_path)
        assert isinstance(dataset2, FreeFormDataset)
    finally:
        os.unlink(temp_path)


def test_load_dataset_invalid_type():
    """Test load_dataset with invalid type raises ValueError."""
    with pytest.raises(ValueError, match="Unknown dataset type"):
        load_dataset("invalid_type", "dummy.json")


def test_dataset_file_not_found():
    """Test that loading non-existent file raises FileNotFoundError."""
    dataset = MultipleChoiceDataset("/nonexistent/file.json")

    with pytest.raises(FileNotFoundError):
        dataset.load()
