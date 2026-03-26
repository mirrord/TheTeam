"""Dataset abstraction layer for benchmarks.

This module provides abstract classes and implementations for loading
and managing different types of benchmark datasets.
"""

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional


@dataclass
class Question:
    """Base class representing a single benchmark question."""

    question_id: int
    question: str
    correct_answer: str
    metadata: dict[str, Any]

    def to_dict(self) -> dict:
        """Convert question to dictionary."""
        return {
            "question_id": self.question_id,
            "question": self.question,
            "correct_answer": self.correct_answer,
            **self.metadata,
        }


@dataclass
class MultipleChoiceQuestion(Question):
    """Multiple choice question with options."""

    choices: list[str]
    correct_letter: Optional[str] = None

    def to_dict(self) -> dict:
        """Convert question to dictionary."""
        base = super().to_dict()
        base["multiple_choice"] = self.choices
        if self.correct_letter:
            base["correct_letter"] = self.correct_letter
        return base


class BenchmarkDataset(ABC):
    """Abstract base class for benchmark datasets."""

    def __init__(self, dataset_path: str):
        """Initialize dataset with path to data file."""
        self.dataset_path = Path(dataset_path)
        self.questions: list[Question] = []
        self._loaded = False

    @abstractmethod
    def load(self) -> list[Question]:
        """Load questions from the dataset file.

        Returns:
            List of Question objects
        """
        pass

    def get_questions(self) -> list[Question]:
        """Get all questions, loading if necessary.

        Returns:
            List of Question objects
        """
        if not self._loaded:
            self.questions = self.load()
            self._loaded = True
        return self.questions

    def __len__(self) -> int:
        """Return number of questions in dataset."""
        return len(self.get_questions())

    def __getitem__(self, idx: int) -> Question:
        """Get question by index."""
        return self.get_questions()[idx]

    def to_json(self, output_path: str):
        """Save dataset to JSON file."""
        questions_data = [q.to_dict() for q in self.get_questions()]
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(questions_data, f, indent=2, ensure_ascii=False)


class MultipleChoiceDataset(BenchmarkDataset):
    """Dataset for multiple choice questions."""

    def load(self) -> list[MultipleChoiceQuestion]:
        """Load multiple choice questions from JSON file.

        Expected JSON format:
        [
            {
                "question": "Question text",
                "correct_answer": "Correct answer text",
                "multiple_choice": ["Option 1", "Option 2", "Option 3", "Option 4"]
            },
            ...
        ]

        Returns:
            List of MultipleChoiceQuestion objects
        """
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {self.dataset_path}")

        with open(self.dataset_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        questions = []
        for idx, item in enumerate(data):
            # Extract standard fields
            question_text = item.get("question", "")
            correct_answer = item.get("correct_answer", "")
            choices = item.get("multiple_choice", [])

            # Store any additional fields as metadata
            metadata = {
                k: v
                for k, v in item.items()
                if k not in ["question", "correct_answer", "multiple_choice"]
            }

            questions.append(
                MultipleChoiceQuestion(
                    question_id=idx,
                    question=question_text,
                    correct_answer=correct_answer,
                    choices=choices,
                    metadata=metadata,
                )
            )

        return questions


class FreeFormDataset(BenchmarkDataset):
    """Dataset for free-form questions (no set answer choices)."""

    def load(self) -> list[Question]:
        """Load free-form questions from JSON file.

        Expected JSON format:
        [
            {
                "question": "Question text",
                "correct_answer": "Correct answer text",
                "evaluation_criteria": "How to evaluate answers",
                ...
            },
            ...
        ]

        Returns:
            List of Question objects
        """
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {self.dataset_path}")

        with open(self.dataset_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        questions = []
        for idx, item in enumerate(data):
            question_text = item.get("question", "")
            correct_answer = item.get("correct_answer", "")

            # Store all other fields as metadata
            metadata = {
                k: v for k, v in item.items() if k not in ["question", "correct_answer"]
            }

            questions.append(
                Question(
                    question_id=idx,
                    question=question_text,
                    correct_answer=correct_answer,
                    metadata=metadata,
                )
            )

        return questions


def load_dataset(dataset_type: str, dataset_path: str) -> BenchmarkDataset:
    """Factory function to load appropriate dataset type.

    Args:
        dataset_type: Type of dataset ("multiple_choice", "free_form", etc.)
        dataset_path: Path to dataset file

    Returns:
        Loaded BenchmarkDataset instance

    Raises:
        ValueError: If dataset_type is not recognized
    """
    dataset_types = {
        "multiple_choice": MultipleChoiceDataset,
        "free_form": FreeFormDataset,
    }

    dataset_class = dataset_types.get(dataset_type)
    if not dataset_class:
        raise ValueError(
            f"Unknown dataset type: {dataset_type}. "
            f"Available types: {', '.join(dataset_types.keys())}"
        )

    return dataset_class(dataset_path)
