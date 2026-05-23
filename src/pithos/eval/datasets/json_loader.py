"""JSON-backed datasets — multiple-choice and free-form."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Iterable, Optional

from ..models import TaskCase
from .base import Dataset


class JsonDataset(Dataset):
    """Base class for JSON-list-of-records datasets."""

    def __init__(self, path: str) -> None:
        self.path = Path(path)
        self._raw: Optional[list[dict]] = None

    def _load_raw(self) -> list[dict]:
        if self._raw is None:
            if not self.path.exists():
                raise FileNotFoundError(f"Dataset file not found: {self.path}")
            with self.path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, list):
                raise ValueError(
                    f"Dataset {self.path} root must be a list, got {type(data).__name__}"
                )
            self._raw = data
        return self._raw

    def cases(self) -> Iterable[TaskCase]:  # pragma: no cover - overridden
        raise NotImplementedError


class MultipleChoiceDataset(JsonDataset):
    """Multiple-choice dataset.

    Each record is expected to look like::

        {"question": "...", "correct_answer": "...",
         "multiple_choice": ["a", "b", "c", "d"], ...}

    If ``shuffle_choices`` is true (default), choice order is randomized
    per case (the dataset uses the global ``random`` state, so callers
    should seed it for reproducibility).
    """

    LETTERS = ("A", "B", "C", "D")

    def __init__(self, path: str, shuffle_choices: bool = True) -> None:
        super().__init__(path)
        self.shuffle_choices = shuffle_choices

    def cases(self) -> Iterable[TaskCase]:
        for idx, item in enumerate(self._load_raw()):
            question = str(item.get("question", "")).strip()
            correct = str(item.get("correct_answer", "")).strip()
            choices = list(item.get("multiple_choice", []) or [])
            if not choices or correct not in choices:
                continue
            if self.shuffle_choices:
                random.shuffle(choices)
            correct_idx = choices.index(correct)
            correct_letter = self.LETTERS[correct_idx]
            rendered = "\n".join(
                f"{letter}. {choice}" for letter, choice in zip(self.LETTERS, choices)
            )
            sample_letter = random.choice(self.LETTERS)
            prompt = (
                "QUESTION\n"
                f"{question}\n\n"
                "ANSWERS\n"
                f"{rendered}\n\n"
                "Provide an explanation for your thinking and then select a "
                "single choice from ANSWERS that answers the QUESTION. "
                "Return in JSON format, for example:\n"
                f'{{"ANSWER": "{sample_letter}"}}\n'
            )
            metadata = {
                k: v
                for k, v in item.items()
                if k not in ("question", "correct_answer", "multiple_choice")
            }
            metadata["choices"] = choices
            metadata["correct_answer"] = correct
            yield TaskCase(
                case_id=f"mc_{idx}",
                task_type="multiple_choice",
                prompt=prompt,
                expected=correct_letter,
                metadata=metadata,
            )


class FreeFormDataset(JsonDataset):
    """Free-form dataset.

    Each record needs ``question`` and ``correct_answer``; any remaining
    keys are stored on :attr:`TaskCase.metadata`.
    """

    def cases(self) -> Iterable[TaskCase]:
        for idx, item in enumerate(self._load_raw()):
            question = str(item.get("question", "")).strip()
            answer = item.get("correct_answer", "")
            metadata = {
                k: v for k, v in item.items() if k not in ("question", "correct_answer")
            }
            yield TaskCase(
                case_id=f"ff_{idx}",
                task_type="free_form",
                prompt=question,
                expected=answer,
                metadata=metadata,
            )


def load_dataset(dataset_type: str, path: str, **kwargs) -> Dataset:
    """Convenience factory used by tests and the CLI."""
    if dataset_type == "multiple_choice":
        return MultipleChoiceDataset(path, **kwargs)
    if dataset_type == "free_form":
        return FreeFormDataset(path)
    raise ValueError(f"Unknown dataset type: {dataset_type!r}")
