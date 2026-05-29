"""Dataset ABC + registry.

Built-in dataset types are registered at import time. Third parties
may register additional types via :func:`register_dataset_type`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable, Iterable

from ..models import TaskCase


class Dataset(ABC):
    """A collection of :class:`TaskCase` objects."""

    @abstractmethod
    def cases(self) -> Iterable[TaskCase]:
        """Return an iterable over all cases in the dataset."""

    def __iter__(self):
        return iter(self.cases())

    def __len__(self) -> int:
        return sum(1 for _ in self.cases())


DatasetFactory = Callable[[dict], Dataset]
_REGISTRY: dict[str, DatasetFactory] = {}


def register_dataset_type(type_name: str, factory: DatasetFactory) -> None:
    """Register a dataset factory under ``type_name``."""
    _REGISTRY[type_name] = factory


def build_dataset(spec: dict) -> Dataset:
    """Construct a :class:`Dataset` from a config dict.

    ``spec`` must include ``type`` (e.g. ``multiple_choice``,
    ``free_form``, ``tool_use``, ``memory_recall``, ``self_reflection``)
    plus any type-specific keys (``path`` or ``builtin``).
    """
    type_name = spec.get("type")
    if type_name not in _REGISTRY:
        raise ValueError(f"Unknown dataset type: {type_name!r}")
    return _REGISTRY[type_name](spec)


def _register_builtins() -> None:
    from .capability_suites import (
        _path_from_spec,
        build_memory_recall_dataset,
        build_self_reflection_dataset,
        build_tool_use_dataset,
    )
    from .json_loader import FreeFormDataset, MultipleChoiceDataset

    register_dataset_type(
        "multiple_choice",
        lambda s: MultipleChoiceDataset(
            path=_path_from_spec(s),
            shuffle_choices=s.get("shuffle_choices", True),
        ),
    )
    register_dataset_type(
        "free_form",
        lambda s: FreeFormDataset(path=_path_from_spec(s)),
    )
    register_dataset_type("tool_use", build_tool_use_dataset)
    register_dataset_type("memory_recall", build_memory_recall_dataset)
    register_dataset_type("self_reflection", build_self_reflection_dataset)


_register_builtins()
