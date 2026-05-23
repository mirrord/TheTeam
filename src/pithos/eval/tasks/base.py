"""Task ABC + dispatcher.

A :class:`Task` couples a dataset (source of :class:`TaskCase` objects)
with a grader (scoring strategy). The runner iterates ``task.cases()``,
asks each :class:`Subject` to run the case, then calls ``task.grade()``
on the result.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable, Iterable, Optional

from ..config import TaskSpec
from ..datasets import build_dataset
from ..datasets.base import Dataset
from ..graders import build_grader
from ..graders.base import Grader
from ..models import GradeResult, TaskCase


class Task(ABC):
    """Abstract task — dataset + grader bundle."""

    def __init__(
        self,
        name: str,
        dataset: Dataset,
        grader: Grader,
        config: Optional[dict] = None,
    ) -> None:
        self.name = name
        self.dataset = dataset
        self.grader = grader
        self.config = dict(config or {})

    def cases(self) -> Iterable[TaskCase]:
        """Iterate cases in dataset order."""
        return self.dataset.cases()

    @abstractmethod
    def grade(
        self, case: TaskCase, output: str, ctx: Optional[Any] = None
    ) -> GradeResult:
        """Score *output* for *case*."""


TaskFactory = Callable[[TaskSpec], Task]
_REGISTRY: dict[str, TaskFactory] = {}


def register_task_type(type_name: str, factory: TaskFactory) -> None:
    _REGISTRY[type_name] = factory


def build_task(spec: TaskSpec) -> Task:
    """Construct a :class:`Task` from a :class:`TaskSpec`."""
    if not _REGISTRY:
        from .free_form import FreeFormTask
        from .memory_recall import MemoryRecallTask
        from .multiple_choice import MultipleChoiceTask
        from .self_reflection import SelfReflectionTask
        from .tool_use import ToolUseTask

        def _make(cls):
            return lambda s: cls(
                name=s.name,
                dataset=build_dataset(s.dataset),
                grader=build_grader(s.grader),
                config=s.config,
            )

        _REGISTRY["multiple_choice"] = _make(MultipleChoiceTask)
        _REGISTRY["free_form"] = _make(FreeFormTask)
        _REGISTRY["tool_use"] = _make(ToolUseTask)
        _REGISTRY["memory_recall"] = _make(MemoryRecallTask)
        _REGISTRY["self_reflection"] = _make(SelfReflectionTask)
    if spec.type not in _REGISTRY:
        raise ValueError(f"Unknown task type: {spec.type!r}")
    return _REGISTRY[spec.type](spec)
