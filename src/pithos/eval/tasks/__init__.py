"""Tasks — bind a :class:`Dataset` to a :class:`Grader`."""

from .base import Task, build_task
from .free_form import FreeFormTask
from .memory_recall import MemoryRecallTask
from .multiple_choice import MultipleChoiceTask
from .self_reflection import SelfReflectionTask
from .tool_use import ToolUseTask

__all__ = [
    "Task",
    "build_task",
    "FreeFormTask",
    "MultipleChoiceTask",
    "ToolUseTask",
    "MemoryRecallTask",
    "SelfReflectionTask",
]
