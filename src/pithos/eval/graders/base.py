"""Grader ABC + registry."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable, Optional

from ..config import GraderSpec
from ..models import GradeResult, TaskCase


class Grader(ABC):
    """Abstract base for all graders."""

    grader_name: str = "grader"

    def __init__(self, config: Optional[dict] = None) -> None:
        self.config = dict(config or {})

    @abstractmethod
    def grade(
        self,
        output: str,
        expected: Any,
        *,
        case: Optional[TaskCase] = None,
        ctx: Optional[Any] = None,
    ) -> GradeResult:
        """Score *output* against *expected*."""


GraderFactory = Callable[[GraderSpec], Grader]


GraderRegistry: dict[str, GraderFactory] = {}


def register_grader(type_name: str, factory: GraderFactory) -> None:
    """Register a grader factory under *type_name*."""
    GraderRegistry[type_name] = factory


def _ensure_builtins_registered() -> None:
    if GraderRegistry:
        return
    from .composite import CompositeGrader
    from .exact_match import ExactMatchGrader
    from .letter_match import LetterMatchGrader
    from .llm_judge import OllamaJudge
    from .memory_recall import MemoryRecallGrader
    from .regex_match import RegexMatchGrader
    from .tool_trace import ToolTraceGrader

    GraderRegistry["letter_match"] = lambda s: LetterMatchGrader(s.config)
    GraderRegistry["exact_match"] = lambda s: ExactMatchGrader(s.config)
    GraderRegistry["regex_match"] = lambda s: RegexMatchGrader(s.config)
    GraderRegistry["llm_judge"] = lambda s: OllamaJudge(s.config)
    GraderRegistry["composite"] = lambda s: CompositeGrader(s.config)
    GraderRegistry["tool_trace"] = lambda s: ToolTraceGrader(s.config)
    GraderRegistry["memory_recall"] = lambda s: MemoryRecallGrader(s.config)


def build_grader(spec: GraderSpec) -> Grader:
    """Construct a :class:`Grader` from a :class:`GraderSpec`."""
    _ensure_builtins_registered()
    if spec.type not in GraderRegistry:
        raise ValueError(f"Unknown grader type: {spec.type!r}")
    return GraderRegistry[spec.type](spec)
