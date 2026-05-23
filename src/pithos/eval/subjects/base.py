"""Base Subject ABC and the dispatcher that builds subjects from specs."""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Callable, Optional

from ..config import SubjectSpec
from ..models import RunContext, SubjectRun, TaskCase


class Subject(ABC):
    """Abstract base for evaluatable subjects.

    Concrete subjects (agent, flowchart, team) are responsible for
    wiring up metrics + tracing around the underlying runtime call so
    callers only see a uniform :class:`SubjectRun`.
    """

    def __init__(self, name: str, config: Optional[dict] = None) -> None:
        self.name = name
        self.config = dict(config or {})

    @abstractmethod
    def run(self, case: TaskCase, ctx: RunContext) -> SubjectRun:
        """Execute *case* against the subject and return the result."""

    # ------------------------------------------------------------------
    # Convenience helpers used by concrete subjects
    # ------------------------------------------------------------------

    @staticmethod
    def _now() -> datetime:
        return datetime.now()


SubjectFactory = Callable[[SubjectSpec], Subject]


_REGISTRY: dict[str, SubjectFactory] = {}


def register_subject_type(type_name: str, factory: SubjectFactory) -> None:
    """Register a :class:`Subject` factory under *type_name*."""
    _REGISTRY[type_name] = factory


def build_subject(spec: SubjectSpec) -> Subject:
    """Construct a :class:`Subject` from a :class:`SubjectSpec`.

    Falls back to importing the concrete subject classes lazily so
    importing :mod:`pithos.eval.subjects` does not force flowchart /
    team modules to load if they are not needed.
    """
    if spec.type not in _REGISTRY:
        # Lazy registration of built-ins on first call.
        from .agent import AgentSubject
        from .flowchart import FlowchartSubject
        from .team import TeamSubject

        _REGISTRY.setdefault("agent", lambda s: AgentSubject(s.name, s.config))
        _REGISTRY.setdefault("flowchart", lambda s: FlowchartSubject(s.name, s.config))
        _REGISTRY.setdefault("team", lambda s: TeamSubject(s.name, s.config))

    if spec.type not in _REGISTRY:
        raise ValueError(f"Unknown subject type: {spec.type!r}")
    return _REGISTRY[spec.type](spec)
