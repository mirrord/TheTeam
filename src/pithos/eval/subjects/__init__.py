"""Subject adapters — uniform interface over agents, flowcharts, and teams.

A :class:`Subject` is anything that can take a :class:`~pithos.eval.models.TaskCase`
and produce a :class:`~pithos.eval.models.SubjectRun`. The subject is
responsible for attaching a fresh :class:`~pithos.metrics.MetricsCollector`
and (where applicable) enabling tracing before invoking the underlying
runtime, then ingesting the captured runtime trace into the unified
:class:`~pithos.eval.models.EvalTrace`.
"""

from .base import Subject, SubjectFactory, build_subject
from .agent import AgentSubject
from .flowchart import FlowchartSubject
from .team import TeamSubject

__all__ = [
    "Subject",
    "SubjectFactory",
    "AgentSubject",
    "FlowchartSubject",
    "TeamSubject",
    "build_subject",
]
