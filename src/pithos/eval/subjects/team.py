"""Team subject — wraps :class:`~pithos.team.AgentTeam`.

The team subject delegates the case prompt to the team's coordinator
(via ``team.run`` if available; otherwise via the team's default agent)
and synthesizes a pseudo-trace where each step represents one agent's
turn.
"""

from __future__ import annotations

from typing import Any, Optional

from ..models import RunContext, SubjectRun, TaskCase
from ..trace.ingest import build_eval_trace_from_agent
from .base import Subject


class TeamSubject(Subject):
    """Multi-agent team subject under test."""

    def __init__(self, name: str, config: Optional[dict] = None) -> None:
        super().__init__(name, config)
        self._team: Any = None

    def _build_team(self) -> Any:
        if "instance" in self.config and self.config["instance"] is not None:
            return self.config["instance"]
        raise ValueError(
            f"TeamSubject {self.name!r}: provide a preconstructed "
            "team via config['instance']; programmatic team loading "
            "from YAML is not yet implemented."
        )

    def run(self, case: TaskCase, ctx: RunContext) -> SubjectRun:
        from pithos.metrics import MetricsCollector

        if self._team is None:
            self._team = self._build_team()
        team = self._team

        collector = MetricsCollector()
        # Attach metrics to every member agent we can reach.
        agents_iter = getattr(team, "agents", {})
        if isinstance(agents_iter, dict):
            for agent in agents_iter.values():
                try:
                    agent.attach_metrics(collector)
                except AttributeError:
                    pass

        started = self._now()
        error: Optional[str] = None
        output = ""
        try:
            if hasattr(team, "run"):
                output = team.run(case.prompt)
            elif hasattr(team, "send"):
                output = team.send(case.prompt)
            else:
                raise AttributeError(
                    "AgentTeam instance has neither .run() nor .send()"
                )
            if not isinstance(output, str):
                output = str(output)
        except Exception as exc:  # pragma: no cover
            error = f"{type(exc).__name__}: {exc}"
        ended = self._now()

        trace = build_eval_trace_from_agent(
            subject_name=self.name,
            case_id=case.case_id,
            collector=collector,
            started_at=started,
            ended_at=ended,
            output=output,
        )

        return SubjectRun(
            subject_name=self.name,
            case_id=case.case_id,
            output=output,
            metrics=collector,
            trace=trace,
            started_at=started,
            ended_at=ended,
            error=error,
        )
