"""Flowchart subject — wraps a :class:`~pithos.flowchart.Flowchart`.

Config keys:

* ``flowchart`` — registered flowchart config name.
* ``agents`` — mapping of agent name -> registered agent config name,
  or list of registered agent names (used as both alias + config name).
* ``instance`` *(optional, programmatic)* — preconstructed Flowchart.
* ``agents_instance`` *(optional, programmatic)* — dict[name, Agent]
  passed straight through to ``Flowchart.run``.
* ``max_steps`` *(optional, default 100)* — passed to ``Flowchart.run``.
"""

from __future__ import annotations

from typing import Any, Optional

from ..models import RunContext, SubjectRun, TaskCase
from ..trace.ingest import build_eval_trace_from_flowchart
from .base import Subject


class FlowchartSubject(Subject):
    """Flowchart-based subject under test."""

    def __init__(self, name: str, config: Optional[dict] = None) -> None:
        super().__init__(name, config)
        self._flowchart: Any = None
        self._agents: Optional[dict[str, Any]] = None

    def _build_flowchart(self) -> Any:
        if "instance" in self.config and self.config["instance"] is not None:
            return self.config["instance"]

        from pithos import ConfigManager, Flowchart

        cm = self.config.get("config_manager") or ConfigManager()
        flow_name = self.config.get("flowchart")
        if not flow_name:
            raise ValueError(
                f"FlowchartSubject {self.name!r}: 'flowchart' config key required"
            )
        return Flowchart.from_registered(flow_name, cm)

    def _build_agents(self) -> dict[str, Any]:
        if (
            "agents_instance" in self.config
            and self.config["agents_instance"] is not None
        ):
            return dict(self.config["agents_instance"])

        from pithos import ConfigManager
        from pithos.agent import OllamaAgent

        cm = self.config.get("config_manager") or ConfigManager()
        raw = self.config.get("agents", {})
        agents: dict[str, Any] = {}
        if isinstance(raw, list):
            for alias in raw:
                agents[alias] = OllamaAgent.from_config(alias, cm)
        elif isinstance(raw, dict):
            for alias, registered_name in raw.items():
                agents[alias] = OllamaAgent.from_config(registered_name or alias, cm)
        return agents

    def run(self, case: TaskCase, ctx: RunContext) -> SubjectRun:
        from pithos.metrics import MetricsCollector

        if self._flowchart is None:
            self._flowchart = self._build_flowchart()
        if self._agents is None:
            self._agents = self._build_agents()

        flowchart = self._flowchart
        agents = self._agents

        collector = MetricsCollector()
        flowchart.attach_metrics(collector, name=self.name)
        for agent in agents.values():
            try:
                agent.attach_metrics(collector)
            except AttributeError:
                pass

        if hasattr(flowchart, "enable_trace"):
            flowchart.enable_trace()

        flowchart.reset()

        started = self._now()
        error: Optional[str] = None
        output = ""
        try:
            output = flowchart.run(
                agents=agents,
                start_node=flowchart.start_node,
                initial_input=case.prompt,
                max_steps=int(self.config.get("max_steps", 100)),
            )
        except Exception as exc:  # pragma: no cover
            error = f"{type(exc).__name__}: {exc}"
        ended = self._now()

        runtime_trace = None
        if hasattr(flowchart, "get_execution_trace"):
            runtime_trace = flowchart.get_execution_trace()

        trace = build_eval_trace_from_flowchart(
            subject_name=self.name,
            case_id=case.case_id,
            collector=collector,
            runtime_trace=runtime_trace,
            started_at=started,
            ended_at=ended,
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
