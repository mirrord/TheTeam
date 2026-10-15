"""Agent subject — wraps a single pithos :class:`~pithos.agent.Agent`.

Config keys (from ``EvalConfig.subjects.<name>``):

* ``agent`` *(optional)* — registered agent config name to load via
  :meth:`Agent.from_config`.  If omitted, a bare agent is built with the
  supplied ``model`` / ``provider`` / ``system_prompt`` / ``temperature``.
* ``model`` *(optional)* — default model identifier.
* ``provider`` *(optional)* — model provider (``ollama`` (default),
  ``llamacpp`` or ``exl``).
* ``system_prompt`` *(optional)* — system prompt override.
* ``temperature`` *(optional)* — sampling temperature override.
"""

from __future__ import annotations

from typing import Any, Optional

from ..models import RunContext, SubjectRun, TaskCase
from ..trace.ingest import build_eval_trace_from_agent
from .base import Subject


class AgentSubject(Subject):
    """Single-agent subject under test."""

    def __init__(self, name: str, config: Optional[dict] = None) -> None:
        super().__init__(name, config)
        self._agent: Any = None  # lazily constructed

    def _build_agent(self) -> Any:
        """Construct or fetch the underlying agent.

        Built lazily so importing the eval package does not require a
        model backend to be installed/available. The ``provider`` config
        key (``ollama`` (default), ``llamacpp`` or ``exl``) selects the
        backend.
        """
        from pithos.agent import Agent, build_agent

        if "instance" in self.config and self.config["instance"] is not None:
            # Test/programmatic injection path.
            return self.config["instance"]

        registered = self.config.get("agent")
        if registered:
            from pithos import ConfigManager

            cm = self.config.get("config_manager") or ConfigManager()
            agent = Agent.from_config(registered, cm)
        else:
            agent = build_agent(
                model=self.config.get("model", "llama3"),
                provider=self.config.get("provider"),
                system_prompt=self.config.get("system_prompt") or "",
                temperature=self.config.get("temperature", 0.0),
            )
        return agent

    def run(self, case: TaskCase, ctx: RunContext) -> SubjectRun:
        from pithos.metrics import MetricsCollector

        if self._agent is None:
            self._agent = self._build_agent()
        agent = self._agent

        collector = MetricsCollector()
        agent.attach_metrics(collector)

        started = self._now()
        error: Optional[str] = None
        output = ""
        try:
            for setup in case.setup_prompts:
                # Setup turns prime the agent (memory / history) but are
                # not graded; we still let metrics accumulate so the
                # trace reflects the full conversation cost.
                agent.send(setup)
            output = agent.send(case.prompt)
        except Exception as exc:  # pragma: no cover - exercised via runner tests
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
