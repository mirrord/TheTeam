"""Configuration model for ``pithos.eval``.

YAML schema (``configs/eval/*.yaml``)::

    name: "My Eval"

    subjects:
      my_agent:
        type: agent              # agent | flowchart | team
        agent: planner           # registered agent config name
        model: glm-4.7-flash
      my_flow:
        type: flowchart
        flowchart: simple_reflect
        agents: [planner]

    tasks:
      linguistic_mc:
        type: multiple_choice
        dataset: { type: multiple_choice, path: ... }
        grader: { type: letter_match }
      tool_basic:
        type: tool_use
        dataset: { type: tool_use, path: ... }
        grader: { type: tool_trace }

    analyzers:
      - { type: loop_detector }
      - { type: redundancy }
      - { type: tool_hallucination }
      - { type: cost, price_map: configs/eval/model_prices.yaml }
      - { type: latency }
      - { type: stability }

    execution:
      rounds: 3
      num_retries: 1
      parallelism: 4

    output:
      base_dir: ./results
      save_traces: true
      create_charts: true
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import yaml


@dataclass
class SubjectSpec:
    name: str
    type: str
    """One of ``agent``, ``flowchart``, ``team``."""
    config: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, name: str, data: dict) -> "SubjectSpec":
        data = dict(data or {})
        stype = data.pop("type", "agent")
        return cls(name=name, type=stype, config=data)


@dataclass
class GraderSpec:
    type: str
    config: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict) -> "GraderSpec":
        data = dict(data or {})
        gtype = data.pop("type", "exact_match")
        return cls(type=gtype, config=data)


@dataclass
class TaskSpec:
    name: str
    type: str
    dataset: dict[str, Any] = field(default_factory=dict)
    grader: GraderSpec = field(default_factory=lambda: GraderSpec(type="exact_match"))
    config: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, name: str, data: dict) -> "TaskSpec":
        data = dict(data or {})
        ttype = data.pop("type", "multiple_choice")
        dataset = data.pop("dataset", {}) or {}
        grader_data = data.pop("grader", {}) or {}
        return cls(
            name=name,
            type=ttype,
            dataset=dataset,
            grader=GraderSpec.from_dict(grader_data),
            config=data,
        )


@dataclass
class AnalyzerSpec:
    type: str
    config: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict) -> "AnalyzerSpec":
        data = dict(data or {})
        atype = data.pop("type")
        return cls(type=atype, config=data)


@dataclass
class EvalExecutionConfig:
    rounds: int = 3
    num_retries: int = 1
    parallelism: int = 1
    timeout: int = 300

    @classmethod
    def from_dict(cls, data: dict) -> "EvalExecutionConfig":
        data = data or {}
        return cls(
            rounds=int(data.get("rounds", 3)),
            num_retries=int(data.get("num_retries", 1)),
            parallelism=int(data.get("parallelism", 1)),
            timeout=int(data.get("timeout", 300)),
        )


@dataclass
class EvalOutputConfig:
    base_dir: str = "./results"
    save_traces: bool = True
    save_metrics: bool = True
    create_charts: bool = True

    @classmethod
    def from_dict(cls, data: dict) -> "EvalOutputConfig":
        data = data or {}
        return cls(
            base_dir=data.get("base_dir", "./results"),
            save_traces=bool(data.get("save_traces", True)),
            save_metrics=bool(data.get("save_metrics", True)),
            create_charts=bool(data.get("create_charts", True)),
        )


@dataclass
class EvalConfig:
    """Top-level evaluation configuration loaded from YAML."""

    name: str
    subjects: dict[str, SubjectSpec]
    tasks: dict[str, TaskSpec]
    analyzers: list[AnalyzerSpec] = field(default_factory=list)
    execution: EvalExecutionConfig = field(default_factory=EvalExecutionConfig)
    output: EvalOutputConfig = field(default_factory=EvalOutputConfig)
    _timestamp: str = field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d"))

    @property
    def folder_name(self) -> str:
        return f"{self._timestamp}-{self.name}"

    @property
    def run_dir(self) -> str:
        return os.path.join(self.output.base_dir, self.folder_name)

    @property
    def cases_dir(self) -> str:
        return os.path.join(self.run_dir, "cases")

    @property
    def stats_dir(self) -> str:
        return os.path.join(self.run_dir, "stats")

    @classmethod
    def from_dict(cls, data: dict) -> "EvalConfig":
        subjects_raw = data.get("subjects", {}) or {}
        tasks_raw = data.get("tasks", {}) or {}
        analyzers_raw = data.get("analyzers", []) or []
        return cls(
            name=data.get("name", "Eval"),
            subjects={
                name: SubjectSpec.from_dict(name, spec)
                for name, spec in subjects_raw.items()
            },
            tasks={
                name: TaskSpec.from_dict(name, spec) for name, spec in tasks_raw.items()
            },
            analyzers=[AnalyzerSpec.from_dict(a) for a in analyzers_raw],
            execution=EvalExecutionConfig.from_dict(data.get("execution", {})),
            output=EvalOutputConfig.from_dict(data.get("output", {})),
        )

    @classmethod
    def from_yaml(cls, path: str) -> "EvalConfig":
        with open(path, "r", encoding="utf-8") as f:
            return cls.from_dict(yaml.safe_load(f) or {})

    @classmethod
    def from_yaml_or_default(cls, path: Optional[str] = None) -> "EvalConfig":
        if path and os.path.exists(path):
            return cls.from_yaml(path)
        default = (
            Path(__file__).parent.parent.parent.parent
            / "configs"
            / "eval"
            / "default_eval.yaml"
        )
        if default.exists():
            return cls.from_yaml(str(default))
        return cls(name="DefaultEval", subjects={}, tasks={})
