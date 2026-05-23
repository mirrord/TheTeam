"""Analyzer ABC + registry.

An analyzer inspects a single :class:`EvalTrace` (per-case) and emits a
list of :class:`TrajectoryIssue` objects. Stateful aggregations across
rounds (e.g. stability) live alongside their own helper rather than the
standard analyzer pipeline.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from ...config import AnalyzerSpec
from ...models import EvalTrace, TrajectoryIssue


@dataclass
class AnalyzerContext:
    """Optional ancillary context passed to analyzers.

    Attributes:
        tool_registry: A :class:`pithos.tools.ToolRegistry`-like object
            (must expose ``list_tool_names() -> Iterable[str]``).
        price_map: Mapping from model name -> dict with
            ``prompt_per_1k`` / ``completion_per_1k`` USD pricing.
        extras: Catch-all for analyzer-specific data.
    """

    tool_registry: Optional[Any] = None
    price_map: dict[str, dict[str, float]] = field(default_factory=dict)
    extras: dict[str, Any] = field(default_factory=dict)


class Analyzer(ABC):
    """Abstract base for trajectory analyzers."""

    analyzer_name: str = "analyzer"

    def __init__(self, config: Optional[dict] = None) -> None:
        self.config = dict(config or {})

    @abstractmethod
    def analyze(
        self, trace: EvalTrace, ctx: Optional[AnalyzerContext] = None
    ) -> list[TrajectoryIssue]:
        """Return zero or more issues for *trace*."""


AnalyzerFactory = Callable[[AnalyzerSpec], Analyzer]


_REGISTRY: dict[str, AnalyzerFactory] = {}


def register_analyzer(type_name: str, factory: AnalyzerFactory) -> None:
    _REGISTRY[type_name] = factory


def _ensure_builtins_registered() -> None:
    if _REGISTRY:
        return
    from .cost import CostAnalyzer
    from .latency import LatencyAnalyzer
    from .loop_detector import LoopDetector
    from .redundancy import RedundancyAnalyzer
    from .security_stub import SecurityStubAnalyzer
    from .tool_hallucination import ToolHallucinationAnalyzer

    _REGISTRY["loop_detector"] = lambda s: LoopDetector(s.config)
    _REGISTRY["redundancy"] = lambda s: RedundancyAnalyzer(s.config)
    _REGISTRY["tool_hallucination"] = lambda s: ToolHallucinationAnalyzer(s.config)
    _REGISTRY["cost"] = lambda s: CostAnalyzer(s.config)
    _REGISTRY["latency"] = lambda s: LatencyAnalyzer(s.config)
    _REGISTRY["security_stub"] = lambda s: SecurityStubAnalyzer(s.config)


def build_analyzer(spec: AnalyzerSpec) -> Analyzer:
    """Construct an :class:`Analyzer` from a :class:`AnalyzerSpec`."""
    _ensure_builtins_registered()
    if spec.type not in _REGISTRY:
        raise ValueError(f"Unknown analyzer type: {spec.type!r}")
    return _REGISTRY[spec.type](spec)
