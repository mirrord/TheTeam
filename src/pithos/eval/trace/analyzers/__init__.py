"""Trajectory analyzers — inspect an :class:`EvalTrace` for issues."""

from .base import Analyzer, AnalyzerContext, build_analyzer, register_analyzer
from .cost import CostAnalyzer
from .latency import LatencyAnalyzer
from .loop_detector import LoopDetector
from .redundancy import RedundancyAnalyzer
from .security_stub import SecurityStubAnalyzer
from .stability import StabilityAnalyzer, stability_for_subject
from .tool_hallucination import ToolHallucinationAnalyzer

__all__ = [
    "Analyzer",
    "AnalyzerContext",
    "build_analyzer",
    "register_analyzer",
    "LoopDetector",
    "RedundancyAnalyzer",
    "ToolHallucinationAnalyzer",
    "CostAnalyzer",
    "LatencyAnalyzer",
    "StabilityAnalyzer",
    "stability_for_subject",
    "SecurityStubAnalyzer",
]
