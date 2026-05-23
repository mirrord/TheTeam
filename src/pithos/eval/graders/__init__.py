"""Graders — scoring functions that produce :class:`GradeResult` objects.

A grader is a small object with a ``grade(output, expected, *, case=None,
ctx=None) -> GradeResult`` method.  Graders are intentionally pure — they
never call external services unless explicitly wired (e.g.
:class:`OllamaJudge` calls the local Ollama server).
"""

from .base import Grader, GraderRegistry, build_grader, register_grader
from .composite import CompositeGrader
from .exact_match import ExactMatchGrader
from .letter_match import LetterMatchGrader, extract_valid_json
from .llm_judge import OllamaJudge
from .memory_recall import MemoryRecallGrader
from .regex_match import RegexMatchGrader
from .tool_trace import ToolTraceGrader

__all__ = [
    "Grader",
    "GraderRegistry",
    "build_grader",
    "register_grader",
    "CompositeGrader",
    "ExactMatchGrader",
    "LetterMatchGrader",
    "OllamaJudge",
    "RegexMatchGrader",
    "ToolTraceGrader",
    "MemoryRecallGrader",
    "extract_valid_json",
]
