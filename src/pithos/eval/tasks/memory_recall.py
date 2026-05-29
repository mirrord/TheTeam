"""Memory-recall task — two-turn evaluation.

Each :class:`TaskCase` carries one or more ``setup_prompts`` (seeded in
turn 1) and a final :attr:`TaskCase.prompt` (the recall query). The
:class:`~pithos.eval.subjects.agent.AgentSubject` is responsible for
sending the setup prompts to the subject before the graded prompt.
"""

from __future__ import annotations

from typing import Any, Optional

from ..models import GradeResult, TaskCase
from .base import Task


class MemoryRecallTask(Task):
    def grade(
        self, case: TaskCase, output: str, ctx: Optional[Any] = None
    ) -> GradeResult:
        return self.grader.grade(output, case.expected, case=case, ctx=ctx)
