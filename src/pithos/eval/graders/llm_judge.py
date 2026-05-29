"""Ollama LLM-as-judge grader.

Wraps a small chat call to a deterministic (temperature=0) Ollama model
that scores ``output`` against ``expected`` on a 0-100 rubric. The judge
is expected to return JSON like::

    {"score": 87, "passed": true, "reasoning": "..."}

If the judge response cannot be parsed, the grader returns score=0,
passed=False, and records the raw response under ``detail``.
"""

from __future__ import annotations

import json
import re
from typing import Any, Optional

from ..models import GradeResult, TaskCase
from .base import Grader

DEFAULT_RUBRIC = (
    "You are an impartial grader. Score the candidate answer against the "
    "expected answer on a 0-100 scale, where 100 is a perfect match in "
    "meaning and correctness. Respond with a single JSON object: "
    '{{"score": <int 0-100>, "passed": <true|false>, '
    '"reasoning": "<short explanation>"}}.\n\n'
    "QUESTION:\n{prompt}\n\n"
    "EXPECTED:\n{expected}\n\n"
    "CANDIDATE:\n{output}\n"
)


_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def _parse_judge_response(raw: str) -> Optional[dict]:
    if not isinstance(raw, str):
        return None
    m = _JSON_RE.search(raw)
    if not m:
        return None
    try:
        data = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict):
        return None
    return data


class OllamaJudge(Grader):
    """LLM-as-judge grader backed by an Ollama chat model.

    Config keys:

    * ``model`` *(str, required)* — Ollama model identifier.
    * ``rubric`` *(str, optional)* — format string with ``{prompt}``,
      ``{expected}``, and ``{output}`` placeholders.
    * ``temperature`` *(float, default 0.0)*.
    * ``pass_threshold`` *(int, default 70)* — used only if the judge
      omits ``passed`` from its response.
    * ``client`` *(any, optional, programmatic)* — injected stand-in for
      :mod:`ollama` exposing ``.chat(model=..., messages=..., options=...)``.
    """

    grader_name = "llm_judge"

    def grade(
        self,
        output: str,
        expected: Any,
        *,
        case: Optional[TaskCase] = None,
        ctx: Optional[Any] = None,
    ) -> GradeResult:
        model = self.config.get("model")
        if not model:
            return GradeResult(
                grader=self.grader_name,
                score=0.0,
                passed=False,
                detail={"error": "judge 'model' not configured"},
            )

        rubric = self.config.get("rubric", DEFAULT_RUBRIC)
        prompt_text = case.prompt if case is not None else ""
        message = rubric.format(
            prompt=prompt_text,
            expected="" if expected is None else str(expected),
            output=output or "",
        )

        client = self.config.get("client")
        if client is None:
            try:
                import ollama  # type: ignore

                client = ollama
            except ImportError:
                return GradeResult(
                    grader=self.grader_name,
                    score=0.0,
                    passed=False,
                    detail={"error": "ollama package not installed"},
                )

        try:
            response = client.chat(
                model=model,
                messages=[{"role": "user", "content": message}],
                options={"temperature": float(self.config.get("temperature", 0.0))},
            )
        except Exception as exc:  # pragma: no cover - defensive
            return GradeResult(
                grader=self.grader_name,
                score=0.0,
                passed=False,
                detail={"error": f"{type(exc).__name__}: {exc}"},
            )

        raw = ""
        if isinstance(response, dict):
            raw = (response.get("message") or {}).get("content", "") or response.get(
                "response", ""
            )
        else:  # ollama-python ChatResponse object
            raw = getattr(getattr(response, "message", None), "content", "") or ""

        parsed = _parse_judge_response(raw)
        if parsed is None:
            return GradeResult(
                grader=self.grader_name,
                score=0.0,
                passed=False,
                detail={"raw": raw, "error": "unparseable judge response"},
            )

        try:
            score = float(parsed.get("score", 0))
        except (TypeError, ValueError):
            score = 0.0
        score = max(0.0, min(100.0, score))

        threshold = float(self.config.get("pass_threshold", 70))
        if "passed" in parsed:
            passed = bool(parsed["passed"])
        else:
            passed = score >= threshold

        return GradeResult(
            grader=self.grader_name,
            score=score,
            passed=passed,
            detail={
                "judge_model": model,
                "reasoning": parsed.get("reasoning", ""),
                "raw": raw,
            },
        )
