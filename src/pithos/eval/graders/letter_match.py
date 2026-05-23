"""Letter-match grader — extracts an "ANSWER" letter from JSON in the output.

Designed for multiple-choice tasks where the prompt instructs the model
to return ``{"ANSWER": "X"}`` for ``X in {A, B, C, D}``.
"""

from __future__ import annotations

import json
import re
from typing import Any, Optional

from ..models import GradeResult, TaskCase
from .base import Grader


def extract_valid_json(string: str) -> Optional[dict]:
    """Extract the first valid ``{"ANSWER": "<letter>"}`` JSON block.

    Extracts the last syntactically valid JSON block embedded in *text*.
    Returns ``None`` if no qualifying block is found.
    """
    if not isinstance(string, str) or not string:
        return None
    string_clean = string.replace("\n", "")
    json_pattern = re.compile(r"\{.*?\}|\[.*?\]", re.DOTALL)
    potential_jsons = json_pattern.findall(string_clean)
    if not potential_jsons:
        return None
    potential_jsons.reverse()
    for pj in potential_jsons:
        try:
            valid_json = json.loads(pj)
        except json.JSONDecodeError:
            continue
        if not isinstance(valid_json, dict):
            continue
        valid_json = {k.upper(): v for k, v in valid_json.items()}
        if "ANSWER" not in valid_json:
            continue
        ans = valid_json["ANSWER"]
        if isinstance(ans, str) and ans in ("A", "B", "C", "D"):
            return valid_json
    return None


class LetterMatchGrader(Grader):
    """Score 100 if the extracted letter matches *expected*, else 0."""

    grader_name = "letter_match"

    def grade(
        self,
        output: str,
        expected: Any,
        *,
        case: Optional[TaskCase] = None,
        ctx: Optional[Any] = None,
    ) -> GradeResult:
        parsed = extract_valid_json(output or "")
        letter = parsed["ANSWER"] if parsed else ""
        expected_letter = (expected or "").upper() if isinstance(expected, str) else ""
        passed = bool(letter) and letter == expected_letter
        return GradeResult(
            grader=self.grader_name,
            score=100.0 if passed else 0.0,
            passed=passed,
            detail={
                "extracted_letter": letter,
                "expected_letter": expected_letter,
                "parsed_json": parsed,
            },
        )
