"""Tests for built-in graders."""

from unittest.mock import MagicMock

from pithos.eval import GraderSpec, TaskCase
from pithos.eval.graders import (
    CompositeGrader,
    ExactMatchGrader,
    LetterMatchGrader,
    OllamaJudge,
    RegexMatchGrader,
    build_grader,
    extract_valid_json,
)

# --------------------------------------------------------------------------
# extract_valid_json
# --------------------------------------------------------------------------


def test_extract_valid_json_finds_answer():
    assert extract_valid_json('Some text {"ANSWER": "B"} trailing')["ANSWER"] == "B"


def test_extract_valid_json_case_insensitive_key():
    assert extract_valid_json('blah {"answer": "C"}')["ANSWER"] == "C"


def test_extract_valid_json_returns_none_when_letter_invalid():
    assert extract_valid_json('{"ANSWER": "Z"}') is None
    assert extract_valid_json("no json here") is None
    assert extract_valid_json("") is None


def test_extract_valid_json_prefers_last_valid_block():
    # Both blocks are valid; the function reverses then iterates, so
    # the LAST occurrence wins.
    result = extract_valid_json('{"ANSWER": "A"} ... {"ANSWER": "D"}')
    assert result["ANSWER"] == "D"


# --------------------------------------------------------------------------
# LetterMatchGrader
# --------------------------------------------------------------------------


def test_letter_match_pass():
    g = LetterMatchGrader()
    r = g.grade('reasoning... {"ANSWER": "B"}', "B")
    assert r.passed is True
    assert r.score == 100.0
    assert r.detail["extracted_letter"] == "B"


def test_letter_match_fail_wrong_letter():
    g = LetterMatchGrader()
    r = g.grade('{"ANSWER": "C"}', "B")
    assert r.passed is False
    assert r.score == 0.0


def test_letter_match_fail_no_json():
    g = LetterMatchGrader()
    r = g.grade("garbage", "B")
    assert r.passed is False
    assert r.detail["extracted_letter"] == ""


# --------------------------------------------------------------------------
# ExactMatchGrader
# --------------------------------------------------------------------------


def test_exact_match_pass_default_case_sensitive():
    r = ExactMatchGrader().grade("Hello ", "Hello")
    assert r.passed is True


def test_exact_match_case_insensitive_config():
    r = ExactMatchGrader({"case_sensitive": False}).grade("HELLO", "hello")
    assert r.passed is True


def test_exact_match_fail_case_sensitive():
    r = ExactMatchGrader().grade("HELLO", "hello")
    assert r.passed is False


# --------------------------------------------------------------------------
# RegexMatchGrader
# --------------------------------------------------------------------------


def test_regex_match_pass_from_expected():
    r = RegexMatchGrader().grade("the answer is 42", r"answer is \d+")
    assert r.passed is True
    assert r.detail["match"] == "answer is 42"


def test_regex_match_pass_from_config_pattern():
    r = RegexMatchGrader({"pattern": r"\bfoo\b"}).grade("foo bar", expected=None)
    assert r.passed is True


def test_regex_match_invalid_pattern():
    r = RegexMatchGrader({"pattern": "[unterminated"}).grade("foo", None)
    assert r.passed is False
    assert "invalid regex" in r.detail["error"]


# --------------------------------------------------------------------------
# OllamaJudge
# --------------------------------------------------------------------------


def test_ollama_judge_parses_judge_json():
    fake_client = MagicMock()
    fake_client.chat.return_value = {
        "message": {"content": '{"score": 87, "passed": true, "reasoning": "good"}'}
    }
    g = OllamaJudge({"model": "judge-model", "client": fake_client})
    case = TaskCase(case_id="q1", task_type="free_form", prompt="P")
    r = g.grade("candidate", "expected", case=case)
    assert r.score == 87.0
    assert r.passed is True
    assert r.detail["reasoning"] == "good"
    # Verify temperature option propagated.
    kwargs = fake_client.chat.call_args.kwargs
    assert kwargs["options"]["temperature"] == 0.0


def test_ollama_judge_unparseable_response():
    fake_client = MagicMock()
    fake_client.chat.return_value = {"message": {"content": "not json"}}
    g = OllamaJudge({"model": "m", "client": fake_client})
    r = g.grade("c", "e")
    assert r.passed is False
    assert r.score == 0.0
    assert "unparseable" in r.detail["error"]


def test_ollama_judge_threshold_applied_when_passed_missing():
    fake_client = MagicMock()
    fake_client.chat.return_value = {
        "message": {"content": '{"score": 55, "reasoning": "ok"}'}
    }
    g = OllamaJudge({"model": "m", "client": fake_client, "pass_threshold": 60})
    r = g.grade("c", "e")
    assert r.score == 55.0
    assert r.passed is False


def test_ollama_judge_missing_model_returns_error():
    g = OllamaJudge({})
    r = g.grade("c", "e")
    assert r.passed is False
    assert "model" in r.detail["error"]


# --------------------------------------------------------------------------
# CompositeGrader
# --------------------------------------------------------------------------


def test_composite_grader_weighted_average():
    grader = CompositeGrader(
        {
            "components": [
                {"type": "exact_match", "weight": 0.5},
                {"type": "exact_match", "case_sensitive": False, "weight": 0.5},
            ],
            "pass_threshold": 60,
        }
    )
    # First component fails (case mismatch), second passes.
    r = grader.grade("HELLO", "hello")
    assert r.score == 50.0
    assert r.passed is False  # 50 < 60
    assert len(r.detail["components"]) == 2


def test_composite_grader_empty_components():
    r = CompositeGrader({"components": []}).grade("a", "a")
    assert r.passed is False
    assert "no components" in r.detail["error"]


# --------------------------------------------------------------------------
# build_grader dispatch
# --------------------------------------------------------------------------


def test_build_grader_dispatches_known_types():
    assert isinstance(build_grader(GraderSpec(type="letter_match")), LetterMatchGrader)
    assert isinstance(build_grader(GraderSpec(type="exact_match")), ExactMatchGrader)
    assert isinstance(
        build_grader(GraderSpec(type="regex_match", config={"pattern": "."})),
        RegexMatchGrader,
    )
    assert isinstance(
        build_grader(GraderSpec(type="llm_judge", config={"model": "m"})),
        OllamaJudge,
    )
    assert isinstance(
        build_grader(GraderSpec(type="composite", config={"components": []})),
        CompositeGrader,
    )


def test_build_grader_rejects_unknown():
    import pytest

    with pytest.raises(ValueError, match="Unknown grader type"):
        build_grader(GraderSpec(type="nope"))
