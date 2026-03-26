"""Tests for benchmark evaluation utilities."""

import os
import tempfile

import pandas as pd

from benchmarks.easy_problems.eval_utils import (
    extract_valid_json,
    model_clean,
    score_and_save,
)


# --- extract_valid_json ---


class TestExtractValidJson:
    def test_simple_valid_json(self):
        """Extract a simple valid answer JSON."""
        result = extract_valid_json('{"ANSWER": "A"}')
        assert result == {"ANSWER": "A"}

    def test_valid_json_in_surrounding_text(self):
        """Extract JSON embedded in LLM prose."""
        text = 'I think the answer is B. Here: {"answer": "B"} done.'
        result = extract_valid_json(text)
        assert result["ANSWER"] == "B"

    def test_case_insensitive_key(self):
        """Keys are uppercased during extraction."""
        result = extract_valid_json('{"answer": "C"}')
        assert result["ANSWER"] == "C"

    def test_invalid_answer_letter(self):
        """Returns None if answer letter is not A-D."""
        assert extract_valid_json('{"ANSWER": "E"}') is None

    def test_no_json_present(self):
        """Returns None when no JSON-like structure exists."""
        assert extract_valid_json("The answer is definitely A.") is None

    def test_empty_string(self):
        """Returns None for empty input."""
        assert extract_valid_json("") is None

    def test_malformed_json(self):
        """Returns None for invalid JSON syntax."""
        assert extract_valid_json("{ANSWER: A}") is None

    def test_multiple_json_prefers_last(self):
        """When multiple JSON objects, prefers the last valid one."""
        text = '{"ANSWER": "A"} some text {"ANSWER": "D"}'
        result = extract_valid_json(text)
        assert result["ANSWER"] == "D"

    def test_json_with_newlines(self):
        """Handles JSON split across newlines."""
        text = '{\n"ANSWER": "B"\n}'
        result = extract_valid_json(text)
        assert result["ANSWER"] == "B"

    def test_non_string_answer_value(self):
        """Returns None when answer value is not a string."""
        assert extract_valid_json('{"ANSWER": 1}') is None

    def test_json_array_ignored(self):
        """JSON arrays are not valid answers."""
        assert extract_valid_json('[{"ANSWER": "A"}]') is None

    def test_json_without_answer_key(self):
        """Returns None when JSON has no ANSWER key."""
        assert extract_valid_json('{"result": "A"}') is None

    def test_extra_keys_preserved(self):
        """Extra keys in the JSON are kept."""
        result = extract_valid_json('{"answer": "A", "confidence": "high"}')
        assert result["ANSWER"] == "A"
        assert result["CONFIDENCE"] == "high"


# --- model_clean ---


class TestModelClean:
    def test_removes_path_prefix(self):
        assert model_clean("library/llama3") == "llama3"

    def test_replaces_dots_and_colons(self):
        assert model_clean("gemma3:27b") == "gemma3_27b"
        assert model_clean("model.v2.1") == "model_v2_1"

    def test_combined(self):
        assert model_clean("org/model.v1:latest") == "model_v1_latest"

    def test_already_clean(self):
        assert model_clean("simple_model") == "simple_model"


# --- score_and_save ---


class TestScoreAndSave:
    def _make_cfg(self, tmpdir):
        """Create a minimal config-like object for score_and_save."""

        class FakeSaveFolders:
            answers_save_path = tmpdir

        class FakeCfg:
            SaveFolders = FakeSaveFolders()
            models = {
                "test_model": {"model": "test", "service": "dummy"},
            }

        return FakeCfg()

    def _make_questions(self, model_name, answers):
        """Build questions list with pre-filled responses."""
        questions = []
        for correct, rsp_text, json_answer in answers:
            q = {
                "question": "What is X?",
                "correct_letter": correct,
                "responses": {
                    model_name: rsp_text,
                    f"{model_name}_choice": json_answer,
                },
            }
            questions.append(q)
        return questions

    def test_correct_scoring(self):
        """Correct answers get 100, incorrect get 0."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = self._make_cfg(tmpdir)
            questions = self._make_questions(
                "test_model",
                [
                    ("A", "answer A", {"ANSWER": "A"}),
                    ("B", "answer A", {"ANSWER": "A"}),
                ],
            )
            result = score_and_save(cfg, questions, round_num=1)
            df = result["test_model"]
            assert df.iloc[0]["score"] == 100
            assert df.iloc[1]["score"] == 0

    def test_none_answer_handled(self):
        """None answers (failed extraction) are scored as empty."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = self._make_cfg(tmpdir)
            questions = self._make_questions(
                "test_model",
                [
                    ("A", "gibberish", None),
                ],
            )
            result = score_and_save(cfg, questions, round_num=1)
            df = result["test_model"]
            assert df.iloc[0]["score"] == 0
            assert df.iloc[0]["invalid_answer_letter"] == 1

    def test_saves_json_file(self):
        """Results are persisted as JSONL files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = self._make_cfg(tmpdir)
            questions = self._make_questions(
                "test_model",
                [("A", "answer A", {"ANSWER": "A"})],
            )
            score_and_save(cfg, questions, round_num=1)
            output_path = os.path.join(
                tmpdir, "round_1", "final_answers-test_model.json"
            )
            assert os.path.exists(output_path)

    def test_omit_models(self):
        """Omitted models are not scored."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = self._make_cfg(tmpdir)
            questions = self._make_questions(
                "test_model",
                [("A", "answer A", {"ANSWER": "A"})],
            )
            result = score_and_save(
                cfg, questions, round_num=1, omit_models=["test_model"]
            )
            assert "test_model" not in result

    def test_models_override(self):
        """models_override replaces cfg.models for iteration."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = self._make_cfg(tmpdir)
            override = {"alt_model": {"model": "alt", "service": "dummy"}}
            questions = self._make_questions(
                "alt_model",
                [("C", "answer C", {"ANSWER": "C"})],
            )
            result = score_and_save(
                cfg, questions, round_num=1, models_override=override
            )
            assert "alt_model" in result
            assert result["alt_model"].iloc[0]["score"] == 100

    def test_invalid_answer_flagged(self):
        """Empty answer letters are flagged as invalid."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = self._make_cfg(tmpdir)
            questions = self._make_questions(
                "test_model",
                [("A", "dunno", {"ANSWER": ""})],
            )
            result = score_and_save(cfg, questions, round_num=1)
            assert result["test_model"].iloc[0]["invalid_answer_letter"] == 1
