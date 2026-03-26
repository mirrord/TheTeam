"""Tests for multiple choice question construction."""

import random

from benchmarks.easy_problems.multiple_choice import construct_multiple_choice_question


def _make_question():
    """Create a sample question dict matching the dataset format."""
    return {
        "question": "What color is the sky?",
        "correct_answer": "Blue",
        "multiple_choice": ["Blue", "Red", "Green", "Yellow"],
    }


class TestConstructMultipleChoiceQuestion:
    def test_returns_prompt_and_correct_letter(self):
        """Returns a (messages, correct_letter) tuple."""
        random.seed(42)
        q = _make_question()
        messages, correct_letter = construct_multiple_choice_question(q)
        assert isinstance(messages, list)
        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        assert correct_letter in ["A", "B", "C", "D"]

    def test_prompt_contains_question(self):
        """The prompt text includes the original question."""
        random.seed(42)
        q = _make_question()
        messages, _ = construct_multiple_choice_question(q)
        assert "What color is the sky?" in messages[0]["content"]

    def test_prompt_contains_all_choices(self):
        """All answer options appear in the prompt."""
        random.seed(42)
        q = _make_question()
        messages, _ = construct_multiple_choice_question(q)
        content = messages[0]["content"]
        for choice in ["Blue", "Red", "Green", "Yellow"]:
            assert choice in content

    def test_correct_letter_matches_correct_answer(self):
        """The returned letter maps to the correct answer in the shuffled list."""
        random.seed(42)
        q = _make_question()
        messages, correct_letter = construct_multiple_choice_question(q)
        content = messages[0]["content"]
        # Find the line with the correct letter
        for line in content.split("\n"):
            if line.startswith(f"{correct_letter}. "):
                assert "Blue" in line
                break
        else:
            raise AssertionError(f"Letter {correct_letter} not found in prompt")

    def test_shuffles_choices(self):
        """Choices are shuffled (not always in original order)."""
        seen_orders = set()
        for seed in range(20):
            random.seed(seed)
            q = _make_question()
            _, letter = construct_multiple_choice_question(q)
            seen_orders.add(letter)
        # With 20 seeds over 4 options, we should see at least 2 different positions
        assert len(seen_orders) >= 2

    def test_prompt_includes_json_format_example(self):
        """The prompt contains a JSON format instruction."""
        random.seed(42)
        q = _make_question()
        messages, _ = construct_multiple_choice_question(q)
        assert '"ANSWER"' in messages[0]["content"]

    def test_all_four_letters_present(self):
        """Options A through D all appear in the prompt."""
        random.seed(42)
        q = _make_question()
        messages, _ = construct_multiple_choice_question(q)
        content = messages[0]["content"]
        for letter in ["A.", "B.", "C.", "D."]:
            assert letter in content
