import random


def construct_multiple_choice_question(question: dict) -> tuple[list[dict[str, str]], str]:
    letters = ["A", "B", "C", "D"]
    random.shuffle(question["multiple_choice"])
    correct_answer_idx = question["multiple_choice"].index(question["correct_answer"])
    correct_letter:str = letters[correct_answer_idx]
    answers = "\n".join(
        [
            f"{letter}. {question}"
            for letter, question in zip(letters, question["multiple_choice"])
        ]
    )
    random_letter = random.choice(letters)

    prompt = f"""QUESTION
{question["question"]}

ANSWERS
{answers}

Provide an explanation for your thinking and then select a single choice from ANSWERS that answer the QUESTION. Return in JSON format, for example:
{{"ANSWER": "{random_letter}"}}
"""

    return [{"role": "user", "content": prompt}], correct_letter
