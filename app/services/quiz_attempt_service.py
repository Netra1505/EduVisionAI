from app.database import get_lecture
from app.config import QUIZ_DIR
from app.services.quiz_service import evaluate_answer, save_attempt
from app.services.analytics_service import get_learning_analytics


def submit_quiz(lecture_id: str, answers: dict) -> dict:
    """
    Evaluate submitted answers, save the attempt, and refresh analytics.

    answers maps question IDs to student-entered answers.
    """
    if not isinstance(answers, dict) or not answers:
        raise ValueError("Submit at least one answer.")

    lecture = get_lecture(lecture_id)
    if lecture is None:
        raise FileNotFoundError(f"Lecture not found: {lecture_id}")

    quiz_path = QUIZ_DIR / f"{lecture_id}.json"
    if not quiz_path.is_file():
        raise FileNotFoundError(
            "No quiz exists for this lecture. Generate one first."
        )

    import json

    with quiz_path.open("r", encoding="utf-8") as handle:
        questions = json.load(handle)

    if not isinstance(questions, list) or not questions:
        raise ValueError("The saved quiz is empty or invalid.")

    questions_by_id = {
        str(question["id"]): question
        for question in questions
        if question.get("id") is not None
    }

    results = []

    for question_id, submitted_answer in answers.items():
        question = questions_by_id.get(str(question_id))

        if question is None:
            raise ValueError(
                f"Question ID {question_id!r} does not belong to this quiz."
            )

        result = evaluate_answer(question, str(submitted_answer))
        result["question_id"] = str(question_id)
        results.append(result)

    attempt = save_attempt(lecture_id, results)
    analytics = get_learning_analytics(lecture_id)

    return {
        "lecture_id": lecture_id,
        "score": sum(bool(item["correct"]) for item in results),
        "answered": len(results),
        "accuracy": (
            round(
                sum(bool(item["correct"]) for item in results)
                / len(results),
                3,
            )
            if results else 0.0
        ),
        "attempt": attempt,
        "analytics": analytics,
    }
