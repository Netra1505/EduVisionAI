import json
from collections import defaultdict

from app.config import ATTEMPT_DIR


def get_learning_analytics(lecture_id: str) -> dict:
    """Summarize saved quiz attempts for one lecture."""
    if (
        not lecture_id
        or "/" in lecture_id
        or "\\" in lecture_id
        or lecture_id in {".", ".."}
    ):
        raise ValueError("Invalid lecture ID")

    path = ATTEMPT_DIR / f"{lecture_id}.json"

    if not path.is_file():
        return {
            "lecture_id": lecture_id,
            "attempt_count": 0,
            "total_questions_answered": 0,
            "overall_accuracy": 0.0,
            "concepts": [],
            "weak_concepts": [],
            "message": "No quiz attempts recorded yet.",
        }

    with path.open("r", encoding="utf-8") as handle:
        attempts = json.load(handle)

    if not isinstance(attempts, list):
        raise ValueError("Quiz attempts file must contain a list")

    concept_totals = defaultdict(lambda: {"correct": 0, "total": 0})
    total_correct = 0
    total_answered = 0

    for attempt in attempts:
        for result in attempt.get("results", []):
            concept = str(result.get("concept") or "general").strip().lower()
            is_correct = bool(result.get("correct", False))

            concept_totals[concept]["total"] += 1
            concept_totals[concept]["correct"] += int(is_correct)

            total_answered += 1
            total_correct += int(is_correct)

    concepts = []

    for concept, counts in concept_totals.items():
        accuracy = counts["correct"] / counts["total"]

        concepts.append({
            "concept": concept,
            "correct": counts["correct"],
            "total": counts["total"],
            "accuracy": round(accuracy, 3),
        })

    concepts.sort(key=lambda item: (item["accuracy"], item["concept"]))

    weak_concepts = [
        {
            **item,
            "recommendation": "Revise this concept and retry related questions.",
        }
        for item in concepts
        if item["accuracy"] < 0.7
    ]

    return {
        "lecture_id": lecture_id,
        "attempt_count": len(attempts),
        "total_questions_answered": total_answered,
        "overall_accuracy": round(
            total_correct / total_answered, 3
        ) if total_answered else 0.0,
        "concepts": concepts,
        "weak_concepts": weak_concepts,
        "message": (
            "Analytics calculated from saved attempts."
            if total_answered
            else "Attempts exist, but no question results are recorded."
        ),
    }
