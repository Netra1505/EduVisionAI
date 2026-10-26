import json
import re
import uuid
from pathlib import Path

from app.config import QUIZ_DIR, ATTEMPT_DIR


STOPWORDS = {
    "the", "and", "for", "are", "was", "were", "this", "that",
    "with", "from", "have", "has", "had", "you", "your", "its",
    "they", "their", "them", "then", "than", "into", "about",
    "what", "when", "where", "which", "while", "will", "would",
    "could", "should", "there", "here", "also", "used", "using",
    "use", "can", "not", "but", "all", "any", "how", "does",
}


def _validate_id(lecture_id: str):
    if (
        not lecture_id
        or Path(lecture_id).name != lecture_id
        or lecture_id in {".", ".."}
        or "/" in lecture_id
        or "\\" in lecture_id
    ):
        raise ValueError("Invalid lecture ID")


def generate_quiz(
    lecture_id: str,
    transcript: list[dict],
    count: int = 10,
) -> list[dict]:
    """Generate quiz questions using real transcript sentences."""
    _validate_id(lecture_id)

    if count < 1:
        raise ValueError("count must be positive")

    questions = []
    seen = set()

    for segment in transcript:
        sentence = re.sub(
            r"\s+", " ", str(segment.get("text", ""))
        ).strip()

        if len(sentence.split()) < 5:
            continue

        candidates = re.findall(
            r"\b[a-zA-Z][a-zA-Z0-9_-]{2,}\b",
            sentence,
        )

        candidates = [
            word for word in candidates
            if word.lower() not in STOPWORDS
        ]

        if not candidates:
            continue

        # Prefer a longer meaningful word to create a useful blank.
        answer = max(candidates, key=len)
        normalized = sentence.lower()

        if normalized in seen:
            continue
        seen.add(normalized)

        question_text = re.sub(
            rf"\b{re.escape(answer)}\b",
            "________",
            sentence,
            count=1,
            flags=re.IGNORECASE,
        )

        questions.append({
            "id": uuid.uuid4().hex[:12],
            "question": f"Complete the sentence: {question_text}",
            "answer": answer,
            "concept": answer.lower(),
            "source_start": float(segment.get("start", 0)),
            "source_end": float(segment.get("end", 0)),
            "type": "fill_in_the_blank",
        })

        if len(questions) >= count:
            break

    output_path = QUIZ_DIR / f"{lecture_id}.json"
    output_path.write_text(
        json.dumps(questions, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    return questions


def evaluate_answer(question: dict, user_answer: str) -> dict:
    """Evaluate a short answer using normalized exact matching."""
    expected = re.sub(
        r"\s+", " ", str(question.get("answer", ""))
    ).strip().casefold()

    submitted = re.sub(
        r"\s+", " ", str(user_answer or "")
    ).strip().casefold()

    correct = bool(expected) and submitted == expected

    return {
        "correct": correct,
        "expected_answer": question.get("answer", ""),
        "submitted_answer": user_answer,
        "concept": question.get("concept", "unknown"),
        "source_start": question.get("source_start", 0),
    }


def save_attempt(lecture_id: str, results: list[dict]) -> Path:
    """Append a quiz attempt to persistent per-lecture history."""
    _validate_id(lecture_id)

    path = ATTEMPT_DIR / f"{lecture_id}.json"

    if path.exists():
        attempts = json.loads(path.read_text(encoding="utf-8"))
    else:
        attempts = []

    attempts.append({
        "attempt_id": uuid.uuid4().hex,
        "results": results,
        "correct": sum(bool(r.get("correct")) for r in results),
        "total": len(results),
    })

    path.write_text(
        json.dumps(attempts, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return path


def get_weak_concepts(lecture_id: str, threshold: float = 0.7) -> list[dict]:
    """Identify concepts with low accuracy from recorded quiz attempts."""
    _validate_id(lecture_id)

    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be between 0 and 1")

    path = ATTEMPT_DIR / f"{lecture_id}.json"

    if not path.exists():
        return []

    attempts = json.loads(path.read_text(encoding="utf-8"))
    stats = {}

    for attempt in attempts:
        for result in attempt.get("results", []):
            concept = str(result.get("concept", "unknown")).lower()
            record = stats.setdefault(
                concept, {"correct": 0, "total": 0}
            )
            record["total"] += 1
            record["correct"] += int(bool(result.get("correct")))

    weak = []

    for concept, record in stats.items():
        accuracy = record["correct"] / record["total"]

        if accuracy < threshold:
            weak.append({
                "concept": concept,
                "accuracy": round(accuracy, 3),
                "correct": record["correct"],
                "total": record["total"],
                "recommendation": "Needs revision",
            })

    return sorted(weak, key=lambda item: item["accuracy"])
