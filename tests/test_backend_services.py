import pytest

from app.database import save_lecture, get_lecture
from app.services.code_service import validate_python
from app.services.quiz_service import evaluate_answer


def test_lecture_metadata_persists():
    lecture_id = "test_backend_metadata_123"

    save_lecture({
        "lecture_id": lecture_id,
        "status": "test",
    })

    result = get_lecture(lecture_id)

    assert result is not None
    assert result["lecture_id"] == lecture_id
    assert result["status"] == "test"


def test_invalid_lecture_id_is_rejected():
    with pytest.raises(ValueError):
        get_lecture("../invalid")


def test_python_syntax_validation():
    assert validate_python("print(1 + 1)")["valid"] is True
    assert validate_python("if True print('bad')")["valid"] is False


def test_quiz_answer_exact_match():
    question = {
        "answer": "performance",
        "concept": "performance",
        "source_start": 12.0,
    }

    result = evaluate_answer(question, "performance")

    assert result["correct"] is True


def test_quiz_answer_wrong():
    question = {
        "answer": "performance",
        "concept": "performance",
        "source_start": 12.0,
    }

    result = evaluate_answer(question, "temperature")

    assert result["correct"] is False
