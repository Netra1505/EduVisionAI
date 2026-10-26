from pathlib import Path

from app.database import get_lecture, update_lecture
from app.services.transcription_service import transcribe_lecture
from app.services.search_service import build_search_index
from app.services.whiteboard_service import extract_whiteboards
from app.services.concept_service import build_concept_graph
from app.services.quiz_service import generate_quiz
from app.services.diarization_service import diarize_audio, assign_speakers
from app.config import TRANSCRIPT_DIR


def process_lecture(lecture_id: str) -> dict:
    """
    Process an ingested lecture through transcription and learning features.

    A lecture is marked ready only when the required stages succeed.
    Optional diarization and OCR failures are reported separately.
    """
    metadata = get_lecture(lecture_id)

    if metadata is None:
        raise FileNotFoundError(f"Lecture not found: {lecture_id}")

    update_lecture(
        lecture_id,
        status="processing",
        pipeline_results={},
        pipeline_errors=[],
    )

    results = {}
    errors = []

    def run_stage(name, operation, required=True):
        try:
            value = operation()
            results[name] = {
                "status": "completed",
                "details": value,
            }
            return value
        except Exception as error:
            message = f"{name}: {type(error).__name__}: {error}"
            results[name] = {
                "status": "failed",
                "error": message,
                "required": required,
            }
            errors.append(message)
            return None

    # 1. Transcription
    transcription = run_stage(
        "transcription",
        lambda: transcribe_lecture(lecture_id),
    )

    if transcription is None:
        update_lecture(
            lecture_id,
            status="failed",
            pipeline_results=results,
            pipeline_errors=errors,
        )
        return {
            "lecture_id": lecture_id,
            "status": "failed",
            "results": results,
            "errors": errors,
        }

    transcript = transcription["segments"]

    # 2. Speaker diarization (optional)
    def diarization_stage():
        audio_path = metadata.get("audio_path")
        if not audio_path or not Path(audio_path).is_file():
            return {"status": "unavailable", "reason": "Audio unavailable"}

        diarization = diarize_audio(audio_path)

        if diarization["status"] == "completed":
            updated = assign_speakers(transcript, diarization["segments"])
            transcript_path = TRANSCRIPT_DIR / f"{lecture_id}.json"
            import json
            transcript_path.write_text(
                json.dumps(updated, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            transcript[:] = updated

        return {
            "status": diarization["status"],
            "reason": diarization.get("reason"),
            "speaker_count": len(diarization.get("speakers", [])),
        }

    run_stage("diarization", diarization_stage, required=False)

    # 3. Semantic search
    run_stage(
        "semantic_search",
        lambda: {
            "index": build_search_index(lecture_id, transcript).get("index_path"),
            "documents": len(transcript),
        },
    )

    # 4. Whiteboard frame extraction and OCR (optional)
    def ocr_stage():
        video_path = metadata.get("video_path")
        if not video_path or not Path(video_path).is_file():
            return {"status": "unavailable", "reason": "Video unavailable"}

        records = extract_whiteboards(lecture_id, video_path)
        return {
            "frames_examined": len(records),
            "frames_with_text": sum(
                bool(record.get("has_readable_text")) for record in records
            ),
        }

    run_stage("whiteboard_ocr", ocr_stage, required=False)

    # 5. Concept graph
    run_stage(
        "concept_graph",
        lambda: {
            "nodes": len(
                build_concept_graph(lecture_id, transcript).get("nodes", [])
            ),
        },
    )

    # 6. Dynamic quiz
    run_stage(
        "quiz",
        lambda: {
            "questions": len(generate_quiz(lecture_id, transcript, count=10)),
        },
    )

    required_stages = [
        "transcription",
        "semantic_search",
        "concept_graph",
        "quiz",
    ]

    ready = all(
        results.get(stage, {}).get("status") == "completed"
        for stage in required_stages
    )

    final_status = "ready" if ready else "needs_attention"

    update_lecture(
        lecture_id,
        status=final_status,
        pipeline_results=results,
        pipeline_errors=errors,
    )

    return {
        "lecture_id": lecture_id,
        "status": final_status,
        "results": results,
        "errors": errors,
    }
