from app.services.ingestion_service import ingest_video
from app.services.transcription_service import transcribe_lecture


def create_lecture_from_video(video_path: str) -> dict:
    """
    Ingest a video and transcribe it.

    Returns the lecture ID and transcription results.
    """
    metadata = ingest_video(video_path)
    lecture_id = metadata["lecture_id"]

    try:
        transcription = transcribe_lecture(lecture_id)
    except Exception:
        # Preserve the ingestion record and the transcription failure status.
        raise

    return {
        "lecture": metadata,
        "transcription": {
            "status": transcription["status"],
            "language": transcription["language"],
            "segment_count": transcription["segment_count"],
            "transcript_path": transcription["transcript_path"],
        },
    }
