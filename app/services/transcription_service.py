import json
import os
import tempfile
from pathlib import Path

from faster_whisper import WhisperModel

from app.config import TRANSCRIPT_DIR, WHISPER_MODEL
from app.database import get_lecture, update_lecture


def _load_whisper():
    """Load the configured Whisper model."""
    device = os.getenv("EDUVISION_DEVICE", "cpu")
    compute_type = "int8" if device == "cpu" else "float16"

    return WhisperModel(
        WHISPER_MODEL,
        device=device,
        compute_type=compute_type,
    )


def transcribe_lecture(
    lecture_id: str,
    model=None,
) -> dict:
    """
    Transcribe the audio extracted during ingestion.

    Saves timestamped segments to data/transcripts/<lecture_id>.json.
    """
    metadata = get_lecture(lecture_id)

    if metadata is None:
        raise FileNotFoundError(
            f"No lecture metadata found for {lecture_id}"
        )

    audio_path = Path(metadata.get("audio_path", ""))

    if not audio_path.is_file():
        raise FileNotFoundError(
            f"Extracted audio not found: {audio_path}"
        )

    update_lecture(
        lecture_id,
        status="transcribing",
        transcription_error=None,
    )

    try:
        whisper = model if model is not None else _load_whisper()

        segments, info = whisper.transcribe(
            str(audio_path),
            beam_size=5,
            vad_filter=True,
        )

        transcript = []

        # Consume the generator to finish inference.
        for segment in segments:
            text = segment.text.strip()

            if not text:
                continue

            transcript.append({
                "start": round(float(segment.start), 3),
                "end": round(float(segment.end), 3),
                "text": text,
                "speaker": "Unknown",
            })

        if not transcript:
            raise ValueError(
                "Whisper returned no speech segments. "
                "The video may contain silence or unclear audio."
            )

        output_path = TRANSCRIPT_DIR / f"{lecture_id}.json"

        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            suffix=".tmp",
            dir=TRANSCRIPT_DIR,
            delete=False,
        ) as handle:
            json.dump(
                transcript,
                handle,
                indent=2,
                ensure_ascii=False,
            )
            handle.write("\n")
            temporary_path = Path(handle.name)

        temporary_path.replace(output_path)

        update_lecture(
            lecture_id,
            status="transcribed",
            transcript_path=str(output_path),
            transcript_segments=len(transcript),
            detected_language=getattr(info, "language", None),
            transcription_error=None,
        )

        return {
            "lecture_id": lecture_id,
            "status": "transcribed",
            "language": getattr(info, "language", None),
            "segment_count": len(transcript),
            "transcript_path": str(output_path),
            "segments": transcript,
        }

    except Exception as error:
        update_lecture(
            lecture_id,
            status="transcription_failed",
            transcription_error=f"{type(error).__name__}: {error}",
        )
        raise
