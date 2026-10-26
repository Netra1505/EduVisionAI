import os
from pathlib import Path


def diarize_audio(audio_path: str) -> dict:
    """
    Attempt speaker diarization using pyannote.audio.

    Requires an authorized Hugging Face token and access to the
    selected diarization model. Returns an explicit status if unavailable.
    """
    audio = Path(audio_path)

    if not audio.is_file():
        raise FileNotFoundError(f"Audio not found: {audio}")

    token = os.getenv("HF_TOKEN")

    if not token:
        return {
            "status": "unavailable",
            "reason": "HF_TOKEN is not configured",
            "speakers": [],
            "segments": [],
        }

    try:
        from pyannote.audio import Pipeline
    except ImportError:
        return {
            "status": "unavailable",
            "reason": (
                "pyannote.audio is not installed. "
                "Install a compatible version before enabling diarization."
            ),
            "speakers": [],
            "segments": [],
        }

    try:
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-community-1",
            token=token,
        )

        if pipeline is None:
            raise RuntimeError("Diarization pipeline could not be loaded")

        result = pipeline(str(audio))
        annotation = getattr(result, "speaker_diarization", result)

        segments = []

        for turn, _, speaker in annotation.itertracks(yield_label=True):
            segments.append({
                "start": round(float(turn.start), 2),
                "end": round(float(turn.end), 2),
                "speaker": str(speaker),
            })

        return {
            "status": "completed",
            "reason": None,
            "speakers": sorted({
                segment["speaker"] for segment in segments
            }),
            "segments": segments,
        }

    except Exception as error:
        return {
            "status": "failed",
            "reason": f"{type(error).__name__}: {error}",
            "speakers": [],
            "segments": [],
        }


def assign_speakers(
    transcript: list[dict],
    diarization_segments: list[dict],
) -> list[dict]:
    """Assign the speaker with the greatest temporal overlap per segment."""
    output = []

    for item in transcript:
        record = dict(item)
        start = float(record.get("start", 0))
        end = float(record.get("end", start))

        overlaps = []

        for speaker_segment in diarization_segments:
            overlap = max(
                0.0,
                min(end, float(speaker_segment["end"]))
                - max(start, float(speaker_segment["start"])),
            )

            if overlap > 0:
                overlaps.append((overlap, speaker_segment["speaker"]))

        record["speaker"] = (
            max(overlaps, key=lambda item: item[0])[1]
            if overlaps else "Unknown"
        )
        output.append(record)

    return output
