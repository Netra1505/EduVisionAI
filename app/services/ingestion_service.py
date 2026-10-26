import json
import shutil
import subprocess
import uuid
from pathlib import Path

from app.config import UPLOAD_DIR, AUDIO_DIR
from app.database import save_lecture


SUPPORTED_EXTENSIONS = {
    ".mp4", ".mov", ".mkv", ".avi", ".webm", ".mpeg", ".mpg"
}

MAX_UPLOAD_BYTES = 2 * 1024 * 1024 * 1024  # 2 GB


def ingest_video(source_path: str) -> dict:
    """
    Register a video, validate that FFmpeg can read it, extract WAV audio,
    and persist metadata. Accepts a local file path.
    """
    source = Path(source_path).expanduser().resolve()

    if not source.is_file():
        raise FileNotFoundError(f"Video file not found: {source}")

    if source.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported extension {source.suffix!r}. "
            f"Supported extensions: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )

    if source.stat().st_size == 0:
        raise ValueError("The selected video file is empty.")

    if source.stat().st_size > MAX_UPLOAD_BYTES:
        raise ValueError("Video exceeds the 2 GB ingestion limit.")

    ffmpeg = shutil.which("ffmpeg")
    ffprobe = shutil.which("ffprobe")

    if not ffmpeg or not ffprobe:
        raise RuntimeError(
            "FFmpeg and FFprobe are required. Install them with: "
            "sudo apt update && sudo apt install ffmpeg"
        )

    # Validate the container and obtain its duration.
    probe = subprocess.run(
        [
            ffprobe, "-v", "error",
            "-show_entries", "format=duration,format_name",
            "-of", "json",
            str(source),
        ],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    if probe.returncode != 0:
        raise ValueError(
            "FFprobe could not read this file: "
            + probe.stderr[-1000:]
        )

    try:
        media_info = json.loads(probe.stdout)
        duration = float(media_info["format"]["duration"])
    except (ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
        raise ValueError("Could not determine video duration.") from error

    if duration <= 0:
        raise ValueError("Video duration must be greater than zero.")

    lecture_id = uuid.uuid4().hex[:12]
    stored_video = UPLOAD_DIR / f"{lecture_id}{source.suffix.lower()}"
    stored_audio = AUDIO_DIR / f"{lecture_id}.wav"

    # Copy the original into managed storage.
    shutil.copy2(source, stored_video)

    try:
        # Extract mono 16 kHz WAV, suitable for speech-recognition models.
        extraction = subprocess.run(
            [
                ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "error",
                "-y", "-i", str(stored_video),
                "-vn", "-ac", "1", "-ar", "16000",
                "-c:a", "pcm_s16le",
                str(stored_audio),
            ],
            capture_output=True,
            text=True,
            timeout=max(120, int(duration * 3)),
            check=False,
        )

        if extraction.returncode != 0:
            raise ValueError(
                "Audio extraction failed. The video may contain no "
                "usable audio stream. FFmpeg: "
                + extraction.stderr[-1500:]
            )

        if not stored_audio.is_file() or stored_audio.stat().st_size == 0:
            raise ValueError("FFmpeg produced no usable audio file.")

        metadata = {
            "lecture_id": lecture_id,
            "original_filename": source.name,
            "video_path": str(stored_video),
            "audio_path": str(stored_audio),
            "duration_seconds": round(duration, 2),
            "format_name": media_info["format"].get("format_name"),
            "status": "ingested",
        }

        save_lecture(metadata)
        return metadata

    except Exception:
        stored_audio.unlink(missing_ok=True)
        stored_video.unlink(missing_ok=True)
        raise
