
import subprocess
from pathlib import Path


def extract_audio(video_path: str, audio_path: str) -> str:
    video = Path(video_path).resolve()
    audio = Path(audio_path).resolve()

    if not video.is_file():
        raise FileNotFoundError(f"Video not found: {video}")

    audio.parent.mkdir(parents=True, exist_ok=True)

    command = [
        "ffmpeg",
        "-y",
        "-i", str(video),
        "-vn",
        "-ac", "1",
        "-ar", "16000",
        "-c:a", "pcm_s16le",
        str(audio),
    ]

    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        timeout=600,
    )

    if result.returncode != 0 or not audio.exists():
        raise RuntimeError(
            "Audio extraction failed: " + result.stderr[-2000:]
        )

    return str(audio)
