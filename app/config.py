from pathlib import Path
import os

# Project root: the directory containing this app/ folder.
ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"

UPLOAD_DIR = DATA_DIR / "uploads"
AUDIO_DIR = DATA_DIR / "audio"
TRANSCRIPT_DIR = DATA_DIR / "transcripts"
INDEX_DIR = DATA_DIR / "indexes"
FRAME_DIR = DATA_DIR / "frames"
WHITEBOARD_DIR = DATA_DIR / "whiteboards"
QUIZ_DIR = DATA_DIR / "quizzes"
REPORT_DIR = DATA_DIR / "reports"
ATTEMPT_DIR = DATA_DIR / "attempts"

WHISPER_MODEL = os.getenv("WHISPER_MODEL", "small")
EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "sentence-transformers/all-MiniLM-L6-v2",
)
DEVICE = os.getenv("EDUVISION_DEVICE", "cpu")

for directory in (
    UPLOAD_DIR,
    AUDIO_DIR,
    TRANSCRIPT_DIR,
    INDEX_DIR,
    FRAME_DIR,
    WHITEBOARD_DIR,
    QUIZ_DIR,
    REPORT_DIR,
    ATTEMPT_DIR,
):
    directory.mkdir(parents=True, exist_ok=True)
