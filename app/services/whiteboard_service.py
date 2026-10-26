import json
from pathlib import Path

import cv2
import pytesseract

from app.config import FRAME_DIR, WHITEBOARD_DIR


def extract_whiteboards(
    lecture_id: str,
    video_path: str,
    interval: int = 5,
    min_text_chars: int = 3,
) -> list[dict]:
    """Sample lecture frames, run OCR, and persist timestamped results."""

    if (
        not lecture_id
        or Path(lecture_id).name != lecture_id
        or lecture_id in {".", ".."}
        or "/" in lecture_id
        or "\\" in lecture_id
    ):
        raise ValueError("Invalid lecture ID")

    video = Path(video_path).resolve()

    if not video.is_file():
        raise FileNotFoundError(f"Video not found: {video}")

    if interval < 1:
        raise ValueError("Frame sampling interval must be at least 1 second")

    if min_text_chars < 1:
        raise ValueError("min_text_chars must be positive")

    frame_dir = FRAME_DIR / lecture_id
    output_dir = WHITEBOARD_DIR / lecture_id
    frame_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    capture = cv2.VideoCapture(str(video))

    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video}")

    fps = capture.get(cv2.CAP_PROP_FPS)
    frame_count = capture.get(cv2.CAP_PROP_FRAME_COUNT)

    if not fps or fps <= 0:
        capture.release()
        raise RuntimeError("Could not determine the video's frame rate")

    duration = frame_count / fps if frame_count else 0
    records = []

    try:
        for second in range(0, int(duration) + 1, interval):
            capture.set(cv2.CAP_PROP_POS_MSEC, second * 1000)
            ok, frame = capture.read()

            if not ok:
                continue

            frame_path = frame_dir / f"frame_{second:06d}.jpg"

            if not cv2.imwrite(str(frame_path), frame):
                continue

            # Improve OCR readability by converting to grayscale and
            # applying thresholding. The original frame is retained.
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            processed = cv2.threshold(
                gray, 0, 255,
                cv2.THRESH_BINARY + cv2.THRESH_OTSU,
            )[1]

            text = pytesseract.image_to_string(
                processed,
                config="--psm 11",
            ).strip()

            cleaned_text = "\n".join(
                line.strip()
                for line in text.splitlines()
                if line.strip()
            )

            records.append({
                "timestamp_seconds": second,
                "timestamp_label": (
                    f"{second // 60:02d}:{second % 60:02d}"
                ),
                "frame_path": str(frame_path),
                "ocr_text": cleaned_text,
                "has_readable_text": len(cleaned_text) >= min_text_chars,
            })
    finally:
        capture.release()

    output_path = output_dir / "whiteboards.json"
    output_path.write_text(
        json.dumps(records, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    readable = sum(item["has_readable_text"] for item in records)

    print(
        f"Sampled {len(records)} frames; "
        f"{readable} frames contain potentially readable text."
    )
    print(f"Saved OCR results to: {output_path}")

    return records
