import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from app.config import UPLOAD_DIR


def _metadata_path(lecture_id: str) -> Path:
    # Prevent lecture IDs from escaping the uploads directory.
    if (
        not lecture_id
        or Path(lecture_id).name != lecture_id
        or lecture_id in {".", ".."}
        or "/" in lecture_id
        or "\\" in lecture_id
    ):
        raise ValueError("Invalid lecture ID")

    return UPLOAD_DIR / f"{lecture_id}.json"


def save_lecture(metadata: dict) -> dict:
    if not isinstance(metadata, dict):
        raise TypeError("Lecture metadata must be a dictionary")

    lecture_id = metadata.get("lecture_id")
    path = _metadata_path(lecture_id)

    record = dict(metadata)
    record["updated_at"] = datetime.now(timezone.utc).isoformat()

    # Write a temporary file first to avoid partially written JSON.
    fd, temp_name = tempfile.mkstemp(
        prefix=f".{lecture_id}_",
        suffix=".tmp",
        dir=str(UPLOAD_DIR),
    )

    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(record, handle, indent=2, ensure_ascii=False)
            handle.write("\n")

        os.replace(temp_name, path)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)

    return record


def get_lecture(lecture_id: str) -> dict | None:
    path = _metadata_path(lecture_id)

    if not path.is_file():
        return None

    with path.open("r", encoding="utf-8") as handle:
        record = json.load(handle)

    if not isinstance(record, dict):
        raise ValueError(f"Invalid lecture metadata: {path}")

    return record


def list_lectures() -> list[dict]:
    records = []

    for path in UPLOAD_DIR.glob("*.json"):
        try:
            with path.open("r", encoding="utf-8") as handle:
                record = json.load(handle)

            if isinstance(record, dict) and record.get("lecture_id"):
                records.append(record)
        except (OSError, json.JSONDecodeError):
            # Ignore unreadable metadata rather than stopping the listing.
            continue

    return sorted(
        records,
        key=lambda item: item.get("updated_at", ""),
        reverse=True,
    )


def update_lecture(lecture_id: str, **changes) -> dict:
    record = get_lecture(lecture_id)

    if record is None:
        raise FileNotFoundError(f"Lecture not found: {lecture_id}")

    # Keep the record's identity stable.
    if (
        "lecture_id" in changes
        and changes["lecture_id"] != lecture_id
    ):
        raise ValueError("Cannot change a lecture's ID")

    record.update(changes)
    return save_lecture(record)
