import argparse
import json
import sys

from app.services.ingestion_service import ingest_video
from app.services.pipeline_service import process_lecture


def main():
    parser = argparse.ArgumentParser(
        description="Process a video with EduVision AI."
    )
    parser.add_argument("video", help="Path to the input video")
    args = parser.parse_args()

    try:
        lecture = ingest_video(args.video)
        print("Lecture ID:", lecture["lecture_id"])
        print("Video ingested. Starting processing...")

        result = process_lecture(lecture["lecture_id"])

        print("\n===== PROCESSING REPORT =====")
        print(json.dumps(result, indent=2, ensure_ascii=False))

        if result["status"] != "ready":
            sys.exit(1)

    except Exception as error:
        print(f"ERROR: {type(error).__name__}: {error}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
