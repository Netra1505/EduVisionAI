import json
import subprocess
import time
from pathlib import Path

import cv2
import faiss
import numpy as np
import pytesseract
from faster_whisper import WhisperModel
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parents[1]
VIDEO = ROOT / "data/uploads/a49ef79fd3d9.mp4"
AUDIO = ROOT / "data/audio/a49ef79fd3d9.wav"
TRANSCRIPT = ROOT / "data/transcripts/a49ef79fd3d9.json"
INDEX = ROOT / "data/indexes/a49ef79fd3d9.faiss"
DOCUMENTS = ROOT / "data/indexes/a49ef79fd3d9_documents.json"
FRAMES = ROOT / "data/frames/a49ef79fd3d9"

def main():
    if not VIDEO.is_file():
        raise FileNotFoundError(f"Lecture not found: {VIDEO}")

    print("\n[1/5] Extracting audio with FFmpeg...")
    result = subprocess.run([
        "ffmpeg", "-y", "-i", str(VIDEO),
        "-vn", "-ac", "1", "-ar", "16000",
        "-c:a", "pcm_s16le", str(AUDIO)
    ], capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(result.stderr[-2000:])
    print(f"PASS: Audio saved ({AUDIO.stat().st_size:,} bytes)")

    print("\n[2/5] Transcribing lecture with Whisper...")
    model = WhisperModel("small", device="cpu", compute_type="int8")
    segments, language = model.transcribe(
        str(AUDIO), beam_size=5, vad_filter=True
    )
    records = []
    for seg in segments:
        text = seg.text.strip()
        if text:
            records.append({
                "start": round(float(seg.start), 2),
                "end": round(float(seg.end), 2),
                "text": text,
                "speaker": "Unknown",
            })
            print(
                f"  [{seg.start:6.1f}s - {seg.end:6.1f}s] {text}"
            )

    if not records:
        raise RuntimeError("No speech detected in the lecture.")

    TRANSCRIPT.write_text(
        json.dumps(records, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"PASS: {len(records)} segments; language={language}")

    print("\n[3/5] Building FAISS semantic search index...")
    embedder = SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2"
    )
    vectors = embedder.encode(
        [r["text"] for r in records],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    ).astype("float32")

    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    faiss.write_index(index, str(INDEX))
    DOCUMENTS.write_text(
        json.dumps(records, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    query = "What are the main topics explained in this lecture?"
    query_vector = embedder.encode(
        [query], normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype("float32")
    scores, ids = index.search(query_vector, min(3, len(records)))

    print(f"PASS: Indexed {index.ntotal} transcript segments")
    print("Sample semantic search:")
    for score, idx in zip(scores[0], ids[0]):
        if idx >= 0:
            print(
                f"  score={score:.3f}, "
                f"time={records[int(idx)]['start']:.1f}s: "
                f"{records[int(idx)]['text']}"
            )

    print("\n[4/5] Sampling video frames and testing OCR...")
    FRAMES.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(VIDEO))
    if not cap.isOpened():
        raise RuntimeError("OpenCV could not open the lecture video.")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    duration = count / fps if count else 0
    sampled = 0

    for seconds in range(0, int(duration) + 1, 5):
        cap.set(cv2.CAP_PROP_POS_MSEC, seconds * 1000)
        ok, frame = cap.read()
        if not ok:
            continue

        path = FRAMES / f"frame_{seconds:06d}.jpg"
        if cv2.imwrite(str(path), frame):
            sampled += 1
        if seconds == 0:
            text = pytesseract.image_to_string(frame).strip()
            print("First-frame OCR:", text[:300] or "(no readable text)")

    cap.release()
    print(f"PASS: Sampled {sampled} frames; duration={duration:.1f}s")

    print("\n[5/5] Saving processing report...")
    report = {
        "video": VIDEO.name,
        "language": str(language),
        "duration_seconds": round(duration, 2),
        "transcript_segments": len(records),
        "faiss_vectors": index.ntotal,
        "embedding_dimensions": vectors.shape[1],
        "sampled_frames": sampled,
        "speaker_labels": "Unknown; diarization not run",
        "completed": True,
    }
    report_path = ROOT / "data/reports/preflight_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(json.dumps(report, indent=2))
    print("\nSUCCESS: Terminal pipeline test completed.")

if __name__ == "__main__":
    main()
