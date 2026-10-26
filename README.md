# EduVision AI

An AI-assisted lecture processing and learning application.

## Features
- Video upload and persistent lecture records
- FFmpeg audio extraction
- Whisper transcription with timestamps
- Optional pyannote speaker diarization
- Per-lecture FAISS semantic search
- Timestamp-based search results
- Video frame sampling and OCR
- Transcript-derived concept graph
- Transcript-derived quiz generation
- Quiz-attempt-based weak-concept estimates
- Python code editor with subprocess timeout
- Processing status and local JSON reports

## Requirements
- Python 3.11
- FFmpeg
- Tesseract OCR
- Dependencies in requirements.txt

## Run
Activate the virtual environment and run:

    streamlit run app/main.py

## Optional speaker diarization
Install pyannote.audio, configure the required Hugging Face access,
and set HF_TOKEN in your environment. Without it, speaker labels remain
Unknown; they are not guessed.

## Limitations
- OCR can fail on handwriting, diagrams, and equations.
- Concept relationships are based on transcript word co-occurrence.
- Generated quiz questions require human review.
- Search relevance and timestamp correctness must be evaluated.
- The local code runner is not a secure sandbox and must not be exposed
  to untrusted users.
