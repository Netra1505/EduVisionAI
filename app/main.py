import os
import re
import sys
import json
import uuid
import time
import shutil
import subprocess
from pathlib import Path
from collections import Counter
from datetime import datetime

import cv2
import faiss
import networkx as nx
import numpy as np
import pandas as pd
import pytesseract
import streamlit as st
from sentence_transformers import SentenceTransformer
from faster_whisper import WhisperModel

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
DIRS = {
    "uploads": DATA / "uploads",
    "audio": DATA / "audio",
    "transcripts": DATA / "transcripts",
    "indexes": DATA / "indexes",
    "frames": DATA / "frames",
    "whiteboards": DATA / "whiteboards",
    "quizzes": DATA / "quizzes",
    "reports": DATA / "reports",
}
for folder in DIRS.values():
    folder.mkdir(parents=True, exist_ok=True)

st.set_page_config(
    page_title="EduVision AI",
    page_icon="🎓",
    layout="wide",
)

STOPWORDS = set("""
a an the and or but if then of to in on at by for from with is are was
were be been being it this that these those as into about than so such
we you they he she them their our your i can could should would will
may might have has had do does did not no yes lecture student professor
""".split())

@st.cache_resource
def get_whisper():
    # CPU mode works without an NVIDIA GPU.
    return WhisperModel(
        os.getenv("WHISPER_MODEL", "small"),
        device="cpu",
        compute_type="int8",
    )

@st.cache_resource
def get_embedder():
    return SentenceTransformer(
        os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    )

def read_json(path, default=None):
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default

def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

def timestamp(seconds):
    seconds = max(0, int(seconds))
    h, remainder = divmod(seconds, 3600)
    m, s = divmod(remainder, 60)
    return f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"

def lecture_path(lecture_id):
    return DIRS["uploads"] / f"{lecture_id}.json"

def get_lectures():
    records = []
    for path in DIRS["uploads"].glob("*.json"):
        item = read_json(path)
        if item and Path(item.get("video_path", "")).is_file():
            records.append(item)
    return sorted(records, key=lambda x: x.get("created_at", ""), reverse=True)

def load_transcript(lecture_id):
    return read_json(DIRS["transcripts"] / f"{lecture_id}.json", [])

def extract_audio(video_path, audio_path):
    command = [
        "ffmpeg", "-y", "-i", str(video_path),
        "-vn", "-ac", "1", "-ar", "16000",
        "-c:a", "pcm_s16le", str(audio_path),
    ]
    result = subprocess.run(
        command, capture_output=True, text=True, timeout=1800
    )
    if result.returncode:
        raise RuntimeError(result.stderr[-2500:])
    if not Path(audio_path).is_file():
        raise RuntimeError("FFmpeg did not create an audio file.")
    return str(audio_path)

def transcribe(audio_path):
    model = get_whisper()
    segments, info = model.transcribe(
        str(audio_path),
        beam_size=5,
        vad_filter=True,
    )
    output = []
    for segment in segments:
        text = segment.text.strip()
        if text:
            output.append({
                "start": round(float(segment.start), 2),
                "end": round(float(segment.end), 2),
                "text": text,
                "speaker": "Unknown",
            })
    if not output:
        raise RuntimeError(
            "No speech was detected. Check the audio and try another model."
        )
    return output, info.language

def diarize_if_configured(audio_path, transcript):
    """
    Optional real speaker diarization.
    Set HF_TOKEN and install pyannote.audio to enable it.
    Without those, speaker labels remain Unknown rather than being invented.
    """
    token = os.getenv("HF_TOKEN")
    if not token:
        return transcript, "Speaker diarization not configured; labels are Unknown."

    try:
        from pyannote.audio import Pipeline
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-community-1",
            token=token,
        )
        diarization = pipeline(str(audio_path))
        annotations = getattr(diarization, "speaker_diarization", diarization)
        turns = []
        for turn, _, speaker in annotations.itertracks(yield_label=True):
            turns.append((float(turn.start), float(turn.end), str(speaker)))

        for segment in transcript:
            midpoint = (segment["start"] + segment["end"]) / 2
            matches = [
                (end - start, speaker)
                for start, end, speaker in turns
                if start <= midpoint <= end
            ]
            if matches:
                segment["speaker"] = min(matches)[1]
        return transcript, "Speaker diarization completed."
    except Exception as exc:
        return transcript, f"Diarization unavailable: {exc}"

def split_text(text, max_words=90):
    words = text.split()
    chunks = []
    for start in range(0, len(words), max_words):
        chunk = " ".join(words[start:start + max_words]).strip()
        if chunk:
            chunks.append(chunk)
    return chunks

def build_search_index(lecture_id, transcript):
    # Store embeddings and the FAISS index independently for each lecture.
    documents = []
    for seg in transcript:
        documents.append({
            "start": seg["start"],
            "end": seg["end"],
            "text": seg["text"],
            "speaker": seg.get("speaker", "Unknown"),
        })

    if not documents:
        raise ValueError("Cannot index an empty transcript.")

    model = get_embedder()
    texts = [d["text"] for d in documents]
    vectors = model.encode(
        texts, convert_to_numpy=True, normalize_embeddings=True,
        show_progress_bar=False,
    ).astype("float32")
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)

    faiss.write_index(index, str(DIRS["indexes"] / f"{lecture_id}.faiss"))
    np.save(DIRS["indexes"] / f"{lecture_id}.npy", vectors)
    write_json(DIRS["indexes"] / f"{lecture_id}_documents.json", documents)
    return len(documents)

def search_lecture(lecture_id, query, k=5):
    index_path = DIRS["indexes"] / f"{lecture_id}.faiss"
    docs_path = DIRS["indexes"] / f"{lecture_id}_documents.json"
    index = faiss.read_index(str(index_path))
    documents = read_json(docs_path, [])
    vector = get_embedder().encode(
        [query], convert_to_numpy=True, normalize_embeddings=True
    ).astype("float32")
    scores, ids = index.search(vector, min(k, len(documents)))
    results = []
    for score, idx in zip(scores[0], ids[0]):
        if idx >= 0:
            result = dict(documents[int(idx)])
            result["score"] = round(float(score), 4)
            results.append(result)
    return results

def extract_whiteboards(lecture_id, video_path, interval=5):
    output_dir = DIRS["frames"] / lecture_id
    output_dir.mkdir(parents=True, exist_ok=True)
    board_dir = DIRS["whiteboards"] / lecture_id
    board_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError("OpenCV could not open the video.")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
    duration = frame_count / fps if fps else 0
    saved = []
    seconds = 0

    while seconds < duration:
        cap.set(cv2.CAP_PROP_POS_MSEC, seconds * 1000)
        ok, frame = cap.read()
        if not ok:
            seconds += interval
            continue

        # Resize large frames to keep processing and storage manageable.
        height, width = frame.shape[:2]
        if width > 1280:
            scale = 1280 / width
            frame = cv2.resize(frame, (1280, int(height * scale)))

        image_path = output_dir / f"frame_{seconds:06d}.jpg"
        cv2.imwrite(str(image_path), frame)

        # Basic OCR: actual extracted text, not fabricated equations.
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )[1]
        try:
            text = pytesseract.image_to_string(gray).strip()
        except Exception:
            text = ""

        record = {
            "timestamp": seconds,
            "image_path": str(image_path),
            "text": text,
        }
        if text:
            saved.append(record)
        seconds += interval

    cap.release()
    write_json(board_dir / "whiteboards.json", saved)
    return saved, duration

def extract_concepts(transcript):
    words = Counter()
    for seg in transcript:
        tokens = re.findall(r"[A-Za-z][A-Za-z0-9_-]{2,}", seg["text"].lower())
        words.update(w for w in tokens if w not in STOPWORDS)
    return words.most_common(20)

def make_concept_graph(lecture_id, transcript):
    concepts = extract_concepts(transcript)
    graph = nx.Graph()
    terms = [word for word, _ in concepts[:15]]
    for term, count in concepts[:15]:
        graph.add_node(term, frequency=count)

    # Link concepts that appear in the same transcript segment.
    for seg in transcript:
        tokens = set(
            re.findall(r"[A-Za-z][A-Za-z0-9_-]{2,}", seg["text"].lower())
        ) & set(terms)
        for a in tokens:
            for b in tokens:
                if a < b:
                    if graph.has_edge(a, b):
                        graph[a][b]["weight"] += 1
                    else:
                        graph.add_edge(a, b, weight=1)

    nodes = [
        {"id": n, "frequency": graph.nodes[n].get("frequency", 1)}
        for n in graph.nodes
    ]
    edges = [
        {"source": a, "target": b, "weight": graph[a][b]["weight"]}
        for a, b in graph.edges
    ]
    result = {"nodes": nodes, "edges": edges}
    write_json(DIRS["reports"] / f"{lecture_id}_concept_graph.json", result)
    return graph, result

def generate_quiz(lecture_id, transcript, count=10):
    # Make answerable questions from actual transcript sentences.
    # Each question has its source text and timestamp for auditability.
    candidates = []
    for seg in transcript:
        text = seg["text"].strip()
        words = re.findall(r"\b[A-Za-z][A-Za-z0-9_-]{3,}\b", text)
        if len(words) >= 5:
            candidates.append((seg, words))

    questions = []
    seen = set()
    for seg, words in candidates:
        candidates_for_blank = [
            w for w in words if w.lower() not in STOPWORDS
        ]
        if not candidates_for_blank:
            continue
        answer = max(candidates_for_blank, key=len)
        key = (seg["text"], answer.lower())
        if key in seen:
            continue
        seen.add(key)
        masked = re.sub(
            rf"\b{re.escape(answer)}\b", "________",
            seg["text"], count=1, flags=re.IGNORECASE
        )
        questions.append({
            "question": f"Complete the lecture statement: {masked}",
            "answer": answer,
            "start": seg["start"],
            "concept": answer.lower(),
            "source_text": seg["text"],
        })
        if len(questions) >= count:
            break

    write_json(DIRS["quizzes"] / f"{lecture_id}.json", questions)
    return questions

def load_attempts(lecture_id):
    return read_json(DIRS["quizzes"] / f"{lecture_id}_attempts.json", [])

def save_attempts(lecture_id, attempts):
    write_json(DIRS["quizzes"] / f"{lecture_id}_attempts.json", attempts)

def weak_concepts(lecture_id):
    attempts = load_attempts(lecture_id)
    scores = {}
    for item in attempts:
        concept = item.get("concept", "unknown")
        scores.setdefault(concept, []).append(bool(item.get("correct")))
    result = []
    for concept, outcomes in scores.items():
        accuracy = sum(outcomes) / len(outcomes)
        result.append({
            "concept": concept,
            "attempts": len(outcomes),
            "accuracy": round(accuracy * 100, 1),
            "status": "Needs revision" if accuracy < 0.7 else "On track",
        })
    return sorted(result, key=lambda x: x["accuracy"])

def run_code(code):
    # Local prototype only. This is not a secure sandbox for untrusted code.
    # Do not expose this endpoint publicly or execute arbitrary student code.
    if len(code) > 10000:
        return "", "Code exceeds the 10,000-character limit."
    tmp = DIRS["reports"] / f"code_{uuid.uuid4().hex}.py"
    try:
        tmp.write_text(code, encoding="utf-8")
        result = subprocess.run(
            [sys.executable, "-I", str(tmp)],
            capture_output=True, text=True, timeout=5,
            cwd=str(DIRS["reports"]),
            env={"PATH": os.environ.get("PATH", "")},
        )
        return result.stdout[-5000:], result.stderr[-5000:]
    except subprocess.TimeoutExpired:
        return "", "Execution timed out after 5 seconds."
    except Exception as exc:
        return "", str(exc)
    finally:
        tmp.unlink(missing_ok=True)

def process_lecture(lecture):
    lecture_id = lecture["lecture_id"]
    audio_path = DIRS["audio"] / f"{lecture_id}.wav"
    transcript_path = DIRS["transcripts"] / f"{lecture_id}.json"

    lecture["status"] = "extracting_audio"
    write_json(lecture_path(lecture_id), lecture)
    extract_audio(lecture["video_path"], audio_path)

    lecture["status"] = "transcribing"
    write_json(lecture_path(lecture_id), lecture)
    transcript, language = transcribe(audio_path)
    transcript, diarization_status = diarize_if_configured(
        audio_path, transcript
    )
    write_json(transcript_path, transcript)

    lecture["status"] = "indexing"
    write_json(lecture_path(lecture_id), lecture)
    build_search_index(lecture_id, transcript)

    lecture["status"] = "extracting_whiteboards"
    write_json(lecture_path(lecture_id), lecture)
    boards, duration = extract_whiteboards(
        lecture_id, lecture["video_path"]
    )
    graph, graph_data = make_concept_graph(lecture_id, transcript)
    quiz = generate_quiz(lecture_id, transcript)

    lecture.update({
        "status": "ready",
        "language": language,
        "duration_seconds": round(duration, 2),
        "transcript_segments": len(transcript),
        "whiteboard_items": len(boards),
        "concept_count": len(graph_data["nodes"]),
        "quiz_questions": len(quiz),
        "diarization_status": diarization_status,
        "processed_at": datetime.now().isoformat(timespec="seconds"),
    })
    write_json(lecture_path(lecture_id), lecture)
    return lecture

# ------------------------- User interface -------------------------

st.title("🎓 EduVision AI")
st.caption("Video-grounded lecture intelligence and personalized learning")

with st.sidebar:
    st.header("Project status")
    st.write("Transcription: faster-whisper")
    st.write("Semantic search: Sentence Transformers + FAISS")
    st.write("Frames and OCR: OpenCV + Tesseract")
    st.write("Concepts: NetworkX")
    st.write("Storage: local JSON, images, and FAISS indexes")
    st.warning(
        "Local prototype. Code execution is not isolated securely; "
        "only run code you trust."
    )

st.header("1. Upload a lecture")
upload = st.file_uploader(
    "Choose a lecture video",
    type=["mp4", "mov", "mkv", "webm"],
)
if upload is not None:
    st.video(upload)
    st.write(f"Size: {upload.size / (1024 * 1024):.2f} MB")
    if upload.size == 0:
        st.error("This file is empty.")
    elif upload.size > 500 * 1024 * 1024:
        st.error("Maximum upload size is 500 MB.")
    elif st.button("Save lecture", type="primary"):
        lecture_id = uuid.uuid4().hex[:12]
        extension = Path(upload.name).suffix.lower()
        video_path = DIRS["uploads"] / f"{lecture_id}{extension}"
        video_path.write_bytes(upload.getvalue())
        lecture = {
            "lecture_id": lecture_id,
            "original_filename": upload.name,
            "video_path": str(video_path),
            "status": "uploaded",
            "created_at": datetime.now().isoformat(timespec="seconds"),
        }
        write_json(lecture_path(lecture_id), lecture)
        st.session_state["lecture_id"] = lecture_id
        st.success(f"Saved lecture: {lecture_id}")
        st.rerun()

lectures = get_lectures()
st.divider()
st.header("2. Select and process a lecture")

if not lectures:
    st.info("Save a lecture above to begin.")
    st.stop()

labels = {
    f"{x['original_filename']} — {x['lecture_id']} [{x.get('status', 'uploaded')}]":
    x["lecture_id"] for x in lectures
}
label_list = list(labels.keys())
saved_id = st.session_state.get("lecture_id")
selected_index = next(
    (i for i, label in enumerate(label_list) if labels[label] == saved_id), 0
)
selected_label = st.selectbox("Saved lectures", label_list, index=selected_index)
lecture_id = labels[selected_label]
lecture = next(x for x in lectures if x["lecture_id"] == lecture_id)

col1, col2, col3 = st.columns(3)
col1.metric("Status", lecture.get("status", "uploaded"))
col2.metric("Transcript segments", lecture.get("transcript_segments", 0))
col3.metric("Whiteboard OCR items", lecture.get("whiteboard_items", 0))

st.video(lecture["video_path"])

if st.button("🚀 Process full lecture", type="primary"):
    try:
        with st.status("Processing the actual uploaded lecture...", expanded=True) as status:
            st.write("Extracting audio and transcribing speech...")
            st.write("Building the per-lecture FAISS index...")
            st.write("Sampling video frames and extracting visible text...")
            st.write("Building concept graph and quiz...")
            updated = process_lecture(lecture)
            status.update(label="Lecture processing completed", state="complete")
        st.success(
            f"Processed {updated['transcript_segments']} transcript segments."
        )
        st.rerun()
    except Exception as exc:
        lecture["status"] = "error"
        lecture["last_error"] = str(exc)
        write_json(lecture_path(lecture_id), lecture)
        st.error(f"Processing failed: {exc}")
        st.info(
            "The terminal may contain additional details. "
            "Fix the reported issue and retry."
        )

transcript = load_transcript(lecture_id)
transcript_ready = bool(transcript)
index_ready = (DIRS["indexes"] / f"{lecture_id}.faiss").exists()
board_records = read_json(
    DIRS["whiteboards"] / lecture_id / "whiteboards.json", []
)
quiz_questions = read_json(DIRS["quizzes"] / f"{lecture_id}.json", [])

st.divider()
st.header("3. EduVision AI workspace")
tab_transcript, tab_search, tab_board, tab_code, tab_graph, tab_quiz, tab_analytics = st.tabs([
    "Transcript", "FAISS Search", "Whiteboard AI", "Code Lab",
    "Concept Graph", "Quiz", "Learning Analytics"
])

with tab_transcript:
    st.subheader("Speaker-labelled timestamped transcript")
    if transcript_ready:
        query = st.text_input("Filter transcript", key="transcript_filter")
        speaker_names = sorted(set(
            seg.get("speaker", "Unknown") for seg in transcript
        ))
        speaker_filter = st.selectbox(
            "Speaker filter", ["All speakers"] + speaker_names
        )
        for seg in transcript:
            if query.lower() not in seg["text"].lower():
                continue
            if speaker_filter != "All speakers" and seg.get("speaker") != speaker_filter:
                continue
            st.markdown(
                f"**[{timestamp(seg['start'])} – {timestamp(seg['end'])}]** "
                f"**{seg.get('speaker', 'Unknown')}**"
            )
            st.write(seg["text"])
    else:
        st.info("Process this lecture to generate a real transcript.")

with tab_search:
    st.subheader("Timestamp-based semantic search")
    st.caption("Search uses embeddings and a lecture-specific FAISS index.")
    search_query = st.text_input(
        "Ask a question about this lecture",
        placeholder="What does the professor explain about ...?",
        key="semantic_query",
    )
    top_k = st.slider("Results", 1, 10, 5)
    if st.button("Search lecture"):
        if not index_ready:
            st.warning("Process the lecture first to build its index.")
        elif not search_query.strip():
            st.warning("Enter a question.")
        else:
            try:
                results = search_lecture(lecture_id, search_query, top_k)
                for result in results:
                    with st.container(border=True):
                        st.markdown(
                            f"**[{timestamp(result['start'])}]** "
                            f"Relevance score: `{result['score']:.3f}`"
                        )
                        st.write(result["text"])
                        st.caption(
                            f"Segment ends at {timestamp(result['end'])}"
                        )
            except Exception as exc:
                st.error(f"Search failed: {exc}")

    st.markdown("**Evaluation queries**")
    st.write(
        "Enter ten questions grounded in this lecture and verify each returned "
        "timestamp against the video. Search results are candidates, not "
        "guaranteed correct answers."
    )
    eval_text = st.text_area(
        "One question per line (up to 10)",
        placeholder="What is the main definition?\nWhich steps did the professor explain?",
    )
    if st.button("Run 10-query evaluation"):
        questions = [q.strip() for q in eval_text.splitlines() if q.strip()][:10]
        if not index_ready:
            st.warning("Process the lecture first.")
        elif not questions:
            st.warning("Enter at least one question.")
        else:
            rows = []
            for question in questions:
                results = search_lecture(lecture_id, question, 1)
                best = results[0] if results else {}
                rows.append({
                    "question": question,
                    "predicted_timestamp": timestamp(best.get("start", 0)),
                    "retrieved_text": best.get("text", ""),
                    "relevance_score": best.get("score", 0),
                    "human_verified_correct": None,
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True)
            st.download_button(
                "Download query evaluation CSV",
                pd.DataFrame(rows).to_csv(index=False),
                file_name=f"{lecture_id}_query_evaluation.csv",
                mime="text/csv",
            )

with tab_board:
    st.subheader("Extracted frames and equation/text OCR")
    st.caption(
        "Frames are sampled every 5 seconds. OCR reads visible printed or "
        "handwritten text imperfectly; mathematical notation may need correction."
    )
    if board_records:
        for item in board_records:
            with st.expander(
                f"Frame at {timestamp(item['timestamp'])} "
                f"— {'Text detected' if item.get('text') else 'No text detected'}"
            ):
                image_path = Path(item["image_path"])
                if image_path.is_file():
                    st.image(str(image_path), use_container_width=True)
                st.write(item.get("text") or "No readable text detected.")
    else:
        st.info("Process the lecture to sample frames and run OCR.")

with tab_code:
    st.subheader("Whiteboard-to-code lab")
    board_texts = [
        item.get("text", "") for item in board_records if item.get("text")
    ]
    if board_texts:
        selected_board = st.selectbox(
            "Choose extracted whiteboard text",
            range(len(board_texts)),
            format_func=lambda i: board_texts[i][:100],
        )
        source_text = board_texts[selected_board]
    else:
        source_text = ""

    st.write("Use extracted text as a starting point, then review the code.")
    default_code = (
        "# Starter example based on the lecture's whiteboard text\n"
        "# Verify the algorithm and fill in missing details before running.\n\n"
        f"# Whiteboard text: {source_text[:300]!r}\n\n"
        "def solve(values):\n"
        "    # TODO: implement the algorithm described in the lecture\n"
        "    return values\n\n"
        "if __name__ == '__main__':\n"
        "    print(solve([1, 2, 3]))\n"
    )
    code = st.text_area(
        "Python code editor",
        value=default_code,
        height=300,
        key=f"code_editor_{lecture_id}",
    )
    st.warning(
        "Prototype warning: execution uses a subprocess with a timeout, "
        "not an OS-level security sandbox. Run only code you trust."
    )
    if st.button("Run Python code"):
        stdout, stderr = run_code(code)
        if stdout:
            st.markdown("**Output**")
            st.code(stdout)
        if stderr:
            st.markdown("**Errors / diagnostics**")
            st.code(stderr)
        if not stdout and not stderr:
            st.info("Program completed with no printed output.")
        if stderr:
            st.caption(
                "Use the error message to correct your code, then run it again. "
                "Automatic AI repair is not enabled in this local version."
            )

with tab_graph:
    st.subheader("Lecture concept graph")
    graph_path = DIRS["reports"] / f"{lecture_id}_concept_graph.json"
    graph_data = read_json(graph_path, {"nodes": [], "edges": []})
    if graph_data["nodes"]:
        st.graphviz_chart(
            "graph EduVision { "
            + " ".join(
                f'"{n["id"]}" [label="{n["id"]}\\nfrequency {n["frequency"]}"];'
                for n in graph_data["nodes"]
            )
            + " "
            + " ".join(
                f'"{e["source"]}" -- "{e["target"]}" [label="{e["weight"]}"];'
                for e in graph_data["edges"]
            )
            + " }"
        )
        st.caption(
            "This first graph is based on frequent words co-occurring in transcript "
            "segments. It is a useful starting point, not a validated ontology."
        )
    else:
        st.info("Process the lecture to build its concept graph.")

with tab_quiz:
    st.subheader("Dynamic transcript-based quiz")
    st.caption(
        "Questions are generated from actual transcript sentences. "
        "Review the source before treating a generated question as authoritative."
    )
    if quiz_questions:
        attempts = load_attempts(lecture_id)
        with st.form(f"quiz_{lecture_id}"):
            answers = {}
            for i, question in enumerate(quiz_questions):
                st.markdown(f"**Q{i + 1}.** {question['question']}")
                answers[str(i)] = st.text_input(
                    "Your answer", key=f"answer_{lecture_id}_{i}"
                )
            submitted = st.form_submit_button("Submit quiz")
        if submitted:
            new_attempts = []
            for i, question in enumerate(quiz_questions):
                given = answers[str(i)].strip()
                correct = given.casefold() == question["answer"].casefold()
                new_attempts.append({
                    "concept": question["concept"],
                    "given_answer": given,
                    "correct_answer": question["answer"],
                    "correct": correct,
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                    "source_start": question["start"],
                })
            attempts.extend(new_attempts)
            save_attempts(lecture_id, attempts)
            score = sum(x["correct"] for x in new_attempts)
            st.success(f"Score: {score}/{len(new_attempts)}")
            st.rerun()
    else:
        st.info("Process the lecture to generate quiz questions.")

with tab_analytics:
    st.subheader("Learning analytics and weak concepts")
    weak = weak_concepts(lecture_id)
    if weak:
        st.dataframe(pd.DataFrame(weak), use_container_width=True)
        st.bar_chart(pd.DataFrame(weak).set_index("concept")["accuracy"])
        st.caption(
            "Weak concepts are estimated from recorded quiz attempts. "
            "More attempts improve the usefulness of this estimate."
        )
    else:
        st.info("Submit a quiz to begin tracking concept-level performance.")

    st.subheader("Processing report")
    st.json(lecture)

st.divider()
st.caption(
    "EduVision AI | Local prototype | Results depend on video quality, "
    "audio clarity, OCR quality, and model performance."
)
