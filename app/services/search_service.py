import json
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from app.config import INDEX_DIR, EMBEDDING_MODEL


def get_embedder():
    """Load the configured sentence-embedding model."""
    return SentenceTransformer(EMBEDDING_MODEL)


def _paths(lecture_id: str):
    if (
        not lecture_id
        or Path(lecture_id).name != lecture_id
        or lecture_id in {".", ".."}
        or "/" in lecture_id
        or "\\" in lecture_id
    ):
        raise ValueError("Invalid lecture ID")

    index_path = INDEX_DIR / f"{lecture_id}.faiss"
    documents_path = INDEX_DIR / f"{lecture_id}_documents.json"
    return index_path, documents_path


def build_search_index(
    lecture_id: str,
    transcript: list[dict],
    embedder=None,
) -> dict:
    """Build a persistent cosine-similarity index from transcript segments."""
    if not transcript:
        raise ValueError("Cannot index an empty transcript")

    documents = []

    for segment in transcript:
        text = str(segment.get("text", "")).strip()
        if not text:
            continue

        documents.append({
            "start": float(segment.get("start", 0)),
            "end": float(segment.get("end", 0)),
            "text": text,
            "speaker": segment.get("speaker", "Unknown"),
        })

    if not documents:
        raise ValueError("No non-empty transcript segments to index")

    embedder = embedder or get_embedder()
    vectors = embedder.encode(
        [item["text"] for item in documents],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    vectors = np.asarray(vectors, dtype=np.float32)

    if vectors.ndim != 2 or vectors.shape[0] != len(documents):
        raise ValueError("Embedding model returned an invalid vector shape")

    if not np.isfinite(vectors).all():
        raise ValueError("Embedding model returned non-finite values")

    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)

    index_path, documents_path = _paths(lecture_id)
    faiss.write_index(index, str(index_path))
    documents_path.write_text(
        json.dumps(documents, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    return {
        "lecture_id": lecture_id,
        "indexed_segments": len(documents),
        "embedding_dimensions": int(vectors.shape[1]),
        "index_path": str(index_path),
        "documents_path": str(documents_path),
    }


def search_lecture(
    lecture_id: str,
    query: str,
    k: int = 5,
    embedder=None,
) -> list[dict]:
    """Retrieve the most semantically relevant timestamped segments."""
    query = query.strip() if isinstance(query, str) else ""

    if not query:
        raise ValueError("Enter a question to search the lecture")

    if k < 1:
        raise ValueError("k must be at least 1")

    index_path, documents_path = _paths(lecture_id)

    if not index_path.is_file() or not documents_path.is_file():
        raise FileNotFoundError(
            f"Search index not found for lecture {lecture_id}. "
            "Build the index first."
        )

    index = faiss.read_index(str(index_path))
    documents = json.loads(documents_path.read_text(encoding="utf-8"))

    if index.ntotal != len(documents):
        raise ValueError("Search index and transcript records do not match")

    if not documents:
        return []

    embedder = embedder or get_embedder()
    vector = embedder.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    vector = np.asarray(vector, dtype=np.float32)

    if vector.shape != (1, index.d):
        raise ValueError("Query embedding dimensions do not match the index")

    scores, ids = index.search(vector, min(k, index.ntotal))

    results = []
    for score, idx in zip(scores[0], ids[0]):
        if idx < 0:
            continue

        item = dict(documents[int(idx)])
        item["score"] = float(score)
        item["timestamp_label"] = (
            f"{int(item['start'] // 60):02d}:"
            f"{int(item['start'] % 60):02d}"
        )
        results.append(item)

    return results
