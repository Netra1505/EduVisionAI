import json
import re
from collections import Counter
from itertools import combinations

import networkx as nx

from app.config import REPORT_DIR


STOPWORDS = {
    "the", "and", "for", "are", "was", "were", "this", "that",
    "with", "from", "have", "has", "had", "you", "your", "its",
    "they", "their", "them", "then", "than", "into", "about",
    "what", "when", "where", "which", "while", "will", "would",
    "could", "should", "there", "here", "also", "used", "using",
    "use", "can", "not", "but", "all", "any", "how", "does",
    "did", "been", "being", "our", "out", "each", "more", "most",
    "such", "only", "other", "some", "very", "its", "it's",
}


def extract_concepts(transcript: list[dict], top_n: int = 20) -> list[dict]:
    """Extract candidate concepts from words in the lecture transcript."""
    if top_n < 1:
        raise ValueError("top_n must be positive")

    counts = Counter()
    first_seen = {}

    for segment in transcript:
        text = str(segment.get("text", ""))
        start = float(segment.get("start", 0))

        words = re.findall(r"\b[a-zA-Z][a-zA-Z0-9_-]{2,}\b", text.lower())

        for word in words:
            if word in STOPWORDS:
                continue

            counts[word] += 1
            first_seen.setdefault(word, start)

    return [
        {
            "concept": word,
            "frequency": frequency,
            "first_seen_seconds": round(first_seen[word], 2),
        }
        for word, frequency in counts.most_common(top_n)
    ]


def build_concept_graph(
    lecture_id: str,
    transcript: list[dict],
    top_n: int = 20,
) -> dict:
    """Create a graph of concepts that occur together in transcript segments."""
    concepts = extract_concepts(transcript, top_n=top_n)
    selected = {item["concept"] for item in concepts}

    graph = nx.Graph()

    for item in concepts:
        graph.add_node(
            item["concept"],
            frequency=item["frequency"],
            first_seen_seconds=item["first_seen_seconds"],
        )

    for segment in transcript:
        words = set(
            re.findall(
                r"\b[a-zA-Z][a-zA-Z0-9_-]{2,}\b",
                str(segment.get("text", "")).lower(),
            )
        )
        words &= selected
        words -= STOPWORDS

        for left, right in combinations(sorted(words), 2):
            if graph.has_edge(left, right):
                graph[left][right]["weight"] += 1
            else:
                graph.add_edge(left, right, weight=1)

    output = {
        "lecture_id": lecture_id,
        "method": "word frequency and co-occurrence; candidate concepts, not verified ontology",
        "nodes": [
            {"id": node, **attrs}
            for node, attrs in graph.nodes(data=True)
        ],
        "edges": [
            {"source": left, "target": right, **attrs}
            for left, right, attrs in graph.edges(data=True)
        ],
    }

    path = REPORT_DIR / f"{lecture_id}_concept_graph.json"
    path.write_text(
        json.dumps(output, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    output["output_path"] = str(path)
    return output
