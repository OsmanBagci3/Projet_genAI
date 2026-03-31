from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List

from rank_bm25 import BM25Okapi

BASE_DIR = Path(__file__).resolve().parent
INPUT_FILE = BASE_DIR / "context_chunks.json"
TOP_K = 5


def load_chunks(input_path: Path) -> List[Dict[str, Any]]:
    if not input_path.exists():
        raise FileNotFoundError(f"Fichier introuvable : {input_path}")

    with input_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("Le fichier JSON doit contenir une liste de chunks.")

    return data


def tokenize(text: str) -> List[str]:
    """
    Tokenisation simple :
    - minuscules
    - extraction des mots alphanumériques
    """
    return re.findall(r"\b\w+\b", text.lower())


class SparseRetriever:
    def __init__(self, chunks: List[Dict[str, Any]]) -> None:
        self.chunks = chunks
        self.tokenized_corpus = [tokenize(chunk.get("text", "")) for chunk in chunks]
        self.bm25 = BM25Okapi(self.tokenized_corpus)

    def search(self, query: str, top_k: int = TOP_K) -> List[Dict[str, Any]]:
        query_tokens = tokenize(query)
        scores = self.bm25.get_scores(query_tokens)

        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)

        results: List[Dict[str, Any]] = []
        for rank, (idx, score) in enumerate(ranked[:top_k], start=1):
            chunk = self.chunks[idx]

            results.append(
                {
                    "rank": rank,
                    "score": float(score),
                    "content_id": chunk.get("content_id"),
                    "source_file": chunk.get("source_file"),
                    "chunk_type": chunk.get("chunk_type"),
                    "entity": chunk.get("entity"),
                    "subtype": chunk.get("subtype"),
                    "priority": chunk.get("priority"),
                    "text": chunk.get("text"),
                }
            )

        return results


def print_results(query: str, results: List[Dict[str, Any]]) -> None:
    print("\n" + "=" * 100)
    print(f"QUESTION : {query}")
    print("=" * 100)

    if not results:
        print("Aucun résultat trouvé.")
        return

    for item in results:
        print(f"\nRank       : {item['rank']}")
        print(f"Score      : {item['score']:.4f}")
        print(f"content_id : {item['content_id']}")
        print(f"type       : {item['chunk_type']}")
        print(f"entity     : {item['entity']}")
        print(f"subtype    : {item['subtype']}")
        print(f"priority   : {item['priority']}")
        print(f"source     : {item['source_file']}")
        print("Text preview:")
        print((item["text"] or "")[:500])
        print("-" * 100)


def main() -> None:
    chunks = load_chunks(INPUT_FILE)
    retriever = SparseRetriever(chunks)

    test_queries = [
        "Which patients received Paracetamol?",
        "Which doctors work in Cardiology?",
        "How many appointments did each doctor have?",
        "Show appointments with patient and doctor names.",
        "Which table contains prescriptions?",
    ]

    for query in test_queries:
        results = retriever.search(query, top_k=TOP_K)
        print_results(query, results)


if __name__ == "__main__":
    main()
