from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, List

from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

BASE_DIR = Path(__file__).resolve().parent
QDRANT_PATH = BASE_DIR / "qdrant_data"
COLLECTION_NAME = "hospital_context"

EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"
TOP_K = 5


class DenseRetriever:
    def __init__(
        self,
        qdrant_path: Path,
        collection_name: str,
        embedding_model_name: str,
    ) -> None:
        self.client = self._create_client(qdrant_path)
        self.collection_name = collection_name
        self.model = SentenceTransformer(embedding_model_name)

    def _create_client(self, qdrant_path: Path) -> QdrantClient:
        try:
            return QdrantClient(path=str(qdrant_path))
        except RuntimeError as exc:
            # When another Python process holds qdrant_data/.lock, use a per-process copy.
            if "already accessed by another instance" not in str(exc):
                raise

            fallback_root = Path(tempfile.gettempdir()) / "qdrant_local_fallback"
            fallback_root.mkdir(parents=True, exist_ok=True)
            fallback_path = fallback_root / f"{qdrant_path.name}_{os.getpid()}"

            if fallback_path.exists():
                shutil.rmtree(fallback_path)

            # Ignore the lock file itself when cloning a fallback storage copy.
            shutil.copytree(
                qdrant_path,
                fallback_path,
                ignore=shutil.ignore_patterns(".lock"),
            )

            print(
                f"[Qdrant] Local storage is locked; using fallback copy: {fallback_path}"
            )
            return QdrantClient(path=str(fallback_path))

    def embed_query(self, query: str) -> List[float]:
        vector = self.model.encode(
            query,
            normalize_embeddings=True,
        )
        return vector.tolist()

    def search(self, query: str, top_k: int = TOP_K) -> List[Dict[str, Any]]:
        query_vector = self.embed_query(query)

        response = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=top_k,
        )

        results = response.points

        formatted_results: List[Dict[str, Any]] = []
        for rank, result in enumerate(results, start=1):
            payload = result.payload or {}

            formatted_results.append(
                {
                    "rank": rank,
                    "score": result.score,
                    "content_id": payload.get("content_id"),
                    "source_file": payload.get("source_file"),
                    "chunk_type": payload.get("chunk_type"),
                    "entity": payload.get("entity"),
                    "subtype": payload.get("subtype"),
                    "priority": payload.get("priority"),
                    "text": payload.get("text"),
                }
            )

        return formatted_results

    def close(self) -> None:
        try:
            self.client.close()
        except Exception:
            # Ignore shutdown edge cases from underlying portalocker.
            pass


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
    retriever = DenseRetriever(
        qdrant_path=QDRANT_PATH,
        collection_name=COLLECTION_NAME,
        embedding_model_name=EMBEDDING_MODEL_NAME,
    )

    test_queries = [
        "Which patients received Paracetamol?",
        "Which doctors work in Cardiology?",
        "How many appointments did each doctor have?",
        "Show appointments with patient and doctor names.",
        "Which table contains prescriptions?",
    ]

    try:
        for query in test_queries:
            results = retriever.search(query, top_k=TOP_K)
            print_results(query, results)
    finally:
        retriever.close()


if __name__ == "__main__":
    main()
