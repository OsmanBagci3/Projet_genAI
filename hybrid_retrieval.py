from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List

from dense_retrieval import DenseRetriever, QDRANT_PATH, COLLECTION_NAME, EMBEDDING_MODEL_NAME
from sparse_retrieval import SparseRetriever, load_chunks


TOP_K_DENSE = 5
TOP_K_SPARSE = 5
TOP_K_FINAL = 5
RRF_K = 60


def reciprocal_rank_fusion(rank: int, k: int = RRF_K) -> float:
    return 1.0 / (k + rank)


class HybridRetriever:
    def __init__(self) -> None:
        self.dense_retriever = DenseRetriever(
            qdrant_path=QDRANT_PATH,
            collection_name=COLLECTION_NAME,
            embedding_model_name=EMBEDDING_MODEL_NAME,
        )

        chunks = load_chunks(__import__("pathlib").Path(__file__).resolve().parent / "context_chunks.json")
        self.sparse_retriever = SparseRetriever(chunks)

    def search(
        self,
        query: str,
        top_k_dense: int = TOP_K_DENSE,
        top_k_sparse: int = TOP_K_SPARSE,
        top_k_final: int = TOP_K_FINAL,
    ) -> List[Dict[str, Any]]:
        dense_results = self.dense_retriever.search(query, top_k=top_k_dense)
        sparse_results = self.sparse_retriever.search(query, top_k=top_k_sparse)

        fused: Dict[str, Dict[str, Any]] = defaultdict(dict)

        for item in dense_results:
            content_id = item["content_id"]
            fused[content_id]["content_id"] = content_id
            fused[content_id]["text"] = item["text"]
            fused[content_id]["source_file"] = item["source_file"]
            fused[content_id]["chunk_type"] = item["chunk_type"]
            fused[content_id]["entity"] = item["entity"]
            fused[content_id]["subtype"] = item["subtype"]
            fused[content_id]["priority"] = item["priority"]
            fused[content_id]["dense_rank"] = item["rank"]
            fused[content_id]["dense_score"] = item["score"]

        for item in sparse_results:
            content_id = item["content_id"]
            fused[content_id]["content_id"] = content_id
            fused[content_id]["text"] = item["text"]
            fused[content_id]["source_file"] = item["source_file"]
            fused[content_id]["chunk_type"] = item["chunk_type"]
            fused[content_id]["entity"] = item["entity"]
            fused[content_id]["subtype"] = item["subtype"]
            fused[content_id]["priority"] = item["priority"]
            fused[content_id]["sparse_rank"] = item["rank"]
            fused[content_id]["sparse_score"] = item["score"]

        final_results: List[Dict[str, Any]] = []

        for _, item in fused.items():
            dense_rank = item.get("dense_rank")
            sparse_rank = item.get("sparse_rank")

            rrf_score = 0.0
            if dense_rank is not None:
                rrf_score += reciprocal_rank_fusion(dense_rank)
            if sparse_rank is not None:
                rrf_score += reciprocal_rank_fusion(sparse_rank)

            item["rrf_score"] = rrf_score
            final_results.append(item)

        final_results.sort(key=lambda x: x["rrf_score"], reverse=True)

        for idx, item in enumerate(final_results[:top_k_final], start=1):
            item["hybrid_rank"] = idx

        return final_results[:top_k_final]

    def close(self) -> None:
        self.dense_retriever.close()


def print_results(query: str, results: List[Dict[str, Any]]) -> None:
    print("\n" + "=" * 100)
    print(f"QUESTION : {query}")
    print("=" * 100)

    if not results:
        print("Aucun résultat trouvé.")
        return

    for item in results:
        print(f"\nHybrid rank : {item['hybrid_rank']}")
        print(f"RRF score   : {item['rrf_score']:.6f}")
        print(f"content_id  : {item['content_id']}")
        print(f"type        : {item['chunk_type']}")
        print(f"entity      : {item['entity']}")
        print(f"subtype     : {item['subtype']}")
        print(f"priority    : {item['priority']}")
        print(f"source      : {item['source_file']}")
        print(f"dense_rank  : {item.get('dense_rank')}")
        print(f"sparse_rank : {item.get('sparse_rank')}")
        print("Text preview:")
        print((item["text"] or "")[:500])
        print("-" * 100)


def main() -> None:
    retriever = HybridRetriever()

    test_queries = [
        "Which patients received Paracetamol?",
        "Which doctors work in Cardiology?",
        "How many appointments did each doctor have?",
        "Show appointments with patient and doctor names.",
        "Which table contains prescriptions?",
    ]

    try:
        for query in test_queries:
            results = retriever.search(query)
            print_results(query, results)
    finally:
        retriever.close()

if __name__ == "__main__":
    main()