from __future__ import annotations

from typing import Any, Dict, List

from hybrid_retrieval import HybridRetriever

TOP_K_FINAL = 5


class HeuristicReranker:
    def rerank(
        self, query: str, results: List[Dict[str, Any]], top_k_final: int = TOP_K_FINAL
    ) -> List[Dict[str, Any]]:
        query_lower = query.lower()

        reranked: List[Dict[str, Any]] = []

        for item in results:
            score = item["rrf_score"]
            chunk_type = item.get("chunk_type", "")
            entity = (item.get("entity") or "").lower()
            priority = item.get("priority", "low")

            bonus = 0.0

            # 1. Bonus selon priorité
            if priority == "high":
                bonus += 0.20
            elif priority == "medium":
                bonus += 0.10

            # 2. Bonus selon type de chunk
            if chunk_type == "example":
                bonus += 0.30
            elif chunk_type == "join_pattern":
                bonus += 0.25
            elif chunk_type == "schema":
                bonus += 0.20
            elif chunk_type == "rule":
                bonus += 0.15
            elif chunk_type == "description":
                bonus += 0.10
            elif chunk_type == "vocabulary":
                bonus += 0.05

            # 3. Bonus si l'entité semble correspondre à la question
            if (
                any(word in query_lower for word in ["patient", "patients"])
                and "patient" in entity
            ):
                bonus += 0.15

            if (
                any(
                    word in query_lower
                    for word in ["doctor", "doctors", "médecin", "médecins"]
                )
                and "doctor" in entity
            ):
                bonus += 0.15

            if (
                any(
                    word in query_lower
                    for word in [
                        "appointment",
                        "appointments",
                        "consultation",
                        "consultations",
                    ]
                )
                and "appointment" in entity
            ):
                bonus += 0.15

            if (
                any(
                    word in query_lower
                    for word in [
                        "prescription",
                        "prescriptions",
                        "medication",
                        "medications",
                        "paracetamol",
                    ]
                )
                and "prescription" in entity
            ):
                bonus += 0.20

            if (
                any(
                    word in query_lower
                    for word in ["department", "departments", "cardiology", "neurology"]
                )
                and "department" in entity
            ):
                bonus += 0.15

            # 4. Bonus contextuel sur le texte
            text_lower = (item.get("text") or "").lower()

            if "paracetamol" in query_lower and "paracetamol" in text_lower:
                bonus += 0.25

            if "cardiology" in query_lower and "cardiology" in text_lower:
                bonus += 0.20

            # Score final
            final_score = score + bonus

            enriched = dict(item)
            enriched["rerank_bonus"] = bonus
            enriched["final_score"] = final_score
            reranked.append(enriched)

        reranked.sort(key=lambda x: x["final_score"], reverse=True)

        for idx, item in enumerate(reranked[:top_k_final], start=1):
            item["rerank_rank"] = idx

        return reranked[:top_k_final]


def print_results(query: str, results: List[Dict[str, Any]]) -> None:
    print("\n" + "=" * 100)
    print(f"QUESTION : {query}")
    print("=" * 100)

    if not results:
        print("Aucun résultat trouvé.")
        return

    for item in results:
        print(f"\nRerank rank : {item['rerank_rank']}")
        print(f"Final score : {item['final_score']:.6f}")
        print(f"RRF score   : {item['rrf_score']:.6f}")
        print(f"Bonus       : {item['rerank_bonus']:.6f}")
        print(f"content_id  : {item['content_id']}")
        print(f"type        : {item['chunk_type']}")
        print(f"entity      : {item['entity']}")
        print(f"subtype     : {item['subtype']}")
        print(f"priority    : {item['priority']}")
        print(f"source      : {item['source_file']}")
        print("Text preview:")
        print((item["text"] or "")[:500])
        print("-" * 100)


def main() -> None:
    hybrid = HybridRetriever()
    reranker = HeuristicReranker()

    test_queries = [
        "Which patients received Paracetamol?",
        "Which doctors work in Cardiology?",
        "How many appointments did each doctor have?",
        "Show appointments with patient and doctor names.",
        "Which table contains prescriptions?",
    ]

    for query in test_queries:
        hybrid_results = hybrid.search(query)
        reranked_results = reranker.rerank(query, hybrid_results)
        print_results(query, reranked_results)


if __name__ == "__main__":
    main()
