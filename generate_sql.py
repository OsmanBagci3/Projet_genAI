import requests

from hybrid_retrieval import HybridRetriever
from query_construction import QueryConstructor
from rerank_results import HeuristicReranker

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "mistral"


class SQLGenerator:
    def generate(self, prompt: str) -> str:
        response = requests.post(
            OLLAMA_URL, json={"model": MODEL_NAME, "prompt": prompt, "stream": False}
        )

        data = response.json()
        return data.get("response", "").strip()


def main():
    hybrid = HybridRetriever()
    reranker = HeuristicReranker()
    constructor = QueryConstructor()
    generator = SQLGenerator()

    queries = [
        "Which patients received Paracetamol?",
        "Which doctors work in Cardiology?",
        "How many appointments did each doctor have?",
    ]

    for query in queries:
        print("\n" + "=" * 100)
        print(f"QUESTION: {query}")
        print("=" * 100)

        # Retrieval + rerank
        hybrid_results = hybrid.search(query)
        reranked = reranker.rerank(query, hybrid_results)

        # Prompt construction
        prompt = constructor.build_context(query, reranked)

        # LLM generation
        sql = generator.generate(prompt)

        print("\nGenerated SQL:\n")
        print(sql)
        print("=" * 100)


if __name__ == "__main__":
    main()
