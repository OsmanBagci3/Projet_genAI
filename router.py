from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Literal, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer

RouteType = Literal["SQL_QUERY", "SCHEMA_HELP", "OUT_OF_SCOPE"]


EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"


@dataclass
class RouteExample:
    text: str
    route: RouteType


class SemanticRouter:
    def __init__(self, model_name: str = EMBEDDING_MODEL_NAME) -> None:
        self.model = SentenceTransformer(model_name)

        self.examples: List[RouteExample] = [
            # SQL_QUERY
            RouteExample("How many patients are in the database?", "SQL_QUERY"),
            RouteExample("Which doctor has the most appointments?", "SQL_QUERY"),
            RouteExample("Show appointments in Cardiology.", "SQL_QUERY"),
            RouteExample("List all medications prescribed.", "SQL_QUERY"),
            RouteExample("Count prescriptions by medication.", "SQL_QUERY"),
            RouteExample("Which patients received Paracetamol?", "SQL_QUERY"),
            RouteExample("Who is the busiest doctor?", "SQL_QUERY"),
            RouteExample("Show patient names with their doctors.", "SQL_QUERY"),
            # SCHEMA_HELP
            RouteExample("Which table contains prescriptions?", "SCHEMA_HELP"),
            RouteExample("What column stores diagnosis?", "SCHEMA_HELP"),
            RouteExample("Show the database schema.", "SCHEMA_HELP"),
            RouteExample(
                "What is the structure of the appointments table?", "SCHEMA_HELP"
            ),
            RouteExample("Which columns exist in doctors?", "SCHEMA_HELP"),
            RouteExample("What fields are stored in patients?", "SCHEMA_HELP"),
            # OUT_OF_SCOPE
            RouteExample("What is diabetes?", "OUT_OF_SCOPE"),
            RouteExample("Explain hypertension.", "OUT_OF_SCOPE"),
            RouteExample("Give me medical advice for fever.", "OUT_OF_SCOPE"),
            RouteExample("What is the best treatment for migraine?", "OUT_OF_SCOPE"),
            RouteExample("Explain the causes of heart disease.", "OUT_OF_SCOPE"),
        ]

        self.example_texts = [ex.text for ex in self.examples]
        self.example_routes = [ex.route for ex in self.examples]

        self.example_embeddings = self.model.encode(
            self.example_texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

    def route(self, query: str) -> Tuple[RouteType, float, Dict[RouteType, float]]:
        query_embedding = self.model.encode(
            query,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        similarities = np.dot(self.example_embeddings, query_embedding)

        # meilleur score par classe
        route_scores: Dict[RouteType, float] = {
            "SQL_QUERY": -1.0,
            "SCHEMA_HELP": -1.0,
            "OUT_OF_SCOPE": -1.0,
        }

        for sim, route in zip(similarities, self.example_routes):
            if sim > route_scores[route]:
                route_scores[route] = float(sim)

        # route principale
        sorted_routes = sorted(route_scores.items(), key=lambda x: x[1], reverse=True)
        best_route, best_score = sorted_routes[0]
        second_route, second_score = sorted_routes[1]

        # fallback si trop ambigu
        if best_score - second_score < 0.03:
            heuristic_route = self._heuristic_fallback(query)
            return heuristic_route, best_score, route_scores

        return best_route, best_score, route_scores

    def _heuristic_fallback(self, query: str) -> RouteType:
        q = query.lower().strip()

        schema_terms = [
            "which table",
            "what table",
            "schema",
            "column",
            "columns",
            "structure",
            "definition",
            "fields",
        ]
        if any(term in q for term in schema_terms):
            return "SCHEMA_HELP"

        db_terms = [
            "patient",
            "patients",
            "doctor",
            "doctors",
            "appointment",
            "appointments",
            "prescription",
            "prescriptions",
            "medication",
            "medications",
            "diagnosis",
            "department",
            "cardiology",
        ]
        if any(term in q for term in db_terms):
            return "SQL_QUERY"

        return "OUT_OF_SCOPE"


def print_route(
    query: str, route: RouteType, score: float, route_scores: Dict[RouteType, float]
) -> None:
    print("\n" + "=" * 100)
    print(f"QUESTION : {query}")
    print("=" * 100)
    print(f"ROUTE CHOISIE : {route}")
    print(f"SCORE         : {score:.4f}")
    print("DETAIL SCORES :")
    for route_name, route_score in route_scores.items():
        print(f"  - {route_name}: {route_score:.4f}")


def main() -> None:
    router = SemanticRouter()

    test_queries = [
        "give me number of patients",
        "how many patients do we have",
        "Which table contains prescriptions?",
        "What is diabetes?",
        "the doctor who has the most appointments",
        "show all medications prescribed",
        "what fields are stored in patients",
        "explain hypertension",
        "count appointments per doctor",
    ]

    for query in test_queries:
        route, score, route_scores = router.route(query)
        print_route(query, route, score, route_scores)


class QueryRouter:
    """Wrapper compatible avec execute_sql.py — expose .route(query) -> RouteType."""

    def __init__(self) -> None:
        self._router = SemanticRouter()

    def route(self, query: str) -> RouteType:
        route, _, _ = self._router.route(query)
        return route


if __name__ == "__main__":
    main()
