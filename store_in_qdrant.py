from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams


BASE_DIR = Path(__file__).resolve().parent
INPUT_FILE = BASE_DIR / "context_chunks_embedded.json"
QDRANT_PATH = BASE_DIR / "qdrant_data"
COLLECTION_NAME = "hospital_context"


def load_embedded_chunks(input_path: Path) -> List[Dict[str, Any]]:
    if not input_path.exists():
        raise FileNotFoundError(f"Fichier introuvable : {input_path}")

    with input_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("Le fichier doit contenir une liste de chunks enrichis.")

    return data


def create_qdrant_collection(client: QdrantClient, collection_name: str, vector_size: int) -> None:
    existing = [c.name for c in client.get_collections().collections]

    if collection_name in existing:
        client.delete_collection(collection_name)

    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(
            size=vector_size,
            distance=Distance.COSINE,
        ),
    )


def build_points(chunks: List[Dict[str, Any]]) -> List[PointStruct]:
    points: List[PointStruct] = []

    for idx, chunk in enumerate(chunks):
        vector = chunk.get("embedding")
        if not vector:
            raise ValueError(f"Chunk sans embedding : {chunk.get('content_id', 'unknown')}")

        payload = {
            "content_id": chunk.get("content_id"),
            "text": chunk.get("text"),
            "source_file": chunk.get("source_file"),
            "chunk_type": chunk.get("chunk_type"),
            "entity": chunk.get("entity"),
            "subtype": chunk.get("subtype"),
            "priority": chunk.get("priority"),
            "embedding_model": chunk.get("embedding_model"),
            "embedding_dimension": chunk.get("embedding_dimension"),
        }

        points.append(
            PointStruct(
                id=idx,
                vector=vector,
                payload=payload,
            )
        )

    return points


def main() -> None:
    chunks = load_embedded_chunks(INPUT_FILE)

    if not chunks:
        raise ValueError("Aucun chunk trouvé dans le fichier d'entrée.")

    vector_size = chunks[0].get("embedding_dimension")
    if not vector_size:
        raise ValueError("Impossible de déterminer la dimension des embeddings.")

    client = QdrantClient(path=str(QDRANT_PATH))

    print("Création de la collection Qdrant...")
    create_qdrant_collection(client, COLLECTION_NAME, vector_size)

    print("Préparation des points...")
    points = build_points(chunks)

    print(f"Insertion de {len(points)} points dans Qdrant...")
    client.upsert(
        collection_name=COLLECTION_NAME,
        points=points,
    )

    collection_info = client.get_collection(COLLECTION_NAME)

    print("\nStockage terminé avec succès.")
    print(f"Collection       : {COLLECTION_NAME}")
    print(f"Nombre de points : {collection_info.points_count}")
    print(f"Dimension        : {vector_size}")
    print(f"Dossier Qdrant   : {QDRANT_PATH}")


if __name__ == "__main__":
    main()