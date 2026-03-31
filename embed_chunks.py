from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from sentence_transformers import SentenceTransformer

BASE_DIR = Path(__file__).resolve().parent
INPUT_FILE = BASE_DIR / "context_chunks.json"
OUTPUT_FILE = BASE_DIR / "context_chunks_embedded.json"

EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"


def load_chunks(input_path: Path) -> List[Dict[str, Any]]:
    if not input_path.exists():
        raise FileNotFoundError(f"Fichier introuvable : {input_path}")

    with input_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("Le fichier JSON doit contenir une liste de chunks.")

    return data


def add_embeddings_to_chunks(
    chunks: List[Dict[str, Any]],
    model_name: str,
) -> List[Dict[str, Any]]:
    print(f"Chargement du modèle d'embedding : {model_name}")
    model = SentenceTransformer(model_name)

    texts = []
    for chunk in chunks:
        text = chunk.get("text", "").strip()
        if not text:
            raise ValueError(
                f"Chunk sans texte détecté : {chunk.get('content_id', 'unknown')}"
            )
        texts.append(text)

    print(f"Génération des embeddings pour {len(texts)} chunks...")
    vectors = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    enriched_chunks: List[Dict[str, Any]] = []

    for chunk, vector in zip(chunks, vectors):
        enriched_chunk = dict(chunk)
        enriched_chunk["embedding_model"] = model_name
        enriched_chunk["embedding_dimension"] = int(len(vector))
        enriched_chunk["embedding"] = vector.tolist()
        enriched_chunks.append(enriched_chunk)

    return enriched_chunks


def save_embedded_chunks(
    embedded_chunks: List[Dict[str, Any]],
    output_path: Path,
) -> None:
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(embedded_chunks, f, indent=2, ensure_ascii=False)

    print(f"Fichier généré : {output_path}")


def print_summary(embedded_chunks: List[Dict[str, Any]]) -> None:
    print("\nRésumé des embeddings générés :")
    print(f"Nombre total de chunks : {len(embedded_chunks)}")

    if embedded_chunks:
        first = embedded_chunks[0]
        print(f"Premier content_id : {first.get('content_id')}")
        print(f"Modèle utilisé     : {first.get('embedding_model')}")
        print(f"Dimension vecteur  : {first.get('embedding_dimension')}")

        preview = first.get("embedding", [])[:5]
        print(f"Premières valeurs  : {preview}")


def main() -> None:
    chunks = load_chunks(INPUT_FILE)
    embedded_chunks = add_embeddings_to_chunks(chunks, EMBEDDING_MODEL_NAME)
    save_embedded_chunks(embedded_chunks, OUTPUT_FILE)
    print_summary(embedded_chunks)


if __name__ == "__main__":
    main()
