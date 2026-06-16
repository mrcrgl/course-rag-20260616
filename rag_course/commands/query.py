"""CLI command for querying embedded chunks from Qdrant."""

from __future__ import annotations

from typing import Any

from qdrant_client import QdrantClient, models

from rag_course.config import AppConfig
from rag_course.embeddings import build_client, create_embedding


def run_query(config: AppConfig, search_term: str) -> list[models.ScoredPoint]:
    """Embed a search term, query Qdrant, and print compact result blocks."""

    term = search_term.strip()
    if not term:
        raise ValueError("search_term must be non-empty.")

    client = build_client(config)
    query_embedding = create_embedding(client, text=term, model=config.embedding_model)

    qdrant_client = QdrantClient(url=config.qdrant_url, api_key=config.qdrant_api_key)
    response = qdrant_client.query_points(
        collection_name=config.qdrant_collection_name,
        query=query_embedding.vector,
        limit=15,
        score_threshold=0.55,
        with_payload=True,
        with_vectors=False,
    )

    points = list(response.points)
    if not points:
        print("No results found.")
        return []

    for index, point in enumerate(points, start=1):
        print(_format_point(index, point))

    return points


def _format_point(index: int, point: models.ScoredPoint) -> str:
    payload: dict[str, Any] = point.payload or {}
    text = payload.get("text")
    if not isinstance(text, str) or not text.strip():
        text = "<no text>"
    snippet = _shorten(text)
    return "\n".join(
        [
            f"[{index}] score={point.score:.4f}",
            f"id={point.id}",
            f"text={snippet}",
        ]
    )


def _shorten(text: str, *, limit: int = 240) -> str:
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 1].rstrip() + "…"
