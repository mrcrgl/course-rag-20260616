"""CLI commands for embeddings."""

from __future__ import annotations

import json

from rag_course.config import AppConfig
from rag_course.embeddings import build_client, cosine_similarity, create_embedding


def run_embed(config: AppConfig, text: str) -> list[float]:
    """Create an embedding and print the raw vector."""

    client = build_client(config)
    result = create_embedding(client, text=text, model=config.embedding_model)
    print(json.dumps(result.vector))
    return result.vector


def run_similarity(config: AppConfig) -> float:
    """Prompt for two texts and print their cosine similarity."""

    client = build_client(config)
    first_text = _prompt_text("First text: ")
    second_text = _prompt_text("Second text: ")

    first_embedding = create_embedding(client, text=first_text, model=config.embedding_model)
    second_embedding = create_embedding(client, text=second_text, model=config.embedding_model)
    similarity = cosine_similarity(first_embedding.vector, second_embedding.vector)

    print(similarity)
    return similarity


def _prompt_text(prompt: str) -> str:
    try:
        import readline  # noqa: F401
    except ImportError:
        pass
    return input(prompt).strip()
