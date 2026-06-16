"""Reusable embedding helpers for OpenAI-compatible endpoints."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

from openai import OpenAI

from rag_course.config import AppConfig


@dataclass(frozen=True, slots=True)
class EmbeddingResult:
    """Normalized embedding output."""

    vector: list[float]
    model: str


@dataclass(frozen=True, slots=True)
class EmbeddingBatchResult:
    """Normalized embedding output for a batch of inputs."""

    vectors: list[list[float]]
    model: str


def build_client(config: AppConfig) -> OpenAI:
    """Create an OpenAI client using the configured endpoint and token."""

    if not config.openai_api_key:
        raise ValueError(
            "OPENAI_API_KEY is required for embedding commands. "
            "Set it in `.env` or your shell environment."
        )

    kwargs: dict[str, str] = {}
    if config.openai_base_url:
        kwargs["base_url"] = config.openai_base_url
    kwargs["api_key"] = config.openai_api_key
    return OpenAI(**kwargs)


def create_embedding(client: OpenAI, *, text: str, model: str) -> EmbeddingResult:
    """Request a single embedding vector for the provided text."""

    response = client.embeddings.create(
        input=text,
        model=model,
    )
    vector = list(response.data[0].embedding)
    return EmbeddingResult(vector=vector, model=response.model)


def create_embeddings(client: OpenAI, *, texts: Sequence[str], model: str) -> EmbeddingBatchResult:
    """Request embedding vectors for a batch of texts."""

    response = client.embeddings.create(
        input=list(texts),
        model=model,
    )
    vectors = [list(item.embedding) for item in response.data]
    return EmbeddingBatchResult(vectors=vectors, model=response.model)


def cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    """Compute cosine similarity between two vectors."""

    if len(left) != len(right):
        raise ValueError("Vectors must have the same length.")

    dot_product = sum(l * r for l, r in zip(left, right))
    left_magnitude = math.sqrt(sum(value * value for value in left))
    right_magnitude = math.sqrt(sum(value * value for value in right))

    if left_magnitude == 0 or right_magnitude == 0:
        raise ValueError("Cosine similarity is undefined for zero-length vectors.")

    return dot_product / (left_magnitude * right_magnitude)
