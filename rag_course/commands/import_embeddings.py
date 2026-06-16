"""CLI command for importing embedded chunks into Qdrant."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

from qdrant_client import QdrantClient, models
from yaml import safe_load

from rag_course.chunker import ChunkMetadata
from rag_course.config import AppConfig


@dataclass(frozen=True, slots=True)
class ImportedEmbeddedChunk:
    """Embedded chunk payload read from disk before import."""

    text: str
    metadata: ChunkMetadata
    embedding: list[float]


def run_import_embeddings(config: AppConfig, input_path: str) -> int:
    """Import embedded chunks from YAML into Qdrant."""

    input_file = Path(input_path).expanduser().resolve()
    payload = safe_load(input_file.read_text(encoding="utf-8"))
    embedded_chunks = _load_embedded_chunks(payload)

    client = QdrantClient(url=config.qdrant_url, api_key=config.qdrant_api_key)
    _ensure_collection(client, config)

    total = 0
    for batch in _batched(embedded_chunks, 256):
        for index, chunk in enumerate(batch):
            if len(chunk.embedding) != config.qdrant_vector_size:
                raise ValueError(
                    f"Chunk {index} has embedding length {len(chunk.embedding)} "
                    f"but qdrant_vector_size is {config.qdrant_vector_size}."
                )
        points = [
            models.PointStruct(
                id=chunk.metadata.uuid,
                vector=chunk.embedding,
                payload=_payload_from_metadata(chunk),
            )
            for chunk in batch
        ]
        client.upload_points(
            collection_name=config.qdrant_collection_name,
            points=points,
            wait=True,
        )
        total += len(points)

    print(
        f"Imported {total} chunks into {config.qdrant_collection_name} "
        f"at {config.qdrant_url}"
    )
    return total


def _ensure_collection(client: QdrantClient, config: AppConfig) -> None:
    if client.collection_exists(collection_name=config.qdrant_collection_name):
        return

    client.create_collection(
        collection_name=config.qdrant_collection_name,
        vectors_config=models.VectorParams(
            size=config.qdrant_vector_size,
            distance=models.Distance.COSINE,
        ),
    )


def _load_embedded_chunks(payload: object) -> list[ImportedEmbeddedChunk]:
    if not isinstance(payload, dict):
        raise ValueError("Embedding file must contain a YAML mapping at the top level.")

    raw_chunks = payload.get("chunks")
    if not isinstance(raw_chunks, list):
        raise ValueError("Embedding file must contain a top-level 'chunks' list.")

    chunks: list[ImportedEmbeddedChunk] = []
    for index, item in enumerate(raw_chunks):
        if not isinstance(item, dict):
            raise ValueError(f"Chunk {index} must be a YAML mapping.")

        text = item.get("text")
        metadata = item.get("metadata")
        embedding = item.get("embedding")
        if not isinstance(text, str):
            raise ValueError(f"Chunk {index} is missing a text string.")
        if not isinstance(metadata, dict):
            raise ValueError(f"Chunk {index} is missing metadata.")
        if not isinstance(embedding, list):
            raise ValueError(f"Chunk {index} is missing an embedding vector.")
        if metadata.get("uuid") is None:
            raise ValueError(f"Chunk {index} is missing metadata.uuid.")

        chunks.append(
            ImportedEmbeddedChunk(
                text=text,
                metadata=ChunkMetadata(**metadata),
                embedding=[float(value) for value in embedding],
            )
        )
    return chunks


def _payload_from_metadata(chunk: ImportedEmbeddedChunk) -> dict[str, Any]:
    payload: dict[str, Any] = {"text": chunk.text}
    for key, value in asdict(chunk.metadata).items():
        if key == "uuid":
            continue
        payload[f"meta_{key}"] = value
    return payload


def _batched(items: Iterable[ImportedEmbeddedChunk], size: int) -> Iterable[list[ImportedEmbeddedChunk]]:
    batch: list[ImportedEmbeddedChunk] = []
    for item in items:
        batch.append(item)
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch
