"""CLI command for embedding chunk YAML into a new YAML artifact."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from yaml import safe_dump, safe_load

from rag_course.chunker import ChunkMetadata
from rag_course.config import AppConfig
from rag_course.embeddings import build_client, create_embeddings


@dataclass(frozen=True, slots=True)
class EmbeddedChunk:
    """Chunk content enriched with its embedding vector."""

    text: str
    metadata: ChunkMetadata
    embedding: list[float]


def run_embed_chunks(
    config: AppConfig,
    input_path: str,
    output_path: str,
    *,
    batch_size: int,
) -> Path:
    """Read chunk YAML, embed each chunk text, and write a new YAML artifact."""

    if batch_size <= 0:
        raise ValueError("batch_size must be greater than zero.")

    input_file = Path(input_path).expanduser().resolve()
    payload = safe_load(input_file.read_text(encoding="utf-8"))
    chunks = _load_chunks(payload)

    client = build_client(config)
    embedded_chunks: list[EmbeddedChunk] = []
    for batch in _batched(chunks, batch_size):
        texts = [chunk.text for chunk in batch]
        batch_result = create_embeddings(client, texts=texts, model=config.embedding_model)
        if len(batch_result.vectors) != len(batch):
            raise ValueError("Embedding response size did not match the input batch size.")

        embedded_chunks.extend(
            EmbeddedChunk(text=chunk.text, metadata=chunk.metadata, embedding=vector)
            for chunk, vector in zip(batch, batch_result.vectors, strict=True)
        )

    output = Path(output_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output_payload: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": payload.get("source"),
        "chunker": payload.get("chunker"),
        "embedding": {
            "model": config.embedding_model,
            "batch_size": batch_size,
        },
        "chunks": [_chunk_to_dict(chunk) for chunk in embedded_chunks],
    }
    output.write_text(safe_dump(output_payload, sort_keys=False, allow_unicode=True), encoding="utf-8")
    print(output)
    return output


def _load_chunks(payload: object) -> list[EmbeddedChunk]:
    if not isinstance(payload, dict):
        raise ValueError("Chunk file must contain a YAML mapping at the top level.")

    raw_chunks = payload.get("chunks")
    if not isinstance(raw_chunks, list):
        raise ValueError("Chunk file must contain a top-level 'chunks' list.")

    chunks: list[EmbeddedChunk] = []
    for index, item in enumerate(raw_chunks):
        if not isinstance(item, dict):
            raise ValueError(f"Chunk {index} must be a YAML mapping.")

        text = item.get("text")
        metadata = item.get("metadata")
        if not isinstance(text, str):
            raise ValueError(f"Chunk {index} is missing a text string.")
        if not isinstance(metadata, dict):
            raise ValueError(f"Chunk {index} is missing metadata.")

        chunks.append(
            EmbeddedChunk(
                text=text,
                metadata=ChunkMetadata(**metadata),
                embedding=[],
            )
        )
    return chunks


def _chunk_to_dict(chunk: EmbeddedChunk) -> dict[str, object]:
    return {
        "text": chunk.text,
        "metadata": asdict(chunk.metadata),
        "embedding": chunk.embedding,
    }


def _batched(items: Iterable[EmbeddedChunk], size: int) -> Iterable[list[EmbeddedChunk]]:
    batch: list[EmbeddedChunk] = []
    for item in items:
        batch.append(item)
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch
