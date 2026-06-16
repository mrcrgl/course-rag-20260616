"""CLI command for chunking plain-text sources into YAML."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from yaml import safe_dump

from rag_course.chunker import ChunkMetadata, ChunkerConfig, chunk_text, chunk_to_dict
from rag_course.config import AppConfig
from rag_course.sources import read_text_source


def run_chunk(
    config: AppConfig,
    source: str,
    output_path: str,
    *,
    max_tokens_per_chunk: int,
    min_words_per_chunk: int,
    overlap_sentences: int,
    target_sentences_per_chunk: int,
    canonical_url: str | None = None,
    author: str | None = None,
    pii_classification: str | None = None,
    embedding_model: str | None = None,
) -> Path:
    """Read a source, chunk it, and write the chunks as YAML."""

    source_data = read_text_source(source)
    chunk_config = ChunkerConfig(
        max_tokens_per_chunk=max_tokens_per_chunk,
        min_words_per_chunk=min_words_per_chunk,
        overlap_sentences=overlap_sentences,
        target_sentences_per_chunk=target_sentences_per_chunk,
    )
    base_metadata = ChunkMetadata(
        canonical_url=canonical_url or source_data.canonical_url,
        last_modified=source_data.last_modified,
        media_type=source_data.media_type,
        author=author,
        pii_classification=pii_classification,
        embedding_model_used=embedding_model or config.embedding_model,
    )
    chunks = chunk_text(source_data.text, config=chunk_config, base_metadata=base_metadata)
    output = Path(output_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": {
            "input": source,
            "canonical_url": canonical_url or source_data.canonical_url,
            "media_type": source_data.media_type,
            "last_modified": source_data.last_modified,
        },
        "chunker": {
            "max_tokens_per_chunk": max_tokens_per_chunk,
            "min_words_per_chunk": min_words_per_chunk,
            "overlap_sentences": overlap_sentences,
            "target_sentences_per_chunk": target_sentences_per_chunk,
        },
        "chunks": [chunk_to_dict(chunk) for chunk in chunks],
    }
    output.write_text(safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8")
    print(output)
    return output
