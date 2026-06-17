"""CLI command for chunking plain-text sources into YAML."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from yaml import safe_dump

from rag_course.chunker import ChunkMetadata, ChunkerConfig, chunk_legal_pdf, chunk_text, chunk_to_dict
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
    chunker: str = "story",
    canonical_url: str | None = None,
    author: str | None = None,
    pii_classification: str | None = None,
    embedding_model: str | None = None,
) -> Path:
    """Read a source, chunk it, and write the chunks as YAML."""

    source_data = read_text_source(source)
    selected_chunker = _normalize_chunker_name(chunker)
    chunk_config = ChunkerConfig(
        max_tokens_per_chunk=max_tokens_per_chunk,
        min_words_per_chunk=min_words_per_chunk,
        overlap_sentences=overlap_sentences,
        target_sentences_per_chunk=target_sentences_per_chunk,
        chunker_type=_chunker_type_for_name(selected_chunker),
    )
    base_metadata = ChunkMetadata(
        canonical_url=canonical_url or source_data.canonical_url,
        last_modified=source_data.last_modified,
        media_type=source_data.media_type,
        author=author,
        pii_classification=pii_classification,
        embedding_model_used=embedding_model or config.embedding_model,
    )
    if selected_chunker == "story":
        chunks = chunk_text(source_data.text, config=chunk_config, base_metadata=base_metadata)
    elif selected_chunker == "legal-pdf":
        if not source_data.pages:
            raise ValueError("The legal-pdf chunker requires a PDF source.")
        chunks = chunk_legal_pdf(source_data.pages, config=chunk_config, base_metadata=base_metadata)
    else:  # pragma: no cover - guarded by CLI choices
        raise ValueError(f"Unsupported chunker: {chunker}")

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
            "name": selected_chunker,
            "type": chunk_config.chunker_type,
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


def _normalize_chunker_name(value: str) -> str:
    normalized = value.strip().lower().replace("_", "-")
    if normalized in {"story", "paragraph-sentence-window"}:
        return "story"
    if normalized in {"legal", "legal-pdf", "legal-pdf-page-window"}:
        return "legal-pdf"
    raise ValueError("chunker must be one of: story, legal-pdf")


def _chunker_type_for_name(name: str) -> str:
    if name == "story":
        return "paragraph-sentence-window"
    if name == "legal-pdf":
        return "legal-pdf-page-window"
    raise ValueError(f"Unsupported chunker: {name}")
