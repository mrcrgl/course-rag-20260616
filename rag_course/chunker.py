"""Plain-text chunking for RAG ingestion."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import math
import re
from typing import Iterable


ContentId = str


@dataclass(frozen=True, slots=True)
class ChunkerConfig:
    """Chunking rules used to build RAG chunks."""

    max_tokens_per_chunk: int = 250
    min_words_per_chunk: int = 2
    overlap_sentences: int = 1
    target_sentences_per_chunk: int = 3
    chunker_version: str = "1.0.0"
    chunker_type: str = "paragraph-sentence-window"
    ingester_version: str = "rag-course/1.0.0"


@dataclass(frozen=True, slots=True)
class ChunkMetadata:
    """Optional metadata attached to each chunk."""

    canonical_url: str | None = None
    references: list[ContentId] = field(default_factory=list)
    page_number: int | None = None
    headline_1: str | None = None
    headline_2: str | None = None
    headline_3: str | None = None
    paragraph_number: int | None = None
    position_start: int | None = None
    position_end: int | None = None
    last_modified: str | None = None
    created_at: str | None = None
    author: str | None = None
    pii_classification: str | None = None
    media_type: str | None = None
    embedding_model_used: str | None = None
    software_version_ingester: str | None = None
    chunker_version: str | None = None
    chunker_type: str | None = None
    max_tokens_per_chunk: int | None = None
    overlap_sentences: int | None = None
    target_sentences_per_chunk: int | None = None


@dataclass(frozen=True, slots=True)
class Chunk:
    """A chunk of plain text and its metadata."""

    text: str
    metadata: ChunkMetadata


@dataclass(frozen=True, slots=True)
class _Sentence:
    text: str
    start: int
    end: int


_HEADING_RE = re.compile(r"^\s*(?P<num>\d+(?:\.\d+)*)\.\s+(?P<title>.+?)\.\((?P<page>\d+)\)\s*$")
_SENTENCE_RE = re.compile(r".+?(?:[.!?]+(?:[»”\"']+)?(?:\s+|$)|$)", re.DOTALL)
_INLINE_MARKUP_RE = re.compile(
    r"(?<!\w)=([^=\n]+?)=(?!\w)|(?<!\w)_([^_\n]+?)_(?!\w)|(?<!\w)~([^~\n]+?)~(?!\w)"
)


def chunk_text(text: str, *, config: ChunkerConfig, base_metadata: ChunkMetadata | None = None) -> list[Chunk]:
    """Chunk plain text into sentence-aligned paragraph windows."""

    metadata_defaults = base_metadata or ChunkMetadata()
    paragraphs = _split_paragraphs(text)
    chunks: list[Chunk] = []
    heading_state = {"headline_1": None, "headline_2": None, "headline_3": None}
    cursor = 0

    for paragraph_number, paragraph in enumerate(paragraphs, start=1):
        paragraph_start = text.find(paragraph, cursor)
        if paragraph_start < 0:
            paragraph_start = cursor
        cursor = paragraph_start + len(paragraph)

        heading_match = _HEADING_RE.match(paragraph)
        if heading_match:
            level = len(heading_match.group("num").split("."))
            title = heading_match.group("title").strip()
            page_number = int(heading_match.group("page"))
            heading_state = _update_headings(heading_state, level, title)
            metadata_defaults = _replace_metadata(
                metadata_defaults,
                page_number=page_number,
                headline_1=heading_state["headline_1"],
                headline_2=heading_state["headline_2"],
                headline_3=heading_state["headline_3"],
            )
            continue

        sentences = _split_sentences(paragraph)
        if not sentences:
            continue

        chunks.extend(
            _chunk_paragraph(
                sentences,
                paragraph_start=paragraph_start,
                paragraph_number=paragraph_number,
                config=config,
                metadata_defaults=metadata_defaults,
            )
        )

    return chunks


def chunk_to_dict(chunk: Chunk) -> dict[str, object]:
    """Convert a chunk into YAML/JSON serializable data."""

    return {"text": chunk.text, "metadata": asdict(chunk.metadata)}


def _split_paragraphs(text: str) -> list[str]:
    return [paragraph.strip() for paragraph in re.split(r"\n\s*\n+", text.strip()) if paragraph.strip()]


def _split_sentences(paragraph: str) -> list[_Sentence]:
    sentences: list[_Sentence] = []
    index = 0
    for match in _SENTENCE_RE.finditer(paragraph):
        sentence = match.group(0).strip()
        if not sentence:
            continue
        start = paragraph.find(sentence, index)
        if start < 0:
            start = index
        end = start + len(sentence)
        index = end
        sentences.append(_Sentence(text=sentence, start=start, end=end))
    return sentences


def _chunk_paragraph(
    sentences: list[_Sentence],
    *,
    paragraph_start: int,
    paragraph_number: int,
    config: ChunkerConfig,
    metadata_defaults: ChunkMetadata,
) -> list[Chunk]:
    chunks: list[Chunk] = []
    start_index = 0
    max_sentences = max(1, config.target_sentences_per_chunk)
    overlap = min(max(0, config.overlap_sentences), max_sentences - 1)

    while start_index < len(sentences):
        end_index = min(len(sentences), start_index + max_sentences)
        while end_index > start_index + 1:
            candidate = _join_sentences(sentences[start_index:end_index])
            if _estimate_tokens(candidate) <= config.max_tokens_per_chunk:
                break
            end_index -= 1

        if end_index <= start_index:
            end_index = start_index + 1

        selected = sentences[start_index:end_index]
        chunk_text = _normalize_inline_markup(_join_sentences(selected))
        chunk_start = paragraph_start + selected[0].start
        chunk_end = paragraph_start + selected[-1].end

        if _word_count(chunk_text) <= config.min_words_per_chunk:
            if end_index >= len(sentences):
                break
            start_index = max(end_index - overlap, start_index + 1)
            continue

        chunks.append(
            Chunk(
                text=chunk_text,
                metadata=_replace_metadata(
                    metadata_defaults,
                    paragraph_number=paragraph_number,
                    position_start=chunk_start,
                    position_end=chunk_end,
                    created_at=datetime.now(timezone.utc).isoformat(),
                    max_tokens_per_chunk=config.max_tokens_per_chunk,
                    overlap_sentences=config.overlap_sentences,
                    target_sentences_per_chunk=config.target_sentences_per_chunk,
                    chunker_version=config.chunker_version,
                    chunker_type=config.chunker_type,
                    software_version_ingester=config.ingester_version,
                ),
            )
        )

        if end_index >= len(sentences):
            break
        start_index = max(end_index - overlap, start_index + 1)

    return chunks


def _join_sentences(sentences: Iterable[_Sentence]) -> str:
    return " ".join(sentence.text.strip() for sentence in sentences).strip()


def _normalize_inline_markup(text: str) -> str:
    """Remove Gutenberg-style inline markup markers while keeping the content."""

    def _replace(match: re.Match[str]) -> str:
        for group in match.groups():
            if group is not None:
                return group
        return match.group(0)

    normalized = _INLINE_MARKUP_RE.sub(_replace, text)
    return re.sub(r"\s{2,}", " ", normalized).strip()


def _word_count(text: str) -> int:
    return len(text.split())


def _estimate_tokens(text: str) -> int:
    words = len(text.split())
    return max(1, math.ceil(words * 1.3))


def _update_headings(
    current: dict[str, str | None],
    level: int,
    title: str,
) -> dict[str, str | None]:
    updated = dict(current)
    if level == 1:
        updated["headline_1"] = title
        updated["headline_2"] = None
        updated["headline_3"] = None
    elif level == 2:
        updated["headline_2"] = title
        updated["headline_3"] = None
    else:
        updated["headline_3"] = title
    return updated


def _replace_metadata(metadata: ChunkMetadata, **updates: object) -> ChunkMetadata:
    data = asdict(metadata)
    data.update(updates)
    return ChunkMetadata(**data)
