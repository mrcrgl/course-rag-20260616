"""Chunking strategies for RAG ingestion."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import math
import re
from uuid import uuid4
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

    uuid: str | None = None
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


@dataclass(frozen=True, slots=True)
class _Paragraph:
    text: str
    start: int
    end: int
    number: int


_HEADING_RE = re.compile(r"^\s*(?P<num>\d+(?:\.\d+)*)\.\s+(?P<title>.+?)\.\((?P<page>\d+)\)\s*$")
_SENTENCE_RE = re.compile(r".+?(?:[.!?]+(?:[»”\"']+)?(?:\s+|$)|$)", re.DOTALL)
_INLINE_MARKUP_RE = re.compile(
    r"(?<!\w)=([^=\n]+?)=(?!\w)|(?<!\w)_([^_\n]+?)_(?!\w)|(?<!\w)~([^~\n]+?)~(?!\w)"
)
_TERMINAL_SECTION_TITLES = {
    "Kleine Gedichte.",
    "VOCABULARY.",
}


def chunk_text(text: str, *, config: ChunkerConfig, base_metadata: ChunkMetadata | None = None) -> list[Chunk]:
    """Chunk prose into sentence-aligned paragraph windows."""

    metadata_defaults = base_metadata or ChunkMetadata()
    paragraphs = _merge_dialogue_paragraphs(text, _extract_paragraphs(text))
    chunks: list[Chunk] = []
    heading_state = {"headline_1": None, "headline_2": None, "headline_3": None}

    for paragraph in paragraphs:
        paragraph_text = paragraph.text

        heading_match = _HEADING_RE.match(paragraph_text)
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

        if _is_terminal_section(paragraph_text):
            break

        sentences = _split_sentences(paragraph_text)
        if not sentences:
            continue

        chunks.extend(
            _chunk_paragraph(
                sentences,
                paragraph_start=paragraph.start,
                paragraph_number=paragraph.number,
                config=config,
                metadata_defaults=metadata_defaults,
            )
        )

    return chunks


def chunk_legal_pdf(
    pages: list[str],
    *,
    config: ChunkerConfig,
    base_metadata: ChunkMetadata | None = None,
) -> list[Chunk]:
    """Chunk legal PDF pages without crossing page boundaries."""

    metadata_defaults = base_metadata or ChunkMetadata()
    chunks: list[Chunk] = []

    for page_number, page_text in enumerate(pages, start=1):
        normalized_page_text = _normalize_pdf_page_text(page_text)
        if not normalized_page_text:
            continue

        for paragraph in _extract_paragraphs(normalized_page_text):
            sentences = _split_sentences(paragraph.text)
            if not sentences:
                continue

            chunks.extend(
                _chunk_paragraph(
                    sentences,
                    paragraph_start=paragraph.start,
                    paragraph_number=paragraph.number,
                    page_number=page_number,
                    config=config,
                    metadata_defaults=metadata_defaults,
                )
            )

    return chunks


def chunk_to_dict(chunk: Chunk) -> dict[str, object]:
    """Convert a chunk into YAML/JSON serializable data."""

    return {"text": chunk.text, "metadata": asdict(chunk.metadata)}


def _extract_paragraphs(text: str) -> list[_Paragraph]:
    paragraphs: list[_Paragraph] = []
    cursor = 0
    for paragraph_number, paragraph in enumerate(_split_paragraphs(text), start=1):
        paragraph_start = text.find(paragraph, cursor)
        if paragraph_start < 0:
            paragraph_start = cursor
        paragraph_end = paragraph_start + len(paragraph)
        cursor = paragraph_end
        paragraphs.append(
            _Paragraph(
                text=paragraph,
                start=paragraph_start,
                end=paragraph_end,
                number=paragraph_number,
            )
        )
    return paragraphs


def _split_paragraphs(text: str) -> list[str]:
    return [paragraph.strip() for paragraph in re.split(r"\n\s*\n+", text.strip()) if paragraph.strip()]


def _merge_dialogue_paragraphs(source_text: str, paragraphs: list[_Paragraph]) -> list[_Paragraph]:
    merged: list[_Paragraph] = []
    index = 0
    while index < len(paragraphs):
        current = paragraphs[index]
        if index + 1 < len(paragraphs) and _should_merge_dialogue_paragraphs(current.text, paragraphs[index + 1].text):
            following = paragraphs[index + 1]
            merged.append(
                _Paragraph(
                    text=source_text[current.start:following.end],
                    start=current.start,
                    end=following.end,
                    number=current.number,
                )
            )
            index += 2
            continue

        merged.append(current)
        index += 1
    return merged


def _should_merge_dialogue_paragraphs(current: str, following: str) -> bool:
    """Keep a dialogue lead-in with the direct speech that follows it."""

    return current.rstrip().endswith(":") and following.lstrip().startswith("»")


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
    page_number: int | None = None,
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
                    page_number=page_number if page_number is not None else metadata_defaults.page_number,
                    paragraph_number=paragraph_number,
                    position_start=chunk_start,
                    position_end=chunk_end,
                    created_at=_rfc3339_now(),
                    uuid=str(uuid4()),
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


def _normalize_pdf_page_text(text: str) -> str:
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    normalized = re.sub(r"-\n(?=\w)", "", normalized)
    normalized = re.sub(r"[ \t]+\n", "\n", normalized)
    normalized = re.sub(r"\n{3,}", "\n\n", normalized)
    return normalized.strip()


def _is_terminal_section(paragraph: str) -> bool:
    """Stop at auxiliary sections that are not part of the story text.

    The Project Gutenberg reader text used in this course appends poems and a
    vocabulary section after the main story collection. Those sections are
    useful in the source edition but should not be indexed as narrative chunks.
    """

    return paragraph.strip() in _TERMINAL_SECTION_TITLES


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


def _rfc3339_now() -> str:
    return datetime.now(timezone.utc).isoformat()
