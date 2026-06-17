"""CLI command for querying embedded chunks from Qdrant."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from qdrant_client import QdrantClient

from rag_course.config import AppConfig
from rag_course.embeddings import build_client, create_embedding


@dataclass(frozen=True, slots=True)
class RagResult:
    """Normalized retrieval result."""

    point_id: str
    score: float
    text: str
    metadata: dict[str, Any]


def run_query(config: AppConfig, search_term: str) -> list[RagResult]:
    """Embed a search term, query Qdrant, and print compact result blocks."""

    results = retrieve_rag_results(
        config,
        search_term,
        top_k=15,
        score_threshold=0.55,
    )

    if not results:
        print("No results found.")
        return []

    for index, result in enumerate(results, start=1):
        print(format_result_block(index, result))

    return results


def retrieve_rag_results(
    config: AppConfig,
    search_term: str,
    *,
    top_k: int,
    score_threshold: float,
) -> list[RagResult]:
    """Return retrieval results for a prompt."""

    term = search_term.strip()
    if not term:
        raise ValueError("search_term must be non-empty.")

    client = build_client(config)
    query_embedding = create_embedding(client, text=term, model=config.embedding_model)

    qdrant_client = QdrantClient(url=config.qdrant_url, api_key=config.qdrant_api_key)
    response = qdrant_client.query_points(
        collection_name=config.qdrant_collection_name,
        query=query_embedding.vector,
        limit=top_k,
        score_threshold=score_threshold,
        with_payload=True,
        with_vectors=False,
    )

    results: list[RagResult] = []
    for point in response.points:
        payload = point.payload or {}
        text = payload.get("text")
        if not isinstance(text, str):
            text = ""
        metadata = {key: value for key, value in payload.items() if key != "text"}
        results.append(
            RagResult(
                point_id=str(point.id),
                score=float(point.score),
                text=text,
                metadata=metadata,
            )
        )
    return results


def format_result_block(index: int, result: RagResult) -> str:
    """Render a compact human-readable retrieval block."""

    snippet = _shorten(result.text or "<no text>")
    return "\n".join(
        [
            f"[{index}] score={result.score:.4f}",
            f"id={result.point_id}",
            f"text={snippet}",
        ]
    )


def build_rag_context(
    results: list[RagResult],
    *,
    total_token_budget: int,
    per_entry_token_budget: int,
) -> str:
    """Render retrieval results for system-prompt injection."""

    if total_token_budget <= 0 or per_entry_token_budget <= 0:
        return ""

    lines = ["Retrieved context:"]
    remaining_budget = total_token_budget
    included = 0

    for index, result in enumerate(results, start=1):
        entry_lines = _render_context_entry(index, result, per_entry_token_budget)
        entry_text = "\n".join(entry_lines).strip()
        entry_tokens = _estimate_tokens(entry_text)
        if entry_tokens > per_entry_token_budget:
            entry_lines = _render_context_entry(
                index,
                RagResult(
                    point_id=result.point_id,
                    score=result.score,
                    text=_truncate_text_for_budget(
                        result.text,
                        per_entry_token_budget=per_entry_token_budget,
                        metadata=result.metadata,
                    ),
                    metadata=result.metadata,
                ),
                per_entry_token_budget,
            )
            entry_text = "\n".join(entry_lines).strip()
            entry_tokens = _estimate_tokens(entry_text)

        if entry_tokens > remaining_budget:
            break

        lines.extend(entry_lines)
        lines.append("")
        remaining_budget -= entry_tokens
        included += 1

    if included == 0:
        return ""

    return "\n".join(lines).rstrip()


def _render_context_entry(index: int, result: RagResult, per_entry_token_budget: int) -> list[str]:
    text = result.text.strip() or "<no text>"
    metadata_lines = _format_metadata_lines(result.metadata)
    lines = [
        f"Result {index}:",
        f"score: {result.score:.4f}",
        f"id: {result.point_id}",
        "text: |",
    ]
    lines.extend(f"  {line}" for line in _wrap_text(text, per_entry_token_budget))
    if metadata_lines:
        lines.append("metadata:")
        lines.extend(f"  {line}" for line in metadata_lines)
    else:
        lines.append("metadata: {}")
    return lines


def _format_metadata_lines(metadata: dict[str, Any]) -> list[str]:
    lines: list[str] = []
    for key, value in metadata.items():
        lines.extend(_format_key_value_lines(f"meta_{key}", value))
    return lines


def _format_key_value_lines(key: str, value: Any, *, indent: int = 0) -> list[str]:
    prefix = "  " * indent
    if isinstance(value, dict):
        lines = [f"{prefix}{key}:"]
        for child_key, child_value in value.items():
            lines.extend(_format_key_value_lines(str(child_key), child_value, indent=indent + 1))
        return lines
    if isinstance(value, list):
        if not value:
            return [f"{prefix}{key}: []"]
        lines = [f"{prefix}{key}:"]
        for item in value:
            if isinstance(item, (dict, list)):
                lines.extend(_format_key_value_lines("-", item, indent=indent + 1))
            else:
                lines.append(f"{prefix}  - {_format_value(item)}")
        return lines
    return [f"{prefix}{key}: {_format_value(value)}"]


def _format_value(value: Any) -> str:
    if isinstance(value, str):
        return value
    return repr(value)


def _truncate_text_for_budget(
    text: str,
    *,
    per_entry_token_budget: int,
    metadata: dict[str, Any],
) -> str:
    words = text.split()
    if not words:
        return ""

    for end in range(len(words), 0, -1):
        candidate = " ".join(words[:end])
        candidate_entry = "\n".join(
            _render_context_entry(
                1,
                RagResult(point_id="x", score=0.0, text=candidate, metadata=metadata),
                per_entry_token_budget,
            )
        )
        if _estimate_tokens(candidate_entry) <= per_entry_token_budget:
            return candidate
    return words[0]


def _wrap_text(text: str, token_budget: int) -> list[str]:
    words = text.split()
    if not words:
        return [""]

    max_words = max(1, int(token_budget / 1.3))
    if len(words) <= max_words:
        return [" ".join(words)]

    wrapped = " ".join(words[: max_words - 1]).rstrip() + "…"
    return [wrapped]


def _shorten(text: str, *, limit: int = 240) -> str:
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 1].rstrip() + "…"


def _estimate_tokens(text: str) -> int:
    words = len(text.split())
    return max(1, int(words * 1.3))
