"""Configuration and status commands."""

from __future__ import annotations

from rag_course.config import AppConfig


def format_status(config: AppConfig) -> str:
    """Render config as a small human-readable report."""

    lines = [
        f"Application: {config.app_name}",
        f"Environment: {config.app_env}",
        f"Verbose: {config.app_verbose}",
        f"Default name: {config.default_name}",
        f"Chat model: {config.chat_model}",
        f"RAG score threshold: {config.rag_score_threshold}",
        f"RAG top_k: {config.rag_top_k}",
        f"RAG context token budget total: {config.rag_context_token_budget_total}",
        f"RAG context token budget per entry: {config.rag_context_token_budget_per_entry}",
        f"Qdrant URL: {config.qdrant_url}",
        f"Qdrant collection: {config.qdrant_collection_name}",
        f"Qdrant vector size: {config.qdrant_vector_size}",
    ]
    return "\n".join(lines)
