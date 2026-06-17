"""Load and validate application configuration."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path

from dotenv import load_dotenv


@dataclass(frozen=True, slots=True)
class AppConfig:
    """Application settings sourced from environment variables."""

    app_name: str = "RAG Course Demo"
    default_name: str = "World"
    app_env: str = "development"
    app_verbose: bool = False
    openai_base_url: str | None = None
    openai_api_key: str | None = None
    embedding_model: str = "text-embedding-3-small"
    chat_model: str = "gpt-4.1-mini"
    rag_score_threshold: float = 0.5
    rag_top_k: int = 5
    rag_context_token_budget_total: int = 800
    rag_context_token_budget_per_entry: int = 200
    qdrant_url: str = "http://localhost:9333"
    qdrant_collection_name: str = "rag_chunks"
    qdrant_vector_size: int = 1536
    qdrant_api_key: str | None = None


DEFAULT_CONFIG = AppConfig()


def load_config() -> AppConfig:
    """Load `.env` if present and return normalized configuration."""

    load_dotenv(dotenv_path=_project_root() / ".env", override=False)

    return AppConfig(
        app_name=os.getenv("APP_NAME", DEFAULT_CONFIG.app_name),
        default_name=os.getenv("DEFAULT_NAME", DEFAULT_CONFIG.default_name),
        app_env=os.getenv("APP_ENV", DEFAULT_CONFIG.app_env),
        app_verbose=_as_bool(os.getenv("APP_VERBOSE"), default=DEFAULT_CONFIG.app_verbose),
        openai_base_url=_optional_str(os.getenv("OPENAI_BASE_URL")),
        openai_api_key=_optional_str(os.getenv("OPENAI_API_KEY")),
        embedding_model=_value_or_default(
            os.getenv("EMBEDDING_MODEL"),
            default=DEFAULT_CONFIG.embedding_model,
        ),
        chat_model=_value_or_default(
            os.getenv("CHAT_MODEL"),
            default=DEFAULT_CONFIG.chat_model,
        ),
        rag_score_threshold=_as_float(
            os.getenv("RAG_SCORE_THRESHOLD"),
            default=DEFAULT_CONFIG.rag_score_threshold,
        ),
        rag_top_k=_as_int(
            os.getenv("RAG_TOP_K"),
            default=DEFAULT_CONFIG.rag_top_k,
        ),
        rag_context_token_budget_total=_as_int(
            os.getenv("RAG_CONTEXT_TOKEN_BUDGET_TOTAL"),
            default=DEFAULT_CONFIG.rag_context_token_budget_total,
        ),
        rag_context_token_budget_per_entry=_as_int(
            os.getenv("RAG_CONTEXT_TOKEN_BUDGET_PER_ENTRY"),
            default=DEFAULT_CONFIG.rag_context_token_budget_per_entry,
        ),
        qdrant_url=_value_or_default(
            os.getenv("QDRANT_URL"),
            default=DEFAULT_CONFIG.qdrant_url,
        ),
        qdrant_collection_name=_value_or_default(
            os.getenv("QDRANT_COLLECTION_NAME"),
            default=DEFAULT_CONFIG.qdrant_collection_name,
        ),
        qdrant_vector_size=_as_int(
            os.getenv("QDRANT_VECTOR_SIZE"),
            default=DEFAULT_CONFIG.qdrant_vector_size,
        ),
        qdrant_api_key=_optional_str(os.getenv("QDRANT_API_KEY")),
    )


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _as_bool(raw: str | None, *, default: bool) -> bool:
    if raw is None:
        return default
    normalized = raw.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return default


def _optional_str(raw: str | None) -> str | None:
    if raw is None:
        return None
    value = raw.strip()
    return value or None


def _value_or_default(raw: str | None, *, default: str) -> str:
    if raw is None:
        return default
    value = raw.strip()
    return value or default


def _as_int(raw: str | None, *, default: int) -> int:
    if raw is None:
        return default
    value = raw.strip()
    if not value:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _as_float(raw: str | None, *, default: float) -> float:
    if raw is None:
        return default
    value = raw.strip()
    if not value:
        return default
    try:
        return float(value)
    except ValueError:
        return default
