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
    ]
    return "\n".join(lines)

