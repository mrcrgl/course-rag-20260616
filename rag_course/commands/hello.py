"""Greeting-related commands."""

from __future__ import annotations

from dataclasses import dataclass

from rag_course.config import AppConfig


@dataclass(frozen=True, slots=True)
class HelloResult:
    message: str


def run(config: AppConfig, name: str | None = None) -> HelloResult:
    """Build a readable greeting message."""

    target = name or config.default_name
    message = f"Hello, {target}!"
    return HelloResult(message=message)

