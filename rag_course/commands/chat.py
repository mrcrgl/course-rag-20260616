"""CLI command for interactive chat with the LLM."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import logging
from pathlib import Path
from time import perf_counter
from typing import Any
from uuid import uuid4

from openai import OpenAI

from rag_course.commands.query import RagResult, build_rag_context, retrieve_rag_results
from rag_course.config import AppConfig
from rag_course.embeddings import build_client


logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ChatUsage:
    """Token counts reported by the streaming API."""

    prompt_tokens: int | None
    completion_tokens: int | None
    total_tokens: int | None


def run_chat(config: AppConfig) -> int:
    """Run an interactive chat loop and persist per-turn audit logs."""

    base_system_prompt = _read_system_prompt()
    client = build_client(config)
    conversation: list[dict[str, str]] = []
    session_id = _session_id()
    audit_dir = _project_root() / "auditlog"
    audit_dir.mkdir(parents=True, exist_ok=True)

    turn = 0
    while True:
        try:
            user_prompt = _prompt_text("You: ")
        except EOFError:
            print()
            break

        if not user_prompt:
            continue
        if user_prompt.strip().lower() in {"exit", "quit", ":q"}:
            break

        turn += 1
        rag_results = retrieve_rag_results(
            config,
            user_prompt,
            top_k=config.rag_top_k,
            score_threshold=config.rag_score_threshold,
        )
        rag_context = build_rag_context(
            rag_results,
            total_token_budget=config.rag_context_token_budget_total,
            per_entry_token_budget=config.rag_context_token_budget_per_entry,
        )
        effective_system_prompt = _compose_system_prompt(base_system_prompt, rag_context)

        request_messages = [
            {"role": "system", "content": effective_system_prompt},
            *conversation,
            {"role": "user", "content": user_prompt},
        ]
        response_text, usage, latency_seconds = _stream_response(
            client,
            config=config,
            messages=request_messages,
        )
        response_text = _append_response_footer(response_text, rag_results)

        conversation.append({"role": "user", "content": user_prompt})
        conversation.append({"role": "assistant", "content": response_text})

        audit_path = _write_audit_entry(
            audit_dir,
            session_id=session_id,
            turn=turn,
            base_system_prompt=base_system_prompt,
            rag_context=rag_context,
            retrieved_results=rag_results,
            request_messages=request_messages,
            response_text=response_text,
            usage=usage,
            latency_seconds=latency_seconds,
            model=config.chat_model,
        )
        logger.debug(
            "chat turn=%s prompt_tokens=%s completion_tokens=%s total_tokens=%s latency_ms=%.0f rag_results=%s audit=%s",
            turn,
            usage.prompt_tokens if usage else None,
            usage.completion_tokens if usage else None,
            usage.total_tokens if usage else None,
            latency_seconds * 1000.0,
            len(rag_results),
            audit_path,
        )

    return turn


def _compose_system_prompt(base_system_prompt: str, rag_context: str) -> str:
    if not rag_context:
        return base_system_prompt
    return "\n\n".join([base_system_prompt, rag_context]).strip()


def _stream_response(
    client: OpenAI,
    *,
    config: AppConfig,
    messages: list[dict[str, str]],
) -> tuple[str, ChatUsage | None, float]:
    start = perf_counter()
    stream = client.chat.completions.create(
        model=config.chat_model,
        messages=messages,
        stream=True,
        stream_options={"include_usage": True},
    )

    response_parts: list[str] = []
    usage: ChatUsage | None = None
    print("Assistant: ", end="", flush=True)
    for chunk in stream:
        chunk_usage = getattr(chunk, "usage", None)
        if chunk_usage is not None:
            usage = ChatUsage(
                prompt_tokens=getattr(chunk_usage, "prompt_tokens", None),
                completion_tokens=getattr(chunk_usage, "completion_tokens", None),
                total_tokens=getattr(chunk_usage, "total_tokens", None),
            )
        for choice in getattr(chunk, "choices", []):
            delta = getattr(choice, "delta", None)
            content = getattr(delta, "content", None) if delta is not None else None
            if content:
                response_parts.append(content)
                print(content, end="", flush=True)

    print()
    latency_seconds = perf_counter() - start
    response_text = "".join(response_parts).strip()
    return response_text, usage, latency_seconds


def _write_audit_entry(
    audit_dir: Path,
    *,
    session_id: str,
    turn: int,
    base_system_prompt: str,
    rag_context: str,
    retrieved_results: list[RagResult],
    request_messages: list[dict[str, str]],
    response_text: str,
    usage: ChatUsage | None,
    latency_seconds: float,
    model: str,
) -> Path:
    timestamp = datetime.now(timezone.utc).isoformat()
    path = audit_dir / f"{session_id}_turn_{turn:03d}.md"
    content = {
        "session_id": session_id,
        "turn": turn,
        "timestamp": timestamp,
        "model": model,
        "latency_seconds": round(latency_seconds, 6),
        "usage": {
            "prompt_tokens": usage.prompt_tokens if usage else None,
            "completion_tokens": usage.completion_tokens if usage else None,
            "total_tokens": usage.total_tokens if usage else None,
        },
        "base_system_prompt": base_system_prompt,
        "rag_context": rag_context,
        "retrieved_results": retrieved_results,
        "prompt_chain": request_messages,
        "response": response_text,
    }
    path.write_text(_render_audit_markdown(content), encoding="utf-8")
    return path


def _append_response_footer(response_text: str, retrieved_results: list[RagResult]) -> str:
    footer = _render_response_footer(retrieved_results)
    if not footer or "Response Footer:" in response_text:
        return response_text

    print(footer)
    if not response_text.strip():
        return footer
    return "\n\n".join([response_text.rstrip(), footer])


def _render_response_footer(retrieved_results: list[RagResult]) -> str:
    sources: list[str] = []
    seen: set[tuple[str, str]] = set()

    for result in retrieved_results:
        canonical_url = _metadata_lookup(result.metadata, "canonical_url")
        page_number = _metadata_lookup(result.metadata, "page_number")
        if canonical_url is None or page_number is None:
            continue

        source_key = (str(canonical_url), str(page_number))
        if source_key in seen:
            continue
        seen.add(source_key)
        sources.append(f"- [{len(sources) + 1}]({canonical_url}#page={page_number})")

    if not sources:
        return ""

    return "\n".join(
        [
            "Response Footer:",
            "",
            "---",
            "",
            "Sources:",
            *sources,
        ]
    )


def _metadata_lookup(metadata: dict[str, Any], key: str) -> Any:
    if key in metadata:
        return metadata[key]
    prefixed_key = f"meta_{key}"
    if prefixed_key in metadata:
        return metadata[prefixed_key]
    return None


def _render_audit_markdown(content: dict[str, Any]) -> str:
    lines = [
        f"# Chat Turn {content['turn']}",
        "",
        f"- session_id: {content['session_id']}",
        f"- timestamp: {content['timestamp']}",
        f"- model: {content['model']}",
        f"- latency_seconds: {content['latency_seconds']}",
        f"- prompt_tokens: {content['usage']['prompt_tokens']}",
        f"- completion_tokens: {content['usage']['completion_tokens']}",
        f"- total_tokens: {content['usage']['total_tokens']}",
        "",
        "## Prompt Chain",
    ]
    for message in content["prompt_chain"]:
        role = message.get("role", "unknown")
        text = str(message.get("content", "")).rstrip()
        lines.extend(
            [
                f"### {role}",
                "```text",
                text,
                "```",
                "",
            ]
        )
    if content["retrieved_results"]:
        lines.extend(["## Retrieved Results", ""])
        for index, result in enumerate(content["retrieved_results"], start=1):
            lines.extend(
                [
                    f"### Result {index}",
                    f"- score: {result.score:.4f}",
                    f"- id: {result.point_id}",
                    "```text",
                    result.text,
                    "```",
                    "",
                ]
            )
            if result.metadata:
                lines.append("```yaml")
                lines.extend(_render_dict_as_lines(_result_metadata_for_log(result.metadata)))
                lines.append("```")
                lines.append("")
    if content["rag_context"]:
        lines.extend(
            [
                "## RAG Context",
                "```text",
                str(content["rag_context"]).rstrip(),
                "```",
                "",
            ]
        )
    lines.extend(
        [
            "## Response",
            "```text",
            str(content["response"]).rstrip(),
            "```",
            "",
            "## Base System Prompt",
            "```text",
            str(content["base_system_prompt"]).rstrip(),
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def _result_metadata_for_log(metadata: dict[str, Any]) -> dict[str, Any]:
    return {f"meta_{key}": value for key, value in metadata.items()}


def _render_dict_as_lines(value: dict[str, Any], *, indent: int = 0) -> list[str]:
    lines: list[str] = []
    prefix = "  " * indent
    for key, item in value.items():
        if isinstance(item, dict):
            lines.append(f"{prefix}{key}:")
            lines.extend(_render_dict_as_lines(item, indent=indent + 1))
        elif isinstance(item, list):
            lines.append(f"{prefix}{key}:")
            for entry in item:
                if isinstance(entry, dict):
                    lines.extend(_render_dict_as_lines(entry, indent=indent + 1))
                else:
                    lines.append(f"{prefix}  - {entry}")
        else:
            lines.append(f"{prefix}{key}: {item}")
    return lines


def _read_system_prompt() -> str:
    path = _project_root() / "config" / "system_prompt.md"
    if not path.exists():
        raise ValueError(f"Missing system prompt file: {path}")
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"System prompt file is empty: {path}")
    return text


def _session_id() -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}_{uuid4().hex[:8]}"


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _prompt_text(prompt: str) -> str:
    try:
        import readline  # noqa: F401
    except ImportError:
        pass
    return input(prompt).strip()
