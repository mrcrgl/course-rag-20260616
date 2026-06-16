"""Command line interface for the educational CLI."""

from __future__ import annotations

import argparse
from typing import Sequence

from rag_course.commands.chunk import run_chunk
from rag_course.commands.embeddings import run_embed, run_similarity
from rag_course.commands.hello import run as run_hello
from rag_course.commands.status import format_status
from rag_course.config import load_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rag-course",
        description="Educational CLI for demonstrating a maintainable Python project layout.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    hello_parser = subparsers.add_parser("hello", help="Print a greeting.")
    hello_parser.add_argument("name", nargs="?", help="Optional name to greet.")

    subparsers.add_parser("status", help="Show the loaded configuration.")

    embed_parser = subparsers.add_parser("embed", help="Create an embedding for a text.")
    embed_parser.add_argument("text", nargs="+", help="Text to embed.")

    subparsers.add_parser(
        "similarity",
        help="Prompt for two texts and print their cosine similarity.",
    )

    chunk_parser = subparsers.add_parser(
        "chunk",
        help="Read a local file or URL and write chunk metadata to YAML.",
    )
    chunk_parser.add_argument("source", help="Local file path or http(s) URL.")
    chunk_parser.add_argument("output", help="YAML file to write.")
    chunk_parser.add_argument(
        "--max-tokens",
        type=int,
        default=250,
        help="Maximum estimated tokens per chunk.",
    )
    chunk_parser.add_argument(
        "--min-words",
        type=int,
        default=2,
        help="Minimum number of words required for a chunk to be kept.",
    )
    chunk_parser.add_argument(
        "--overlap-sentences",
        type=int,
        default=1,
        help="Number of sentences to overlap between chunks.",
    )
    chunk_parser.add_argument(
        "--target-sentences",
        type=int,
        default=3,
        help="Target number of sentences per chunk.",
    )
    chunk_parser.add_argument(
        "--canonical-url",
        help="Override the canonical URL stored in chunk metadata.",
    )
    chunk_parser.add_argument("--author", help="Optional author metadata.")
    chunk_parser.add_argument(
        "--pii-classification",
        help="Optional PII classification metadata.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    config = load_config()

    try:
        if args.command == "hello":
            result = run_hello(config, name=args.name)
            print(result.message)
            return 0

        if args.command == "status":
            print(format_status(config))
            return 0

        if args.command == "embed":
            text = " ".join(args.text).strip()
            if not text:
                parser.error("embed requires non-empty text.")
            run_embed(config, text)
            return 0

        if args.command == "similarity":
            run_similarity(config)
            return 0

        if args.command == "chunk":
            run_chunk(
                config,
                args.source,
                args.output,
                max_tokens_per_chunk=args.max_tokens,
                min_words_per_chunk=args.min_words,
                overlap_sentences=args.overlap_sentences,
                target_sentences_per_chunk=args.target_sentences,
                canonical_url=args.canonical_url,
                author=args.author,
                pii_classification=args.pii_classification,
                embedding_model=config.embedding_model,
            )
            return 0
    except ValueError as exc:
        parser.error(str(exc))

    parser.error(f"Unsupported command: {args.command}")
    return 2
