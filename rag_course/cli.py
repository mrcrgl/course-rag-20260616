"""Command line interface for the educational CLI."""

from __future__ import annotations

import argparse
from typing import Sequence

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
    except ValueError as exc:
        parser.error(str(exc))

    parser.error(f"Unsupported command: {args.command}")
    return 2
