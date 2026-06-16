from __future__ import annotations

import io
from contextlib import redirect_stdout
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from rag_course.commands.query import run_query
from rag_course.config import AppConfig


class _FakeQdrantClient:
    def __init__(self, *, url: str, api_key: str | None = None) -> None:
        self.url = url
        self.api_key = api_key
        self.calls: list[dict[str, object]] = []

    def query_points(self, **kwargs: object) -> object:
        self.calls.append(kwargs)
        return SimpleNamespace(
            points=[
                SimpleNamespace(
                    id="00000000-0000-0000-0000-000000000001",
                    score=0.91,
                    payload={"text": "First matching chunk with enough details."},
                ),
                SimpleNamespace(
                    id="00000000-0000-0000-0000-000000000002",
                    score=0.73,
                    payload={"text": "Second matching chunk with more text than needed."},
                ),
            ]
        )


class QueryTests(unittest.TestCase):
    def test_run_query_embeds_prompt_and_prints_compact_blocks(self) -> None:
        fake_qdrant = _FakeQdrantClient(url="http://localhost:9333")

        with patch("rag_course.commands.query.build_client", return_value=object()) as build_client:
            with patch("rag_course.commands.query.create_embedding") as create_embedding:
                create_embedding.return_value = SimpleNamespace(vector=[0.1, 0.2, 0.3], model="text-embedding-3-small")
                with patch("rag_course.commands.query.QdrantClient", return_value=fake_qdrant):
                    buffer = io.StringIO()
                    with redirect_stdout(buffer):
                        results = run_query(
                            AppConfig(
                                qdrant_url="http://localhost:9333",
                                qdrant_collection_name="rag_chunks",
                                qdrant_vector_size=3,
                                openai_api_key="sk-test",
                            ),
                            "matching chunk",
                        )

        self.assertEqual(len(results), 2)
        build_client.assert_called_once()
        create_embedding.assert_called_once_with(
            build_client.return_value,
            text="matching chunk",
            model="text-embedding-3-small",
        )
        self.assertEqual(len(fake_qdrant.calls), 1)
        call = fake_qdrant.calls[0]
        self.assertEqual(call["collection_name"], "rag_chunks")
        self.assertEqual(call["query"], [0.1, 0.2, 0.3])
        self.assertEqual(call["limit"], 15)
        self.assertEqual(call["score_threshold"], 0.55)
        self.assertTrue(call["with_payload"])
        self.assertFalse(call["with_vectors"])

        output = buffer.getvalue()
        self.assertIn("score=0.9100", output)
        self.assertIn("id=00000000-0000-0000-0000-000000000001", output)
        self.assertIn("First matching chunk with enough details.", output)


if __name__ == "__main__":
    unittest.main()
