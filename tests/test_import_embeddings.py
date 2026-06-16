from __future__ import annotations

import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

from yaml import safe_dump, safe_load

from rag_course.commands.import_embeddings import run_import_embeddings
from rag_course.config import AppConfig


class _FakeQdrantClient:
    def __init__(self, *, url: str, api_key: str | None = None) -> None:
        self.url = url
        self.api_key = api_key
        self.collection_exists_calls: list[str] = []
        self.create_collection_calls: list[dict[str, object]] = []
        self.upload_points_calls: list[dict[str, object]] = []

    def collection_exists(self, *, collection_name: str) -> bool:
        self.collection_exists_calls.append(collection_name)
        return False

    def create_collection(self, **kwargs: object) -> None:
        self.create_collection_calls.append(kwargs)

    def upload_points(self, **kwargs: object) -> None:
        self.upload_points_calls.append(kwargs)


class ImportEmbeddingsTests(unittest.TestCase):
    def test_run_import_embeddings_creates_collection_and_prefixes_payload(self) -> None:
        input_payload = {
            "generated_at": "2026-06-16T00:00:00+00:00",
            "source": {"input": "chunks.yaml"},
            "chunker": {"target_sentences_per_chunk": 3},
            "embedding": {"model": "text-embedding-3-small", "batch_size": 2},
            "chunks": [
                {
                    "text": "first chunk",
                    "metadata": {
                        "uuid": "00000000-0000-0000-0000-000000000001",
                        "canonical_url": "https://example.com/story.txt",
                        "page_number": 1,
                        "headline_1": "Story",
                    },
                    "embedding": [0.1, 0.2, 0.3],
                },
                {
                    "text": "second chunk",
                    "metadata": {
                        "uuid": "00000000-0000-0000-0000-000000000002",
                        "canonical_url": "https://example.com/story.txt",
                        "page_number": 1,
                        "headline_1": "Story",
                    },
                    "embedding": [0.4, 0.5, 0.6],
                },
            ],
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "embedded.yaml"
            input_path.write_text(safe_dump(input_payload, sort_keys=False, allow_unicode=True), encoding="utf-8")

            fake_client = _FakeQdrantClient(url="http://localhost:9333")

            with patch("rag_course.commands.import_embeddings.QdrantClient", return_value=fake_client):
                imported = run_import_embeddings(
                    AppConfig(
                        qdrant_url="http://localhost:9333",
                        qdrant_collection_name="rag_chunks",
                        qdrant_vector_size=3,
                    ),
                    str(input_path),
                )

            self.assertEqual(imported, 2)
            self.assertEqual(fake_client.collection_exists_calls, ["rag_chunks"])
            self.assertEqual(len(fake_client.create_collection_calls), 1)
            self.assertEqual(len(fake_client.upload_points_calls), 1)

            create_kwargs = fake_client.create_collection_calls[0]
            self.assertEqual(create_kwargs["collection_name"], "rag_chunks")
            self.assertEqual(create_kwargs["vectors_config"].size, 3)

            upload_kwargs = fake_client.upload_points_calls[0]
            self.assertEqual(upload_kwargs["collection_name"], "rag_chunks")
            self.assertTrue(upload_kwargs["wait"])

            points = upload_kwargs["points"]
            self.assertEqual([point.id for point in points], [
                "00000000-0000-0000-0000-000000000001",
                "00000000-0000-0000-0000-000000000002",
            ])
            self.assertEqual([point.payload["text"] for point in points], ["first chunk", "second chunk"])
            self.assertNotIn("meta_uuid", points[0].payload)
            self.assertEqual(points[0].payload["meta_canonical_url"], "https://example.com/story.txt")
            self.assertEqual(points[0].payload["meta_page_number"], 1)
            self.assertEqual(points[0].payload["meta_headline_1"], "Story")

    def test_run_import_embeddings_rejects_vector_size_mismatch(self) -> None:
        input_payload = {
            "chunks": [
                {
                    "text": "first chunk",
                    "metadata": {"uuid": "00000000-0000-0000-0000-000000000001"},
                    "embedding": [0.1, 0.2],
                }
            ]
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "embedded.yaml"
            input_path.write_text(safe_dump(input_payload, sort_keys=False, allow_unicode=True), encoding="utf-8")

            fake_client = _FakeQdrantClient(url="http://localhost:9333")

            with patch("rag_course.commands.import_embeddings.QdrantClient", return_value=fake_client):
                with self.assertRaises(ValueError):
                    run_import_embeddings(
                        AppConfig(
                            qdrant_url="http://localhost:9333",
                            qdrant_collection_name="rag_chunks",
                            qdrant_vector_size=3,
                        ),
                        str(input_path),
                    )


if __name__ == "__main__":
    unittest.main()
