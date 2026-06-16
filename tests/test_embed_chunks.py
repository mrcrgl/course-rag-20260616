from __future__ import annotations

import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

from yaml import safe_dump, safe_load

from rag_course.commands.embed_chunks import run_embed_chunks
from rag_course.config import AppConfig


class EmbedChunksTests(unittest.TestCase):
    def test_run_embed_chunks_batches_texts_and_writes_embeddings(self) -> None:
        input_payload = {
            "generated_at": "2026-06-16T00:00:00+00:00",
            "source": {"input": "source.txt", "canonical_url": "https://example.com/source.txt"},
            "chunker": {"target_sentences_per_chunk": 3},
            "chunks": [
                {"text": "first chunk", "metadata": {"uuid": "00000000-0000-0000-0000-000000000001"}},
                {"text": "second chunk", "metadata": {"uuid": "00000000-0000-0000-0000-000000000002"}},
                {"text": "third chunk", "metadata": {"uuid": "00000000-0000-0000-0000-000000000003"}},
            ],
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "chunks.yaml"
            output_path = Path(tmpdir) / "embedded.yaml"
            input_path.write_text(safe_dump(input_payload, sort_keys=False, allow_unicode=True), encoding="utf-8")

            calls: list[list[str]] = []

            def fake_embeddings(_client: object, *, texts: list[str], model: str):
                calls.append(list(texts))

                class Result:
                    def __init__(self) -> None:
                        self.vectors = [[float(len(text))] for text in texts]
                        self.model = model

                return Result()

            with patch("rag_course.commands.embed_chunks.build_client", return_value=object()) as build_client:
                with patch("rag_course.commands.embed_chunks.create_embeddings", side_effect=fake_embeddings):
                    result_path = run_embed_chunks(
                        AppConfig(embedding_model="text-embedding-3-small", openai_api_key="sk-test"),
                        str(input_path),
                        str(output_path),
                        batch_size=2,
                    )

            self.assertEqual(result_path, output_path.resolve())
            self.assertEqual(calls, [["first chunk", "second chunk"], ["third chunk"]])
            build_client.assert_called_once()

            output_payload = safe_load(output_path.read_text(encoding="utf-8"))
            self.assertEqual(output_payload["embedding"]["batch_size"], 2)
            self.assertEqual(output_payload["embedding"]["model"], "text-embedding-3-small")
            self.assertEqual(
                [chunk["embedding"] for chunk in output_payload["chunks"]],
                [[11.0], [12.0], [11.0]],
            )


if __name__ == "__main__":
    unittest.main()
