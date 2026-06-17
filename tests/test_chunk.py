from __future__ import annotations

import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

from yaml import safe_load

from rag_course.chunker import Chunk, ChunkMetadata
from rag_course.commands.chunk import run_chunk
from rag_course.config import AppConfig
from rag_course.sources import TextSource


class ChunkCommandTests(unittest.TestCase):
    def test_run_chunk_selects_legal_pdf_chunker(self) -> None:
        fake_chunks = [
            Chunk(
                text="Legal text chunk.",
                metadata=ChunkMetadata(
                    uuid="00000000-0000-0000-0000-000000000001",
                    page_number=12,
                    canonical_url="https://example.com/bgb.pdf",
                ),
            )
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "chunks.yaml"
            with patch(
                "rag_course.commands.chunk.read_text_source",
                return_value=TextSource(
                    text="ignored",
                    canonical_url="https://example.com/bgb.pdf",
                    media_type="application/pdf",
                    pages=["page one"],
                ),
            ) as read_text_source:
                with patch("rag_course.commands.chunk.chunk_legal_pdf", return_value=fake_chunks) as chunk_legal_pdf:
                    result_path = run_chunk(
                        AppConfig(embedding_model="text-embedding-3-small"),
                        "https://example.com/bgb.pdf",
                        str(output_path),
                        max_tokens_per_chunk=250,
                        min_words_per_chunk=2,
                        overlap_sentences=1,
                        target_sentences_per_chunk=3,
                        chunker="legal-pdf",
                        embedding_model="text-embedding-3-small",
                    )

            self.assertEqual(result_path, output_path.resolve())
            read_text_source.assert_called_once_with("https://example.com/bgb.pdf")
            chunk_legal_pdf.assert_called_once()

            payload = safe_load(output_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["chunker"]["name"], "legal-pdf")
            self.assertEqual(payload["chunker"]["type"], "legal-pdf-page-window")
            self.assertEqual(payload["chunks"][0]["metadata"]["page_number"], 12)


if __name__ == "__main__":
    unittest.main()
