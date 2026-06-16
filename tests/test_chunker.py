from __future__ import annotations

from datetime import datetime
import re
import unittest

from rag_course.chunker import ChunkMetadata, ChunkerConfig, chunk_text
from rag_course.sources import _rfc3339_from_http_date


class ChunkerTests(unittest.TestCase):
    def test_chunk_text_stops_before_terminal_sections(self) -> None:
        text = (
            "1. Example Story.(1)\n\n"
            "This is the first sentence. This is the second sentence.\n\n"
            "Kleine Gedichte.\n\n"
            "A poem that should not be chunked.\n\n"
            "VOCABULARY.\n\n"
            "=word=, definition."
        )

        chunks = chunk_text(
            text,
            config=ChunkerConfig(target_sentences_per_chunk=1, max_tokens_per_chunk=250),
            base_metadata=ChunkMetadata(canonical_url="https://example.com/story.txt"),
        )

        self.assertEqual([chunk.text for chunk in chunks], ["This is the first sentence.", "This is the second sentence."])
        self.assertTrue(all(chunk.metadata.headline_1 == "Example Story" for chunk in chunks))

    def test_chunk_text_keeps_dialogue_lead_in_with_quote(self) -> None:
        text = (
            "1. Example Story.(1)\n\n"
            "Durch die Fensterscheiben sah der Vater einen Hasen vorbeispringen. Er sagte:\n\n"
            "»Mein Sohn, da ist ein Hase. Ich möchte ihn zum Mittagessen haben. Sieh, ob du ihn stehlen kannst.«"
        )

        chunks = chunk_text(
            text,
            config=ChunkerConfig(target_sentences_per_chunk=5, max_tokens_per_chunk=250),
            base_metadata=ChunkMetadata(canonical_url="https://example.com/story.txt"),
        )

        self.assertEqual(len(chunks), 1)
        self.assertIn("Er sagte:", chunks[0].text)
        self.assertIn("»Mein Sohn, da ist ein Hase.", chunks[0].text)

    def test_chunk_text_assigns_uuid_and_rfc3339_created_at(self) -> None:
        text = (
            "1. Example Story.(1)\n\n"
            "This is the first sentence. This is the second sentence."
        )

        chunks = chunk_text(
            text,
            config=ChunkerConfig(target_sentences_per_chunk=1, max_tokens_per_chunk=250),
            base_metadata=ChunkMetadata(canonical_url="https://example.com/story.txt"),
        )

        uuids = [chunk.metadata.uuid for chunk in chunks]
        self.assertTrue(all(uuid is not None for uuid in uuids))
        self.assertEqual(len(uuids), len(set(uuids)))
        self.assertTrue(all(re.fullmatch(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", uuid or "") for uuid in uuids))
        self.assertTrue(all(chunk.metadata.created_at is not None for chunk in chunks))
        self.assertTrue(all(datetime.fromisoformat(chunk.metadata.created_at or "") for chunk in chunks))

    def test_http_last_modified_is_normalized_to_rfc3339(self) -> None:
        normalized = _rfc3339_from_http_date("Sun, 14 Jun 2026 15:53:44 GMT")

        self.assertEqual(normalized, "2026-06-14T15:53:44+00:00")


if __name__ == "__main__":
    unittest.main()
