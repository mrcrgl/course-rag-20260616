from __future__ import annotations

import unittest

from rag_course.chunker import ChunkMetadata, ChunkerConfig, chunk_text


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


if __name__ == "__main__":
    unittest.main()
