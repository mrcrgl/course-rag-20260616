from __future__ import annotations

import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

from rag_course.sources import read_text_source


class _FakePdfPage:
    def __init__(self, text: str) -> None:
        self._text = text

    def extract_text(self) -> str:
        return self._text


class _FakePdfReader:
    def __init__(self, _stream: object) -> None:
        self.pages = [_FakePdfPage("First page."), _FakePdfPage("Second page.")]


class SourceTests(unittest.TestCase):
    def test_read_text_source_detects_pdf_and_extracts_pages(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "sample.pdf"
            path.write_bytes(b"%PDF-1.4 fake")

            with patch("pypdf.PdfReader", return_value=_FakePdfReader(object())):
                source = read_text_source(str(path))

            self.assertEqual(source.media_type, "application/pdf")
            self.assertEqual(source.pages, ["First page.", "Second page."])
            self.assertIn("First page.", source.text)
            self.assertIn("Second page.", source.text)


if __name__ == "__main__":
    unittest.main()
