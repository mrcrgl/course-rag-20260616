"""Utilities for reading plain text and PDF sources from local files or HTTP URLs."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timezone
from io import BytesIO
from pathlib import Path
from email.utils import parsedate_to_datetime
import re
from urllib.parse import urlparse
from urllib.request import Request, urlopen


@dataclass(frozen=True, slots=True)
class TextSource:
    """Normalized source content and metadata."""

    text: str
    canonical_url: str | None = None
    media_type: str | None = None
    last_modified: str | None = None
    pages: list[str] | None = None


def read_text_source(source: str) -> TextSource:
    """Read text from a local path or HTTP(S) URL."""

    if _is_http_url(source):
        return _read_http_source(source)
    return _read_file_source(source)


def _is_http_url(value: str) -> bool:
    parsed = urlparse(value)
    return parsed.scheme in {"http", "https"}


def _read_file_source(path_value: str) -> TextSource:
    path = Path(path_value).expanduser().resolve()
    if path.suffix.lower() == ".pdf":
        return _read_pdf_bytes(
            path.read_bytes(),
            canonical_url=path.as_uri(),
            last_modified=None,
        )

    text = path.read_text(encoding="utf-8")
    return TextSource(text=text, canonical_url=path.as_uri(), media_type="text/plain")


def _read_http_source(url: str) -> TextSource:
    request = Request(url, headers={"User-Agent": "rag-course/1.0"})
    with urlopen(request) as response:
        raw_bytes = response.read()
        charset = response.headers.get_content_charset() or "utf-8"
        media_type = response.headers.get_content_type()
        last_modified = _rfc3339_from_http_date(response.headers.get("Last-Modified"))

    if _is_pdf_media_type(media_type) or url.lower().endswith(".pdf"):
        return _read_pdf_bytes(raw_bytes, canonical_url=url, last_modified=last_modified)

    return TextSource(
        text=raw_bytes.decode(charset, errors="replace"),
        canonical_url=url,
        media_type=media_type,
        last_modified=last_modified,
    )


def _read_pdf_bytes(raw_bytes: bytes, *, canonical_url: str, last_modified: str | None) -> TextSource:
    try:
        from pypdf import PdfReader
    except ImportError as exc:  # pragma: no cover - dependency failure is environment-specific
        raise RuntimeError("PDF support requires the pypdf package.") from exc

    reader = PdfReader(BytesIO(raw_bytes))
    pages: list[str] = []
    for page in reader.pages:
        extracted = page.extract_text() or ""
        pages.append(_normalize_pdf_page_text(extracted))

    text = "\n\n".join(page for page in pages if page.strip())
    return TextSource(
        text=text,
        canonical_url=canonical_url,
        media_type="application/pdf",
        last_modified=last_modified,
        pages=pages,
    )


def _is_pdf_media_type(media_type: str) -> bool:
    return media_type == "application/pdf" or media_type.endswith("+pdf")


def _normalize_pdf_page_text(text: str) -> str:
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    normalized = re.sub(r"-\n(?=\w)", "", normalized)
    normalized = re.sub(r"[ \t]+\n", "\n", normalized)
    normalized = re.sub(r"\n{3,}", "\n\n", normalized)
    return normalized.strip()


def _rfc3339_from_http_date(value: str | None) -> str | None:
    if value is None:
        return None

    parsed = parsedate_to_datetime(value)
    if parsed is None:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc).isoformat()
