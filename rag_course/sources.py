"""Utilities for reading plain text from local files or HTTP URLs."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timezone
from pathlib import Path
from email.utils import parsedate_to_datetime
from urllib.parse import urlparse
from urllib.request import Request, urlopen


@dataclass(frozen=True, slots=True)
class TextSource:
    """Normalized source content and metadata."""

    text: str
    canonical_url: str | None = None
    media_type: str | None = None
    last_modified: str | None = None


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
    text = path.read_text(encoding="utf-8")
    return TextSource(text=text, canonical_url=path.as_uri(), media_type="text/plain")


def _read_http_source(url: str) -> TextSource:
    request = Request(url, headers={"User-Agent": "rag-course/1.0"})
    with urlopen(request) as response:
        raw_bytes = response.read()
        charset = response.headers.get_content_charset() or "utf-8"
        media_type = response.headers.get_content_type()
        last_modified = _rfc3339_from_http_date(response.headers.get("Last-Modified"))

    return TextSource(
        text=raw_bytes.decode(charset, errors="replace"),
        canonical_url=url,
        media_type=media_type,
        last_modified=last_modified,
    )


def _rfc3339_from_http_date(value: str | None) -> str | None:
    if value is None:
        return None

    parsed = parsedate_to_datetime(value)
    if parsed is None:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc).isoformat()
