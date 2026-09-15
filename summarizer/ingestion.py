"""Canonical source ingestion with stable provenance."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path


class SourceIngestionError(ValueError):
    """Base class for source-ingestion failures."""


class SourceReadError(SourceIngestionError):
    """Raised when a source cannot be read."""


class SourceDecodeError(SourceIngestionError):
    """Raised when source bytes are not valid in the requested encoding."""


class EmptySourceError(SourceIngestionError):
    """Raised when canonicalization leaves no source content."""


@dataclass(frozen=True)
class SourceDocument:
    """Canonical text and the identity derived from its UTF-8 bytes."""

    text: str
    source_id: str
    path: Path | None = None


def normalize_source_text(text: str) -> str:
    """Normalize transport whitespace without flattening document structure."""
    normalized = text.removeprefix("\ufeff").replace("\r\n", "\n").replace("\r", "\n")
    lines = [line.rstrip(" \t") for line in normalized.split("\n")]
    while lines and not lines[0]:
        lines.pop(0)
    while lines and not lines[-1]:
        lines.pop()
    return "\n".join(lines)


def ingest_text(text: str, *, path: Path | None = None) -> SourceDocument:
    """Create an immutable document from canonicalized source text."""
    canonical_text = normalize_source_text(text)
    if not canonical_text.strip():
        raise EmptySourceError("source is empty after normalization")
    source_id = sha256(canonical_text.encode("utf-8")).hexdigest()
    return SourceDocument(text=canonical_text, source_id=source_id, path=path)


def read_source(path: str | Path, *, encoding: str = "utf-8") -> SourceDocument:
    """Read and canonicalize a text source from disk."""
    source_path = Path(path)
    try:
        text = source_path.read_text(encoding=encoding)
    except UnicodeDecodeError as error:
        raise SourceDecodeError(
            f"could not decode {source_path} using {encoding}"
        ) from error
    except OSError as error:
        raise SourceReadError(
            f"could not read {source_path} using {encoding}"
        ) from error
    return ingest_text(text, path=source_path)
