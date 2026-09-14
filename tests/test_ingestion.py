from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from summarizer.ingestion import (
    EmptySourceError,
    SourceDecodeError,
    SourceReadError,
    ingest_text,
    normalize_source_text,
    read_source,
)


def test_normalization_preserves_structure_and_unicode() -> None:
    source = "\ufeff\r\n# H  \r\n\r\n  - café\t \rNext\t\r\n\r\n"

    normalized = normalize_source_text(source)

    assert normalized == "# H\n\n  - café\nNext"


def test_source_identity_is_based_on_canonical_utf8_text() -> None:
    document = ingest_text("\ufeff# H\r\n\r\n  - café  \r\n")

    assert document.text == "# H\n\n  - café"
    assert document.source_id == ingest_text(document.text).source_id


def test_source_document_is_immutable() -> None:
    document = ingest_text("content")

    with pytest.raises(FrozenInstanceError):
        document.text = "changed"  # type: ignore[misc]


@pytest.mark.parametrize("source", ["", " \t\r\n\n"])
def test_empty_canonical_source_is_rejected(source: str) -> None:
    with pytest.raises(EmptySourceError, match="empty"):
        ingest_text(source)


def test_read_source_decodes_utf8_and_records_path(tmp_path: Path) -> None:
    path = tmp_path / "source.txt"
    path.write_bytes("Résumé\r\n".encode("utf-8"))

    document = read_source(path)

    assert document.text == "Résumé"
    assert document.path == path


def test_decode_error_mentions_path_and_encoding(tmp_path: Path) -> None:
    path = tmp_path / "invalid.txt"
    path.write_bytes(b"\xff")

    with pytest.raises(SourceDecodeError) as caught:
        read_source(path)

    assert str(path) in str(caught.value)
    assert "utf-8" in str(caught.value).lower()
    assert isinstance(caught.value.__cause__, UnicodeDecodeError)


def test_read_error_mentions_path_and_encoding(tmp_path: Path) -> None:
    path = tmp_path / "missing.txt"

    with pytest.raises(SourceReadError) as caught:
        read_source(path)

    assert str(path) in str(caught.value)
    assert "utf-8" in str(caught.value).lower()
    assert isinstance(caught.value.__cause__, OSError)


@pytest.mark.parametrize(
    "source",
    ["\xa0", "\u3000\u3000", "\x0b", "\x0c", "\xa0\t\xa0"],
)
def test_unicode_whitespace_only_input_is_rejected(source: str) -> None:
    with pytest.raises(EmptySourceError, match="empty"):
        ingest_text(source)


def test_interior_nbsp_in_real_prose_is_accepted_with_unchanged_offsets() -> None:
    # U+00A0 is legitimate content inside prose; only the surrounding whitespace
    # should be stripped by normalize_source_text, not interior NBSP characters.
    prose = "Hello\xa0world"
    document = ingest_text(prose)
    assert "\xa0" in document.text
    assert document.text.index("\xa0") == prose.index("\xa0")
