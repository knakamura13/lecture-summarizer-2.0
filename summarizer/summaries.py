from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, field_validator

# Identifies the record shape for cache keys and audit artifacts. Bump it
# whenever a change would make a previously stored record invalid or mean
# something different.
LEAF_SCHEMA_VERSION = "leaf/2"

# A quote is a pull-quote, not a transcription. Both caps bound how much a
# single node's quotations can inflate `serialize_child` (summarizer/merge.py)
# and trip the merge budget check. The per-quote cap applies everywhere an
# `EvidenceItem.quote` appears, since any one of them serializes into the
# node; the per-node cap applies only to `SummaryNode.quotations`, the
# "salient quotations" the master plan bounds. Shared here so the leaf and
# merge prompts state the same limit the validators enforce.
MAX_QUOTE_CHARS = 500
MAX_QUOTATIONS_PER_NODE = 5


class ContentKind(str, Enum):
    """What kind of thing a content unit asserts.

    Deliberately small and genre-neutral: the same vocabulary has to describe
    an article, a transcript, a report, and a narrative.
    """

    FACT = "fact"
    CLAIM = "claim"
    DEFINITION = "definition"
    PROCEDURE = "procedure"
    EXAMPLE = "example"
    OTHER = "other"


class _Record(BaseModel):
    """Base for every structured record built from model output.

    Two settings are load-bearing rather than stylistic. `extra="forbid"` is
    what makes the generated schema emit `additionalProperties: false`, and
    declaring fields without defaults is what makes the generated schema mark
    them required. Together they make `model_json_schema()` directly usable as
    an OpenAI strict schema, so the schema sent to a provider cannot drift from
    the validator that checks the response. Do not add field defaults here.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")


def _reject_blank(value: str) -> str:
    if not value.strip():
        raise ValueError("must not be blank")
    return value


def _empty_when_null(value: Any) -> Any:
    """Treat a null collection as an absent one.

    Only the strict OpenAI path guarantees a present, non-null key. Ollama's
    format argument constrains decoding on a best-effort basis, so a model with
    nothing to say about a field may emit null instead of an empty array.
    """
    return () if value is None else value


class EvidenceItem(_Record):
    """A pointer from an assertion back to the segment that supports it."""

    segment_id: str
    quote: str | None

    _reject_blank_segment_id = field_validator("segment_id")(_reject_blank)

    @field_validator("quote")
    @classmethod
    def _validate_quote(cls, value: str | None) -> str | None:
        """Absent means null, not empty. Also enforces a length cap.

        A blank quote would otherwise pass a verbatim check trivially, since
        every string contains the empty string, and code reading `quote is not
        None` as "has a quotation" would get nothing. The length cap bounds
        one quote's contribution to its node's serialized size: a model asked
        to copy verbatim has no natural stopping point.
        """
        if value is not None:
            if not value.strip():
                raise ValueError("quote must be null rather than blank")
            if len(value) > MAX_QUOTE_CHARS:
                raise ValueError(f"quote must not exceed {MAX_QUOTE_CHARS} characters")
        return value


class ContentUnit(_Record):
    """One assertion carried forward for later merging."""

    text: str
    kind: ContentKind
    evidence: tuple[EvidenceItem, ...]
    qualification: str | None
    uncertain: bool

    _reject_blank_text = field_validator("text")(_reject_blank)
    _empty_evidence = field_validator("evidence", mode="before")(_empty_when_null)


class GroundedAnnotation(_Record):
    """A retained qualification or contradiction with explicit support."""

    text: str
    evidence: tuple[EvidenceItem, ...]

    _reject_blank_text = field_validator("text")(_reject_blank)
    _empty_evidence = field_validator("evidence", mode="before")(_empty_when_null)


class SummaryNode(_Record):
    """A summary of one region of the source.

    `level` is zero for a leaf. Later merge levels reuse this record rather
    than introducing a parallel one, so the hierarchy has a single central
    type.
    """

    summary: str
    content_units: tuple[ContentUnit, ...]
    entities: tuple[str, ...]
    qualifications: tuple[GroundedAnnotation, ...]
    contradictions: tuple[GroundedAnnotation, ...]
    quotations: tuple[EvidenceItem, ...]
    provenance: tuple[str, ...]
    level: int

    _reject_blank_summary = field_validator("summary")(_reject_blank)
    _empty_collections = field_validator(
        "content_units",
        "entities",
        "qualifications",
        "contradictions",
        "quotations",
        "provenance",
        mode="before",
    )(_empty_when_null)

    @field_validator("level")
    @classmethod
    def _reject_negative_level(cls, value: int) -> int:
        if value < 0:
            raise ValueError("level must not be negative")
        return value

    @field_validator("quotations")
    @classmethod
    def _validate_quotation_count(cls, value: tuple[EvidenceItem, ...]) -> tuple[EvidenceItem, ...]:
        if len(value) > MAX_QUOTATIONS_PER_NODE:
            raise ValueError(f"must not exceed {MAX_QUOTATIONS_PER_NODE} quotations")
        return value


def leaf_summary_schema() -> dict[str, Any]:
    """Return the JSON Schema a provider is asked to produce for a leaf."""
    return SummaryNode.model_json_schema()
