"""Versioned, secret-safe audit artifacts for completed summary runs."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from math import isfinite
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import TYPE_CHECKING, Annotated, Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    TypeAdapter,
    field_validator,
    model_validator,
)

from summarizer.hierarchy import TreeNode
from summarizer.providers.base import GenerationResult
from summarizer.safety import redact_text
from summarizer.segmentation import SourceSegment

if TYPE_CHECKING:
    from summarizer.verification import VerificationResult


AUDIT_SCHEMA_VERSION = "audit/2"
AUDIT_SCHEMA_VERSION_V3 = "audit/3"
AUDIT_SCHEMA_VERSION_V4 = "audit/4"


class AuditError(ValueError):
    """An audit record is incomplete, inconsistent, or cannot be written."""


_CLOSED_CODE = re.compile(r"^[a-z][a-z0-9_]*$")
_AUDIT_WORK_ID = re.compile(
    r"^(?:[DS]\d{6}|L\d+N\d{4}|M(?:\d{6}|\d+N\d{4})|V\d{2}(?:[CS]\d{6})?|"
    r"editorial-final|segmentation)$"
)
_VERIFICATION_SPAN_ID = re.compile(r"^V\d{2}S\d{6}$")
_VERIFICATION_CLAIM_ID = re.compile(r"^V\d{2}C\d{6}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SEGMENT_ID = re.compile(r"^[DS]\d{6}$")
_NODE_ID = re.compile(r"^L\d+N\d{4}$")
_SAFE_IDENTITY = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_SAFE_MODEL_IDENTITY = re.compile(
    r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}(?::[A-Za-z0-9][A-Za-z0-9._-]{0,127})?$"
)
_VERIFICATION_PROMPT_VERSION = frozenset(
    {
        "verification-decomposition/1",
        "verification-classification/1",
        "verification-repair/1",
    }
)
_FINISH_STATUSES = frozenset(
    {
        "completed",
        "stop",
        "length",
        "content_filter",
        "tool_calls",
        "cancelled",
        "error",
        "unknown",
    }
)
_CACHE_HIT_CODES = frozenset({"hit"})
_CACHE_MISS_CODES = frozenset({"missing", "corrupt", "wrong_version", "incompatible"})
_CACHE_INVALIDATION_CODES = frozenset(
    {
        "source_changed",
        "prompt_changed",
        "schema_changed",
        "model_changed",
        "behavior_changed",
    }
)
_RETRY_FAILURE_CODES = frozenset(
    {"timeout", "rate_limit", "connection", "server", "transient"}
)


def _audit_identity(value: str) -> str:
    if _SAFE_IDENTITY.fullmatch(value) and redact_text(value) == value:
        return value
    raise ValueError("audit identity must be a non-secret identifier")


def _audit_optional_identity(value: str | None) -> str | None:
    return None if value is None else _audit_identity(value)


def _audit_model_identity(value: str) -> str:
    if _SAFE_MODEL_IDENTITY.fullmatch(value) and redact_text(value) == value:
        return value
    raise ValueError("audit model must be a non-secret model identity")


def _audit_optional_model_identity(value: str | None) -> str | None:
    return None if value is None else _audit_model_identity(value)


def _audit_finish_status(value: str | None) -> str | None:
    if value is None or value in _FINISH_STATUSES:
        return value
    raise ValueError("audit finish_status must be a closed status")


def _audit_prompt_version(value: str) -> str:
    if value in _VERIFICATION_PROMPT_VERSION:
        return value
    raise ValueError("audit prompt_version must be a supported version")


def _audit_source_id(value: str) -> str:
    if _SHA256.fullmatch(value):
        return value
    raise ValueError("audit source_id must be a sha256 identity")


def _audit_segment_id(value: str) -> str:
    if _SEGMENT_ID.fullmatch(value):
        return value
    raise ValueError("audit segment_id must be a segment identity")


def _audit_segment_ids(values: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(_audit_segment_id(value) for value in values)


def _audit_node_id(value: str) -> str:
    if _NODE_ID.fullmatch(value):
        return value
    raise ValueError("audit node_id must be a tree identity")


def _audit_node_ids(values: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(_audit_node_id(value) for value in values)


def _audit_closed_codes(
    values: object, *, allowed: frozenset[str], label: str
) -> tuple[str, ...]:
    if not isinstance(values, (list, tuple)) or any(
        not isinstance(value, str) or value not in allowed for value in values
    ):
        raise ValueError(f"{label} must contain supported closed codes")
    return tuple(values)


def _audit_work_id(value: str) -> str:
    if _AUDIT_WORK_ID.fullmatch(value):
        return value
    raise ValueError("audit work_id must be a safe stable identifier")


def _audit_integer(value: object, *, minimum: int, label: str) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{label} must be a non-boolean integer of at least {minimum}")
    return value


class _AuditRecord(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class AuditSegment(_AuditRecord):
    segment_id: str
    source_id: str
    order: int
    core_start: int
    core_end: int
    context_start: int
    context_end: int
    core_token_count: int
    token_count: int
    leading_overlap_tokens: int
    trailing_overlap_tokens: int
    boundary_kind: Literal[
        "heading", "paragraph", "list", "sentence", "hard", "document"
    ]

    _valid_segment_id = field_validator("segment_id")(_audit_segment_id)
    _valid_source_id = field_validator("source_id")(_audit_source_id)


class AuditCitation(_AuditRecord):
    segment_id: str
    source_id: str
    order: int

    _valid_segment_id = field_validator("segment_id")(_audit_segment_id)
    _valid_source_id = field_validator("source_id")(_audit_source_id)


class AuditUsage(_AuditRecord):
    provider: str
    model: str
    input_tokens: int | None
    output_tokens: int | None
    finish_status: str | None

    _valid_provider = field_validator("provider")(_audit_identity)
    _valid_model = field_validator("model")(_audit_model_identity)
    _valid_finish_status = field_validator("finish_status")(_audit_finish_status)


class AuditVerificationSpan(_AuditRecord):
    span_id: str
    ordinal: int
    start: int
    end: int
    content_hash: str


class AuditVerificationClaim(_AuditRecord):
    claim_id: str
    span_id: str
    ordinal: int
    is_fallback: bool


class AuditVerificationSelection(_AuditRecord):
    claim_id: str
    selected_ids: tuple[str, ...]
    examined_ids: tuple[str, ...]
    omitted_ids: tuple[str, ...]
    token_cost: int
    retrieval_method: Literal[
        "lexical-overlap/1",
        "lexical-overlap/1-escalation",
        "lexical-overlap/1-escalated",
    ]
    retrieval_complete: bool


class AuditVerificationFinding(_AuditRecord):
    claim_id: str
    verdict: Literal[
        "supported",
        "contradicted",
        "insufficiently_supported",
        "not_meaningfully_verifiable",
    ]
    evidence_ids: tuple[str, ...]


class AuditVerificationAssessment(_AuditRecord):
    claim_id: str
    verdict: Literal[
        "supported",
        "contradicted",
        "insufficiently_supported",
        "not_meaningfully_verifiable",
    ]
    verifier_provider: str
    verifier_model: str
    prompt_version: str
    findings: tuple[AuditVerificationFinding, ...]

    _valid_provider = field_validator("verifier_provider")(_audit_identity)
    _valid_model = field_validator("verifier_model")(_audit_model_identity)
    _valid_prompt_version = field_validator("prompt_version")(_audit_prompt_version)


class AuditVerificationPass(_AuditRecord):
    pass_index: int
    complete: bool
    spans: tuple[AuditVerificationSpan, ...]
    claims: tuple[AuditVerificationClaim, ...]
    selections: tuple[AuditVerificationSelection, ...]
    assessments: tuple[AuditVerificationAssessment, ...]


class AuditVerificationRepair(_AuditRecord):
    span_id: str
    original_hash: str
    triggering_claim_ids: tuple[str, ...]
    action: Literal["qualify", "replace", "remove"]


class AuditVerificationUsage(_AuditRecord):
    phase: Literal["decomposition", "classification", "repair"]
    pass_index: int
    prompt_version: str
    provider: str | None
    model: str | None
    input_tokens: int | None
    output_tokens: int | None
    finish_status: str | None

    _valid_provider = field_validator("provider")(_audit_optional_identity)
    _valid_model = field_validator("model")(_audit_optional_model_identity)
    _valid_prompt_version = field_validator("prompt_version")(_audit_prompt_version)
    _valid_finish_status = field_validator("finish_status")(_audit_finish_status)


class AuditVerification(_AuditRecord):
    enabled: bool
    pass_count: int
    exhausted: bool
    failed: bool
    passes: tuple[AuditVerificationPass, ...]
    repairs: tuple[AuditVerificationRepair, ...]
    usage: tuple[AuditVerificationUsage, ...]
    warning_codes: tuple[str, ...]
    limitation_codes: tuple[str, ...]
    failure_codes: tuple[str, ...]


class AuditCacheMetadata(_AuditRecord):
    """Closed cache outcome and descriptor-invalidation codes for audit/3."""

    cache_hits: tuple[str, ...] = ()
    cache_misses: tuple[str, ...] = ()
    invalidation_reasons: tuple[str, ...] = ()

    @field_validator("cache_hits", mode="before")
    @classmethod
    def _validate_cache_hits(cls, value: object) -> tuple[str, ...]:
        return _audit_closed_codes(value, allowed=_CACHE_HIT_CODES, label="cache hits")

    @field_validator("cache_misses", mode="before")
    @classmethod
    def _validate_cache_misses(cls, value: object) -> tuple[str, ...]:
        return _audit_closed_codes(
            value, allowed=_CACHE_MISS_CODES, label="cache misses"
        )

    @field_validator("invalidation_reasons", mode="before")
    @classmethod
    def _validate_invalidation_reasons(cls, value: object) -> tuple[str, ...]:
        return _audit_closed_codes(
            value,
            allowed=_CACHE_INVALIDATION_CODES,
            label="cache invalidation reasons",
        )


class AuditRetryAttempt(_AuditRecord):
    """Attempt metadata for a single work item in audit/3."""

    work_id: str
    attempt_count: int
    failure_reasons: tuple[str, ...] = ()

    _valid_work_id = field_validator("work_id")(_audit_work_id)

    @field_validator("attempt_count", mode="before")
    @classmethod
    def _validate_positive_count(cls, value: object) -> int:
        return _audit_integer(value, minimum=1, label="attempt_count")

    @field_validator("failure_reasons", mode="before")
    @classmethod
    def _validate_failure_reasons(cls, value: object) -> tuple[str, ...]:
        return _audit_closed_codes(
            value, allowed=_RETRY_FAILURE_CODES, label="retry failure reasons"
        )


class AuditReliability(_AuditRecord):
    """Aggregate reliability metadata for audit/3."""

    cache: AuditCacheMetadata | None = None
    resumed: bool | None = None
    reused_count: int | None = None
    recomputed_count: int | None = None
    attempts: tuple[AuditRetryAttempt, ...] = ()

    @field_validator("reused_count", "recomputed_count", mode="before")
    @classmethod
    def _validate_resume_counts(cls, value: object) -> int | None:
        if value is None:
            return None
        return _audit_integer(value, minimum=0, label="resume counts")

    @model_validator(mode="after")
    def _resume_fields_are_complete(self) -> AuditReliability:
        resume_fields = (self.resumed, self.reused_count, self.recomputed_count)
        if any(value is not None for value in resume_fields) and any(
            value is None for value in resume_fields
        ):
            raise ValueError("resume state must include resumed and both counts")
        return self


class AuditEvidence(_AuditRecord):
    """A traceable evidence edge without copying a possibly sensitive quote."""

    segment_id: str
    quoted: bool

    _valid_segment_id = field_validator("segment_id")(_audit_segment_id)


class AuditContentUnit(_AuditRecord):
    kind: Literal["fact", "claim", "definition", "procedure", "example", "other"]
    evidence: tuple[AuditEvidence, ...]
    qualified: bool
    uncertain: bool


class AuditAnnotation(_AuditRecord):
    evidence: tuple[AuditEvidence, ...]


class AuditSummary(_AuditRecord):
    """The structural/evidence view of a summary, never reader-facing text.

    A summary, entity, content-unit, or quotation string can reproduce a
    source credential. The audit needs their links and classifications, not a
    second durable copy of the generated prose, so this intentionally stores
    no free-form source-derived text.
    """

    level: int
    provenance: tuple[str, ...]
    content_units: tuple[AuditContentUnit, ...]
    qualification_evidence: tuple[AuditAnnotation, ...]
    contradiction_evidence: tuple[AuditAnnotation, ...]
    quotation_evidence: tuple[AuditEvidence, ...]

    _valid_provenance = field_validator("provenance")(_audit_segment_ids)


class AuditGroundingSelection(_AuditRecord):
    selected_ids: tuple[str, ...]
    omitted_ids: tuple[str, ...]
    reserve_tokens: int | None = Field(default=None, ge=1)
    request_capacity_tokens: int | None = Field(default=None, ge=1)
    omission_reason: Literal["budget"]

    _valid_selected_ids = field_validator("selected_ids")(_audit_segment_ids)
    _valid_omitted_ids = field_validator("omitted_ids")(_audit_segment_ids)


class AuditNode(_AuditRecord):
    node_id: str
    level: int
    order: int
    children: tuple[str, ...]
    covered_segments: tuple[str, ...]
    summary: AuditSummary
    _valid_node_id = field_validator("node_id")(_audit_node_id)
    _valid_children = field_validator("children")(_audit_node_ids)
    _valid_covered_segments = field_validator("covered_segments")(_audit_segment_ids)


class AuditNodeV4(AuditNode):
    grounding: AuditGroundingSelection | None


class _AuditArtifactBase(_AuditRecord):
    source_id: str
    strategy: Literal["auto", "direct", "hierarchical"]
    model: str
    configuration: dict[str, Any]
    source_segments: tuple[AuditSegment, ...]
    tree_nodes: tuple[AuditNode, ...]
    root_node_id: str
    citations: tuple[AuditCitation, ...]
    usage: tuple[AuditUsage, ...]
    warnings: tuple[str, ...]
    failures: tuple[str, ...]
    verification: AuditVerification

    @field_validator("configuration", mode="before")
    @classmethod
    def _safe_configuration(cls, value: object) -> dict[str, object]:
        return _validate_audit_configuration(value)

    @field_validator("source_id")
    @classmethod
    def _valid_source_id(cls, value: str) -> str:
        return _audit_source_id(value)

    _valid_model = field_validator("model")(_audit_model_identity)
    _valid_root_node_id = field_validator("root_node_id")(_audit_node_id)

    @model_validator(mode="after")
    def _links_resolve(self) -> _AuditArtifactBase:
        segments = {segment.segment_id: segment for segment in self.source_segments}
        if len(segments) != len(self.source_segments):
            raise ValueError("source segment identifiers must be unique")
        if tuple(segment.segment_id for segment in self.source_segments) != tuple(
            segment.segment_id
            for segment in sorted(self.source_segments, key=lambda item: item.order)
        ):
            raise ValueError("source segments must be in source order")
        if any(segment.source_id != self.source_id for segment in self.source_segments):
            raise ValueError("every source segment must belong to the audit source")

        nodes = {node.node_id: node for node in self.tree_nodes}
        if len(nodes) != len(self.tree_nodes):
            raise ValueError("tree node identifiers must be unique")
        root = nodes.get(self.root_node_id)
        if root is None:
            raise ValueError("root_node_id must resolve to a tree node")

        for node in self.tree_nodes:
            unknown_covered = set(node.covered_segments) - set(segments)
            if unknown_covered:
                raise ValueError("tree coverage must resolve to source segments")
            grounding = getattr(node, "grounding", None)
            if isinstance(node, AuditNodeV4):
                if len(node.children) > 1 and grounding is None:
                    raise ValueError("executed merges must record grounding")
                if len(node.children) <= 1 and grounding is not None:
                    raise ValueError("only executed merges may record grounding")
            if grounding is not None:
                selected = set(grounding.selected_ids)
                omitted = set(grounding.omitted_ids)
                if (
                    len(selected) != len(grounding.selected_ids)
                    or len(omitted) != len(grounding.omitted_ids)
                    or selected & omitted
                ):
                    raise ValueError("grounding selection identifiers must not overlap")
                if (selected | omitted) - set(node.covered_segments):
                    raise ValueError(
                        "grounding selection must resolve to tree coverage"
                    )
                if (grounding.reserve_tokens is None) == (
                    grounding.request_capacity_tokens is None
                ):
                    raise ValueError("grounding selection must record one budget mode")
            for child_id in node.children:
                child = nodes.get(child_id)
                if child is None:
                    raise ValueError("tree child must resolve to a tree node")
                if child.level >= node.level:
                    raise ValueError("tree children must be at a lower level")
            references = _summary_references(node.summary)
            if not references <= set(segments):
                raise ValueError("summary evidence must resolve to source segments")

        citation_ids = tuple(citation.segment_id for citation in self.citations)
        if len(set(citation_ids)) != len(citation_ids):
            raise ValueError("citation identifiers must be unique")
        expected_order = tuple(
            segment.segment_id
            for segment in self.source_segments
            if segment.segment_id in citation_ids
        )
        if citation_ids != expected_order:
            raise ValueError("citations must be unique and in source order")
        for citation in self.citations:
            segment = segments.get(citation.segment_id)
            if segment is None or (
                citation.source_id != segment.source_id
                or citation.order != segment.order
            ):
                raise ValueError("citation mappings must resolve to segment metadata")
        if any(
            not _CLOSED_CODE.fullmatch(code)
            for code in (*self.warnings, *self.failures)
        ):
            raise ValueError("audit diagnostics must be closed identifiers")
        _verification_links_resolve(self.verification, set(segments))
        return self


class AuditArtifactV2(_AuditArtifactBase):
    """The unchanged audit/2 wire contract."""

    schema_version: Literal["audit/2"]


class AuditArtifactV3(_AuditArtifactBase):
    """The audit/3 contract with reliability-only metadata."""

    schema_version: Literal["audit/3"]
    reliability: AuditReliability


class AuditArtifactV4(_AuditArtifactBase):
    """The audit/4 contract with per-merge grounding metadata."""

    schema_version: Literal["audit/4"]
    tree_nodes: tuple[AuditNodeV4, ...]
    reliability: AuditReliability | None = None


_AuditArtifactRecord = Annotated[
    AuditArtifactV2 | AuditArtifactV3 | AuditArtifactV4,
    Field(discriminator="schema_version"),
]
_AUDIT_ARTIFACT_ADAPTER = TypeAdapter(_AuditArtifactRecord)


class AuditArtifact:
    """Version-discriminated audit artifact reader compatibility facade."""

    def __new__(cls, **value: object) -> AuditArtifactV2 | AuditArtifactV3 | AuditArtifactV4:
        return _AUDIT_ARTIFACT_ADAPTER.validate_python(value)

    @staticmethod
    def model_validate(value: object) -> AuditArtifactV2 | AuditArtifactV3 | AuditArtifactV4:
        return _AUDIT_ARTIFACT_ADAPTER.validate_python(value)

    @staticmethod
    def model_validate_json(value: str | bytes) -> AuditArtifactV2 | AuditArtifactV3 | AuditArtifactV4:
        return _AUDIT_ARTIFACT_ADAPTER.validate_json(value)


def _verification_links_resolve(
    verification: AuditVerification, segment_ids: set[str]
) -> None:
    """Validate audit-only verification links without retaining prose."""
    if not verification.enabled:
        if (
            verification.pass_count
            or verification.passes
            or verification.repairs
            or verification.usage
            or verification.exhausted
            or verification.failed
            or verification.warning_codes
            or verification.limitation_codes
            or verification.failure_codes
        ):
            raise ValueError("verification disabled record must be empty")
        return
    if verification.pass_count != len(verification.passes):
        raise ValueError("verification pass count must match pass records")
    pass_indexes = tuple(item.pass_index for item in verification.passes)
    if len(set(pass_indexes)) != len(pass_indexes) or pass_indexes != tuple(
        sorted(pass_indexes)
    ):
        raise ValueError("verification passes must be uniquely ordered")
    if any(
        not _CLOSED_CODE.fullmatch(code)
        for codes in (
            verification.warning_codes,
            verification.limitation_codes,
            verification.failure_codes,
        )
        for code in codes
    ):
        raise ValueError("verification diagnostic codes must be closed identifiers")

    spans_by_id: dict[str, AuditVerificationSpan] = {}
    claims_by_id: dict[str, AuditVerificationClaim] = {}
    for record_position, record in enumerate(verification.passes):
        if not 1 <= record.pass_index <= 99:
            raise ValueError("verification pass index must be between 1 and 99")
        prefix = f"V{record.pass_index:02d}"
        spans = {span.span_id: span for span in record.spans}
        claims = {claim.claim_id: claim for claim in record.claims}
        selections = {selection.claim_id: selection for selection in record.selections}
        assessments = {
            assessment.claim_id: assessment for assessment in record.assessments
        }
        if len(spans) != len(record.spans) or len(claims) != len(record.claims):
            raise ValueError("verification pass identifiers must be unique")
        if any(
            not _VERIFICATION_SPAN_ID.fullmatch(span.span_id)
            or not span.span_id.startswith(f"{prefix}S")
            or not _SHA256.fullmatch(span.content_hash)
            for span in record.spans
        ) or any(
            not _VERIFICATION_CLAIM_ID.fullmatch(claim.claim_id)
            or not claim.claim_id.startswith(f"{prefix}C")
            for claim in record.claims
        ):
            raise ValueError("verification identifiers must belong to their pass")
        if tuple(span.ordinal for span in record.spans) != tuple(
            range(1, len(record.spans) + 1)
        ):
            raise ValueError("verification spans must be in ordinal order")
        claims_by_span: dict[str, list[AuditVerificationClaim]] = {}
        for claim in record.claims:
            claims_by_span.setdefault(claim.span_id, []).append(claim)
        if any(
            tuple(claim.ordinal for claim in span_claims)
            != tuple(range(1, len(span_claims) + 1))
            for span_claims in claims_by_span.values()
        ):
            raise ValueError("verification claims must be in span ordinal order")
        if any(
            span.start < 0
            or span.end <= span.start
            or (index and span.start < record.spans[index - 1].end)
            for index, span in enumerate(record.spans)
        ):
            raise ValueError("verification span ranges must be ordered and nonempty")
        if not set(selections) <= set(claims) or len(selections) != len(
            record.selections
        ):
            raise ValueError("verification selections must resolve pass claims")
        if not set(assessments) <= set(claims) or len(assessments) != len(
            record.assessments
        ):
            raise ValueError("verification assessments must resolve pass claims")
        coverage_complete = set(selections) == set(claims) and set(assessments) == set(
            claims
        )
        if record.complete and not coverage_complete:
            raise ValueError(
                "verification pass completion must match recorded coverage"
            )
        if not record.complete and (
            not verification.failed or record_position != len(verification.passes) - 1
        ):
            raise ValueError(
                "only a terminal failed verification may have a partial pass"
            )
        for claim in record.claims:
            if claim.span_id not in spans:
                raise ValueError("verification claim must resolve to a pass span")
        for selection in record.selections:
            selected = set(selection.selected_ids)
            examined = set(selection.examined_ids)
            omitted = set(selection.omitted_ids)
            if len(selected) != len(selection.selected_ids) or len(
                examined | omitted
            ) != len(selection.examined_ids) + len(selection.omitted_ids):
                raise ValueError("verification evidence identifiers must be unique")
            if not selected <= examined or not (examined | omitted) <= segment_ids:
                raise ValueError(
                    "verification evidence must resolve to source segments"
                )
            if selection.retrieval_complete != (not omitted):
                raise ValueError(
                    "verification retrieval completeness must match omissions"
                )
        for assessment in record.assessments:
            finding_ids = [finding.claim_id for finding in assessment.findings]
            if not assessment.findings or any(
                item != assessment.claim_id for item in finding_ids
            ):
                raise ValueError(
                    "verification findings must resolve to their assessment"
                )
            selection = selections[assessment.claim_id]
            for finding in assessment.findings:
                if not set(finding.evidence_ids) <= set(selection.examined_ids):
                    raise ValueError("verification finding evidence must be examined")
        if set(spans_by_id) & set(spans) or set(claims_by_id) & set(claims):
            raise ValueError("verification identifiers must be unique across passes")
        spans_by_id.update(spans)
        claims_by_id.update(claims)

    for repair in verification.repairs:
        span = spans_by_id.get(repair.span_id)
        if span is None or repair.original_hash != span.content_hash:
            raise ValueError(
                "verification repair must resolve to its original pass span"
            )
        if not repair.triggering_claim_ids or any(
            claim_id not in claims_by_id
            or claims_by_id[claim_id].span_id != repair.span_id
            for claim_id in repair.triggering_claim_ids
        ):
            raise ValueError(
                "verification repair triggers must resolve to its pass span"
            )
    if any(usage.pass_index not in pass_indexes for usage in verification.usage):
        raise ValueError("verification usage must resolve to a pass")


@dataclass(frozen=True)
class Citation:
    segment_id: str
    source_id: str
    order: int


def _summary_references(node: AuditSummary) -> set[str]:
    identifiers = set(node.provenance)
    identifiers.update(
        evidence.segment_id for unit in node.content_units for evidence in unit.evidence
    )
    identifiers.update(
        evidence.segment_id
        for annotation in (
            *node.qualification_evidence,
            *node.contradiction_evidence,
        )
        for evidence in annotation.evidence
    )
    identifiers.update(evidence.segment_id for evidence in node.quotation_evidence)
    return identifiers


def resolve_citations(
    provenance: Sequence[str], *, source_id: str, segments: Sequence[SourceSegment]
) -> tuple[Citation, ...]:
    by_id = {segment.segment_id: segment for segment in segments}
    if len(by_id) != len(segments):
        raise AuditError("source segment identifiers must be unique")
    if any(segment.source_id != source_id for segment in segments):
        raise AuditError("every source segment must belong to the supplied source")
    unknown = set(provenance) - set(by_id)
    if unknown:
        raise AuditError("citation provenance contains an unknown source segment")
    cited = set(provenance)
    return tuple(
        Citation(segment.segment_id, segment.source_id, segment.order)
        for segment in sorted(segments, key=lambda item: item.order)
        if segment.segment_id in cited
    )


def render_citations(text: str, citations: Sequence[Citation]) -> str:
    """Return a readable opt-in source list; the default caller skips this."""
    if not citations:
        return text
    identifiers = ", ".join(citation.segment_id for citation in citations)
    return f"{text}\n\nSources: {identifiers}"


def _audit_segment(segment: SourceSegment) -> AuditSegment:
    return AuditSegment(
        segment_id=segment.segment_id,
        source_id=segment.source_id,
        order=segment.order,
        core_start=segment.core_start,
        core_end=segment.core_end,
        context_start=segment.context_start,
        context_end=segment.context_end,
        core_token_count=segment.core_token_count,
        token_count=segment.token_count,
        leading_overlap_tokens=segment.leading_overlap_tokens,
        trailing_overlap_tokens=segment.trailing_overlap_tokens,
        boundary_kind=segment.boundary_kind.value,
    )


def _audit_node(node: TreeNode, *, include_grounding: bool) -> AuditNode | AuditNodeV4:
    summary = node.summary
    audit_summary = AuditSummary(
        level=summary.level,
        provenance=summary.provenance,
        content_units=tuple(
            AuditContentUnit(
                kind=unit.kind.value,
                evidence=tuple(
                    AuditEvidence(
                        segment_id=evidence.segment_id,
                        quoted=evidence.quote is not None,
                    )
                    for evidence in unit.evidence
                ),
                qualified=unit.qualification is not None,
                uncertain=unit.uncertain,
            )
            for unit in summary.content_units
        ),
        qualification_evidence=tuple(
            AuditAnnotation(
                evidence=tuple(
                    AuditEvidence(
                        segment_id=evidence.segment_id,
                        quoted=evidence.quote is not None,
                    )
                    for evidence in annotation.evidence
                )
            )
            for annotation in summary.qualifications
        ),
        contradiction_evidence=tuple(
            AuditAnnotation(
                evidence=tuple(
                    AuditEvidence(
                        segment_id=evidence.segment_id,
                        quoted=evidence.quote is not None,
                    )
                    for evidence in annotation.evidence
                )
            )
            for annotation in summary.contradictions
        ),
        quotation_evidence=tuple(
            AuditEvidence(
                segment_id=evidence.segment_id,
                quoted=evidence.quote is not None,
            )
            for evidence in summary.quotations
        ),
    )
    values = {
        "node_id": node.node_id,
        "level": node.level,
        "order": node.order,
        "children": node.children,
        "covered_segments": node.covered_segments,
        "summary": audit_summary,
    }
    if not include_grounding:
        return AuditNode(**values)
    return AuditNodeV4(
        **values,
        grounding=(
            AuditGroundingSelection(
                selected_ids=node.grounding.selection.selected_ids,
                omitted_ids=node.grounding.selection.omitted_ids,
                reserve_tokens=node.grounding.reserve_tokens,
                request_capacity_tokens=node.grounding.request_capacity_tokens,
                omission_reason="budget",
            )
            if node.grounding is not None
            else None
        ),
    )


def _usage(generation: GenerationResult) -> AuditUsage:
    return AuditUsage(
        provider=redact_text(generation.provider),
        model=redact_text(generation.model),
        input_tokens=generation.input_tokens,
        output_tokens=generation.output_tokens,
        finish_status=(
            redact_text(generation.finish_status)
            if generation.finish_status is not None
            else None
        ),
    )


_SAFE_CONFIGURATION = frozenset(
    {
        "provider",
        "model",
        "timeout_seconds",
        "strategy",
        "context_window",
        "max_output_tokens",
        "safety_margin_tokens",
        "safety_margin_fraction",
        "max_direct_tokens",
        "target_words",
        "max_merge_children",
        "include_citations",
        "evidence_tokens",
        "request_tokens",
        "output_reserve_tokens",
        "max_repair_passes",
        "enabled",
        "context_window_tokens",
        "context_window_assumed",
        "counter_exact",
        "reserved_output_tokens",
        "usable_input_capacity",
        "document_tokens",
        "fits",
        "prompt_version",
    }
)
_SAFE_CONFIGURATION_SECTIONS = frozenset(
    {"app", "strategy", "pipeline", "verification", "budget"}
)
_CONFIG_PROMPT_VERSION = re.compile(r"^[a-z][a-z0-9_-]{0,63}/[1-9][0-9]*$")
_CONFIG_BOOLEAN_FIELDS = frozenset(
    {"include_citations", "enabled", "context_window_assumed", "counter_exact", "fits"}
)
_CONFIG_POSITIVE_INTEGER_FIELDS = frozenset(
    {
        "context_window",
        "max_output_tokens",
        "max_direct_tokens",
        "target_words",
        "max_merge_children",
        "evidence_tokens",
        "request_tokens",
        "output_reserve_tokens",
        "context_window_tokens",
        "usable_input_capacity",
    }
)
_CONFIG_NONNEGATIVE_INTEGER_FIELDS = frozenset(
    {
        "safety_margin_tokens",
        "max_repair_passes",
        "reserved_output_tokens",
        "document_tokens",
    }
)
_CONFIG_NULLABLE_POSITIVE_INTEGER_FIELDS = frozenset(
    {"context_window", "max_direct_tokens", "max_merge_children"}
)


def _configuration_error(message: str) -> ValueError:
    return ValueError(f"audit configuration {message}")


def _safe_configuration_value(
    key: str, value: object
) -> str | int | float | bool | None:
    if isinstance(value, Enum):
        value = value.value
    if key in _CONFIG_BOOLEAN_FIELDS:
        if isinstance(value, bool):
            return value
        raise _configuration_error(f"{key} must be boolean")
    if key == "provider":
        if value in {"openai", "ollama"}:
            return value
        raise _configuration_error("provider must be an allowed identity")
    if key == "strategy":
        if value in {"auto", "direct", "hierarchical"}:
            return value
        raise _configuration_error("strategy must be an allowed identity")
    if key == "prompt_version":
        if isinstance(value, str) and _CONFIG_PROMPT_VERSION.fullmatch(value):
            return value
        raise _configuration_error("prompt_version must be a version identity")
    if key == "model":
        if (
            isinstance(value, str)
            and _SAFE_MODEL_IDENTITY.fullmatch(value)
            and redact_text(value) == value
        ):
            return value
        raise _configuration_error("model must be a non-secret model identity")
    if key == "timeout_seconds":
        if (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and isfinite(value)
            and value > 0
        ):
            return value
        raise _configuration_error("timeout_seconds must be finite and positive")
    if key == "safety_margin_fraction":
        if (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and isfinite(value)
            and 0 <= value < 1
        ):
            return value
        raise _configuration_error("safety_margin_fraction must be in [0, 1)")
    if key in _CONFIG_NULLABLE_POSITIVE_INTEGER_FIELDS and value is None:
        return None
    if key in _CONFIG_POSITIVE_INTEGER_FIELDS:
        if isinstance(value, int) and not isinstance(value, bool) and value > 0:
            return value
        raise _configuration_error(f"{key} must be a positive integer")
    if key in _CONFIG_NONNEGATIVE_INTEGER_FIELDS:
        if (
            isinstance(value, int)
            and not isinstance(value, bool)
            and value >= 0
            and (key != "max_repair_passes" or value <= 49)
        ):
            return value
        raise _configuration_error(f"{key} must be a permitted nonnegative integer")
    raise _configuration_error(f"contains unsupported field {key!r}")


def _validate_configuration_entries(
    configuration: Mapping[object, object],
) -> dict[str, object]:
    unknown = set(configuration) - _SAFE_CONFIGURATION
    if unknown:
        raise _configuration_error("contains unsupported fields")
    return {
        key: _safe_configuration_value(key, value)
        for key, value in configuration.items()
    }


def _validate_audit_configuration(value: object) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise _configuration_error("must be a mapping")
    sections = set(value) & _SAFE_CONFIGURATION_SECTIONS
    if not sections:
        return _validate_configuration_entries(value)
    if set(value) != sections:
        raise _configuration_error("must not mix configuration sections and fields")
    validated: dict[str, object] = {}
    for section in sorted(sections):
        entries = value[section]
        if not isinstance(entries, Mapping):
            raise _configuration_error("section must be a mapping")
        validated[section] = _validate_configuration_entries(entries)
    return validated


def _allowlisted_configuration(
    configuration: Mapping[str, object],
) -> dict[str, object]:
    """Retain only declared safe runtime metadata; never redact-and-keep paths."""
    if not isinstance(configuration, Mapping):
        raise AuditError("audit configuration must be a mapping")
    sections = set(configuration) & _SAFE_CONFIGURATION_SECTIONS
    if not sections:
        return _validate_audit_configuration(
            {
                key: value
                for key, value in configuration.items()
                if key in _SAFE_CONFIGURATION
            }
        )
    allowed: dict[str, object] = {}
    for section in sorted(sections):
        value = configuration[section]
        if not isinstance(value, Mapping):
            raise AuditError("audit configuration section must be a mapping")
        retained = {
            key: item for key, item in value.items() if key in _SAFE_CONFIGURATION
        }
        if retained:
            allowed[section] = retained
    return _validate_audit_configuration(allowed)


def _audit_verification(
    verification: VerificationResult | None, *, enabled: bool
) -> AuditVerification:
    """Project rich verification runtime data without copying any prose fields."""
    if verification is None:
        return AuditVerification(
            enabled=enabled,
            pass_count=0,
            exhausted=False,
            failed=False,
            passes=(),
            repairs=(),
            usage=(),
            warning_codes=(),
            limitation_codes=(),
            failure_codes=(),
        )
    if not enabled:
        if (
            verification.pass_results
            or verification.passes
            or verification.selections
            or verification.repairs
            or verification.generations
            or verification.diagnostic_codes
            or verification.exhausted
            or verification.failed
            or verification.phase_generations
            or verification.warning_codes
            or verification.limitation_codes
            or verification.failure_codes
        ):
            raise AuditError("disabled verification result must not contain activity")
        return AuditVerification(
            enabled=False,
            pass_count=0,
            exhausted=False,
            failed=False,
            passes=(),
            repairs=(),
            usage=(),
            warning_codes=(),
            limitation_codes=(),
            failure_codes=(),
        )
    pass_results = verification.pass_results
    if verification.passes and len(pass_results) != len(verification.passes):
        raise AuditError("verification audit projection requires complete pass records")
    records: list[AuditVerificationPass] = []
    for item in pass_results:
        pass_indexes = {
            generation.pass_index for generation in item.phase_generations
        } or {assessment.pass_index for assessment in item.assessments}
        if not pass_indexes:
            pass_indexes = {int(span.span_id[1:3]) for span in item.spans}
        if len(pass_indexes) != 1:
            raise AuditError("verification pass record does not match its pass index")
        pass_index = pass_indexes.pop()
        records.append(
            AuditVerificationPass(
                pass_index=pass_index,
                complete=(
                    not item.failed
                    and {selection.claim_id for selection in item.selections}
                    == {claim.claim_id for claim in item.claims}
                    and {assessment.claim_id for assessment in item.assessments}
                    == {claim.claim_id for claim in item.claims}
                ),
                spans=tuple(
                    AuditVerificationSpan(
                        span_id=span.span_id,
                        ordinal=span.ordinal,
                        start=span.start,
                        end=span.end,
                        content_hash=span.content_hash,
                    )
                    for span in item.spans
                ),
                claims=tuple(
                    AuditVerificationClaim(
                        claim_id=claim.claim_id,
                        span_id=claim.span_id,
                        ordinal=claim.ordinal,
                        is_fallback=claim.is_fallback,
                    )
                    for claim in item.claims
                ),
                selections=tuple(
                    AuditVerificationSelection(
                        claim_id=selection.claim_id,
                        selected_ids=selection.selected_ids,
                        examined_ids=selection.examined_ids,
                        omitted_ids=selection.omitted_ids,
                        token_cost=selection.token_cost,
                        retrieval_method=selection.retrieval_method,
                        retrieval_complete=selection.retrieval_complete,
                    )
                    for selection in item.selections
                ),
                assessments=tuple(
                    AuditVerificationAssessment(
                        claim_id=assessment.claim_id,
                        verdict=assessment.verdict.value,
                        verifier_provider=redact_text(assessment.verifier_provider),
                        verifier_model=redact_text(assessment.verifier_model),
                        prompt_version=assessment.prompt_version,
                        findings=tuple(
                            AuditVerificationFinding(
                                claim_id=finding.claim_id,
                                verdict=finding.verdict.value,
                                evidence_ids=finding.evidence_ids,
                            )
                            for finding in assessment.findings
                        ),
                    )
                    for assessment in item.assessments
                ),
            )
        )
    phase_generations = verification.phase_generations or tuple(
        phase_generation
        for item in pass_results
        for phase_generation in item.phase_generations
    )
    return AuditVerification(
        enabled=True,
        pass_count=len(records),
        exhausted=verification.exhausted,
        failed=verification.failed,
        passes=tuple(records),
        repairs=tuple(
            AuditVerificationRepair(
                span_id=repair.span_id,
                original_hash=repair.original_hash,
                triggering_claim_ids=repair.triggering_claim_ids,
                action=repair.action.value,
            )
            for repair in verification.repairs
        ),
        usage=tuple(
            AuditVerificationUsage(
                phase=item.phase.value,
                pass_index=item.pass_index,
                prompt_version=item.prompt_version,
                provider=(
                    redact_text(item.generation.provider)
                    if item.generation is not None
                    else None
                ),
                model=(
                    redact_text(item.generation.model)
                    if item.generation is not None
                    else None
                ),
                input_tokens=(
                    item.generation.input_tokens
                    if item.generation is not None
                    else None
                ),
                output_tokens=(
                    item.generation.output_tokens
                    if item.generation is not None
                    else None
                ),
                finish_status=(
                    redact_text(item.generation.finish_status)
                    if item.generation is not None
                    and item.generation.finish_status is not None
                    else None
                ),
            )
            for item in phase_generations
        ),
        warning_codes=tuple(
            dict.fromkeys((*verification.warning_codes, *verification.diagnostic_codes))
        ),
        limitation_codes=verification.limitation_codes,
        failure_codes=verification.failure_codes,
    )


def build_audit_artifact(
    *,
    source_id: str,
    strategy: str,
    model: str,
    configuration: Mapping[str, object],
    segments: Sequence[SourceSegment],
    nodes: Sequence[TreeNode],
    root_node_id: str,
    citations: Sequence[Citation],
    generations: Sequence[GenerationResult] = (),
    warnings: Sequence[str] = (),
    failures: Sequence[str] = (),
    verification: VerificationResult | None = None,
    verification_enabled: bool = False,
    reliability_cache: Mapping[str, Sequence[str]] | None = None,
    reliability_resume: Mapping[str, object] | None = None,
    reliability_attempts: Sequence[Mapping[str, object]] | None = None,
) -> AuditArtifactV2 | AuditArtifactV3 | AuditArtifactV4:
    """Build a validated artifact without retaining source text or request data."""
    # Determine if we have reliability metadata
    has_reliability = (
        reliability_cache is not None
        or reliability_resume is not None
        or reliability_attempts is not None
    )

    # Build reliability object if present
    reliability_obj = None
    if has_reliability:
        cache_obj = None
        if reliability_cache is not None:
            cache_obj = AuditCacheMetadata(
                cache_hits=tuple(reliability_cache.get("cache_hits", ())),
                cache_misses=tuple(reliability_cache.get("cache_misses", ())),
                invalidation_reasons=tuple(
                    reliability_cache.get("invalidation_reasons", ())
                ),
            )

        resume_values: dict[str, object] = {}
        if reliability_resume is not None:
            resume_values = {
                "resumed": reliability_resume.get("resumed", False),
                "reused_count": reliability_resume.get("reused_count", 0),
                "recomputed_count": reliability_resume.get("recomputed_count", 0),
            }

        attempts = ()
        if reliability_attempts is not None:
            attempts = tuple(
                AuditRetryAttempt(
                    work_id=attempt["work_id"],
                    attempt_count=attempt["attempt_count"],
                    failure_reasons=attempt.get("failure_reasons", ()),
                )
                for attempt in reliability_attempts
            )

        reliability_obj = AuditReliability(
            cache=cache_obj, attempts=attempts, **resume_values
        )

    has_merges = any(len(node.children) > 1 for node in nodes)
    has_grounding = any(node.grounding is not None for node in nodes)
    uses_audit_v4 = has_merges or has_grounding
    common = {
        "source_id": source_id,
        "strategy": strategy,
        "model": redact_text(model),
        "configuration": _allowlisted_configuration(configuration),
        "source_segments": tuple(_audit_segment(segment) for segment in segments),
        "tree_nodes": tuple(
            _audit_node(node, include_grounding=uses_audit_v4) for node in nodes
        ),
        "root_node_id": root_node_id,
        "citations": tuple(
            AuditCitation(
                segment_id=citation.segment_id,
                source_id=citation.source_id,
                order=citation.order,
            )
            for citation in citations
        ),
        "usage": tuple(_usage(generation) for generation in generations),
        "warnings": tuple(warnings),
        "failures": tuple(failures),
        "verification": _audit_verification(
            verification,
            enabled=verification_enabled
            or bool(
                verification
                and (
                    verification.pass_results
                    or verification.passes
                    or verification.repairs
                    or verification.phase_generations
                )
            ),
        ),
    }
    if uses_audit_v4:
        return AuditArtifactV4(
            schema_version=AUDIT_SCHEMA_VERSION_V4,
            reliability=reliability_obj,
            **common,
        )
    if reliability_obj is None:
        return AuditArtifactV2(schema_version=AUDIT_SCHEMA_VERSION, **common)
    return AuditArtifactV3(
        schema_version=AUDIT_SCHEMA_VERSION_V3,
        reliability=reliability_obj,
        **common,
    )


def serialize_audit(artifact: AuditArtifactV2 | AuditArtifactV3 | AuditArtifactV4) -> bytes:
    """Serialize canonically and prove the written representation is valid."""
    encoded = json.dumps(
        artifact.model_dump(mode="json"),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    try:
        AuditArtifact.model_validate_json(encoded)
    except ValueError as error:
        raise AuditError("audit serialization failed validation") from error
    return encoded


def write_audit(path: Path, artifact: AuditArtifact) -> None:
    """Atomically replace an artifact only after canonical validation succeeds."""
    payload = serialize_audit(artifact)
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(
        dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as handle:
        temporary = Path(handle.name)
        try:
            handle.write(payload)
        except OSError:
            temporary.unlink(missing_ok=True)
            raise
    temporary.replace(path)
