"""Claim-level verification domain records and strict response parsing."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum

from nltk.tokenize.punkt import PunktSentenceTokenizer
from pydantic import BaseModel, ConfigDict, ValidationError, field_validator

from summarizer.leaf import _describe, _extract_json_object, _sanitize
from summarizer.providers.base import GenerationResult, ModelProvider
from summarizer.tokenization import TokenCounter


_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SPAN_ID = re.compile(r"^V(?P<pass>\d{2})S\d{6}$")
_CLAIM_ID = re.compile(r"^V(?P<pass>\d{2})C\d{6}$")


class VerificationResponseError(ValueError):
    """A provider response violated the verification contract."""


class ClaimVerdict(StrEnum):
    SUPPORTED = "supported"
    CONTRADICTED = "contradicted"
    INSUFFICIENTLY_SUPPORTED = "insufficiently_supported"
    NOT_MEANINGFULLY_VERIFIABLE = "not_meaningfully_verifiable"


class RepairAction(StrEnum):
    QUALIFY = "qualify"
    REPLACE = "replace"
    REMOVE = "remove"


@dataclass(frozen=True)
class VerificationConfig:
    enabled: bool = False
    evidence_tokens: int = 4096
    request_tokens: int = 8192
    output_reserve_tokens: int = 1024
    safety_margin_tokens: int = 256
    max_repair_passes: int = 1

    def __post_init__(self) -> None:
        for name in ("evidence_tokens", "request_tokens", "output_reserve_tokens"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        for name in ("safety_margin_tokens", "max_repair_passes"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must not be negative")


@dataclass(frozen=True)
class VerificationRuntime:
    provider: ModelProvider
    counter: TokenCounter
    model: str
    timeout_seconds: float
    context_window_tokens: int

    def __post_init__(self) -> None:
        if not self.model.strip():
            raise ValueError("model must not be blank")
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if self.context_window_tokens <= 0:
            raise ValueError("context_window_tokens must be positive")


@dataclass(frozen=True)
class DraftSpan:
    span_id: str
    ordinal: int
    start: int
    end: int
    text: str
    content_hash: str

    def __post_init__(self) -> None:
        if not _SPAN_ID.fullmatch(self.span_id):
            raise ValueError("invalid span_id")
        if self.ordinal <= 0 or self.start < 0 or self.end <= self.start:
            raise ValueError("invalid draft span range")
        if not self.text or not _SHA256.fullmatch(self.content_hash):
            raise ValueError("invalid draft span content")
        expected_hash = hashlib.sha256(self.text.encode("utf-8")).hexdigest()
        if self.content_hash != expected_hash:
            raise ValueError("content_hash must match draft span text")


@dataclass(frozen=True)
class Claim:
    claim_id: str
    span_id: str
    ordinal: int
    anchor: str
    is_fallback: bool

    def __post_init__(self) -> None:
        claim_match = _CLAIM_ID.fullmatch(self.claim_id)
        span_match = _SPAN_ID.fullmatch(self.span_id)
        if not claim_match or not span_match:
            raise ValueError("invalid claim or span identifier")
        if claim_match["pass"] != span_match["pass"]:
            raise ValueError("claim and span must belong to the same pass")
        if self.ordinal <= 0 or not self.anchor.strip():
            raise ValueError("invalid claim")


@dataclass(frozen=True)
class EvidenceSelection:
    claim_id: str
    selected_ids: tuple[str, ...]
    examined_ids: tuple[str, ...]
    omitted_ids: tuple[str, ...]
    token_cost: int
    retrieval_method: str
    retrieval_complete: bool

    def __post_init__(self) -> None:
        if not _CLAIM_ID.fullmatch(self.claim_id):
            raise ValueError("invalid claim_id")
        if self.token_cost < 0 or not self.retrieval_method.strip():
            raise ValueError("invalid evidence selection metadata")
        if len(set((*self.examined_ids, *self.omitted_ids))) != len(
            (*self.examined_ids, *self.omitted_ids)
        ):
            raise ValueError("examined and omitted evidence must be unique")
        if not set(self.selected_ids).issubset(self.examined_ids):
            raise ValueError("selected evidence must have been examined")
        if self.retrieval_complete != (not self.omitted_ids):
            raise ValueError("retrieval completeness does not match omissions")


@dataclass(frozen=True)
class BatchFinding:
    claim_id: str
    verdict: ClaimVerdict
    evidence_ids: tuple[str, ...]
    exact_quotes: tuple[str, ...]

    def __post_init__(self) -> None:
        if not _CLAIM_ID.fullmatch(self.claim_id):
            raise ValueError("invalid claim_id")
        if self.verdict in {ClaimVerdict.SUPPORTED, ClaimVerdict.CONTRADICTED}:
            if not self.evidence_ids:
                raise ValueError("verdict requires evidence")
        if len(self.evidence_ids) != len(self.exact_quotes):
            raise ValueError("each evidence item requires one exact quote")
        if len(set(self.evidence_ids)) != len(self.evidence_ids):
            raise ValueError("evidence identifiers must be unique")


@dataclass(frozen=True)
class ClaimAssessment:
    claim_id: str
    verdict: ClaimVerdict
    findings: tuple[BatchFinding, ...]
    pass_index: int
    verifier_provider: str
    verifier_model: str
    prompt_version: str

    def __post_init__(self) -> None:
        claim_match = _CLAIM_ID.fullmatch(self.claim_id)
        if not claim_match:
            raise ValueError("invalid claim assessment identity")
        if self.pass_index <= 0 or self.pass_index > 99:
            raise ValueError("invalid assessment pass")
        if int(claim_match["pass"]) != self.pass_index:
            raise ValueError("assessment pass must match claim pass")
        if not self.findings or any(
            finding.claim_id != self.claim_id for finding in self.findings
        ):
            raise ValueError("assessment findings must belong to the claim")
        for name in ("verifier_provider", "verifier_model", "prompt_version"):
            if not getattr(self, name).strip():
                raise ValueError(f"{name} must not be blank")


@dataclass(frozen=True)
class RepairEvent:
    span_id: str
    original_hash: str
    triggering_claim_ids: tuple[str, ...]
    action: RepairAction

    def __post_init__(self) -> None:
        span_match = _SPAN_ID.fullmatch(self.span_id)
        if not span_match or not _SHA256.fullmatch(
            self.original_hash
        ):
            raise ValueError("invalid repair target")
        if not self.triggering_claim_ids:
            raise ValueError("repair requires a triggering claim")
        for item in self.triggering_claim_ids:
            claim_match = _CLAIM_ID.fullmatch(item)
            if not claim_match:
                raise ValueError("invalid triggering claim")
            if claim_match["pass"] != span_match["pass"]:
                raise ValueError("repair trigger pass must match span pass")


@dataclass(frozen=True)
class VerificationResult:
    text: str
    passes: tuple[tuple[ClaimAssessment, ...], ...]
    selections: tuple[tuple[EvidenceSelection, ...], ...]
    repairs: tuple[RepairEvent, ...]
    generations: tuple[GenerationResult, ...]
    diagnostic_codes: tuple[str, ...]
    exhausted: bool
    failed: bool

    def __post_init__(self) -> None:
        if not self.text.strip():
            raise ValueError("verification result text must not be blank")
        if len(self.passes) != len(self.selections):
            raise ValueError("verification passes and selections must align")


def _pass_prefix(pass_index: int) -> str:
    if pass_index <= 0 or pass_index > 99:
        raise ValueError("pass_index must be between 1 and 99")
    return f"V{pass_index:02d}"


def split_draft_spans(text: str, *, pass_index: int) -> tuple[DraftSpan, ...]:
    """Split text locally while preserving every character exactly once."""
    prefix = _pass_prefix(pass_index)
    if not text:
        raise ValueError("draft text must not be empty")
    raw = list(PunktSentenceTokenizer().span_tokenize(text))
    if not raw:
        raw = [(0, len(text))]
    spans: list[DraftSpan] = []
    for index, (start, _) in enumerate(raw):
        end = raw[index + 1][0] if index + 1 < len(raw) else len(text)
        span_text = text[start:end]
        spans.append(
            DraftSpan(
                span_id=f"{prefix}S{index + 1:06d}",
                ordinal=index + 1,
                start=start,
                end=end,
                text=span_text,
                content_hash=hashlib.sha256(span_text.encode("utf-8")).hexdigest(),
            )
        )
    return tuple(spans)


class _AnchorGroup(BaseModel):
    model_config = ConfigDict(extra="forbid")

    span_id: str
    anchors: list[str]

    @field_validator("span_id")
    @classmethod
    def _span_id_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must not be blank")
        return value

    @field_validator("anchors")
    @classmethod
    def _anchors_are_nonblank(cls, value: list[str]) -> list[str]:
        if any(not anchor.strip() for anchor in value):
            raise ValueError("anchors must not be blank")
        return value


class _AnchorResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    spans: list[_AnchorGroup]


class _FindingEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid")

    segment_id: str
    exact_quote: str


class _Finding(BaseModel):
    model_config = ConfigDict(extra="forbid")

    claim_id: str
    verdict: ClaimVerdict
    evidence: list[_FindingEvidence]


class _FindingResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    findings: list[_Finding]


def _validated_response(text: str, schema: type[BaseModel], *, subject: str) -> BaseModel:
    try:
        payload = json.loads(_extract_json_object(text))
    except (ValueError, json.JSONDecodeError) as error:
        raise VerificationResponseError(
            f"{subject}: response was not a single JSON object ({_sanitize(error)})"
        ) from error
    try:
        return schema.model_validate(payload)
    except ValidationError as error:
        raise VerificationResponseError(
            f"{subject}: response failed validation ({_describe(error)})"
        ) from error


def parse_claim_anchors(
    text: str,
    *,
    spans: Sequence[DraftSpan],
    pass_index: int,
) -> tuple[Claim, ...]:
    response = _validated_response(text, _AnchorResponse, subject="claim-decomposition")
    assert isinstance(response, _AnchorResponse)
    prefix = _pass_prefix(pass_index)
    legal = {span.span_id: span for span in spans}
    if len(response.spans) != len(legal):
        raise VerificationResponseError("claim-decomposition: missing span result")
    if len({group.span_id for group in response.spans}) != len(response.spans):
        raise VerificationResponseError("claim-decomposition: duplicate span result")
    if set(group.span_id for group in response.spans) != set(legal):
        raise VerificationResponseError("claim-decomposition: unknown span result")

    claims: list[Claim] = []
    groups = {group.span_id: group for group in response.spans}
    for span in spans:
        group = groups[span.span_id]
        claimable_text = span.text.rstrip()
        if len(set(group.anchors)) != len(group.anchors):
            raise VerificationResponseError("claim-decomposition: duplicate anchor")
        if any(anchor not in claimable_text for anchor in group.anchors):
            raise VerificationResponseError("claim-decomposition: anchor not in span")
        anchors = list(group.anchors)
        fallback_index = next(
            (index for index, anchor in enumerate(anchors) if anchor == claimable_text),
            None,
        )
        if fallback_index is None:
            anchors.append(claimable_text)
            fallback_index = len(anchors) - 1
        for anchor_index, anchor in enumerate(anchors):
            ordinal = len(claims) + 1
            claims.append(
                Claim(
                    claim_id=f"{prefix}C{ordinal:06d}",
                    span_id=span.span_id,
                    ordinal=anchor_index + 1,
                    anchor=anchor,
                    is_fallback=anchor_index == fallback_index,
                )
            )
    return tuple(claims)


def parse_claim_findings(
    text: str,
    *,
    claims: Sequence[Claim],
    selected: Mapping[str, Mapping[str, str]],
) -> tuple[BatchFinding, ...]:
    response = _validated_response(text, _FindingResponse, subject="claim-verification")
    assert isinstance(response, _FindingResponse)
    legal_claims = {claim.claim_id for claim in claims}
    result_ids = [finding.claim_id for finding in response.findings]
    if len(result_ids) != len(set(result_ids)) or set(result_ids) != legal_claims:
        raise VerificationResponseError("claim-verification: claim results do not match")

    findings: list[BatchFinding] = []
    by_id = {finding.claim_id: finding for finding in response.findings}
    for claim in claims:
        finding = by_id[claim.claim_id]
        legal_evidence = selected.get(claim.claim_id, {})
        evidence_ids: list[str] = []
        quotes: list[str] = []
        for evidence in finding.evidence:
            if evidence.segment_id in evidence_ids:
                raise VerificationResponseError("claim-verification: duplicate evidence")
            passage = legal_evidence.get(evidence.segment_id)
            if passage is None:
                raise VerificationResponseError("claim-verification: unselected evidence")
            if not evidence.exact_quote.strip() or evidence.exact_quote not in passage:
                raise VerificationResponseError("claim-verification: quote not in evidence")
            evidence_ids.append(evidence.segment_id)
            quotes.append(evidence.exact_quote)
        try:
            findings.append(
                BatchFinding(
                    claim_id=claim.claim_id,
                    verdict=finding.verdict,
                    evidence_ids=tuple(evidence_ids),
                    exact_quotes=tuple(quotes),
                )
            )
        except ValueError as error:
            raise VerificationResponseError(
                f"claim-verification: invalid finding ({_sanitize(error)})"
            ) from error
    return tuple(findings)
