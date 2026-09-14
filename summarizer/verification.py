"""Structure-aware source verification and repair."""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from enum import Enum
from typing import Any, TypeVar

from summarizer.cache import CacheDescriptor, CacheStore
from summarizer.checkpoint import CheckpointSession, CompletedRef
from summarizer.ingestion import SourceDocument
from summarizer.reliability import ReliabilityTracker
from summarizer.tokenization import (
    PrefixTokenCounter,
    SuffixTokenCounter,
    TokenAccountingError,
    TokenCounter,
)

_CLAIM_ID = re.compile(r"^V(?P<pass>\d{2})C\d{6}$")

class ClaimVerdict(str, Enum):
    SUPPORTED = "supported"
    CONTRADICTED = "contradicted"
    INSUFFICIENTLY_SUPPORTED = "insufficiently_supported"

class GenerationPhase(str, Enum):
    REPAIR = "repair"
    VERIFY = "verify"

@dataclass(frozen=True)
class BatchFinding:
    claim_id: str
    verdict: ClaimVerdict
    evidence_ids: Sequence[str]
    exact_quotes: Sequence[str]

    def __post_init__(self) -> None:
        if not _CLAIM_ID.fullmatch(self.claim_id):
            raise ValueError("invalid claim_id")

@dataclass(frozen=True)
class EvidenceBundle:
    selection: Any
    passages: Sequence[Any]

@dataclass(frozen=True)
class VerificationGeneration:
    phase: GenerationPhase
    pass_index: int
    generation: Any | None
    prompt_version: str

@dataclass(frozen=True)
class RepairEvent:
    span_id: str
    old_text: str
    new_text: str

@dataclass(frozen=True)
class VerificationResult:
    text: str
    passes: Sequence[Any]
    selections: Sequence[Any]
    repairs: Sequence[RepairEvent]
    generations: Sequence[Any]
    diagnostic_codes: Sequence[str]
    exhausted: bool
    failed: bool
    pass_results: Sequence[Any] = ()
    phase_generations: Sequence[VerificationGeneration] = ()
    failure_codes: Sequence[str] = ()
    limitation_codes: Sequence[str] = ()

@dataclass(frozen=True)
class VerificationConfig:
    enabled: bool
    evidence_tokens: int = 4096
    request_tokens: int = 8192
    output_reserve_tokens: int = 1024
    safety_margin_tokens: int = 256
    max_repair_passes: int = 1

@dataclass(frozen=True)
class VerificationRuntime:
    provider: Any
    counter: TokenCounter

@dataclass(frozen=True)
class RepairWorkItem:
    span: Any
    triggering_claim_ids: Sequence[str]
    evidence: Sequence[Any]
    preserved_anchors: Sequence[str]

class VerificationCapacityError(Exception):
    """Raised when repair request exceeds capacity."""

class ProviderError(Exception):
    """Raised when provider fails."""

class VerificationResponseError(Exception):
    """Raised when provider response is malformed."""

REPAIR_PROMPT_VERSION = "repair/1"

def _redact_generation(generation: Any) -> Any:
    return "[redacted]" if generation is not None else None

def _redact_terminal_pass(item: Any) -> Any:
    return replace(
        item,
        assessments=tuple(
            replace(
                assessment,
                findings=tuple(
                    BatchFinding(
                        claim_id=finding.claim_id,
                        verdict=finding.verdict,
                        evidence_ids=finding.evidence_ids,
                        exact_quotes=tuple("[redacted]" for _ in finding.exact_quotes),
                    )
                    for finding in assessment.findings
                ),
            )
            for assessment in item.assessments
        ),
        bundles=tuple(
            EvidenceBundle(selection=bundle.selection, passages=())
            for bundle in item.bundles
        ),
        generations=tuple(_redact_generation(gen) for gen in item.generations),
    )

def _terminal_result(
    *,
    text: str,
    pass_results: Sequence[Any],
    repairs: Sequence[RepairEvent],
    generations: Sequence[Any],
    phase_generations: Sequence[VerificationGeneration],
    diagnostic_codes: Sequence[str],
    failure_codes: Sequence[str],
    exhausted: bool,
    limitation_codes: Sequence[str] = (),
) -> VerificationResult:
    """Return a failure result with source passages removed from all pass snapshots."""
    redacted_passes = tuple(_redact_terminal_pass(item) for item in pass_results)
    return VerificationResult(
        text=text,
        passes=tuple(item.assessments for item in redacted_passes),
        selections=tuple(item.selections for item in redacted_passes),
        repairs=tuple(repairs),
        generations=tuple(_redact_generation(item) for item in generations),
        diagnostic_codes=tuple(diagnostic_codes),
        exhausted=exhausted,
        failed=True,
        pass_results=redacted_passes,
        phase_generations=phase_generations,
        failure_codes=tuple(failure_codes),
        limitation_codes=tuple(limitation_codes),
    )

def verify_draft_once(
    draft: str,
    *,
    source_id: str,
    source_index: Any,
    runtime: VerificationRuntime,
    config: VerificationConfig,
    pass_index: int,
    terminalize_errors: bool = False,
) -> Any:
    """Mocked for reproduction. In real code, this does the actual verification."""
    pass

def build_repair_request(batch: Any, source_id: str, runtime: VerificationRuntime) -> Any:
    return type('Req', (), {'input_text': 'mock request'})()

def _measure_request_tokens(request: Any, counter: TokenCounter) -> int:
    return 100

def pack_work_items(
    items: Sequence[RepairWorkItem],
    render_request: Callable[[Sequence[RepairWorkItem]], str],
    measure_request: Callable[[Sequence[RepairWorkItem]], int],
    runtime: VerificationRuntime,
    config: VerificationConfig,
) -> Sequence[Sequence[RepairWorkItem]]:
    return [items]

def parse_repair_proposals(text: str, items: Sequence[RepairWorkItem]) -> Sequence[Any]:
    return [type('Prop', (), {'span_id': item.span.span_id, 'new_text': 'fixed'}) for item in items]

def apply_repairs(
    text: str,
    spans: dict[str, Any],
    repairs: Sequence[Any],
    triggering_claim_ids: dict[str, Sequence[str]],
    preserved_anchors: dict[str, Sequence[str]],
) -> tuple[str, Sequence[RepairEvent]]:
    return text, ()

def _verify_and_repair(
    draft: str,
    *,
    source_id: str,
    source_index: Any,
    runtime: VerificationRuntime,
    config: VerificationConfig,
    _pass_index: int = 1,
) -> VerificationResult:
    """Run finite verification/repair orchestration; disabled mode is zero-call."""
    print(f"DEBUG: _verify_and_repair called (pass={_pass_index}, draft={draft[:20]}...)")
    if not config.enabled:
        print("DEBUG: disabled")
        return VerificationResult(
            text=draft,
            passes=(),
            selections=(),
            repairs=(),
            generations=(),
            diagnostic_codes=(),
            exhausted=False,
            failed=False,
        )
    first = verify_draft_once(
        draft,
        source_id=source_id,
        source_index=source_index,
        runtime=runtime,
        config=config,
        pass_index=_pass_index,
        terminalize_errors=True,
    )
    if first.failed:
        print("DEBUG: first failed")
        return _terminal_result(
            text=draft,
            pass_results=(first,),
            repairs=(),
            generations=first.generations,
            diagnostic_codes=first.diagnostic_codes,
            exhausted=False,
            phase_generations=first.phase_generations,
            failure_codes=first.diagnostic_codes,
        )
    contradicted = tuple(
        assessment
        for assessment in first.assessments
        if assessment.verdict is ClaimVerdict.CONTRADICTED
    )
    if not contradicted:
        print("DEBUG: no contradictions")
        return VerificationResult(
            text=draft,
            passes=(first.assessments,),
            selections=(first.selections,),
            repairs=(),
            generations=first.generations,
            diagnostic_codes=first.diagnostic_codes,
            exhausted=False,
            failed=False,
            pass_results=(first,),
            phase_generations=first.phase_generations,
        )
    if config.max_repair_passes == 0:
        print("DEBUG: max_repair_passes == 0")
        return _terminal_result(
            text=draft,
            pass_results=(first,),
            repairs=(),
            generations=first.generations,
            diagnostic_codes=(*first.diagnostic_codes, "repair_disabled"),
            exhausted=True,
            phase_generations=first.phase_generations,
            failure_codes=("material_contradiction",),
        )
    claims = {claim.claim_id: claim for claim in first.claims}
    assessments = {assessment.claim_id: assessment for assessment in first.assessments}
    bundles = {bundle.selection.claim_id: bundle for bundle in first.bundles}
    spans = {span.span_id: span for span in first.spans}
    conflicting_spans = tuple(
        span_id
        for span_id in spans
        if {
            assessments[claim.claim_id].verdict
            for claim in first.claims
            if claim.span_id == span_id
        }
        >= {ClaimVerdict.SUPPORTED, ClaimVerdict.CONTRADICTED}
    )
    if conflicting_spans:
        print("DEBUG: conflicting spans")
        return _terminal_result(
            text=draft,
            pass_results=(first,),
            repairs=(),
            generations=first.generations,
            phase_generations=first.phase_generations,
            diagnostic_codes=(*first.diagnostic_codes, "repair_conflicting_assessment"),
            failure_codes=("material_contradiction",),
            exhausted=False,
            limitation_codes=("repair_conflicting_assessment",),
        )
    item_by_span: dict[str, RepairWorkItem] = {}
    for assessment in contradicted:
        claim = claims[assessment.claim_id]
        if claim.is_fallback:
            continue
        fallback = next(
            candidate
            for candidate in first.claims
            if candidate.span_id == claim.span_id and candidate.is_fallback
        )
        if assessments[fallback.claim_id].verdict is not ClaimVerdict.CONTRADICTED:
            continue
        sibling_anchors = tuple(
            candidate.anchor
            for candidate in first.claims
            if candidate.span_id == claim.span_id
            and assessments[candidate.claim_id].verdict is ClaimVerdict.SUPPORTED
        )
        evidence_ids = {
            evidence_id
            for assessment_id in (claim.claim_id, fallback.claim_id)
            for finding in assessments[assessment_id].findings
            for evidence_id in finding.evidence_ids
        }
        bundle = bundles[claim.claim_id]
        evidence = tuple(
            passage for passage in bundle.passages if passage.segment_id in evidence_ids
        )
        if evidence:
            existing = item_by_span.get(claim.span_id)
            if existing is not None:
                evidence = tuple(
                    dict.fromkeys((*existing.evidence, *evidence))
                )
                sibling_anchors = tuple(
                    dict.fromkeys((*existing.preserved_anchors, *sibling_anchors))
                )
                triggers = tuple(dict.fromkeys((*existing.triggering_claim_ids, claim.claim_id)))
            else:
                triggers = (claim.claim_id,)
            item_by_span[claim.span_id] = RepairWorkItem(
                span=spans[claim.span_id],
                triggering_claim_ids=triggers,
                evidence=evidence,
                preserved_anchors=sibling_anchors,
            )
    items = tuple(item_by_span.values())
    if not items:
        print("DEBUG: no items for repair")
        return _terminal_result(
            text=draft,
            pass_results=(first,),
            repairs=(),
            generations=first.generations,
            phase_generations=first.phase_generations,
            diagnostic_codes=(*first.diagnostic_codes, "repair_not_eligible"),
            failure_codes=("material_contradiction",),
            exhausted=False,
            limitation_codes=("repair_not_eligible",),
        )
    try:
        batches = pack_work_items(
            tuple(items),
            render_request=lambda batch: build_repair_request(
                batch, source_id=source_id, runtime=runtime
            ).input_text,
            measure_request=lambda batch: _measure_request_tokens(
                build_repair_request(batch, source_id=source_id, runtime=runtime),
                runtime.counter,
            ),
            runtime=runtime,
            config=config,
        )
    except VerificationCapacityError:
        print("DEBUG: capacity failed")
        return _terminal_result(
            text=draft,
            pass_results=(first,),
            repairs=(),
            generations=first.generations,
            phase_generations=(
                *first.phase_generations,
                VerificationGeneration(
                    GenerationPhase.REPAIR,
                    _pass_index,
                    None,
                    REPAIR_PROMPT_VERSION,
                ),
            ),
            diagnostic_codes=(*first.diagnostic_codes, "repair_capacity_failed"),
            failure_codes=("repair_capacity_failed",),
            exhausted=False,
        )
    proposals: list[Any] = []
    repair_generations: list[Any] = []

    def repair_failure(code: str, *, attempted: bool = False) -> VerificationResult:
        print(f"DEBUG: repair_failure {code}")
        return _terminal_result(
            text=draft,
            pass_results=(first,),
            repairs=(),
            generations=(*first.generations, *repair_generations),
            diagnostic_codes=(*first.diagnostic_codes, code),
            exhausted=False,
            failure_codes=(code,),
            phase_generations=(
                *first.phase_generations,
                *(
                    VerificationGeneration(
                        GenerationPhase.REPAIR,
                        _pass_index,
                        generation,
                        REPAIR_PROMPT_VERSION,
                    )
                    for generation in repair_generations
                ),
                *(
                    (
                        VerificationGeneration(
                            GenerationPhase.REPAIR,
                            _pass_index,
                            None,
                            REPAIR_PROMPT_VERSION,
                        ),
                    )
                    if attempted
                    else ()
                ),
            ),
        )

    try:
        for batch in batches:
            try:
                generation = runtime.provider.generate(
                    build_repair_request(batch, source_id=source_id, runtime=runtime)
                )
            except (ProviderError, VerificationResponseError):
                return repair_failure("repair_provider_failed", attempted=True)
            repair_generations.append(generation)
            proposals.extend(parse_repair_proposals(generation.text, items=batch))
        repaired, events = apply_repairs(
            draft, spans=first.spans, repairs=proposals,
            triggering_claim_ids={item.span.span_id: item.triggering_claim_ids for item in items},
            preserved_anchors={item.span.span_id: item.preserved_anchors for item in items},
        )
    except VerificationResponseError:
        return repair_failure("repair_failed")
    second = verify_draft_once(
        repaired,
        source_id=source_id,
        source_index=source_index,
        runtime=runtime,
        config=config,
        pass_index=_pass_index + 1,
        terminalize_errors=True,
    )
    if second.failed:
        print("DEBUG: second failed")
        return _terminal_result(
            text=draft,
            pass_results=(first, second),
            repairs=events,
            generations=(*first.generations, *repair_generations, *second.generations),
            diagnostic_codes=(*first.diagnostic_codes, *second.diagnostic_codes),
            exhausted=True,
            phase_generations=(
                *first.phase_generations,
                *(
                    VerificationGeneration(
                        GenerationPhase.REPAIR,
                        _pass_index,
                        generation,
                        REPAIR_PROMPT_VERSION,
                    )
                    for generation in repair_generations
                ),
                *second.phase_generations,
            ),
            failure_codes=second.diagnostic_codes,
        )
    failed = any(
        assessment.verdict in {ClaimVerdict.CONTRADICTED, ClaimVerdict.INSUFFICIENTLY_SUPPORTED}
        for assessment in second.assessments
    )
    if failed and not second.failed and config.max_repair_passes > 1:
        print(f"DEBUG: failed={failed}, continuing recursion...")
        continued = _verify_and_repair(
            repaired,
            source_id=source_id,
            source_index=source_index,
            runtime=runtime,
            config=replace(config, max_repair_passes=config.max_repair_passes - 1),
            _pass_index=_pass_index + 2,
        )
        combined_passes = (first, second, *continued.pass_results)
        combined_generations = (
            *first.generations,
            *repair_generations,
            *second.generations,
            *continued.generations,
        )
        combined_phases = (
            *first.phase_generations,
            *(
                VerificationGeneration(
                    GenerationPhase.REPAIR,
                    _pass_index,
                    generation,
                    REPAIR_PROMPT_VERSION,
                )
                for generation in repair_generations
            ),
            *second.phase_generations,
            *continued.phase_generations,
        )
        combined_diagnostics = (
            *first.diagnostic_codes,
            *second.diagnostic_codes,
            "repair_reverification_failed",
            *continued.diagnostic_codes,
        )
        if continued.failed:
            print("DEBUG: recursive call failed")
            return _terminal_result(
                text=repaired,
                pass_results=combined_passes,
                repairs=(*events, *continued.repairs),
                generations=combined_generations,
                diagnostic_codes=combined_diagnostics,
                exhausted=continued.exhausted,
                phase_generations=combined_phases,
                failure_codes=continued.failure_codes,
                limitation_codes=continued.limitation_codes,
            )
        print("DEBUG: recursive call succeeded")
        return VerificationResult(
            text=continued.text,
            passes=tuple(item.assessments for item in combined_passes),
            selections=tuple(item.selections for item in combined_passes),
            repairs=(*events, *continued.repairs),
            generations=combined_generations,
            diagnostic_codes=combined_diagnostics,
            exhausted=continued.exhausted,
            failed=False,
            pass_results=combined_passes,
            phase_generations=combined_phases,
        )
    print("DEBUG: returning final repaired result")
    return VerificationResult(
        text=repaired,
        passes=(first.assessments, second.assessments),
        selections=(first.selections, second.selections),
        repairs=events,
        generations=(*first.generations, *repair_generations, *second.generations),
        diagnostic_codes=(*first.diagnostic_codes, *second.diagnostic_codes),
        exhausted=False,
        failed=False,
        pass_results=(first, second),
        phase_generations=(
            *first.phase_generations,
            *(
                VerificationGeneration(
                    GenerationPhase.REPAIR,
                    _pass_index,
                    generation,
                    REPAIR_PROMPT_VERSION,
                )
                for generation in repair_generations
            ),
            *second.phase_generations,
        ),
    )
