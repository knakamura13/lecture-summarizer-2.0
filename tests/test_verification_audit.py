import hashlib
import json
from dataclasses import replace

import pytest

from summarizer.audit import AuditArtifact, AuditError, build_audit_artifact, serialize_audit
from summarizer.direct import whole_document_segment
from summarizer.grounding import SourcePassage
from summarizer.hierarchy import TreeNode
from summarizer.ingestion import ingest_text
from summarizer.providers.base import GenerationResult
from summarizer.summaries import SummaryNode
from summarizer.verification import (
    BatchFinding,
    Claim,
    ClaimAssessment,
    ClaimVerdict,
    DraftSpan,
    EvidenceBundle,
    EvidenceSelection,
    GenerationPhase,
    RepairAction,
    RepairEvent,
    VerificationGeneration,
    VerificationPassResult,
    VerificationResult,
)


class CharacterCounter:
    identity = "test:characters"
    exact = True
    monotonic = True

    def count(self, text: str) -> int:
        return len(text)


def _base_inputs():
    document = ingest_text("Source passage says the value is 41.")
    segment = whole_document_segment(document, CharacterCounter())
    summary = SummaryNode.model_validate(
        {
            "summary": "The value is 41.",
            "content_units": [],
            "entities": [],
            "qualifications": [],
            "contradictions": [],
            "quotations": [],
            "provenance": [segment.segment_id],
            "level": 0,
        }
    )
    node = TreeNode("L0N0001", 0, 0, summary, (), (segment.segment_id,))
    return document, segment, node


def _verification(segment_id: str) -> VerificationResult:
    draft = "Claim prose must not persist."
    span = DraftSpan(
        "V01S000001",
        1,
        0,
        len(draft),
        draft,
        hashlib.sha256(draft.encode("utf-8")).hexdigest(),
    )
    claim = Claim("V01C000001", span.span_id, 1, "Claim prose", False)
    selection = EvidenceSelection(
        claim.claim_id,
        (segment_id,),
        (segment_id,),
        (),
        3,
        "lexical-overlap/1",
        True,
    )
    finding = BatchFinding(
        claim.claim_id,
        ClaimVerdict.CONTRADICTED,
        (segment_id,),
        ("Source passage says the value is 41.",),
    )
    assessment = ClaimAssessment(
        claim.claim_id,
        ClaimVerdict.CONTRADICTED,
        (finding,),
        1,
        "verifier-provider",
        "verifier-model",
        "verification-classification/1",
    )
    generation = GenerationResult(
        "provider response containing source prose", "verifier-provider", "verifier-model", 3, 2, "stop"
    )
    phase_generation = VerificationGeneration(
        GenerationPhase.CLASSIFICATION,
        1,
        generation,
        "verification-classification/1",
    )
    result = VerificationPassResult(
        (span,),
        (claim,),
        (assessment,),
        (selection,),
        (EvidenceBundle(selection, (SourcePassage(segment_id, "Source passage says the value is 41."),)),),
        (generation,),
        (phase_generation,),
        (),
    )
    repair = RepairEvent(span.span_id, span.content_hash, (claim.claim_id,), RepairAction.QUALIFY)
    return VerificationResult(
        text=draft,
        passes=((assessment,),),
        selections=((selection,),),
        repairs=(repair,),
        generations=(generation,),
        diagnostic_codes=("retrieval_bounded", "conflicting_evidence"),
        exhausted=False,
        failed=False,
        pass_results=(result,),
        phase_generations=(phase_generation,),
        limitation_codes=("evidence_incomplete",),
        failure_codes=(),
    )


def _artifact(*, verification: VerificationResult | None = None):
    document, segment, node = _base_inputs()
    return build_audit_artifact(
        source_id=document.source_id,
        strategy="direct",
        model="summary-model",
        configuration={
            "provider": "openai",
            "model": "summary-model",
            "timeout_seconds": 30,
            "input_path": "/secret/input.txt",
            "endpoint": "https://user:pass@example.test",
            "api_key": "sk-12345678901234567890",
            "unknown": "must not persist",
        },
        segments=(segment,),
        nodes=(node,),
        root_node_id=node.node_id,
        citations=(),
        verification=verification,
    )


def test_audit_v2_projects_verification_without_prose_or_unsafe_configuration() -> None:
    artifact = _artifact(verification=_verification("D000001"))

    assert artifact.schema_version == "audit/2"
    encoded = serialize_audit(artifact)
    body = json.loads(encoded)
    verification = body["verification"]
    assert verification["enabled"] is True
    assert verification["pass_count"] == 1
    assert verification["warning_codes"] == ["retrieval_bounded", "conflicting_evidence"]
    assert verification["limitation_codes"] == ["evidence_incomplete"]
    assert verification["passes"][0]["claims"] == [
        {"claim_id": "V01C000001", "is_fallback": False, "ordinal": 1, "span_id": "V01S000001"}
    ]
    assert verification["passes"][0]["spans"] == [
        {
            "content_hash": hashlib.sha256(b"Claim prose must not persist.").hexdigest(),
            "end": len("Claim prose must not persist."),
            "ordinal": 1,
            "span_id": "V01S000001",
            "start": 0,
        }
    ]
    assert verification["usage"] == [
        {
            "finish_status": "stop",
            "input_tokens": 3,
            "model": "verifier-model",
            "output_tokens": 2,
            "pass_index": 1,
            "phase": "classification",
            "prompt_version": "verification-classification/1",
            "provider": "verifier-provider",
        }
    ]
    assert body["configuration"] == {
        "model": "summary-model", "provider": "openai", "timeout_seconds": 30
    }
    for forbidden in (
        "Claim prose must not persist",
        "Source passage says the value is 41.",
        "provider response containing source prose",
        "/secret/input.txt",
        "user:pass",
        "sk-12345678901234567890",
        "must not persist",
    ):
        assert forbidden.encode("utf-8") not in encoded


def test_audit_v2_explicitly_records_disabled_verification() -> None:
    body = json.loads(serialize_audit(_artifact()))

    assert body["schema_version"] == "audit/2"
    assert body["verification"] == {
        "enabled": False,
        "exhausted": False,
        "failed": False,
        "failure_codes": [],
        "limitation_codes": [],
        "pass_count": 0,
        "passes": [],
        "repairs": [],
        "usage": [],
        "warning_codes": [],
    }

    disabled_result = VerificationResult(
        text="Disabled draft prose does not persist.",
        passes=(),
        selections=(),
        repairs=(),
        generations=(),
        diagnostic_codes=(),
        exhausted=False,
        failed=False,
    )
    with_result = json.loads(serialize_audit(_artifact(verification=disabled_result)))

    assert with_result["verification"] == body["verification"]


def test_audit_projects_repairs_from_a_failed_result_that_kept_its_own_lineage() -> None:
    """A failed result can still carry non-empty repairs: its own re-verified
    fix survived even though a deeper continuation later failed closed."""
    failed_with_repair = replace(
        _verification("D000001"),
        failed=True,
        exhausted=True,
        failure_codes=("repair_reverification_failed",),
    )

    body = json.loads(serialize_audit(_artifact(verification=failed_with_repair)))

    verification = body["verification"]
    assert verification["failed"] is True
    assert verification["repairs"] == [
        {
            "action": "qualify",
            "original_hash": hashlib.sha256(b"Claim prose must not persist.").hexdigest(),
            "span_id": "V01S000001",
            "triggering_claim_ids": ["V01C000001"],
        }
    ]


@pytest.mark.parametrize(
    "configuration",
    (
        {"model": "/private/source.txt"},
        {"model": "https://user:pass@example.test"},
        {"model": "sk-12345678901234567890"},
        {"model": "source prose must not be retained"},
        {"provider": "openai", "timeout_seconds": 0},
    ),
)
def test_audit_v2_builder_rejects_unsafe_allowlisted_configuration_values(configuration) -> None:
    document, segment, node = _base_inputs()

    with pytest.raises((AuditError, ValueError), match="configuration"):
        build_audit_artifact(
            source_id=document.source_id,
            strategy="direct",
            model="summary-model",
            configuration=configuration,
            segments=(segment,),
            nodes=(node,),
            root_node_id=node.node_id,
            citations=(),
        )


@pytest.mark.parametrize(
    "configuration",
    (
        {"model": "/private/source.txt"},
        {"app": {"provider": "openai", "api_key": "sk-12345678901234567890"}},
        {"provider": "openai", "source_text": "raw source prose"},
        {"model": "source prose must not be retained"},
    ),
)
def test_audit_v2_direct_model_validation_rejects_configuration_leakage(configuration) -> None:
    body = _artifact().model_dump(mode="json")
    body["configuration"] = configuration

    with pytest.raises(ValueError, match="configuration"):
        AuditArtifact.model_validate(body)


def test_audit_v2_accepts_span_local_claim_ordinals() -> None:
    document, segment, node = _base_inputs()
    draft = "First claim. Second claim."
    first = DraftSpan(
        "V01S000001", 1, 0, 13, "First claim. ",
        hashlib.sha256(b"First claim. ").hexdigest(),
    )
    second = DraftSpan(
        "V01S000002", 2, 13, len(draft), "Second claim.",
        hashlib.sha256(b"Second claim.").hexdigest(),
    )
    claims = (
        Claim("V01C000001", first.span_id, 1, "First claim", False),
        Claim("V01C000002", second.span_id, 1, "Second claim", False),
    )
    selections = tuple(
        EvidenceSelection(claim.claim_id, (segment.segment_id,), (segment.segment_id,), (), 1, "lexical-overlap/1", True)
        for claim in claims
    )
    assessments = tuple(
        ClaimAssessment(
            claim.claim_id,
            ClaimVerdict.SUPPORTED,
            (BatchFinding(claim.claim_id, ClaimVerdict.SUPPORTED, (segment.segment_id,), ("source quote",)),),
            1,
            "verifier-provider",
            "verifier-model",
            "verification-classification/1",
        )
        for claim in claims
    )
    generation = GenerationResult("provider response prose", "verifier-provider", "verifier-model")
    phase_generation = VerificationGeneration(
        GenerationPhase.CLASSIFICATION, 1, generation, "verification-classification/1"
    )
    pass_result = VerificationPassResult(
        (first, second),
        claims,
        assessments,
        selections,
        tuple(EvidenceBundle(selection, ()) for selection in selections),
        (generation,),
        (phase_generation,),
        (),
    )
    verification = VerificationResult(
        text=draft,
        passes=(assessments,),
        selections=(selections,),
        repairs=(),
        generations=(generation,),
        diagnostic_codes=(),
        exhausted=False,
        failed=False,
        pass_results=(pass_result,),
        phase_generations=(phase_generation,),
    )

    artifact = build_audit_artifact(
        source_id=document.source_id,
        strategy="direct",
        model="summary-model",
        configuration={"provider": "openai", "model": "summary-model", "timeout_seconds": 30},
        segments=(segment,),
        nodes=(node,),
        root_node_id=node.node_id,
        citations=(),
        verification=verification,
    )

    assert [claim.ordinal for claim in artifact.verification.passes[0].claims] == [1, 1]


def test_audit_v2_records_terminal_failure_as_closed_metadata() -> None:
    failed = replace(
        _verification("D000001"),
        exhausted=True,
        failed=True,
        failure_codes=("repair_failed",),
    )

    verification = json.loads(serialize_audit(_artifact(verification=failed)))["verification"]

    assert verification["exhausted"] is True
    assert verification["failed"] is True
    assert verification["failure_codes"] == ["repair_failed"]


def test_audit_v2_marks_terminal_malformed_classification_pass_incomplete() -> None:
    complete = _verification("D000001")
    partial_pass = replace(complete.pass_results[0], assessments=(), failed=True)
    partial = replace(
        complete,
        passes=((),),
        pass_results=(partial_pass,),
        failed=True,
        failure_codes=("classification_failed",),
    )

    verification = json.loads(serialize_audit(_artifact(verification=partial)))["verification"]

    assert verification["failed"] is True
    assert verification["passes"][0]["complete"] is False
    assert verification["passes"][0]["claims"]
    assert verification["passes"][0]["assessments"] == []
    assert verification["failure_codes"] == ["classification_failed"]


def test_audit_v2_serializes_terminal_decomposition_failure_without_claim_prose() -> None:
    generation = GenerationResult("malformed provider prose", "provider", "model")
    result = VerificationResult(
        text="Draft prose must not persist.",
        passes=((),),
        selections=((),),
        repairs=(),
        generations=(generation,),
        diagnostic_codes=("decomposition_failed",),
        exhausted=False,
        failed=True,
        pass_results=(
            VerificationPassResult(
                (), (), (), (), (), (generation,),
                (
                    VerificationGeneration(
                        GenerationPhase.DECOMPOSITION,
                        1,
                        generation,
                        "verification-decomposition/1",
                    ),
                ),
                ("decomposition_failed",),
                failed=True,
            ),
        ),
        phase_generations=(
            VerificationGeneration(
                GenerationPhase.DECOMPOSITION,
                1,
                generation,
                "verification-decomposition/1",
            ),
        ),
        failure_codes=("decomposition_failed",),
    )

    encoded = serialize_audit(_artifact(verification=result))
    verification = json.loads(encoded)["verification"]

    assert verification["failed"] is True
    assert verification["passes"] == [
        {
            "assessments": [],
            "claims": [],
            "complete": False,
            "pass_index": 1,
            "selections": [],
            "spans": [],
        }
    ]
    assert b"malformed provider prose" not in encoded
    assert b"Draft prose must not persist." not in encoded


def test_audit_v2_retains_no_response_attempt_identity_without_usage_values() -> None:
    phase_generation = VerificationGeneration(
        GenerationPhase.DECOMPOSITION,
        1,
        None,
        "verification-decomposition/1",
    )
    result = VerificationResult(
        text="No-response draft prose does not persist.",
        passes=((),),
        selections=((),),
        repairs=(),
        generations=(),
        diagnostic_codes=("decomposition_provider_failed",),
        exhausted=False,
        failed=True,
        pass_results=(
            VerificationPassResult(
                (), (), (), (), (), (), (phase_generation,),
                ("decomposition_provider_failed",),
                failed=True,
            ),
        ),
        phase_generations=(phase_generation,),
        failure_codes=("decomposition_provider_failed",),
    )

    encoded = serialize_audit(_artifact(verification=result))
    verification = json.loads(encoded)["verification"]

    assert verification["passes"][0]["complete"] is False
    assert verification["usage"] == [
        {
            "finish_status": None,
            "input_tokens": None,
            "model": None,
            "output_tokens": None,
            "pass_index": 1,
            "phase": "decomposition",
            "prompt_version": "verification-decomposition/1",
            "provider": None,
        }
    ]
    assert b"No-response draft prose does not persist." not in encoded


@pytest.mark.parametrize(
    ("path", "value"),
    (
        (("verification", "passes", 0, "selections", 0, "retrieval_method"), "raw prose"),
        (("verification", "passes", 0, "assessments", 0, "verifier_model"), "raw prose"),
        (("verification", "usage", 0, "finish_status"), "raw prose"),
    ),
)
def test_audit_v2_direct_model_validation_rejects_prose_metadata(path, value) -> None:
    body = _artifact(verification=_verification("D000001")).model_dump(mode="json")
    target = body
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value

    with pytest.raises(ValueError):
        AuditArtifact.model_validate(body)


def test_audit_v2_accepts_versioned_model_tags_as_metadata_identities() -> None:
    body = _artifact(verification=_verification("D000001")).model_dump(mode="json")
    body["model"] = "llama3.2:3b"
    body["configuration"]["model"] = "llama3.2:3b"
    body["verification"]["passes"][0]["assessments"][0]["verifier_model"] = "llama3.2:3b"
    body["verification"]["usage"][0]["model"] = "llama3.2:3b"

    artifact = AuditArtifact.model_validate(body)

    assert artifact.model == "llama3.2:3b"


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("verification", "passes", 0, "claims", 0, "span_id"), "V01S999999"),
        (("verification", "passes", 0, "selections", 0, "claim_id"), "V01C999999"),
        (("verification", "passes", 0, "assessments", 0, "findings", 0, "evidence_ids", 0), "D999999"),
        (("verification", "repairs", 0, "span_id"), "V01S999999"),
    ],
)
def test_audit_v2_rejects_unresolved_verification_links(path, value) -> None:
    body = _artifact(verification=_verification("D000001")).model_dump(mode="json")
    target = body
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value

    with pytest.raises(ValueError, match="verification"):
        AuditArtifact.model_validate(body)


def test_audit_v2_rejects_missing_or_duplicate_claim_records() -> None:
    body = _artifact(verification=_verification("D000001")).model_dump(mode="json")
    body["verification"]["passes"][0]["selections"] = []

    with pytest.raises(ValueError, match="verification"):
        AuditArtifact.model_validate(body)


def test_audit_v2_rejects_nonterminal_partial_pass_and_invalid_span_range() -> None:
    body = _artifact(verification=_verification("D000001")).model_dump(mode="json")
    first = body["verification"]["passes"][0]
    second = json.loads(json.dumps(first).replace("V01", "V03"))
    second["pass_index"] = 3
    body["verification"]["passes"].append(second)
    body["verification"]["pass_count"] = 2
    body["verification"]["passes"][0]["assessments"] = []
    body["verification"]["passes"][0]["complete"] = False
    body["verification"]["failed"] = True
    body["verification"]["usage"][0]["pass_index"] = 3

    with pytest.raises(ValueError, match="terminal failed"):
        AuditArtifact.model_validate(body)

    body = _artifact(verification=_verification("D000001")).model_dump(mode="json")
    body["verification"]["passes"][0]["spans"][0]["end"] = 0

    with pytest.raises(ValueError, match="span ranges"):
        AuditArtifact.model_validate(body)

    body = _artifact(verification=_verification("D000001")).model_dump(mode="json")
    claims = body["verification"]["passes"][0]["claims"]
    claims.append(claims[0].copy())

    with pytest.raises(ValueError, match="verification"):
        AuditArtifact.model_validate(body)


def test_audit_v2_rejects_diagnostic_prose() -> None:
    body = _artifact(verification=_verification("D000001")).model_dump(mode="json")
    body["verification"]["warning_codes"] = ["source prose must not persist"]

    with pytest.raises(ValueError, match="closed identifiers"):
        AuditArtifact.model_validate(body)


def test_audit_v2_preserves_noncontiguous_reverification_pass_identifiers() -> None:
    body = _artifact(verification=_verification("D000001")).model_dump(mode="json")
    body = json.loads(json.dumps(body).replace("V01", "V03"))
    body["verification"]["passes"][0]["pass_index"] = 3
    body["verification"]["usage"][0]["pass_index"] = 3

    artifact = AuditArtifact.model_validate(body)

    assert artifact.verification.passes[0].pass_index == 3
