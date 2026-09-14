"""Tests for audit/3 reliability metadata (cache, resume, retry)."""

import json

import pytest

from summarizer.audit import (
    AuditArtifact,
    AuditError,
    build_audit_artifact,
    serialize_audit,
)
from summarizer.direct import whole_document_segment
from summarizer.hierarchy import TreeNode
from summarizer.ingestion import ingest_text
from summarizer.summaries import SummaryNode


class CharacterCounter:
    identity = "test:characters"
    exact = True
    monotonic = True

    def count(self, text: str) -> int:
        return len(text)


def _fixture():
    document = ingest_text("The credential was sk-12345678901234567890.")
    segment = whole_document_segment(document, CharacterCounter())
    summary = SummaryNode.model_validate(
        {
            "summary": "The credential was sk-12345678901234567890.",
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


def test_audit_v3_records_cache_metadata() -> None:
    """audit/3 includes only closed cache outcome and invalidation codes."""
    document, segment, node = _fixture()
    artifact = build_audit_artifact(
        source_id=document.source_id,
        strategy="direct",
        model="gpt-4o-mini",
        configuration={
            "provider": "openai",
            "model": "gpt-4o-mini",
            "timeout_seconds": 30,
        },
        segments=(segment,),
        nodes=(node,),
        root_node_id=node.node_id,
        citations=(),
        generations=(),
        reliability_cache={
            "cache_hits": ("hit",),
            "cache_misses": ("missing",),
            "invalidation_reasons": ("source_changed",),
        },
    )

    payload = serialize_audit(artifact)
    body = json.loads(payload)

    assert body["schema_version"] == "audit/3"
    assert "cache" in body.get("reliability", {})
    assert body["reliability"]["cache"]["cache_hits"] == ["hit"]
    assert body["reliability"]["cache"]["cache_misses"] == ["missing"]
    assert body["reliability"]["cache"]["invalidation_reasons"] == ["source_changed"]


def test_audit_v3_records_resume_state() -> None:
    """audit/3 includes resume state and reference count."""
    document, segment, node = _fixture()
    artifact = build_audit_artifact(
        source_id=document.source_id,
        strategy="direct",
        model="gpt-4o-mini",
        configuration={
            "provider": "openai",
            "model": "gpt-4o-mini",
            "timeout_seconds": 30,
        },
        segments=(segment,),
        nodes=(node,),
        root_node_id=node.node_id,
        citations=(),
        generations=(),
        reliability_resume={
            "resumed": True,
            "reused_count": 3,
            "recomputed_count": 1,
        },
    )

    payload = serialize_audit(artifact)
    body = json.loads(payload)

    assert body["schema_version"] == "audit/3"
    assert body["reliability"]["resumed"] is True
    assert body["reliability"]["reused_count"] == 3
    assert body["reliability"]["recomputed_count"] == 1


def test_audit_v3_records_retry_attempts() -> None:
    """audit/3 includes retry attempt counts and reasons."""
    document, segment, node = _fixture()
    artifact = build_audit_artifact(
        source_id=document.source_id,
        strategy="direct",
        model="gpt-4o-mini",
        configuration={
            "provider": "openai",
            "model": "gpt-4o-mini",
            "timeout_seconds": 30,
        },
        segments=(segment,),
        nodes=(node,),
        root_node_id=node.node_id,
        citations=(),
        generations=(),
        reliability_attempts=[
            {"work_id": "D000001", "attempt_count": 1, "failure_reasons": []},
            {
                "work_id": "segmentation",
                "attempt_count": 2,
                "failure_reasons": ["timeout", "timeout"],
            },
        ],
    )

    payload = serialize_audit(artifact)
    body = json.loads(payload)

    assert body["schema_version"] == "audit/3"
    attempts = body["reliability"]["attempts"]
    assert len(attempts) == 2
    assert attempts[0]["work_id"] == "D000001"
    assert attempts[0]["attempt_count"] == 1
    assert attempts[1]["attempt_count"] == 2
    assert "timeout" in attempts[1]["failure_reasons"]


def test_audit_v2_compatibility_no_reliability_fields() -> None:
    """audit/2 artifacts without reliability metadata serialize correctly."""
    document, segment, node = _fixture()
    artifact = build_audit_artifact(
        source_id=document.source_id,
        strategy="direct",
        model="gpt-4o-mini",
        configuration={
            "provider": "openai",
            "model": "gpt-4o-mini",
            "timeout_seconds": 30,
        },
        segments=(segment,),
        nodes=(node,),
        root_node_id=node.node_id,
        citations=(),
        generations=(),
    )

    payload = serialize_audit(artifact)
    body = json.loads(payload)

    # audit/2 default when no reliability data
    assert body["schema_version"] == "audit/2"
    assert "reliability" not in body
    assert "grounding" not in body["tree_nodes"][0]


def test_audit_v3_all_reliability_fields_optional() -> None:
    """audit/3 handles partial reliability metadata."""
    document, segment, node = _fixture()
    artifact = build_audit_artifact(
        source_id=document.source_id,
        strategy="direct",
        model="gpt-4o-mini",
        configuration={
            "provider": "openai",
            "model": "gpt-4o-mini",
            "timeout_seconds": 30,
        },
        segments=(segment,),
        nodes=(node,),
        root_node_id=node.node_id,
        citations=(),
        generations=(),
        reliability_cache={
            "cache_hits": (),
            "cache_misses": (),
            "invalidation_reasons": (),
        },
        # resume and attempts omitted
    )

    payload = serialize_audit(artifact)
    body = json.loads(payload)

    assert body["schema_version"] == "audit/3"
    assert body["reliability"]["cache"]["cache_hits"] == []
    assert body["reliability"]["resumed"] is None
    assert body["reliability"]["reused_count"] is None
    assert body["reliability"]["recomputed_count"] is None
    assert body["reliability"]["attempts"] == []


def test_audit_v3_materializes_for_explicitly_empty_retry_metadata() -> None:
    document, segment, node = _fixture()
    artifact = build_audit_artifact(
        source_id=document.source_id,
        strategy="direct",
        model="gpt-4o-mini",
        configuration={
            "provider": "openai",
            "model": "gpt-4o-mini",
            "timeout_seconds": 30,
        },
        segments=(segment,),
        nodes=(node,),
        root_node_id=node.node_id,
        citations=(),
        reliability_attempts=(),
    )

    body = json.loads(serialize_audit(artifact))

    assert body["schema_version"] == "audit/3"
    assert body["reliability"]["attempts"] == []


def test_audit_closed_codes_validate_format() -> None:
    """Invalidation reasons must be lowercase identifiers."""
    document, segment, node = _fixture()

    # Invalid invalidation reason with uppercase
    with pytest.raises((ValueError, AuditError)):
        build_audit_artifact(
            source_id=document.source_id,
            strategy="direct",
            model="gpt-4o-mini",
            configuration={
                "provider": "openai",
                "model": "gpt-4o-mini",
                "timeout_seconds": 30,
            },
            segments=(segment,),
            nodes=(node,),
            root_node_id=node.node_id,
            citations=(),
            generations=(),
            reliability_cache={
                "cache_hits": (),
                "cache_misses": (),
                "invalidation_reasons": ("Invalid_Code",),
            },
        )


def test_audit_v3_redacts_no_secrets_in_reliability() -> None:
    """Reliability metadata contains no raw source text, prompts, or credentials."""
    document, segment, node = _fixture()
    artifact = build_audit_artifact(
        source_id=document.source_id,
        strategy="direct",
        model="gpt-4o-mini",
        configuration={
            "provider": "openai",
            "model": "gpt-4o-mini",
            "timeout_seconds": 30,
        },
        segments=(segment,),
        nodes=(node,),
        root_node_id=node.node_id,
        citations=(),
        generations=(),
        reliability_cache={
            "cache_hits": ("hit",),
            "cache_misses": (),
            "invalidation_reasons": (),
        },
        reliability_resume={"resumed": False, "reused_count": 0, "recomputed_count": 0},
        reliability_attempts=[],
    )

    payload = serialize_audit(artifact)

    # No credential patterns in serialized output
    assert b"sk-" not in payload
    assert b"sk_" not in payload
    assert b"ghp_" not in payload


@pytest.mark.parametrize(
    ("path", "value"),
    (
        (("cache", "cache_hits"), ["/private/cache"]),
        (("cache", "cache_misses"), ["https://user:pass@example.test"]),
        (("cache", "invalidation_reasons"), ["source prose must not persist"]),
        (("attempts", 0, "work_id"), "C:/cache-root"),
        (("attempts", 0, "failure_reasons"), ["prompt contents"]),
    ),
)
def test_audit_v3_direct_validation_rejects_unsafe_reliability_metadata(
    path, value
) -> None:
    document, segment, node = _fixture()
    artifact = build_audit_artifact(
        source_id=document.source_id,
        strategy="direct",
        model="gpt-4o-mini",
        configuration={
            "provider": "openai",
            "model": "gpt-4o-mini",
            "timeout_seconds": 30,
        },
        segments=(segment,),
        nodes=(node,),
        root_node_id=node.node_id,
        citations=(),
        reliability_cache={
            "cache_hits": ("hit",),
            "cache_misses": (),
            "invalidation_reasons": (),
        },
        reliability_attempts=[
            {"work_id": "D000001", "attempt_count": 1, "failure_reasons": []}
        ],
    )
    body = json.loads(serialize_audit(artifact))
    target = body["reliability"]
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value

    with pytest.raises(ValueError):
        AuditArtifact.model_validate(body)


def test_audit_v3_rejects_credential_like_retry_work_ids() -> None:
    document, segment, node = _fixture()
    kwargs = {
        "source_id": document.source_id,
        "strategy": "direct",
        "model": "gpt-4o-mini",
        "configuration": {
            "provider": "openai",
            "model": "gpt-4o-mini",
            "timeout_seconds": 30,
        },
        "segments": (segment,),
        "nodes": (node,),
        "root_node_id": node.node_id,
        "citations": (),
        "reliability_attempts": [
            {
                "work_id": "Dsk-12345678901234567890",
                "attempt_count": 1,
                "failure_reasons": (),
            }
        ],
    }

    with pytest.raises(ValueError, match="work_id"):
        build_audit_artifact(**kwargs)

    valid = build_audit_artifact(
        **{
            **kwargs,
            "reliability_attempts": [
                {
                    "work_id": "D000001",
                    "attempt_count": 1,
                    "failure_reasons": (),
                }
            ],
        }
    )
    body = json.loads(serialize_audit(valid))
    body["reliability"]["attempts"][0]["work_id"] = "Dsk-12345678901234567890"

    with pytest.raises(ValueError, match="work_id"):
        AuditArtifact.model_validate(body)


@pytest.mark.parametrize(
    "work_id",
    (
        "Dignore-previous-instructions",
        "Dsystem:reveal-secret",
        "Dlocalhost:8080",
    ),
)
def test_audit_v3_rejects_prompt_and_host_retry_work_ids(work_id) -> None:
    document, segment, node = _fixture()
    kwargs = {
        "source_id": document.source_id,
        "strategy": "direct",
        "model": "gpt-4o-mini",
        "configuration": {
            "provider": "openai",
            "model": "gpt-4o-mini",
            "timeout_seconds": 30,
        },
        "segments": (segment,),
        "nodes": (node,),
        "root_node_id": node.node_id,
        "citations": (),
    }

    with pytest.raises(ValueError, match="work_id"):
        build_audit_artifact(
            **{
                **kwargs,
                "reliability_attempts": [
                    {
                        "work_id": work_id,
                        "attempt_count": 1,
                        "failure_reasons": (),
                    }
                ],
            }
        )

    valid = build_audit_artifact(
        **{
            **kwargs,
            "reliability_attempts": [
                {
                    "work_id": "D000001",
                    "attempt_count": 1,
                    "failure_reasons": (),
                }
            ],
        }
    )
    body = json.loads(serialize_audit(valid))
    body["reliability"]["attempts"][0]["work_id"] = work_id

    with pytest.raises(ValueError, match="work_id"):
        AuditArtifact.model_validate(body)


@pytest.mark.parametrize(
    "work_id",
    (
        "D000001",
        "S000001",
        "L0N0001",
        "M0N0001",
        "M000001",
        "V01",
        "V01C000001",
        "V01S000001",
        "editorial-final",
        "segmentation",
    ),
)
def test_audit_v3_accepts_established_reliability_identifiers(work_id) -> None:
    document, segment, node = _fixture()
    artifact = build_audit_artifact(
        source_id=document.source_id,
        strategy="direct",
        model="gpt-4o-mini",
        configuration={
            "provider": "openai",
            "model": "gpt-4o-mini",
            "timeout_seconds": 30,
        },
        segments=(segment,),
        nodes=(node,),
        root_node_id=node.node_id,
        citations=(),
        reliability_attempts=[
            {"work_id": work_id, "attempt_count": 1, "failure_reasons": ()}
        ],
    )

    body = json.loads(serialize_audit(artifact))

    assert body["reliability"]["attempts"][0]["work_id"] == work_id
    assert AuditArtifact.model_validate(body).reliability.attempts[0].work_id == work_id


@pytest.mark.parametrize("value", (1.9, True, "2"))
def test_audit_v3_rejects_noninteger_retry_attempt_counts(value) -> None:
    document, segment, node = _fixture()
    kwargs = {
        "source_id": document.source_id,
        "strategy": "direct",
        "model": "gpt-4o-mini",
        "configuration": {
            "provider": "openai",
            "model": "gpt-4o-mini",
            "timeout_seconds": 30,
        },
        "segments": (segment,),
        "nodes": (node,),
        "root_node_id": node.node_id,
        "citations": (),
    }

    with pytest.raises(ValueError, match="attempt_count"):
        build_audit_artifact(
            **{
                **kwargs,
                "reliability_attempts": [
                    {
                        "work_id": "D000001",
                        "attempt_count": value,
                        "failure_reasons": (),
                    }
                ],
            }
        )

    valid = build_audit_artifact(
        **{
            **kwargs,
            "reliability_attempts": [
                {
                    "work_id": "D000001",
                    "attempt_count": 1,
                    "failure_reasons": (),
                }
            ],
        }
    )
    body = json.loads(serialize_audit(valid))
    body["reliability"]["attempts"][0]["attempt_count"] = value

    with pytest.raises(ValueError, match="attempt_count"):
        AuditArtifact.model_validate(body)


@pytest.mark.parametrize("field", ("reused_count", "recomputed_count"))
@pytest.mark.parametrize("value", (1.9, True, "2"))
def test_audit_v3_rejects_noninteger_resume_counts(field, value) -> None:
    document, segment, node = _fixture()
    kwargs = {
        "source_id": document.source_id,
        "strategy": "direct",
        "model": "gpt-4o-mini",
        "configuration": {
            "provider": "openai",
            "model": "gpt-4o-mini",
            "timeout_seconds": 30,
        },
        "segments": (segment,),
        "nodes": (node,),
        "root_node_id": node.node_id,
        "citations": (),
    }
    resume = {"resumed": False, "reused_count": 0, "recomputed_count": 0}
    resume[field] = value

    with pytest.raises(ValueError, match="resume counts"):
        build_audit_artifact(**{**kwargs, "reliability_resume": resume})

    valid = build_audit_artifact(
        **{
            **kwargs,
            "reliability_resume": {
                "resumed": False,
                "reused_count": 0,
                "recomputed_count": 0,
            },
        }
    )
    body = json.loads(serialize_audit(valid))
    body["reliability"][field] = value

    with pytest.raises(ValueError, match="resume counts"):
        AuditArtifact.model_validate(body)


def test_audit_versions_reject_cross_version_reliability_fields() -> None:
    document, segment, node = _fixture()
    artifact = build_audit_artifact(
        source_id=document.source_id,
        strategy="direct",
        model="gpt-4o-mini",
        configuration={
            "provider": "openai",
            "model": "gpt-4o-mini",
            "timeout_seconds": 30,
        },
        segments=(segment,),
        nodes=(node,),
        root_node_id=node.node_id,
        citations=(),
    )
    body = json.loads(serialize_audit(artifact))
    body["reliability"] = {}

    with pytest.raises(ValueError):
        AuditArtifact.model_validate(body)


def test_audit_v3_requires_reliability_and_rejects_unknown_versions() -> None:
    document, segment, node = _fixture()
    artifact = build_audit_artifact(
        source_id=document.source_id,
        strategy="direct",
        model="gpt-4o-mini",
        configuration={
            "provider": "openai",
            "model": "gpt-4o-mini",
            "timeout_seconds": 30,
        },
        segments=(segment,),
        nodes=(node,),
        root_node_id=node.node_id,
        citations=(),
        reliability_resume={"resumed": False, "reused_count": 0, "recomputed_count": 1},
    )
    body = json.loads(serialize_audit(artifact))
    body.pop("reliability")

    with pytest.raises(ValueError):
        AuditArtifact.model_validate(body)

    body["schema_version"] = "audit/4"
    with pytest.raises(ValueError):
        AuditArtifact.model_validate(body)
