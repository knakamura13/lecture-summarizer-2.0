"""Tests for audit/3 reliability metadata (cache, resume, retry)."""

import json
import pytest

from summarizer.audit import (
    AuditError,
    build_audit_artifact,
    serialize_audit,
)
from summarizer.direct import whole_document_segment
from summarizer.hierarchy import TreeNode
from summarizer.ingestion import ingest_text
from summarizer.providers.base import GenerationResult
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
    """audit/3 includes cache hit/miss work items and invalidation reasons."""
    document, segment, node = _fixture()
    artifact = build_audit_artifact(
        source_id=document.source_id,
        strategy="direct",
        model="gpt-4o-mini",
        configuration={"provider": "openai", "model": "gpt-4o-mini", "timeout_seconds": 30},
        segments=(segment,),
        nodes=(node,),
        root_node_id=node.node_id,
        citations=(),
        generations=(),
        reliability_cache={
            "cache_hits": ("segmentation", "leaf_L0N0001"),
            "cache_misses": ("merge_L1N0001",),
            "invalidation_reasons": (),
        },
    )

    payload = serialize_audit(artifact)
    body = json.loads(payload)

    assert body["schema_version"] == "audit/3"
    assert "cache" in body.get("reliability", {})
    assert body["reliability"]["cache"]["cache_hits"] == ["segmentation", "leaf_L0N0001"]
    assert body["reliability"]["cache"]["cache_misses"] == ["merge_L1N0001"]
    assert body["reliability"]["cache"]["invalidation_reasons"] == []


def test_audit_v3_records_resume_state() -> None:
    """audit/3 includes resume state and reference count."""
    document, segment, node = _fixture()
    artifact = build_audit_artifact(
        source_id=document.source_id,
        strategy="direct",
        model="gpt-4o-mini",
        configuration={"provider": "openai", "model": "gpt-4o-mini", "timeout_seconds": 30},
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
        configuration={"provider": "openai", "model": "gpt-4o-mini", "timeout_seconds": 30},
        segments=(segment,),
        nodes=(node,),
        root_node_id=node.node_id,
        citations=(),
        generations=(),
        reliability_attempts=[
            {"work_id": "D000001", "attempt_count": 1, "failure_reasons": []},
            {"work_id": "segmentation", "attempt_count": 2, "failure_reasons": ["timeout", "timeout"]},
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
        configuration={"provider": "openai", "model": "gpt-4o-mini", "timeout_seconds": 30},
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
    # reliability field should be null for audit/2 (or absent if using exclude_none)
    assert body.get("reliability") is None


def test_audit_v3_all_reliability_fields_optional() -> None:
    """audit/3 handles partial reliability metadata."""
    document, segment, node = _fixture()
    artifact = build_audit_artifact(
        source_id=document.source_id,
        strategy="direct",
        model="gpt-4o-mini",
        configuration={"provider": "openai", "model": "gpt-4o-mini", "timeout_seconds": 30},
        segments=(segment,),
        nodes=(node,),
        root_node_id=node.node_id,
        citations=(),
        generations=(),
        reliability_cache={"cache_hits": (), "cache_misses": (), "invalidation_reasons": ()},
        # resume and attempts omitted
    )

    payload = serialize_audit(artifact)
    body = json.loads(payload)

    assert body["schema_version"] == "audit/3"
    assert body["reliability"]["cache"]["cache_hits"] == []
    assert body["reliability"]["resume"] is None
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
            configuration={"provider": "openai", "model": "gpt-4o-mini", "timeout_seconds": 30},
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
        configuration={"provider": "openai", "model": "gpt-4o-mini", "timeout_seconds": 30},
        segments=(segment,),
        nodes=(node,),
        root_node_id=node.node_id,
        citations=(),
        generations=(),
        reliability_cache={"cache_hits": ("segmentation",), "cache_misses": (), "invalidations": ()},
        reliability_resume={"resumed": False, "reused_count": 0, "recomputed_count": 0},
        reliability_attempts=[],
    )

    payload = serialize_audit(artifact)

    # No credential patterns in serialized output
    assert b"sk-" not in payload
    assert b"sk_" not in payload
    assert b"ghp_" not in payload
