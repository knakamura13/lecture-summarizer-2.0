import json

import pytest

from summarizer.audit import (
    AuditError,
    build_audit_artifact,
    render_citations,
    resolve_citations,
    serialize_audit,
    write_audit,
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


def fixture():
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
    citations = resolve_citations(
        summary.provenance, source_id=document.source_id, segments=(segment,)
    )
    return document, segment, node, citations


def test_audit_is_canonical_redacted_and_contains_only_segment_metadata(tmp_path) -> None:
    document, segment, node, citations = fixture()
    artifact = build_audit_artifact(
        source_id=document.source_id,
        strategy="direct",
        model="m",
        configuration={
            "app": {
                "provider": "openai",
                "model": "m",
                "timeout_seconds": 30,
                "input_path": "/private/input.txt",
                "ollama_host": "https://user:pass@example.test",
                "openai_api_key": "sk-12345678901234567890",
            }
        },
        segments=(segment,),
        nodes=(node,),
        root_node_id=node.node_id,
        citations=citations,
        generations=(GenerationResult("ok", "fake", "m", 3, 4, "stop", "unstable-id"),),
        warnings=("bounded_retrieval",),
    )

    first = serialize_audit(artifact)
    second = serialize_audit(artifact)
    assert first == second
    assert b"sk-12345678901234567890" not in first
    assert b"user:pass" not in first
    assert b"unstable-id" not in first
    assert b"/private/input.txt" not in first
    assert b"ghp_123456789012345678901234567890123456" not in first
    body = json.loads(first)
    assert "text" not in body["source_segments"][0]
    assert "text" not in body["tree_nodes"][0]["summary"]
    assert body["configuration"] == {
        "app": {"model": "m", "provider": "openai", "timeout_seconds": 30}
    }
    assert body["citations"] == [{"order": 0, "segment_id": "D000001", "source_id": document.source_id}]

    path = tmp_path / "audit.json"
    write_audit(path, artifact)
    assert path.read_bytes() == first


def test_citations_are_source_ordered_and_unknown_provenance_fails() -> None:
    document, segment, _, _ = fixture()
    assert render_citations("Text.", resolve_citations((segment.segment_id,), source_id=document.source_id, segments=(segment,))) == "Text.\n\nSources: D000001"
    with pytest.raises(AuditError, match="unknown"):
        resolve_citations(("S999999",), source_id=document.source_id, segments=(segment,))


def test_audit_accepts_openai_completed_finish_status() -> None:
    document, segment, node, citations = fixture()
    artifact = build_audit_artifact(
        source_id=document.source_id,
        strategy="direct",
        model="gpt-4o-mini",
        configuration={"provider": "openai", "model": "gpt-4o-mini", "timeout_seconds": 30},
        segments=(segment,),
        nodes=(node,),
        root_node_id=node.node_id,
        citations=citations,
        generations=(GenerationResult("ok", "openai", "gpt-4o-mini", finish_status="completed"),),
    )

    body = json.loads(serialize_audit(artifact))

    assert body["usage"] == [
        {
            "finish_status": "completed",
            "input_tokens": None,
            "model": "gpt-4o-mini",
            "output_tokens": None,
            "provider": "openai",
        }
    ]
