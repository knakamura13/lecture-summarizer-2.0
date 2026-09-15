import json
from dataclasses import replace

import pytest

from summarizer.audit import (
    AuditArtifact,
    AuditError,
    build_audit_artifact,
    render_citations,
    resolve_citations,
    serialize_audit,
    write_audit,
)
from summarizer.direct import whole_document_segment
from summarizer.grounding import GroundingPolicy
from summarizer.hierarchy import TreeNode, build_hierarchy
from summarizer.ingestion import ingest_text
from summarizer.providers.base import GenerationResult
from summarizer.segmentation import SegmentationConfig, segment_document
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


@pytest.mark.parametrize(
    ("mutate", "message"),
    (
        (
            lambda body: body["tree_nodes"][0].update({"covered_segments": ["S000001"]}),
            "tree coverage",
        ),
        (
            lambda body: body.update({"root_node_id": "L9N9999"}),
            "root_node_id",
        ),
        (
            lambda body: body["citations"][0].update({"order": 1}),
            "citation mappings",
        ),
        (
            lambda body: body["source_segments"].append(body["source_segments"][0]),
            "source segment identifiers",
        ),
        (
            lambda body: body.update({"warnings": ["not a closed code"]}),
            "closed identifiers",
        ),
    ),
)
def test_audit_rejects_invalid_link_invariants(mutate, message: str) -> None:
    document, segment, node, citations = fixture()
    body = build_audit_artifact(
        source_id=document.source_id,
        strategy="direct",
        model="m",
        configuration={},
        segments=(segment,),
        nodes=(node,),
        root_node_id=node.node_id,
        citations=citations,
    ).model_dump(mode="json")
    mutate(body)

    with pytest.raises(ValueError, match=message):
        AuditArtifact.model_validate(body)


def test_audit_records_merge_grounding_omissions_from_a_small_reserve() -> None:
    class Provider:
        def generate(self, request):
            level = int((request.operation_id or "merge-L1").rsplit("L", 1)[1])
            grounded_id = next(
                segment.segment_id
                for segment in segments
                if f'"segment_id":"{segment.segment_id}","text":' in request.input_text
            )
            return GenerationResult(
                json.dumps(
                    {
                        "summary": "Merged.",
                        "content_units": [],
                        "entities": [],
                        "qualifications": [],
                        "contradictions": [],
                        "quotations": [],
                        "provenance": [grounded_id],
                        "level": level,
                    }
                ),
                "fake",
                request.model,
            )

    document = ingest_text(
        "one two three four five six seven eight nine ten " * 12
    )
    segments = segment_document(
        document, CharacterCounter(), SegmentationConfig(max_tokens=20)
    )
    leaves = tuple(
        SummaryNode.model_validate(
            {
                "summary": "Leaf.",
                "content_units": [],
                "entities": [],
                "qualifications": [],
                "contradictions": [],
                "quotations": [],
                "provenance": [segment.segment_id],
                "level": 0,
            }
        )
        for segment in segments
    )
    root, nodes, _ = build_hierarchy(
        leaves,
        Provider(),
        CharacterCounter(),
        source_id=document.source_id,
        covered=tuple((segment.segment_id,) for segment in segments),
        attributable={segment.segment_id: segment.text for segment in segments},
        usable_tokens=100_000,
        model="m",
        timeout_seconds=30,
        max_merge_children=3,
        grounding_policy=GroundingPolicy(max_tokens=160),
    )

    artifact = build_audit_artifact(
        source_id=document.source_id,
        strategy="hierarchical",
        model="m",
        configuration={},
        segments=segments,
        nodes=nodes,
        root_node_id=root.node_id,
        citations=(),
    )

    assert artifact.schema_version == "audit/4"
    selections = [
        node.grounding for node in artifact.tree_nodes if node.grounding
    ]
    serialized = json.loads(serialize_audit(artifact))
    merge_nodes = [
        node for node in serialized["tree_nodes"] if len(node["children"]) > 1
    ]
    assert len(selections) == len(merge_nodes)
    assert any(node["grounding"]["omitted_ids"] for node in merge_nodes)
    assert all(
        set(node["grounding"])
        == {
            "omission_reason",
            "omitted_ids",
            "request_capacity_tokens",
            "reserve_tokens",
            "selected_ids",
        }
        for node in merge_nodes
    )
    recorded_ids = {segment.segment_id for segment in artifact.source_segments}
    for selection in selections:
        assert set(selection.selected_ids) <= recorded_ids
        assert set(selection.omitted_ids) <= recorded_ids

    invalid = artifact.model_dump(mode="json")
    merge_index = next(
        index
        for index, node in enumerate(invalid["tree_nodes"])
        if node["grounding"]
    )
    sibling_segment_id = next(
        segment_id
        for segment_id in recorded_ids
        if segment_id not in invalid["tree_nodes"][merge_index]["covered_segments"]
    )
    invalid["tree_nodes"][merge_index]["grounding"]["omitted_ids"] = [
        sibling_segment_id
    ]
    with pytest.raises(
        ValueError, match="grounding selection must resolve to tree coverage"
    ):
        type(artifact).model_validate(invalid)

    invalid = artifact.model_dump(mode="json")
    original_grounding = invalid["tree_nodes"][merge_index]["grounding"]
    invalid["tree_nodes"][merge_index]["grounding"] = None
    with pytest.raises(ValueError, match="executed merges must record grounding"):
        type(artifact).model_validate(invalid)

    leaf_index = next(
        index
        for index, node in enumerate(invalid["tree_nodes"])
        if not node["children"]
    )
    invalid["tree_nodes"][merge_index]["grounding"] = original_grounding
    invalid["tree_nodes"][leaf_index]["grounding"] = original_grounding
    with pytest.raises(ValueError, match="only executed merges may record grounding"):
        type(artifact).model_validate(invalid)

    adaptive_root, adaptive_nodes, _ = build_hierarchy(
        leaves,
        Provider(),
        CharacterCounter(),
        source_id=document.source_id,
        covered=tuple((segment.segment_id,) for segment in segments),
        attributable={segment.segment_id: segment.text for segment in segments},
        usable_tokens=100_000,
        model="m",
        timeout_seconds=30,
        max_merge_children=3,
    )
    adaptive_artifact = build_audit_artifact(
        source_id=document.source_id,
        strategy="hierarchical",
        model="m",
        configuration={},
        segments=segments,
        nodes=adaptive_nodes,
        root_node_id=adaptive_root.node_id,
        citations=(),
    )
    adaptive_merge = next(
        node.grounding
        for node in adaptive_artifact.tree_nodes
        if node.grounding is not None
    )
    assert adaptive_merge.reserve_tokens is None
    assert adaptive_merge.request_capacity_tokens == 100_000

    unrecorded_nodes = tuple(
        replace(node, grounding=None) if len(node.children) > 1 else node
        for node in nodes
    )
    with pytest.raises(ValueError, match="executed merges must record grounding"):
        build_audit_artifact(
            source_id=document.source_id,
            strategy="hierarchical",
            model="m",
            configuration={},
            segments=segments,
            nodes=unrecorded_nodes,
            root_node_id=root.node_id,
            citations=(),
        )


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
