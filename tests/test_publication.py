"""Tests for audit-first, summary-last publication with a completion witness."""

import hashlib
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from summarizer.audit import AuditArtifact, build_audit_artifact, serialize_audit
from summarizer.checkpoint import CheckpointStore, PublicationState, RunPlan
from summarizer.direct import whole_document_segment
from summarizer.finalization import (
    FinalizationResult,
    PublicationError,
    publish_final_output,
    read_published_summary,
)
from summarizer.hierarchy import TreeNode
from summarizer.ingestion import ingest_text
from summarizer.summaries import SummaryNode


class CharacterCounter:
    identity = "test:characters"
    exact = True
    monotonic = True

    def count(self, text: str) -> int:
        return len(text)


def _digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _plan(run_id: str) -> RunPlan:
    return RunPlan(
        run_id, _digest(b"descriptor"), _digest(b"source"), ("segmentation",)
    )


def _result(text: str = "Final summary text.") -> FinalizationResult:
    document = ingest_text("Publication test content.")
    segment = whole_document_segment(document, CharacterCounter())
    summary = SummaryNode.model_validate(
        {
            "summary": "Publication test content.",
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
    artifact = build_audit_artifact(
        source_id=document.source_id,
        strategy="direct",
        model="test-model",
        configuration={
            "provider": "openai",
            "model": "test-model",
            "timeout_seconds": 30,
        },
        segments=(segment,),
        nodes=(node,),
        root_node_id=node.node_id,
        citations=(),
        reliability_resume={"resumed": False, "reused_count": 0, "recomputed_count": 1},
    )
    return FinalizationResult(text, (), artifact)


def _manifest(cache_root: Path, run_id: str) -> dict[str, object]:
    return json.loads((cache_root / "runs" / f"{run_id}.json").read_text())


def test_publication_writes_audit_first_before_summary(tmp_path: Path) -> None:
    order: list[str] = []
    audit_path, summary_path = tmp_path / "audit.json", tmp_path / "summary.txt"

    def replace(path: Path, payload: bytes) -> None:
        order.append(path.name)
        path.write_bytes(payload)

    with CheckpointStore(tmp_path / "cache").open(
        _plan("ordered-publication"), resume=False
    ) as session:
        publish_final_output(
            _result(),
            summary_path=summary_path,
            audit_path=audit_path,
            session=session,
            atomic_replace=replace,
        )
    assert order == ["audit.json", "summary.txt"]
    assert (
        read_published_summary(summary_path, audit_path, session.manifest)
        == "Final summary text."
    )


def test_publication_rejects_unreliable_audit_v4(tmp_path: Path) -> None:
    result = _result()
    assert result.audit is not None
    body = json.loads(serialize_audit(result.audit))
    body["schema_version"] = "audit/4"
    body.pop("reliability")
    for node in body["tree_nodes"]:
        node["grounding"] = None
    audit_v4 = AuditArtifact.model_validate(body)

    with CheckpointStore(tmp_path / "cache").open(
        _plan("reject-unreliable-v4"), resume=False
    ) as session, pytest.raises(PublicationError, match="reliable audit"):
        publish_final_output(
            FinalizationResult(result.text, result.citations, audit_v4),
            summary_path=tmp_path / "summary.txt",
            audit_path=tmp_path / "audit.json",
            session=session,
        )


def test_audit_failure_never_writes_summary(tmp_path: Path) -> None:
    summary_path = tmp_path / "summary.txt"

    def fail_audit(path: Path, payload: bytes) -> None:
        raise OSError("audit unavailable")

    with CheckpointStore(tmp_path / "cache").open(
        _plan("audit-failure"), resume=False
    ) as session:
        with pytest.raises(OSError, match="audit unavailable"):
            publish_final_output(
                _result(),
                summary_path=summary_path,
                audit_path=tmp_path / "audit.json",
                session=session,
                atomic_replace=fail_audit,
            )
        assert session.manifest.publication is PublicationState.INCOMPLETE
    assert not summary_path.exists()


def test_publication_rejects_one_path_for_both_outputs(tmp_path: Path) -> None:
    output_path = tmp_path / "output"
    with (
        CheckpointStore(tmp_path / "cache").open(
            _plan("same-path"), resume=False
        ) as session,
        pytest.raises(PublicationError, match="must differ"),
    ):
        publish_final_output(
            _result(),
            summary_path=output_path,
            audit_path=output_path,
            session=session,
        )
    assert not output_path.exists()


def test_summary_failure_leaves_audit_staged_with_both_digests(tmp_path: Path) -> None:
    audit_path, summary_path = tmp_path / "audit.json", tmp_path / "summary.txt"

    def fail_summary(path: Path, payload: bytes) -> None:
        if path == summary_path:
            raise OSError("summary unavailable")
        path.write_bytes(payload)

    cache_root = tmp_path / "cache"
    with CheckpointStore(cache_root).open(
        _plan("summary-failure"), resume=False
    ) as session:
        with pytest.raises(OSError, match="summary unavailable"):
            publish_final_output(
                _result(),
                summary_path=summary_path,
                audit_path=audit_path,
                session=session,
                atomic_replace=fail_summary,
            )
    manifest = _manifest(cache_root, "summary-failure")
    assert manifest["publication"] == "audit_staged"
    assert len(str(manifest["audit_sha256"])) == 64
    assert len(str(manifest["summary_sha256"])) == 64
    assert audit_path.exists() and not summary_path.exists()


def test_resume_completes_marker_when_digest_matched_files_exist(
    tmp_path: Path,
) -> None:
    cache_root = tmp_path / "cache"
    audit_path, summary_path = tmp_path / "audit.json", tmp_path / "summary.txt"
    plan = _plan("marker-recovery")
    with CheckpointStore(cache_root).open(plan, resume=False) as session:
        original_write = session._write_manifest

        def fail_complete(manifest) -> None:
            if manifest.publication is PublicationState.COMPLETE:
                raise OSError("marker unavailable")
            original_write(manifest)

        session._write_manifest = fail_complete
        with pytest.raises(OSError, match="marker unavailable"):
            publish_final_output(
                _result(),
                summary_path=summary_path,
                audit_path=audit_path,
                session=session,
            )

    writes: list[Path] = []
    with CheckpointStore(cache_root).open(plan, resume=True) as resumed:
        publish_final_output(
            _result(),
            summary_path=summary_path,
            audit_path=audit_path,
            session=resumed,
            atomic_replace=lambda path, payload: writes.append(path),
        )
        assert resumed.manifest.publication is PublicationState.COMPLETE
    assert writes == []


def test_reader_rejects_incomplete_or_digest_mismatched_publication(
    tmp_path: Path,
) -> None:
    audit_path, summary_path = tmp_path / "audit.json", tmp_path / "summary.txt"
    with CheckpointStore(tmp_path / "cache").open(
        _plan("reader-check"), resume=False
    ) as session:
        with pytest.raises(PublicationError, match="incomplete"):
            read_published_summary(summary_path, audit_path, session.manifest)
        publish_final_output(
            _result(), summary_path=summary_path, audit_path=audit_path, session=session
        )
        summary_path.write_text("tampered", encoding="utf-8")
        with pytest.raises(PublicationError, match="incomplete"):
            read_published_summary(summary_path, audit_path, session.manifest)


@pytest.mark.parametrize("damage", ["missing", "tampered"])
def test_resume_republishes_when_audit_is_missing_or_tampered(
    tmp_path: Path, damage: str
) -> None:
    cache_root = tmp_path / "cache"
    audit_path, summary_path = tmp_path / "audit.json", tmp_path / "summary.txt"
    plan = _plan(f"audit-{damage}")
    with CheckpointStore(cache_root).open(plan, resume=False) as session:
        publish_final_output(
            _result(),
            summary_path=summary_path,
            audit_path=audit_path,
            session=session,
        )
    if damage == "missing":
        audit_path.unlink()
    else:
        audit_path.write_text("tampered", encoding="utf-8")

    writes: list[str] = []

    def replace(path: Path, payload: bytes) -> None:
        writes.append(path.name)
        path.write_bytes(payload)

    with CheckpointStore(cache_root).open(plan, resume=True) as session:
        publish_final_output(
            _result(),
            summary_path=summary_path,
            audit_path=audit_path,
            session=session,
            atomic_replace=replace,
        )
        assert (
            read_published_summary(summary_path, audit_path, session.manifest)
            == "Final summary text."
        )
    assert writes == ["audit.json", "summary.txt"]


def test_runs_sharing_output_paths_cannot_interleave_publication(
    tmp_path: Path,
) -> None:
    audit_path, summary_path = tmp_path / "audit.json", tmp_path / "summary.txt"
    first_audit_written = threading.Event()
    release_first = threading.Event()
    second_replace_entered = threading.Event()
    second_started = threading.Event()
    order: list[str] = []
    order_guard = threading.Lock()

    def publisher(run_id: str, text: str) -> None:
        with CheckpointStore(tmp_path / run_id).open(
            _plan(run_id), resume=False
        ) as session:

            def replace(path: Path, payload: bytes) -> None:
                with order_guard:
                    order.append(f"{run_id}:{path.name}")
                if run_id == "first-run" and path == audit_path:
                    path.write_bytes(payload)
                    first_audit_written.set()
                    assert release_first.wait(2)
                    return
                if run_id == "second-run":
                    second_replace_entered.set()
                path.write_bytes(payload)

            if run_id == "second-run":
                second_started.set()
            publish_final_output(
                _result(text),
                summary_path=summary_path,
                audit_path=audit_path,
                session=session,
                atomic_replace=replace,
            )

    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(publisher, "first-run", "first")
        assert first_audit_written.wait(2)
        second = executor.submit(publisher, "second-run", "second")
        assert second_started.wait(2)
        assert not second_replace_entered.wait(0.1)
        release_first.set()
        first.result(timeout=2)
        second.result(timeout=2)

    assert order == [
        "first-run:audit.json",
        "first-run:summary.txt",
        "second-run:audit.json",
        "second-run:summary.txt",
    ]


def test_audit_path_none_needs_no_publication_protocol() -> None:
    assert _result().audit is not None
