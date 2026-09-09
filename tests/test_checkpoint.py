import hashlib
import json
import os
from pathlib import Path

import pytest

from summarizer.cache import CacheDescriptor, CacheStore
from summarizer.checkpoint import (
    CheckpointError,
    CheckpointReason,
    CheckpointStore,
    CompletedRef,
    NonReusableReason,
    NonReusableRef,
    ReuseReason,
    RunPlan,
)


def _digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _plan(**overrides: object) -> RunPlan:
    values: dict[str, object] = {
        "run_id": "run-20260907",
        "descriptor_sha256": _digest(b"run descriptor"),
        "source_sha256": _digest(b"canonical source"),
        "work_ids": ("segmentation", "S000001", "editorial-final"),
    }
    values.update(overrides)
    return RunPlan(**values)  # type: ignore[arg-type]


def _descriptor(**overrides: object) -> CacheDescriptor:
    values: dict[str, object] = {
        "source_id": _digest(b"canonical source"),
        "input_hash": _digest(b"segment one"),
        "stage": "leaf",
        "work_id": "S000001",
        "prompt_version": "leaf/1",
        "schema_version": "summary/1",
        "provider": "openai",
        "model": "gpt-4o-mini",
        "counter_identity": "tiktoken:o200k_base",
        "counter_exact": True,
        "context_window_tokens": 128_000,
        "behavior": {"max_output_tokens": 1024},
    }
    values.update(overrides)
    return CacheDescriptor(**values)  # type: ignore[arg-type]


def _validate_summary(payload: object) -> dict[str, str]:
    if not isinstance(payload, dict) or set(payload) != {"summary"}:
        raise ValueError("expected one summary")
    if not isinstance(payload["summary"], str) or not payload["summary"]:
        raise ValueError("summary must be nonblank")
    return {"summary": payload["summary"]}


def _must_not_validate(payload: object) -> object:
    raise AssertionError(f"unexpected cache load: {payload!r}")


def test_new_run_persists_a_private_versioned_manifest(tmp_path: Path) -> None:
    store = CheckpointStore(tmp_path / "cache")

    with store.open(_plan(), resume=False) as session:
        session.checkpoint(
            completed=(CompletedRef(work_id="S000001", cache_key=_descriptor().key),),
            descriptors={"S000001": _descriptor()},
            metadata={"cache_hits": 1, "retry_exhausted": False},
        )
        session.stage_publication(
            audit_sha256=_digest(b"audit"), summary_sha256=_digest(b"summary")
        )

    path = tmp_path / "cache" / "runs" / "run-20260907.json"
    manifest = json.loads(path.read_text())

    assert manifest == {
        "completed": [{"cache_key": _descriptor().key, "work_id": "S000001"}],
        "descriptor_sha256": _plan().descriptor_sha256,
        "format_version": "run/1",
        "metadata": {"cache_hits": 1, "retry_exhausted": False},
        "non_reusable": [],
        "publication": "audit_staged",
        "audit_sha256": _digest(b"audit"),
        "summary_sha256": _digest(b"summary"),
        "run_id": "run-20260907",
        "source_sha256": _plan().source_sha256,
        "terminal_failure": False,
        "terminal_failure_work_id": None,
        "work_ids": ["segmentation", "S000001", "editorial-final"],
    }
    assert os.stat(path).st_mode & 0o777 == 0o600
    assert os.stat(path.parent).st_mode & 0o777 == 0o700


@pytest.mark.parametrize("publication", ["audit_staged", "complete"])
def test_manifest_rejects_published_state_without_digests(
    tmp_path: Path, publication: str
) -> None:
    store = CheckpointStore(tmp_path / "cache")
    with store.open(_plan(), resume=False):
        pass
    path = tmp_path / "cache" / "runs" / "run-20260907.json"
    manifest = json.loads(path.read_text())
    manifest["publication"] = publication
    path.write_text(json.dumps(manifest))

    with pytest.raises(CheckpointError) as raised:
        with store.open(_plan(), resume=True):
            pass
    assert raised.value.reason is CheckpointReason.CORRUPT


def test_manifest_rejects_incomplete_state_with_publication_digests(
    tmp_path: Path,
) -> None:
    store = CheckpointStore(tmp_path / "cache")
    with store.open(_plan(), resume=False):
        pass
    path = tmp_path / "cache" / "runs" / "run-20260907.json"
    manifest = json.loads(path.read_text())
    manifest.update(audit_sha256=_digest(b"audit"), summary_sha256=_digest(b"summary"))
    path.write_text(json.dumps(manifest))

    with pytest.raises(CheckpointError) as raised:
        with store.open(_plan(), resume=True):
            pass
    assert raised.value.reason is CheckpointReason.CORRUPT


def test_resume_rejects_a_descriptor_or_source_mismatch(tmp_path: Path) -> None:
    store = CheckpointStore(tmp_path / "cache")
    with store.open(_plan(), resume=False):
        pass

    with pytest.raises(CheckpointError) as raised:
        with store.open(_plan(source_sha256=_digest(b"changed source")), resume=True):
            pass

    assert raised.value.reason is CheckpointReason.INCOMPATIBLE


def test_resume_reports_a_corrupt_manifest_without_reusing_it(tmp_path: Path) -> None:
    store = CheckpointStore(tmp_path / "cache")
    with store.open(_plan(), resume=False):
        pass
    (tmp_path / "cache" / "runs" / "run-20260907.json").write_text("not json")

    with pytest.raises(CheckpointError) as raised:
        with store.open(_plan(), resume=True):
            pass

    assert raised.value.reason is CheckpointReason.CORRUPT


def test_resume_rejects_a_manifest_missing_its_version(tmp_path: Path) -> None:
    store = CheckpointStore(tmp_path / "cache")
    with store.open(_plan(), resume=False):
        pass
    path = tmp_path / "cache" / "runs" / "run-20260907.json"
    manifest = json.loads(path.read_text())
    del manifest["format_version"]
    path.write_text(json.dumps(manifest))

    with pytest.raises(CheckpointError) as raised:
        with store.open(_plan(), resume=True):
            pass

    assert raised.value.reason is CheckpointReason.CORRUPT


def test_second_active_holder_fails_clearly(tmp_path: Path) -> None:
    store = CheckpointStore(tmp_path / "cache")
    with store.open(_plan(), resume=False):
        with pytest.raises(CheckpointError) as raised:
            with store.open(_plan(), resume=True):
                pass

    assert raised.value.reason is CheckpointReason.RUN_ACTIVE


def test_resume_rejects_a_relaxed_lock_file(tmp_path: Path) -> None:
    store = CheckpointStore(tmp_path / "cache")
    with store.open(_plan(), resume=False):
        pass
    lock_path = tmp_path / "cache" / "runs" / "run-20260907.lock"
    lock_path.chmod(0o644)

    with pytest.raises(CheckpointError) as raised:
        with store.open(_plan(), resume=True):
            pass

    assert raised.value.reason is CheckpointReason.CORRUPT


def test_resume_rejects_a_symlinked_lock_file(tmp_path: Path) -> None:
    store = CheckpointStore(tmp_path / "cache")
    with store.open(_plan(), resume=False):
        pass
    lock_path = tmp_path / "cache" / "runs" / "run-20260907.lock"
    outside = tmp_path / "outside-lock"
    lock_path.unlink()
    outside.write_text("not a lock")
    lock_path.symlink_to(outside)

    with pytest.raises(CheckpointError) as raised:
        with store.open(_plan(), resume=True):
            pass

    assert raised.value.reason is CheckpointReason.CORRUPT


def test_resume_reuses_only_descriptor_compatible_validated_references(
    tmp_path: Path,
) -> None:
    root = tmp_path / "cache"
    descriptor = _descriptor()
    CacheStore(root).store(descriptor, {"summary": "cached"}, _validate_summary)
    store = CheckpointStore(root)
    with store.open(_plan(), resume=False) as session:
        session.checkpoint(
            completed=(CompletedRef(work_id="S000001", cache_key=descriptor.key),),
            descriptors={"S000001": descriptor},
        )

    with store.open(_plan(), resume=True) as session:
        reusable = session.reusable(
            descriptors={"S000001": descriptor},
            validators={"S000001": _validate_summary},
        )
        incompatible = session.reusable(
            descriptors={"S000001": _descriptor(behavior={"max_output_tokens": 2048})},
            validators={"S000001": _validate_summary},
        )

    assert reusable[0].payload == {"summary": "cached"}
    assert reusable[0].reason is None
    assert incompatible[0].payload is None
    assert incompatible[0].reason is ReuseReason.INCOMPATIBLE


def test_resume_accepts_an_extended_manifest_with_the_same_seed_prefix(
    tmp_path: Path,
) -> None:
    store = CheckpointStore(tmp_path / "cache")
    seed = _plan(work_ids=("segmentation",))
    full = ("segmentation", "S000001", "editorial-final")

    with store.open(seed, resume=False) as session:
        session.ensure_work_prefix(full)

    with store.open(seed, resume=True) as session:
        assert session.manifest.work_ids == full


def test_resume_rejects_a_divergent_extended_manifest_prefix(tmp_path: Path) -> None:
    store = CheckpointStore(tmp_path / "cache")
    seed = _plan(work_ids=("segmentation",))
    with store.open(seed, resume=False) as session:
        session.ensure_work_prefix(("segmentation", "S000001"))

    with pytest.raises(CheckpointError) as raised:
        with store.open(_plan(work_ids=("segmentation", "S000002")), resume=True):
            pass

    assert raised.value.reason is CheckpointReason.INCOMPATIBLE


def test_resume_does_not_reuse_a_reference_to_another_source(tmp_path: Path) -> None:
    root = tmp_path / "cache"
    other_source = _digest(b"other source")
    descriptor = _descriptor(source_id=other_source)
    CacheStore(root).store(descriptor, {"summary": "wrong source"}, _validate_summary)
    store = CheckpointStore(root)
    with store.open(_plan(source_sha256=other_source), resume=False) as session:
        session.checkpoint(
            completed=(CompletedRef(work_id="S000001", cache_key=descriptor.key),),
            descriptors={"S000001": descriptor},
        )
    path = root / "runs" / "run-20260907.json"
    manifest = json.loads(path.read_text())
    manifest["source_sha256"] = _plan().source_sha256
    path.write_text(json.dumps(manifest))

    with store.open(_plan(), resume=True) as session:
        results = session.reusable(
            descriptors={"S000001": descriptor},
            validators={"S000001": _must_not_validate},
        )

    assert results[0].payload is None
    assert results[0].reason is ReuseReason.INCOMPATIBLE


def test_checkpoint_rejects_a_reference_to_another_source(tmp_path: Path) -> None:
    descriptor = _descriptor(source_id=_digest(b"other source"))
    store = CheckpointStore(tmp_path / "cache")

    with store.open(_plan(), resume=False) as session:
        with pytest.raises(CheckpointError) as raised:
            session.checkpoint(
                completed=(CompletedRef(work_id="S000001", cache_key=descriptor.key),),
                descriptors={"S000001": descriptor},
            )

    assert raised.value.reason is CheckpointReason.INCOMPATIBLE


def test_resume_marks_corrupt_cache_reference_non_reusable(tmp_path: Path) -> None:
    root = tmp_path / "cache"
    descriptor = _descriptor()
    path = CacheStore(root).store(descriptor, {"summary": "cached"}, _validate_summary)
    store = CheckpointStore(root)
    with store.open(_plan(), resume=False) as session:
        session.checkpoint(
            completed=(CompletedRef(work_id="S000001", cache_key=descriptor.key),),
            descriptors={"S000001": descriptor},
        )
    path.write_text("not json")

    with store.open(_plan(), resume=True) as session:
        results = session.reusable(
            descriptors={"S000001": descriptor},
            validators={"S000001": _validate_summary},
        )

    assert results[0].payload is None
    assert results[0].reason is ReuseReason.CORRUPT


def test_checkpoint_records_closed_non_reusable_scheduler_state(tmp_path: Path) -> None:
    store = CheckpointStore(tmp_path / "cache")

    with store.open(_plan(), resume=False) as session:
        session.checkpoint_scheduler_state(
            non_reusable=(
                NonReusableRef(
                    work_id="segmentation", reason=NonReusableReason.CANCELLED
                ),
                NonReusableRef(
                    work_id="S000001", reason=NonReusableReason.UNOBSERVABLE
                ),
                NonReusableRef(
                    work_id="editorial-final", reason=NonReusableReason.UNKNOWN
                ),
            ),
            terminal_failure=True,
            terminal_failure_work_id="S000001",
        )

    manifest = json.loads(
        (tmp_path / "cache" / "runs" / "run-20260907.json").read_text()
    )
    assert manifest["non_reusable"] == [
        {"reason": "cancelled", "work_id": "segmentation"},
        {"reason": "unobservable", "work_id": "S000001"},
        {"reason": "unknown", "work_id": "editorial-final"},
    ]
    assert manifest["terminal_failure_work_id"] == "S000001"
    assert manifest["terminal_failure"] is True
