import hashlib
import json
import os
from pathlib import Path
import sys

import pytest

from summarizer.cache import (
    CACHE_FORMAT_VERSION,
    CacheDescriptor,
    CacheMissReason,
    CacheStore,
)


def _descriptor(**overrides: object) -> CacheDescriptor:
    values: dict[str, object] = {
        "source_id": hashlib.sha256(b"canonical source").hexdigest(),
        "input_hash": hashlib.sha256(b"canonical input").hexdigest(),
        "stage": "leaf",
        "work_id": "S000001",
        "prompt_version": "leaf/1",
        "schema_version": "summary/1",
        "provider": "openai",
        "model": "gpt-4o-mini",
        "counter_identity": "tiktoken:o200k_base",
        "counter_exact": True,
        "context_window_tokens": 128_000,
        "behavior": {"max_output_tokens": 1024, "overlap_tokens": 0},
    }
    values.update(overrides)
    return CacheDescriptor(**values)  # type: ignore[arg-type]


def _validate_payload(payload: object) -> dict[str, str]:
    if not isinstance(payload, dict) or set(payload) != {"summary"}:
        raise ValueError("payload must be one summary")
    summary = payload["summary"]
    if not isinstance(summary, str) or not summary:
        raise ValueError("summary must be nonblank")
    return {"summary": summary}


def test_descriptor_key_is_canonical_and_excludes_unsafe_operational_values() -> None:
    descriptor = _descriptor(
        behavior={"max_output_tokens": 1024, "overlap_tokens": 0},
    )
    reordered = _descriptor(
        behavior={"overlap_tokens": 0, "max_output_tokens": 1024},
    )

    assert descriptor.canonical_bytes() == reordered.canonical_bytes()
    assert descriptor.key == reordered.key
    assert descriptor.key == hashlib.sha256(descriptor.canonical_bytes()).hexdigest()
    assert b"sk-12345678901234567890" not in descriptor.canonical_bytes()
    assert b"https://provider.example/v1" not in descriptor.canonical_bytes()
    assert b".summarizer-cache" not in descriptor.canonical_bytes()


@pytest.mark.parametrize(
    "field",
    (
        "credential",
        "endpoint",
        "endpoint_url",
        "host",
        "cache_root",
        "source_path",
        "path",
        "prompt",
        "request",
        "run_id",
        "source_text",
    ),
)
def test_descriptor_rejects_unsafe_behavior_keys(field: str) -> None:
    with pytest.raises(ValueError, match="unsafe"):
        _descriptor(behavior={field: "unsafe"})


@pytest.mark.parametrize(
    "behavior",
    (
        {"credentials": "sk-test-secret"},
        {"apiKey": "sk-test-secret"},
        {"access_token": "token-value"},
        {"baseURL": "safe-identity"},
        {"rawSource": "source prose"},
        {"nested": {"authorization": "Bearer token-value"}},
        {"nested": {"source": {"text": "source prose"}}},
        {"api": {"key": "token-value"}},
        {"access": {"token": "token-value"}},
        {"base": {"url": "safe-identity"}},
        {"raw": {"source": "source prose"}},
    ),
)
def test_descriptor_rejects_unsafe_aliases_recursively(
    behavior: dict[str, object],
) -> None:
    with pytest.raises(ValueError, match="unsafe"):
        _descriptor(behavior=behavior)


@pytest.mark.parametrize(
    "value",
    (
        "sk-test-secret-value",
        "https://provider.example/v1",
        "file:///private/source.txt",
        "/private/source.txt",
    ),
)
def test_descriptor_rejects_sensitive_values_under_safe_looking_keys(value: str) -> None:
    with pytest.raises(ValueError, match="unsafe"):
        _descriptor(behavior={"max_output_tokens": value})


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("source_id", "raw source prose"),
        ("provider", "https://provider.example/v1"),
        ("model", "sk-test-secret-value"),
        ("counter_identity", "/private/tokenizer.json"),
    ),
)
def test_descriptor_rejects_unsafe_direct_identity_values(field: str, value: str) -> None:
    with pytest.raises(ValueError, match="unsafe"):
        _descriptor(**{field: value})


def test_descriptor_rejects_host_and_non_openai_token_identities() -> None:
    """Test that descriptor validation rejects host:port and credential patterns."""
    test_cases = [
        ("provider", "provider.example:443"),
        ("model", "ghp_" + "9" * 36),  # GitHub personal access token
        ("model", "xoxb-" + "6" * 6 + "-" + "6" * 6 + "-" + "a" * 20),  # Slack bot token
        ("model", "glpat-" + "x" * 24),  # GitLab personal access token
        ("model", "hf_" + "x" * 24),  # Hugging Face token
        ("model", "secret"),
    ]
    for field, value in test_cases:
        with pytest.raises(ValueError, match="unsafe"):
            _descriptor(**{field: value})


@pytest.mark.parametrize(
    "behavior",
    (
        {"max_output_tokens": "provider.example:443"},
        {"nested": {"model": "ghp_123456789012345678901234567890123456"}},
        {"nested": {"model": "secret"}},
    ),
)
def test_descriptor_rejects_opaque_host_and_secret_behavior_strings(
    behavior: dict[str, object],
) -> None:
    with pytest.raises(ValueError, match="unsafe"):
        _descriptor(behavior=behavior)


@pytest.mark.parametrize("model", ("repair-model", "org/model:tag", "llama3.2:3b"))
def test_descriptor_accepts_established_model_identity_forms(model: str) -> None:
    assert _descriptor(model=model).key


@pytest.mark.parametrize("counter_identity", ("estimate:utf8-bytes", "tiktoken:o200k_base"))
def test_descriptor_accepts_established_counter_identity_forms(
    counter_identity: str,
) -> None:
    assert _descriptor(counter_identity=counter_identity).key


def test_descriptor_uses_a_validated_behavior_snapshot_after_construction() -> None:
    behavior: dict[str, object] = {"max_output_tokens": 1024}
    descriptor = _descriptor(behavior=behavior)
    original_key = descriptor.key

    behavior["credentials"] = "sk-test-secret"

    assert descriptor.key == original_key
    assert b"sk-test-secret" not in descriptor.canonical_bytes()


def test_descriptor_rejects_noncanonical_nonfinite_behavior_values() -> None:
    with pytest.raises(ValueError, match="unsafe"):
        _descriptor(behavior={"max_output_tokens": float("nan")})


def test_descriptor_accepts_all_approved_output_affecting_configuration() -> None:
    descriptor = _descriptor(
        stage="verification",
        work_id="V01C000001",
        prompt_version="verification/1",
        schema_version="verification-result/1",
        provider="ollama",
        model="llama3.2:3b",
        counter_identity="utf8-conservative",
        behavior={
            "segmentation": {"max_tokens": 1024, "overlap_tokens": 32},
            "strategy_config": {
                "strategy": "hierarchical",
                "context_window": 16_384,
                "max_output_tokens": 1024,
                "safety_margin_tokens": 256,
            },
            "budget": {
                "max_output_tokens": 1024,
                "safety_margin_fraction": 0.02,
            },
            "verification": {
                "verification_enabled": True,
                "evidence_tokens": 4096,
                "request_tokens": 8192,
                "output_reserve_tokens": 1024,
                "max_repair_passes": 1,
            },
            "target_words": 300,
            "timeout_seconds": 42.5,
            "max_merge_children": 4,
            "grounding_policy": "root-provenance/1",
            "editorial_version": "editorial/1",
            "include_citations": False,
        },
    )

    assert descriptor.key

    assert _descriptor(stage="segmentation", work_id="segmentation").key


def test_descriptor_key_changes_for_every_output_affecting_input() -> None:
    base = _descriptor()
    variants = (
        _descriptor(source_id=hashlib.sha256(b"other source").hexdigest()),
        _descriptor(input_hash=hashlib.sha256(b"other input").hexdigest()),
        _descriptor(stage="merge"),
        _descriptor(work_id="L1N0001"),
        _descriptor(prompt_version="leaf/2"),
        _descriptor(schema_version="summary/2"),
        _descriptor(provider="ollama"),
        _descriptor(model="llama3.2:3b"),
        _descriptor(counter_identity="utf8-conservative"),
        _descriptor(counter_exact=False),
        _descriptor(context_window_tokens=32_768),
        _descriptor(behavior={"max_output_tokens": 2048, "overlap_tokens": 0}),
    )

    assert all(item.key != base.key for item in variants)


def test_store_writes_validated_payload_once_with_private_sharded_permissions(
    tmp_path: Path,
) -> None:
    store = CacheStore(tmp_path / "cache")
    descriptor = _descriptor()

    path = store.store(descriptor, {"summary": "first"}, _validate_payload)
    store.store(descriptor, {"summary": "second"}, _validate_payload)

    assert path == (
        tmp_path
        / "cache"
        / "objects"
        / descriptor.key[:2]
        / f"{descriptor.key}.json"
    )
    assert store.load(descriptor, _validate_payload).payload == {"summary": "first"}
    assert os.stat(tmp_path / "cache").st_mode & 0o777 == 0o700
    assert os.stat(path.parent).st_mode & 0o777 == 0o700
    assert os.stat(path).st_mode & 0o777 == 0o600


@pytest.mark.skipif(sys.platform != "darwin", reason="macOS path semantics")
def test_store_canonicalizes_a_logical_var_temp_root(tmp_path: Path) -> None:
    physical_root = tmp_path.resolve()
    try:
        relative_root = physical_root.relative_to("/private/var")
    except ValueError:
        pytest.skip("temporary directory is not under macOS /private/var")
    logical_root = Path("/var") / relative_root / "cache"

    path = CacheStore(logical_root).store(
        _descriptor(),
        {"summary": "ok"},
        _validate_payload,
    )

    assert path.is_relative_to(physical_root)


@pytest.mark.parametrize(
    ("node", "mode"),
    (
        ("root", 0o755),
        ("objects", 0o755),
        ("shard", 0o755),
        ("object", 0o644),
        ("lock", 0o644),
    ),
)
def test_load_rejects_cache_nodes_with_relaxed_permissions(
    tmp_path: Path,
    node: str,
    mode: int,
) -> None:
    store = CacheStore(tmp_path / "cache")
    descriptor = _descriptor()
    path = store.store(descriptor, {"summary": "ok"}, _validate_payload)
    nodes = {
        "root": tmp_path / "cache",
        "objects": tmp_path / "cache" / "objects",
        "shard": path.parent,
        "object": path,
        "lock": path.with_suffix(".lock"),
    }
    os.chmod(nodes[node], mode)

    result = store.load(descriptor, _validate_payload)

    assert not result.hit
    assert result.miss_reason is CacheMissReason.CORRUPT


def test_store_rejects_invalid_payload_before_creating_an_object(tmp_path: Path) -> None:
    store = CacheStore(tmp_path / "cache")
    descriptor = _descriptor()

    with pytest.raises(ValueError, match="summary"):
        store.store(descriptor, {"summary": ""}, _validate_payload)

    assert not (tmp_path / "cache" / "objects").exists()


def test_load_reports_missing_corrupt_wrong_version_and_incompatible_objects(
    tmp_path: Path,
) -> None:
    store = CacheStore(tmp_path / "cache")
    descriptor = _descriptor()

    assert store.load(descriptor, _validate_payload).miss_reason is CacheMissReason.MISSING

    path = store.object_path(descriptor)
    path.parent.mkdir(parents=True)
    for directory in (tmp_path / "cache", tmp_path / "cache" / "objects", path.parent):
        os.chmod(directory, 0o700)
    path.write_text("not-json")
    os.chmod(path, 0o600)
    assert store.load(descriptor, _validate_payload).miss_reason is CacheMissReason.CORRUPT

    path.write_text(
        json.dumps(
            {
                "format_version": "cache/unknown",
                "descriptor": json.loads(descriptor.canonical_bytes()),
                "payload": {"summary": "ok"},
                "payload_kind": descriptor.schema_version,
                "payload_sha256": "0" * 64,
            }
        )
    )
    assert store.load(descriptor, _validate_payload).miss_reason is CacheMissReason.WRONG_VERSION

    path.write_text(json.dumps({"format_version": "cache/older"}))
    assert store.load(descriptor, _validate_payload).miss_reason is CacheMissReason.WRONG_VERSION

    path.write_text(
        json.dumps(
            {
                "format_version": CACHE_FORMAT_VERSION,
                "descriptor": json.loads(_descriptor(model="gpt-4o").canonical_bytes()),
                "payload": {"summary": "ok"},
                "payload_kind": descriptor.schema_version,
                "payload_sha256": "0" * 64,
            }
        )
    )
    assert store.load(descriptor, _validate_payload).miss_reason is CacheMissReason.INCOMPATIBLE


def test_load_rejects_envelopes_with_undeclared_fields(tmp_path: Path) -> None:
    store = CacheStore(tmp_path / "cache")
    descriptor = _descriptor()
    path = store.store(descriptor, {"summary": "ok"}, _validate_payload)
    envelope = json.loads(path.read_text())
    envelope["endpoint"] = "https://provider.example/v1"
    path.write_text(json.dumps(envelope))

    assert store.load(descriptor, _validate_payload).miss_reason is CacheMissReason.CORRUPT


@pytest.mark.parametrize(
    "component",
    ("root", "objects", "shard", "object", "lock"),
)
def test_store_rejects_symlinked_cache_components_without_touching_target(
    tmp_path: Path, component: str
) -> None:
    root = tmp_path / "cache"
    root.mkdir(mode=0o700)
    outside = tmp_path / "outside"
    outside.mkdir(mode=0o755)
    os.chmod(outside, 0o755)
    sentinel = outside / "keep.txt"
    sentinel.write_text("unchanged")
    descriptor = _descriptor()
    if component == "root":
        root.rmdir()
        root.symlink_to(outside, target_is_directory=True)
    else:
        objects = root / "objects"
        if component == "objects":
            objects.symlink_to(outside, target_is_directory=True)
        else:
            objects.mkdir(mode=0o700)
            shard = objects / descriptor.key[:2]
            if component == "shard":
                shard.symlink_to(outside, target_is_directory=True)
            else:
                shard.mkdir(mode=0o700)
                suffix = ".json" if component == "object" else ".lock"
                (shard / f"{descriptor.key}{suffix}").symlink_to(sentinel)

    with pytest.raises(ValueError, match="unsafe cache"):
        CacheStore(root).store(descriptor, {"summary": "ok"}, _validate_payload)

    assert sentinel.read_text() == "unchanged"
    assert list(outside.iterdir()) == [sentinel]
    assert os.stat(outside).st_mode & 0o777 == 0o755


def test_load_maps_an_object_directory_to_a_corrupt_miss(tmp_path: Path) -> None:
    store = CacheStore(tmp_path / "cache")
    descriptor = _descriptor()
    store.object_path(descriptor).mkdir(parents=True)

    result = store.load(descriptor, _validate_payload)

    assert result.miss_reason is CacheMissReason.CORRUPT


def test_store_flushes_and_syncs_file_and_parent_before_returning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = CacheStore(tmp_path / "cache")
    calls: list[int] = []
    original_fsync = os.fsync

    def record_fsync(file_descriptor: int) -> None:
        calls.append(file_descriptor)
        original_fsync(file_descriptor)

    monkeypatch.setattr("summarizer.cache.os.fsync", record_fsync)

    store.store(_descriptor(), {"summary": "ok"}, _validate_payload)

    assert len(calls) >= 2
