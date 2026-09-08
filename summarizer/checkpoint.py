"""Private, locked run manifests for compatible local resume."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
import errno
import fcntl
import json
import os
from pathlib import Path
import re

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from summarizer.cache import (
    CacheDescriptor,
    CacheMissReason,
    CacheStore,
    _UnsafeCachePath,
)


CHECKPOINT_FORMAT_VERSION = "run/1"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_RUN_ID = re.compile(r"^[a-z][a-z0-9-]{0,63}$")
_WORK_ID = re.compile(r"^(?:[DSLVM][A-Za-z0-9:_-]*|editorial-final|segmentation)$")
_METADATA_KEY = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
_MANIFEST_FIELDS = frozenset(
    {
        "completed",
        "descriptor_sha256",
        "format_version",
        "metadata",
        "publication",
        "run_id",
        "source_sha256",
        "work_ids",
    }
)


class PublicationState(str, Enum):
    INCOMPLETE = "incomplete"
    AUDIT_STAGED = "audit_staged"
    COMPLETE = "complete"


class CheckpointReason(str, Enum):
    MISSING = "missing"
    CORRUPT = "corrupt"
    INCOMPATIBLE = "incompatible"
    RUN_ACTIVE = "run_active"
    ALREADY_EXISTS = "already_exists"


class ReuseReason(str, Enum):
    MISSING = "missing"
    CORRUPT = "corrupt"
    INCOMPATIBLE = "incompatible"


class CheckpointError(ValueError):
    def __init__(self, reason: CheckpointReason) -> None:
        self.reason = reason
        super().__init__(reason.value)


def _require_sha256(value: str, *, name: str) -> None:
    if _SHA256.fullmatch(value) is None:
        raise ValueError(f"{name} must be a SHA-256 digest")


def _require_work_id(value: str) -> None:
    if _WORK_ID.fullmatch(value) is None:
        raise ValueError("work_id must be a stable identifier")


@dataclass(frozen=True)
class RunPlan:
    run_id: str
    descriptor_sha256: str
    source_sha256: str
    work_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if _RUN_ID.fullmatch(self.run_id) is None:
            raise ValueError("run_id must be a safe identifier")
        _require_sha256(self.descriptor_sha256, name="descriptor_sha256")
        _require_sha256(self.source_sha256, name="source_sha256")
        if not self.work_ids or len(set(self.work_ids)) != len(self.work_ids):
            raise ValueError("work_ids must be nonempty and unique")
        for work_id in self.work_ids:
            _require_work_id(work_id)


class CompletedRef(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    work_id: str
    cache_key: str

    @field_validator("work_id")
    @classmethod
    def _stable_work_id(cls, value: str) -> str:
        _require_work_id(value)
        return value

    @field_validator("cache_key")
    @classmethod
    def _cache_key(cls, value: str) -> str:
        _require_sha256(value, name="cache_key")
        return value


class RunManifest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    format_version: str = CHECKPOINT_FORMAT_VERSION
    run_id: str
    descriptor_sha256: str
    source_sha256: str
    work_ids: tuple[str, ...]
    completed: tuple[CompletedRef, ...] = ()
    metadata: dict[str, bool | int] = Field(default_factory=dict)
    publication: PublicationState = PublicationState.INCOMPLETE

    @field_validator("format_version")
    @classmethod
    def _format_version(cls, value: str) -> str:
        if value != CHECKPOINT_FORMAT_VERSION:
            raise ValueError("unsupported checkpoint format")
        return value

    @field_validator("run_id")
    @classmethod
    def _run_id(cls, value: str) -> str:
        if _RUN_ID.fullmatch(value) is None:
            raise ValueError("run_id must be a safe identifier")
        return value

    @field_validator("descriptor_sha256", "source_sha256")
    @classmethod
    def _digest(cls, value: str) -> str:
        _require_sha256(value, name="manifest digest")
        return value

    @field_validator("work_ids")
    @classmethod
    def _work_ids(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        if not values or len(set(values)) != len(values):
            raise ValueError("work_ids must be nonempty and unique")
        for value in values:
            _require_work_id(value)
        return values

    @field_validator("metadata", mode="before")
    @classmethod
    def _safe_metadata(cls, value: object) -> object:
        if not isinstance(value, dict):
            raise ValueError("metadata must be an object")
        for name, item in value.items():
            if (
                not isinstance(name, str)
                or _METADATA_KEY.fullmatch(name) is None
                or not (isinstance(item, bool) or type(item) is int)
            ):
                raise ValueError("metadata must contain safe scalar values")
        return value

    @model_validator(mode="after")
    def _references_follow_plan_order(self) -> RunManifest:
        completed_ids = tuple(reference.work_id for reference in self.completed)
        if len(set(completed_ids)) != len(completed_ids):
            raise ValueError("completed references must be unique")
        positions = {work_id: index for index, work_id in enumerate(self.work_ids)}
        if any(work_id not in positions for work_id in completed_ids):
            raise ValueError("completed reference is not planned")
        if tuple(sorted(completed_ids, key=positions.__getitem__)) != completed_ids:
            raise ValueError("completed references must follow planned work order")
        return self


@dataclass(frozen=True)
class ReuseResult:
    reference: CompletedRef
    payload: object | None
    reason: ReuseReason | None


class CheckpointSession:
    def __init__(
        self,
        manifest: RunManifest,
        cache: CacheStore,
        write_manifest: Callable[[RunManifest], None],
    ) -> None:
        self.manifest = manifest
        self._cache = cache
        self._write_manifest = write_manifest

    def checkpoint(
        self,
        *,
        completed: tuple[CompletedRef, ...] | None = None,
        descriptors: Mapping[str, CacheDescriptor] | None = None,
        metadata: Mapping[str, bool | int] | None = None,
        publication: PublicationState | None = None,
    ) -> None:
        if completed is not None:
            self._validate_completed_references(completed, descriptors)
        values = self.manifest.model_dump(mode="python")
        if completed is not None:
            values["completed"] = completed
        if metadata is not None:
            values["metadata"] = dict(metadata)
        if publication is not None:
            values["publication"] = publication
        manifest = RunManifest.model_validate(values)
        self._write_manifest(manifest)
        self.manifest = manifest

    def _validate_completed_references(
        self,
        completed: tuple[CompletedRef, ...],
        descriptors: Mapping[str, CacheDescriptor] | None,
    ) -> None:
        if descriptors is None:
            raise CheckpointError(CheckpointReason.INCOMPATIBLE)
        for reference in completed:
            descriptor = descriptors.get(reference.work_id)
            if (
                descriptor is None
                or descriptor.work_id != reference.work_id
                or descriptor.key != reference.cache_key
                or descriptor.source_id != self.manifest.source_sha256
            ):
                raise CheckpointError(CheckpointReason.INCOMPATIBLE)

    def reusable(
        self,
        *,
        descriptors: Mapping[str, CacheDescriptor],
        validators: Mapping[str, Callable[[object], object]],
    ) -> tuple[ReuseResult, ...]:
        results: list[ReuseResult] = []
        for reference in self.manifest.completed:
            descriptor = descriptors.get(reference.work_id)
            validator = validators.get(reference.work_id)
            if (
                descriptor is None
                or validator is None
                or descriptor.work_id != reference.work_id
                or descriptor.key != reference.cache_key
                or descriptor.source_id != self.manifest.source_sha256
            ):
                results.append(ReuseResult(reference, None, ReuseReason.INCOMPATIBLE))
                continue
            lookup = self._cache.load(descriptor, validator)
            if lookup.hit:
                results.append(ReuseResult(reference, lookup.payload, None))
                continue
            results.append(
                ReuseResult(reference, None, _reuse_reason(lookup.miss_reason))
            )
        return tuple(results)


def _reuse_reason(reason: CacheMissReason | None) -> ReuseReason:
    if reason is CacheMissReason.MISSING:
        return ReuseReason.MISSING
    if reason is CacheMissReason.CORRUPT:
        return ReuseReason.CORRUPT
    return ReuseReason.INCOMPATIBLE


class CheckpointStore:
    """Open one local run manifest at a time under an advisory run lock."""

    def __init__(self, root: Path) -> None:
        self._cache = CacheStore(root)

    @contextmanager
    def open(self, plan: RunPlan, *, resume: bool) -> Iterator[CheckpointSession]:
        with self._open_runs_directory(create=True) as runs_fd:
            with self._run_lock(runs_fd, plan.run_id):
                manifest = self._read_manifest(runs_fd, plan.run_id)
                if manifest is None:
                    if resume:
                        raise CheckpointError(CheckpointReason.MISSING)
                    manifest = RunManifest(
                        run_id=plan.run_id,
                        descriptor_sha256=plan.descriptor_sha256,
                        source_sha256=plan.source_sha256,
                        work_ids=plan.work_ids,
                    )
                    self._write_manifest(runs_fd, manifest)
                elif not resume:
                    raise CheckpointError(CheckpointReason.ALREADY_EXISTS)
                elif not _compatible(manifest, plan):
                    raise CheckpointError(CheckpointReason.INCOMPATIBLE)
                yield CheckpointSession(
                    manifest,
                    self._cache,
                    lambda next_manifest: self._write_manifest(runs_fd, next_manifest),
                )

    @contextmanager
    def _open_runs_directory(self, *, create: bool) -> Iterator[int]:
        root_fd = self._cache._open_root_directory(create=create)
        runs_fd: int | None = None
        try:
            runs_fd = self._cache._open_directory(
                root_fd, "runs", create=create, require_private=True
            )
            yield runs_fd
        finally:
            if runs_fd is not None:
                os.close(runs_fd)
            os.close(root_fd)

    @contextmanager
    def _run_lock(self, runs_fd: int, run_id: str) -> Iterator[None]:
        name = f"{run_id}.lock"
        lock_fd: int | None = None
        created = False
        try:
            try:
                lock_fd = os.open(
                    name,
                    os.O_CREAT | os.O_EXCL | os.O_RDWR | os.O_NOFOLLOW,
                    0o600,
                    dir_fd=runs_fd,
                )
                created = True
            except FileExistsError:
                lock_fd = self._cache._open_regular_file(
                    runs_fd, name, os.O_RDWR, expected_mode=0o600
                )
            if created:
                os.fchmod(lock_fd, 0o600)
            if os.fstat(lock_fd).st_mode & 0o777 != 0o600:
                raise _UnsafeCachePath("unsafe checkpoint lock")
        except _UnsafeCachePath:
            if lock_fd is not None:
                os.close(lock_fd)
            raise CheckpointError(CheckpointReason.CORRUPT) from None
        except OSError as error:
            if lock_fd is not None:
                os.close(lock_fd)
            raise CheckpointError(CheckpointReason.CORRUPT) from error
        assert lock_fd is not None
        try:
            try:
                fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as error:
                if error.errno in {errno.EACCES, errno.EAGAIN}:
                    raise CheckpointError(CheckpointReason.RUN_ACTIVE) from None
                raise CheckpointError(CheckpointReason.CORRUPT) from error
            yield
        finally:
            fcntl.flock(lock_fd, fcntl.LOCK_UN)
            os.close(lock_fd)

    def _read_manifest(self, runs_fd: int, run_id: str) -> RunManifest | None:
        try:
            manifest_fd = self._cache._open_regular_file(
                runs_fd, f"{run_id}.json", os.O_RDONLY, expected_mode=0o600
            )
        except FileNotFoundError:
            return None
        except _UnsafeCachePath:
            raise CheckpointError(CheckpointReason.CORRUPT) from None
        try:
            with os.fdopen(manifest_fd, "rb") as handle:
                payload = json.loads(handle.read())
            if not isinstance(payload, dict) or set(payload) != _MANIFEST_FIELDS:
                raise ValueError("invalid checkpoint manifest shape")
            return RunManifest.model_validate(payload)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError):
            raise CheckpointError(CheckpointReason.CORRUPT) from None

    def _write_manifest(self, runs_fd: int, manifest: RunManifest) -> None:
        encoded = json.dumps(
            manifest.model_dump(mode="json"), separators=(",", ":"), sort_keys=True
        ).encode("utf-8")
        self._cache._atomic_write(runs_fd, f"{manifest.run_id}.json", encoded)


def _compatible(manifest: RunManifest, plan: RunPlan) -> bool:
    return (
        manifest.run_id == plan.run_id
        and manifest.descriptor_sha256 == plan.descriptor_sha256
        and manifest.source_sha256 == plan.source_sha256
        and manifest.work_ids == plan.work_ids
    )
