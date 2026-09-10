"""Private, descriptor-keyed storage for validated successful stage results."""

from __future__ import annotations

import errno
from collections.abc import Callable, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import secrets
import stat
from typing import TypeVar


CACHE_FORMAT_VERSION = "cache/1"
_PROJECTION_FORMAT_VERSION = "cache-projection/1"
_ENVELOPE_FIELDS = frozenset(
    {"descriptor", "format_version", "payload", "payload_kind", "payload_sha256"}
)
_PROJECTION_ENVELOPE_FIELDS = frozenset({"descriptor", "format_version"})
_Payload = TypeVar("_Payload")
_STAGES = frozenset(
    {"segmentation", "direct", "leaf", "merge", "editorial", "verification"}
)
_PROVIDERS = frozenset({"openai", "ollama"})
_VERSION = re.compile(r"^[a-z][a-z0-9-]*/[1-9][0-9]*$")
_MODEL = re.compile(
    r"^[A-Za-z0-9][A-Za-z0-9._/-]*(?::[A-Za-z0-9][A-Za-z0-9._/-]*)?$"
)
_HOST_PORT = re.compile(r"^[A-Za-z0-9.-]+:[0-9]{2,5}$")
_CREDENTIAL_TOKEN = re.compile(
    r"^(?:gh[opsu]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|"
    r"sk-[A-Za-z0-9_-]{8,}|AIza[A-Za-z0-9_-]{12,}|AKIA[A-Z0-9]{12,}|"
    r"xox[baprs]-[0-9]{6,}-[0-9]{6,}-[A-Za-z0-9-]{16,}|"
    r"glpat-[A-Za-z0-9_-]{20,}|hf_[A-Za-z0-9]{20,})$"
)
_COUNTER = re.compile(
    r"^(?:(?:tiktoken|estimate|test):[a-z0-9][a-z0-9._-]*|utf8-conservative)$"
)
_WORK_ID = re.compile(
    r"^(?:[DSLVM][A-Za-z0-9:_-]*|editorial-final|segmentation)$"
)
_STRATEGIES = frozenset({"auto", "direct", "hierarchical"})
_BOOLEAN_FIELDS = frozenset(
    {"counter_exact", "include_citations", "verification_enabled"}
)
_POSITIVE_INTEGER_FIELDS = frozenset(
    {
        "context_window",
        "evidence_tokens",
        "max_in_flight",
        "max_output_tokens",
        "max_tokens",
        "request_tokens",
        "target_words",
    }
)
_POSITIVE_NUMBER_FIELDS = frozenset({"timeout_seconds"})
_NONNEGATIVE_INTEGER_FIELDS = frozenset(
    {"max_repair_passes", "overlap_tokens", "safety_margin_tokens"}
)
_NULLABLE_POSITIVE_INTEGER_FIELDS = frozenset(
    {"max_direct_tokens", "max_merge_children"}
)
_VERSION_FIELDS = frozenset({"editorial_version", "grounding_policy"})
_BEHAVIOR_GROUPS = {
    "grounding": frozenset({"max_tokens"}),
    "segmentation": frozenset({"max_tokens", "overlap_tokens"}),
    "strategy_config": frozenset(
        {
            "context_window",
            "max_direct_tokens",
            "max_output_tokens",
            "safety_margin_tokens",
            "strategy",
        }
    ),
    "budget": frozenset(
        {"context_window", "max_output_tokens", "safety_margin_fraction", "safety_margin_tokens"}
    ),
    "verification": frozenset(
        {
            "evidence_tokens",
            "max_repair_passes",
            "output_reserve_tokens",
            "request_tokens",
            "safety_margin_tokens",
            "verification_enabled",
        }
    ),
}
_ALL_BEHAVIOR_FIELDS = (
    _BOOLEAN_FIELDS
    | _POSITIVE_INTEGER_FIELDS
    | _POSITIVE_NUMBER_FIELDS
    | _NONNEGATIVE_INTEGER_FIELDS
    | _NULLABLE_POSITIVE_INTEGER_FIELDS
    | _VERSION_FIELDS
    | {"output_reserve_tokens", "safety_margin_fraction", "strategy"}
)


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    ).encode("utf-8")


def _require_json_value(value: object, *, label: str) -> None:
    try:
        _canonical_json(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must be JSON serializable") from error


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


def _is_credential_token(value: str) -> bool:
    return _CREDENTIAL_TOKEN.fullmatch(value) is not None


def _is_safe_model(value: str) -> bool:
    return (
        _MODEL.fullmatch(value) is not None
        and _HOST_PORT.fullmatch(value) is None
        and not _is_credential_token(value)
        and value != "secret"
    )


def _is_integer(value: object, *, minimum: int) -> bool:
    return (
        isinstance(value, int)
        and not isinstance(value, bool)
        and value >= minimum
    )


def _canonical_cache_root(root: Path) -> Path:
    if root == Path(".") or root == Path(root.anchor) or ".." in root.parts:
        raise _UnsafeCachePath("unsafe cache root")
    missing: list[str] = []
    candidate = root
    while True:
        try:
            metadata = candidate.lstat()
        except FileNotFoundError:
            missing.append(candidate.name)
            candidate = candidate.parent
            continue
        if not missing:
            if stat.S_ISLNK(metadata.st_mode):
                raise _UnsafeCachePath("unsafe cache root")
            physical_parent = candidate.parent.resolve(strict=True)
            missing.append(candidate.name)
        else:
            physical_parent = candidate.resolve(strict=True)
        if not physical_parent.is_dir():
            raise _UnsafeCachePath("unsafe cache root")
        return physical_parent.joinpath(*reversed(missing))


def _validate_behavior_field(name: str, value: object) -> None:
    if name in _BOOLEAN_FIELDS:
        valid = isinstance(value, bool)
    elif name in _POSITIVE_INTEGER_FIELDS:
        valid = _is_integer(value, minimum=1)
    elif name in _POSITIVE_NUMBER_FIELDS:
        valid = (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and value > 0
        )
    elif name in _NONNEGATIVE_INTEGER_FIELDS:
        valid = _is_integer(value, minimum=0)
    elif name in _NULLABLE_POSITIVE_INTEGER_FIELDS:
        valid = value is None or _is_integer(value, minimum=1)
    elif name == "output_reserve_tokens":
        valid = _is_integer(value, minimum=1)
    elif name == "safety_margin_fraction":
        valid = (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and 0 <= value < 1
        )
    elif name == "strategy":
        valid = value in _STRATEGIES
    elif name in _VERSION_FIELDS:
        valid = isinstance(value, str) and _VERSION.fullmatch(value) is not None
    else:
        valid = False
    if not valid:
        raise ValueError(f"unsafe descriptor behavior value for {name}")


def _validate_behavior(behavior: Mapping[str, object]) -> None:
    for name, value in behavior.items():
        if name in _ALL_BEHAVIOR_FIELDS:
            _validate_behavior_field(name, value)
            continue
        fields = _BEHAVIOR_GROUPS.get(name)
        if fields is None or not isinstance(value, Mapping):
            raise ValueError(f"unsafe descriptor behavior key: {name}")
        if not set(value).issubset(fields):
            raise ValueError(f"unsafe descriptor behavior key: {name}")
        for nested_name, nested_value in value.items():
            _validate_behavior_field(nested_name, nested_value)


@dataclass(frozen=True)
class CacheDescriptor:
    """All safe, output-affecting identities for one cached stage result."""

    source_id: str
    input_hash: str
    stage: str
    work_id: str
    prompt_version: str
    schema_version: str
    provider: str
    model: str
    counter_identity: str
    counter_exact: bool
    context_window_tokens: int
    behavior: Mapping[str, object]
    _behavior_bytes: bytes = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.stage not in _STAGES:
            raise ValueError("unsafe descriptor stage")
        if _WORK_ID.fullmatch(self.work_id) is None:
            raise ValueError("unsafe descriptor work_id")
        if _VERSION.fullmatch(self.prompt_version) is None:
            raise ValueError("unsafe descriptor prompt_version")
        if _VERSION.fullmatch(self.schema_version) is None:
            raise ValueError("unsafe descriptor schema_version")
        if self.provider not in _PROVIDERS:
            raise ValueError("unsafe descriptor provider")
        if not _is_safe_model(self.model):
            raise ValueError("unsafe descriptor model")
        if _COUNTER.fullmatch(self.counter_identity) is None:
            raise ValueError("unsafe descriptor counter_identity")
        if not _is_sha256(self.source_id):
            raise ValueError("unsafe descriptor source_id")
        if not _is_sha256(self.input_hash):
            raise ValueError("input_hash must be a lowercase SHA-256 digest")
        if not _is_integer(self.context_window_tokens, minimum=1):
            raise ValueError("context_window_tokens must be positive")
        if not isinstance(self.counter_exact, bool):
            raise ValueError("unsafe descriptor counter_exact")
        if not isinstance(self.behavior, Mapping):
            raise ValueError("unsafe descriptor behavior")
        _validate_behavior(self.behavior)
        _require_json_value(self.behavior, label="descriptor behavior")
        object.__setattr__(self, "_behavior_bytes", _canonical_json(self.behavior))

    def canonical_value(self) -> dict[str, object]:
        return {
            "behavior": json.loads(self._behavior_bytes),
            "context_window_tokens": self.context_window_tokens,
            "counter_exact": self.counter_exact,
            "counter_identity": self.counter_identity,
            "input_hash": self.input_hash,
            "model": self.model,
            "prompt_version": self.prompt_version,
            "provider": self.provider,
            "schema_version": self.schema_version,
            "source_id": self.source_id,
            "stage": self.stage,
            "work_id": self.work_id,
        }

    def canonical_bytes(self) -> bytes:
        return _canonical_json(self.canonical_value())

    @property
    def key(self) -> str:
        return hashlib.sha256(self.canonical_bytes()).hexdigest()


def _descriptor_invalidation_reasons(
    previous: CacheDescriptor, current: CacheDescriptor
) -> tuple[str, ...]:
    reasons: list[str] = []
    if previous.source_id != current.source_id:
        reasons.append("source_changed")
    if previous.prompt_version != current.prompt_version:
        reasons.append("prompt_changed")
    if previous.schema_version != current.schema_version:
        reasons.append("schema_changed")
    if (previous.provider, previous.model) != (current.provider, current.model):
        reasons.append("model_changed")
    behavior_changed = (
        previous.counter_identity != current.counter_identity
        or previous.counter_exact != current.counter_exact
        or previous.context_window_tokens != current.context_window_tokens
        or previous._behavior_bytes != current._behavior_bytes
    )
    if behavior_changed:
        reasons.append("behavior_changed")
    return tuple(reasons)


class CacheMissReason(str, Enum):
    MISSING = "missing"
    CORRUPT = "corrupt"
    WRONG_VERSION = "wrong_version"
    INCOMPATIBLE = "incompatible"


@dataclass(frozen=True)
class CacheLookup:
    payload: object | None
    miss_reason: CacheMissReason | None

    @property
    def hit(self) -> bool:
        return self.miss_reason is None


class _UnsafeCachePath(ValueError):
    pass


class CacheStore:
    """Read and atomically publish validated objects under a narrow key lock."""

    def __init__(self, root: Path) -> None:
        self._root = _canonical_cache_root(root)

    def object_path(self, descriptor: CacheDescriptor) -> Path:
        return self._root / "objects" / descriptor.key[:2] / f"{descriptor.key}.json"

    def load(
        self, descriptor: CacheDescriptor, validate: Callable[[object], _Payload]
    ) -> CacheLookup:
        return self._load(descriptor, validate)

    def invalidation_reasons(self, descriptor: CacheDescriptor) -> tuple[str, ...]:
        """Compare a safe prior projection for this logical cache work item."""
        try:
            with self._open_projection_directory(create=False) as directory_fd:
                previous = self._load_projection(
                    directory_fd, self._projection_key(descriptor)
                )
        except (FileNotFoundError, _UnsafeCachePath):
            return ()
        if previous is None:
            return ()
        return _descriptor_invalidation_reasons(previous, descriptor)

    def record_descriptor_projection(self, descriptor: CacheDescriptor) -> None:
        """Advance the private safe projection after a validated terminal result."""
        projection_key = self._projection_key(descriptor)
        envelope = {
            "descriptor": descriptor.canonical_value(),
            "format_version": _PROJECTION_FORMAT_VERSION,
        }
        with self._open_projection_directory(create=True) as directory_fd:
            with self._key_lock(directory_fd, projection_key):
                self._atomic_write(
                    directory_fd,
                    f"{projection_key}.json",
                    _canonical_json(envelope),
                )

    def store(
        self,
        descriptor: CacheDescriptor,
        payload: object,
        validate: Callable[[object], _Payload],
    ) -> Path:
        path, _ = self._store_winner(descriptor, payload, validate)
        return path

    def store_winner(
        self,
        descriptor: CacheDescriptor,
        payload: object,
        validate: Callable[[object], _Payload],
    ) -> _Payload:
        """Store one validated payload and return the locked first-writer value."""
        _, winner = self._store_winner(descriptor, payload, validate)
        return winner

    def _store_winner(
        self,
        descriptor: CacheDescriptor,
        payload: object,
        validate: Callable[[object], _Payload],
    ) -> tuple[Path, _Payload]:
        validated = validate(payload)
        _require_json_value(validated, label="validated payload")
        path = self.object_path(descriptor)
        with self._open_shard_directory(descriptor, create=True) as directory_fd:
            with self._key_lock(directory_fd, descriptor.key):
                existing = self._load_from_directory(
                    directory_fd,
                    descriptor,
                    validate,
                    strict_paths=True,
                )
                if existing.hit:
                    return path, validate(existing.payload)
                envelope = {
                    "descriptor": descriptor.canonical_value(),
                    "format_version": CACHE_FORMAT_VERSION,
                    "payload": validated,
                    "payload_kind": descriptor.schema_version,
                    "payload_sha256": hashlib.sha256(_canonical_json(validated)).hexdigest(),
                }
                self._atomic_write(
                    directory_fd,
                    f"{descriptor.key}.json",
                    _canonical_json(envelope),
                )
        return path, validated

    def _load(
        self, descriptor: CacheDescriptor, validate: Callable[[object], _Payload]
    ) -> CacheLookup:
        try:
            with self._open_shard_directory(descriptor, create=False) as directory_fd:
                return self._load_from_directory(directory_fd, descriptor, validate)
        except FileNotFoundError:
            return CacheLookup(None, CacheMissReason.MISSING)
        except _UnsafeCachePath:
            return CacheLookup(None, CacheMissReason.CORRUPT)

    @staticmethod
    def _projection_key(descriptor: CacheDescriptor) -> str:
        return hashlib.sha256(
            _canonical_json({"stage": descriptor.stage, "work_id": descriptor.work_id})
        ).hexdigest()

    def _load_projection(
        self, directory_fd: int, projection_key: str
    ) -> CacheDescriptor | None:
        try:
            projection_fd = self._open_regular_file(
                directory_fd,
                f"{projection_key}.json",
                os.O_RDONLY,
                expected_mode=0o600,
            )
        except FileNotFoundError:
            return None
        try:
            with os.fdopen(projection_fd, "rb") as handle:
                envelope = json.loads(handle.read())
            if (
                not isinstance(envelope, dict)
                or set(envelope) != _PROJECTION_ENVELOPE_FIELDS
                or envelope.get("format_version") != _PROJECTION_FORMAT_VERSION
                or not isinstance(envelope.get("descriptor"), dict)
            ):
                return None
            descriptor = CacheDescriptor(**envelope["descriptor"])
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError):
            return None
        if self._projection_key(descriptor) != projection_key:
            return None
        return descriptor

    def _load_from_directory(
        self,
        directory_fd: int,
        descriptor: CacheDescriptor,
        validate: Callable[[object], _Payload],
        *,
        strict_paths: bool = False,
    ) -> CacheLookup:
        object_name = f"{descriptor.key}.json"
        if not strict_paths:
            try:
                self._check_existing_lock(directory_fd, descriptor.key)
            except _UnsafeCachePath:
                return CacheLookup(None, CacheMissReason.CORRUPT)
        try:
            descriptor_fd = self._open_regular_file(
                directory_fd,
                object_name,
                os.O_RDONLY,
                expected_mode=0o600,
            )
        except FileNotFoundError:
            return CacheLookup(None, CacheMissReason.MISSING)
        except _UnsafeCachePath:
            if strict_paths:
                raise
            return CacheLookup(None, CacheMissReason.CORRUPT)
        try:
            with os.fdopen(descriptor_fd, "rb") as handle:
                encoded = handle.read()
        except OSError:
            return CacheLookup(None, CacheMissReason.CORRUPT)
        try:
            envelope = json.loads(encoded)
            if not isinstance(envelope, dict):
                raise ValueError("envelope must be an object")
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError):
            return CacheLookup(None, CacheMissReason.CORRUPT)
        if envelope.get("format_version") != CACHE_FORMAT_VERSION:
            return CacheLookup(None, CacheMissReason.WRONG_VERSION)
        if set(envelope) != _ENVELOPE_FIELDS:
            return CacheLookup(None, CacheMissReason.CORRUPT)
        if envelope.get("descriptor") != descriptor.canonical_value():
            return CacheLookup(None, CacheMissReason.INCOMPATIBLE)
        if envelope.get("payload_kind") != descriptor.schema_version:
            return CacheLookup(None, CacheMissReason.INCOMPATIBLE)
        payload = envelope.get("payload")
        digest = envelope.get("payload_sha256")
        try:
            payload_digest = hashlib.sha256(_canonical_json(payload)).hexdigest()
        except (TypeError, ValueError):
            return CacheLookup(None, CacheMissReason.CORRUPT)
        if not isinstance(digest, str) or digest != payload_digest:
            return CacheLookup(None, CacheMissReason.CORRUPT)
        try:
            validated = validate(payload)
        except (TypeError, ValueError):
            return CacheLookup(None, CacheMissReason.CORRUPT)
        return CacheLookup(validated, None)

    @contextmanager
    def _open_shard_directory(self, descriptor: CacheDescriptor, *, create: bool):
        root_fd: int | None = None
        objects_fd: int | None = None
        shard_fd: int | None = None
        try:
            root_fd = self._open_root_directory(create=create)
            objects_fd = self._open_directory(
                root_fd,
                "objects",
                create=create,
                require_private=True,
            )
            shard_fd = self._open_directory(
                objects_fd,
                descriptor.key[:2],
                create=create,
                require_private=True,
            )
            yield shard_fd
        finally:
            for descriptor_fd in (shard_fd, objects_fd, root_fd):
                if descriptor_fd is not None:
                    os.close(descriptor_fd)

    @contextmanager
    def _open_projection_directory(self, *, create: bool):
        root_fd: int | None = None
        projections_fd: int | None = None
        try:
            root_fd = self._open_root_directory(create=create)
            projections_fd = self._open_directory(
                root_fd,
                "projections",
                create=create,
                require_private=True,
            )
            yield projections_fd
        finally:
            if projections_fd is not None:
                os.close(projections_fd)
            if root_fd is not None:
                os.close(root_fd)

    def _open_root_directory(self, *, create: bool) -> int:
        root = self._root
        if root == Path(".") or ".." in root.parts:
            raise _UnsafeCachePath("unsafe cache root")
        parts = root.parts[1:] if root.is_absolute() else root.parts
        if not parts:
            raise _UnsafeCachePath("unsafe cache root")
        directory_fd = os.open(
            root.anchor if root.is_absolute() else ".",
            os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
        )
        try:
            for index, component in enumerate(parts):
                child_fd = self._open_directory(
                    directory_fd,
                    component,
                    create=create,
                    require_private=index == len(parts) - 1,
                )
                os.close(directory_fd)
                directory_fd = child_fd
            return directory_fd
        except BaseException:
            os.close(directory_fd)
            raise

    @staticmethod
    def _open_directory(
        parent_fd: int,
        name: str,
        *,
        create: bool,
        require_private: bool,
    ) -> int:
        created = False
        try:
            metadata = os.lstat(name, dir_fd=parent_fd)
        except FileNotFoundError:
            if not create:
                raise
            try:
                os.mkdir(name, mode=0o700, dir_fd=parent_fd)
                created = True
            except FileExistsError:
                pass
            metadata = CacheStore._lstat(parent_fd, name)
        except OSError as error:
            raise _UnsafeCachePath("unsafe cache directory") from error
        if not stat.S_ISDIR(metadata.st_mode):
            raise _UnsafeCachePath("unsafe cache directory")
        try:
            directory_fd = os.open(
                name,
                os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                dir_fd=parent_fd,
            )
        except OSError as error:
            raise _UnsafeCachePath("unsafe cache directory") from error
        actual = os.fstat(directory_fd)
        if not stat.S_ISDIR(actual.st_mode) or (
            actual.st_dev,
            actual.st_ino,
        ) != (metadata.st_dev, metadata.st_ino):
            os.close(directory_fd)
            raise _UnsafeCachePath("unsafe cache directory")
        if require_private:
            if created:
                os.fchmod(directory_fd, 0o700)
            elif actual.st_mode & 0o777 != 0o700:
                os.close(directory_fd)
                raise _UnsafeCachePath("unsafe cache directory")
        return directory_fd

    @staticmethod
    def _lstat(parent_fd: int, name: str) -> os.stat_result:
        try:
            return os.lstat(name, dir_fd=parent_fd)
        except FileNotFoundError:
            raise
        except OSError as error:
            raise _UnsafeCachePath("unsafe cache path") from error

    @staticmethod
    def _open_regular_file(
        parent_fd: int,
        name: str,
        flags: int,
        *,
        expected_mode: int,
    ) -> int:
        metadata = CacheStore._lstat(parent_fd, name)
        if not stat.S_ISREG(metadata.st_mode):
            raise _UnsafeCachePath("unsafe cache object")
        try:
            descriptor_fd = os.open(
                name,
                flags | os.O_NOFOLLOW,
                dir_fd=parent_fd,
            )
        except OSError as error:
            raise _UnsafeCachePath("unsafe cache object") from error
        actual = os.fstat(descriptor_fd)
        if (
            not stat.S_ISREG(actual.st_mode)
            or actual.st_mode & 0o777 != expected_mode
            or (
                actual.st_dev,
                actual.st_ino,
            ) != (metadata.st_dev, metadata.st_ino)
        ):
            os.close(descriptor_fd)
            raise _UnsafeCachePath("unsafe cache object")
        return descriptor_fd

    def _check_existing_lock(self, directory_fd: int, key: str) -> None:
        try:
            descriptor_fd = self._open_regular_file(
                directory_fd,
                f"{key}.lock",
                os.O_RDONLY,
                expected_mode=0o600,
            )
        except FileNotFoundError:
            return
        os.close(descriptor_fd)

    @contextmanager
    def _key_lock(self, directory_fd: int, key: str):
        lock_name = f"{key}.lock"
        created = False
        try:
            descriptor = os.open(
                lock_name,
                os.O_CREAT | os.O_EXCL | os.O_RDWR | os.O_NOFOLLOW,
                0o600,
                dir_fd=directory_fd,
            )
            created = True
        except FileExistsError:
            descriptor = self._open_regular_file(
                directory_fd,
                lock_name,
                os.O_RDWR,
                expected_mode=0o600,
            )
        except OSError as error:
            raise _UnsafeCachePath("unsafe cache lock") from error
        try:
            if created:
                os.fchmod(descriptor, 0o600)
            if os.fstat(descriptor).st_mode & 0o777 != 0o600:
                raise _UnsafeCachePath("unsafe cache lock")
            fcntl.flock(descriptor, fcntl.LOCK_EX)
            yield
        finally:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
            os.close(descriptor)

    def _atomic_write(
        self,
        directory_fd: int,
        object_name: str,
        payload: bytes,
    ) -> None:
        temporary_name = f".{object_name}.{secrets.token_hex(16)}.tmp"
        temporary_fd: int | None = None
        try:
            temporary_fd = os.open(
                temporary_name,
                os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_NOFOLLOW,
                0o600,
                dir_fd=directory_fd,
            )
            os.fchmod(temporary_fd, 0o600)
            with os.fdopen(temporary_fd, "wb") as handle:
                temporary_fd = None
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(
                temporary_name,
                object_name,
                src_dir_fd=directory_fd,
                dst_dir_fd=directory_fd,
            )
            temporary_name = ""
            self._sync_directory(directory_fd)
        finally:
            if temporary_fd is not None:
                os.close(temporary_fd)
            if temporary_name:
                try:
                    os.unlink(temporary_name, dir_fd=directory_fd)
                except FileNotFoundError:
                    pass

    @staticmethod
    def _sync_directory(descriptor: int) -> None:
        try:
            os.fsync(descriptor)
        except OSError as error:
            if error.errno not in {errno.EINVAL, errno.ENOTSUP}:
                raise
