"""Compose final writing, optional citations, and optional audit output."""

from __future__ import annotations

import hashlib
import os
import threading
import weakref
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile

from summarizer.audit import (
    AuditArtifact,
    Citation,
    build_audit_artifact,
    render_citations,
    resolve_citations,
    serialize_audit,
    write_audit,
)
from summarizer.checkpoint import (
    CheckpointSession,
    PublicationState,
    RunManifest,
)
from summarizer.editorial import write_editorial
from summarizer.hierarchy import TreeNode
from summarizer.providers.base import GenerationResult, ModelProvider
from summarizer.reliability import ReliabilityTracker
from summarizer.segmentation import CacheCoordinator, SourceSegment
from summarizer.summaries import SummaryNode
from summarizer.tokenization import TokenCounter
from summarizer.verification import (
    VerificationConfig,
    VerificationResult,
    VerificationRuntime,
    build_source_lexical_index,
    verify_and_repair,
)

_DEFAULT_VERIFICATION_CONFIG = VerificationConfig()
_PUBLICATION_LOCKS_GUARD = threading.Lock()
_PUBLICATION_LOCKS: weakref.WeakValueDictionary[tuple[str, str], threading.RLock] = (
    weakref.WeakValueDictionary()
)


@dataclass(frozen=True)
class FinalizationResult:
    text: str
    citations: tuple[Citation, ...]
    audit: AuditArtifact | None


class FinalizationVerificationError(RuntimeError):
    """Verification closed without a safe reader-facing final summary."""


class PublicationError(RuntimeError):
    """A final output pair is absent, incomplete, or does not match its witness."""


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _atomic_replace(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with NamedTemporaryFile(
            dir=path.parent, prefix=f".{path.name}.", delete=False
        ) as handle:
            temporary = Path(handle.name)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except OSError:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise


def _path_key(summary_path: Path, audit_path: Path) -> tuple[str, str]:
    summary_key = str(summary_path.resolve(strict=False))
    audit_key = str(audit_path.resolve(strict=False))
    return (
        (summary_key, audit_key)
        if summary_key <= audit_key
        else (audit_key, summary_key)
    )


@contextmanager
def _publication_lock(summary_path: Path, audit_path: Path) -> Iterator[None]:
    key = _path_key(summary_path, audit_path)
    with _PUBLICATION_LOCKS_GUARD:
        lock = _PUBLICATION_LOCKS.get(key)
        if lock is None:
            lock = threading.RLock()
            _PUBLICATION_LOCKS[key] = lock
    with lock:
        yield


def _read_bytes(path: Path) -> bytes | None:
    try:
        return path.read_bytes()
    except OSError:
        return None


def _matches(path: Path, digest: str) -> bool:
    payload = _read_bytes(path)
    return payload is not None and _sha256(payload) == digest


def publish_final_output(
    result: FinalizationResult,
    *,
    summary_path: Path,
    audit_path: Path,
    session: CheckpointSession,
    atomic_replace: Callable[[Path, bytes], None] = _atomic_replace,
) -> None:
    """Publish a reliable audit before using the summary as its witness."""
    if summary_path.resolve(strict=False) == audit_path.resolve(strict=False):
        raise PublicationError("summary_path and audit_path must differ")
    if (
        result.audit is None
        or result.audit.schema_version not in {"audit/3", "audit/4"}
        or (
            result.audit.schema_version == "audit/4"
            and result.audit.reliability is None
        )
    ):
        raise PublicationError("reliable publication requires a validated reliable audit")
    audit_payload = serialize_audit(result.audit)
    summary_payload = result.text.encode("utf-8")
    audit_digest = _sha256(audit_payload)
    summary_digest = _sha256(summary_payload)
    with _publication_lock(summary_path, audit_path):
        manifest = session.manifest
        expected = (manifest.audit_sha256, manifest.summary_sha256)
        files_match = _matches(audit_path, audit_digest) and _matches(
            summary_path, summary_digest
        )
        if (
            expected == (audit_digest, summary_digest)
            and manifest.publication is PublicationState.COMPLETE
            and files_match
        ):
            return
        if (
            expected == (audit_digest, summary_digest)
            and manifest.publication is PublicationState.AUDIT_STAGED
            and files_match
        ):
            session.complete_publication()
            return

        atomic_replace(audit_path, audit_payload)
        session.stage_publication(
            audit_sha256=audit_digest, summary_sha256=summary_digest
        )
        atomic_replace(summary_path, summary_payload)
        session.complete_publication()


def read_published_summary(
    summary_path: Path, audit_path: Path, manifest: RunManifest
) -> str:
    """Read a final summary only when the manifest witnesses both exact files."""
    with _publication_lock(summary_path, audit_path):
        audit_payload = _read_bytes(audit_path)
        summary_payload = _read_bytes(summary_path)
        if (
            manifest.publication is not PublicationState.COMPLETE
            or manifest.audit_sha256 is None
            or manifest.summary_sha256 is None
            or audit_payload is None
            or summary_payload is None
            or _sha256(audit_payload) != manifest.audit_sha256
            or _sha256(summary_payload) != manifest.summary_sha256
        ):
            raise PublicationError("final output publication is incomplete")
        try:
            return summary_payload.decode("utf-8")
        except UnicodeDecodeError as error:
            raise PublicationError("final summary is unreadable") from error


def _verification_runtime(
    *,
    provider: ModelProvider,
    counter: TokenCounter | None,
    model: str,
    timeout_seconds: float,
    context_window_tokens: int | None,
    injected: VerificationRuntime | None,
) -> VerificationRuntime:
    """Resolve the default dependencies or use an injected complete runtime."""
    if injected is None:
        if counter is None or context_window_tokens is None:
            raise ValueError(
                "enabled verification requires a counter and context window"
            )
        runtime = VerificationRuntime(
            provider=provider,
            counter=counter,
            model=model,
            timeout_seconds=timeout_seconds,
            context_window_tokens=context_window_tokens,
        )
    else:
        runtime = injected
    return runtime


def _build_audit(
    *,
    audit_path: Path | None,
    source_id: str,
    strategy: str,
    model: str,
    audit_configuration: Mapping[str, object] | None,
    segments: Sequence[SourceSegment],
    nodes: Sequence[TreeNode],
    root_node_id: str,
    citations: Sequence[Citation],
    generations: Sequence[GenerationResult],
    warnings: Sequence[str],
    failures: Sequence[str],
    verification: VerificationResult | None,
    verification_enabled: bool,
    reliability_resume: Mapping[str, object] | None = None,
    reliability_tracker: ReliabilityTracker | None = None,
    materialize: bool,
) -> AuditArtifact | None:
    if audit_path is None:
        return None
    snapshot = reliability_tracker.snapshot() if reliability_tracker else None
    if snapshot is not None:
        reliability_resume = {
            "resumed": snapshot.resumed,
            "reused_count": snapshot.reused_count,
            "recomputed_count": snapshot.recomputed_count,
        }
    artifact = build_audit_artifact(
        source_id=source_id,
        strategy=strategy,
        model=model,
        configuration=audit_configuration or {},
        segments=segments,
        nodes=nodes,
        root_node_id=root_node_id,
        citations=citations,
        generations=generations,
        warnings=warnings,
        failures=failures,
        verification=verification,
        verification_enabled=verification_enabled,
        reliability_resume=reliability_resume,
        reliability_cache=snapshot.cache if snapshot is not None else None,
        reliability_attempts=snapshot.attempts if snapshot is not None else None,
    )
    if materialize:
        write_audit(audit_path, artifact)
    return artifact


def _finalize_summary(
    root: SummaryNode,
    provider: ModelProvider,
    *,
    source_id: str,
    model: str,
    timeout_seconds: float,
    target_words: int,
    strategy: str,
    segments: Sequence[SourceSegment],
    nodes: Sequence[TreeNode],
    root_node_id: str,
    include_citations: bool = False,
    audit_configuration: Mapping[str, object] | None = None,
    audit_path: Path | None = None,
    generations: Sequence[GenerationResult] = (),
    warnings: Sequence[str] = (),
    failures: Sequence[str] = (),
    counter: TokenCounter | None = None,
    source_cores: Mapping[str, str] | None = None,
    verification: VerificationConfig = _DEFAULT_VERIFICATION_CONFIG,
    verification_runtime: VerificationRuntime | None = None,
    verification_context_window_tokens: int | None = None,
    verification_coordinator: CacheCoordinator | None = None,
    reliability_resume: Mapping[str, object] | None = None,
    reliability_tracker: ReliabilityTracker | None = None,
    materialize_audit: bool,
) -> FinalizationResult:
    """Run the final editor and materialize optional safe output views.

    The root's own validated provenance determines citations. The writer does
    not choose or invent source identifiers, so output formatting cannot leave
    a citation dangling from the recorded source metadata.
    """
    editorial = write_editorial(
        root,
        provider,
        source_id=source_id,
        model=model,
        timeout_seconds=timeout_seconds,
        target_words=target_words,
    )
    verification_result: VerificationResult | None = None
    if verification.enabled:
        if source_cores is None:
            raise ValueError(
                "enabled verification requires root-provenance source cores"
            )
        runtime = _verification_runtime(
            provider=provider,
            counter=counter,
            model=model,
            timeout_seconds=timeout_seconds,
            context_window_tokens=verification_context_window_tokens,
            injected=verification_runtime,
        )
        verification_result = verify_and_repair(
            editorial.text,
            source_id=source_id,
            source_index=build_source_lexical_index(
                provenance_ids=root.provenance,
                source=source_cores,
            ),
            runtime=runtime,
            config=verification,
            coordinator=verification_coordinator,
        )
        if verification_result.failed:
            _build_audit(
                audit_path=audit_path,
                source_id=source_id,
                strategy=strategy,
                model=model,
                audit_configuration=audit_configuration,
                segments=segments,
                nodes=nodes,
                root_node_id=root_node_id,
                citations=(),
                generations=(*generations, editorial.generation),
                warnings=warnings,
                failures=failures,
                verification=verification_result,
                verification_enabled=True,
                reliability_resume=reliability_resume,
                reliability_tracker=reliability_tracker,
                materialize=True,
            )
            raise FinalizationVerificationError(
                "verification did not produce a safe final summary"
            )

    citations = resolve_citations(
        root.provenance, source_id=source_id, segments=segments
    )
    final_text = verification_result.text if verification_result else editorial.text
    text = render_citations(final_text, citations) if include_citations else final_text

    artifact = _build_audit(
        audit_path=audit_path,
        source_id=source_id,
        strategy=strategy,
        model=model,
        audit_configuration=audit_configuration,
        segments=segments,
        nodes=nodes,
        root_node_id=root_node_id,
        citations=citations,
        generations=(*generations, editorial.generation),
        warnings=warnings,
        failures=failures,
        verification=verification_result,
        verification_enabled=verification.enabled,
        reliability_resume=reliability_resume,
        reliability_tracker=reliability_tracker,
        materialize=materialize_audit,
    )
    return FinalizationResult(text=text, citations=citations, audit=artifact)


def finalize_summary(
    root: SummaryNode,
    provider: ModelProvider,
    *,
    source_id: str,
    model: str,
    timeout_seconds: float,
    target_words: int,
    strategy: str,
    segments: Sequence[SourceSegment],
    nodes: Sequence[TreeNode],
    root_node_id: str,
    include_citations: bool = False,
    audit_configuration: Mapping[str, object] | None = None,
    audit_path: Path | None = None,
    generations: Sequence[GenerationResult] = (),
    warnings: Sequence[str] = (),
    failures: Sequence[str] = (),
    counter: TokenCounter | None = None,
    source_cores: Mapping[str, str] | None = None,
    verification: VerificationConfig = _DEFAULT_VERIFICATION_CONFIG,
    verification_runtime: VerificationRuntime | None = None,
    verification_context_window_tokens: int | None = None,
    verification_coordinator: CacheCoordinator | None = None,
    reliability_resume: Mapping[str, object] | None = None,
    reliability_tracker: ReliabilityTracker | None = None,
) -> FinalizationResult:
    """Run finalization and materialize its requested audit output."""
    return _finalize_summary(
        root,
        provider,
        source_id=source_id,
        model=model,
        timeout_seconds=timeout_seconds,
        target_words=target_words,
        strategy=strategy,
        segments=segments,
        nodes=nodes,
        root_node_id=root_node_id,
        include_citations=include_citations,
        audit_configuration=audit_configuration,
        audit_path=audit_path,
        generations=generations,
        warnings=warnings,
        failures=failures,
        counter=counter,
        source_cores=source_cores,
        verification=verification,
        verification_runtime=verification_runtime,
        verification_context_window_tokens=verification_context_window_tokens,
        verification_coordinator=verification_coordinator,
        reliability_resume=reliability_resume,
        reliability_tracker=reliability_tracker,
        materialize_audit=True,
    )
