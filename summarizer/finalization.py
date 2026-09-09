"""Compose final writing, optional citations, and optional audit output."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from summarizer.audit import (
    AuditArtifact,
    Citation,
    build_audit_artifact,
    render_citations,
    resolve_citations,
    write_audit,
)
from summarizer.editorial import write_editorial
from summarizer.hierarchy import TreeNode
from summarizer.providers.base import GenerationResult, ModelProvider
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


@dataclass(frozen=True)
class FinalizationResult:
    text: str
    citations: tuple[Citation, ...]
    audit: AuditArtifact | None


class FinalizationVerificationError(RuntimeError):
    """Verification closed without a safe reader-facing final summary."""


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


def _write_audit(
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
) -> AuditArtifact | None:
    if audit_path is None:
        return None
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
    )
    write_audit(audit_path, artifact)
    return artifact


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
            raise ValueError("enabled verification requires root-provenance source cores")
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
            _write_audit(
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
            )
            raise FinalizationVerificationError("verification did not produce a safe final summary")

    citations = resolve_citations(
        root.provenance, source_id=source_id, segments=segments
    )
    final_text = verification_result.text if verification_result else editorial.text
    text = render_citations(final_text, citations) if include_citations else final_text

    artifact = _write_audit(
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
    )
    return FinalizationResult(text=text, citations=citations, audit=artifact)
