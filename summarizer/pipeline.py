"""Orchestrate canonical source ingestion through final editorial output."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from threading import Lock

from summarizer.budget import (
    BudgetError,
    BudgetReport,
    measure_overhead,
    resolve_context_window,
    select_strategy,
    usable_input_capacity,
)
from summarizer.cache import CacheStore
from summarizer.checkpoint import CheckpointStore, RunPlan
from summarizer.config import AppConfig, CacheConfig, ReliabilityConfig, StrategyConfig
from summarizer.direct import summarize_direct, whole_document_segment
from summarizer.finalization import (
    FinalizationResult,
    _finalize_summary,
    publish_final_output,
)
from summarizer.hierarchy import TreeNode, build_hierarchy
from summarizer.ingestion import SourceDocument
from summarizer.leaf import summarize_segments
from summarizer.providers.base import (
    GenerationRequest,
    GenerationResult,
    ModelProvider,
    ProviderRetriesExhaustedError,
)
from summarizer.reliability import ReliabilityTracker
from summarizer.segmentation import (
    CacheCoordinator,
    SegmentationConfig,
    cached_segment_document,
)
from summarizer.tokenization import TokenCounter
from summarizer.verification import VerificationConfig, VerificationRuntime


@dataclass(frozen=True)
class PipelineConfig:
    target_words: int = 300
    segmentation: SegmentationConfig | None = None
    max_merge_children: int | None = None
    include_citations: bool = False
    audit_path: Path | None = None
    verification: VerificationConfig = field(default_factory=VerificationConfig)
    verification_runtime: VerificationRuntime | None = None
    cache: CacheConfig = field(default_factory=CacheConfig)
    reliability: ReliabilityConfig = field(default_factory=ReliabilityConfig)

    def __post_init__(self) -> None:
        if self.target_words <= 0:
            raise ValueError("target_words must be positive")
        if self.verification_runtime is not None and not self.verification.enabled:
            raise ValueError("verification runtime requires enabled verification")


@dataclass(frozen=True)
class PipelineResult:
    final: FinalizationResult
    strategy: BudgetReport
    root: TreeNode
    nodes: tuple[TreeNode, ...]


_DEFAULT_PIPELINE_CONFIG = PipelineConfig()
_DEFAULT_SEGMENT_CAPACITY_DIVISOR = 4


@dataclass(frozen=True)
class _RecordedGeneration:
    work_id: str | None
    sequence: int
    result: GenerationResult


class _RecordingProvider:
    """Capture completed logical calls without exposing request prompts to audit."""

    def __init__(
        self,
        delegate: ModelProvider,
        coordinator: CacheCoordinator | None = None,
        reliability_tracker: ReliabilityTracker | None = None,
    ) -> None:
        self._delegate = delegate
        self.cache_coordinator = coordinator
        self._reliability_tracker = reliability_tracker
        self._generations: list[_RecordedGeneration] = []
        self._next_sequence: dict[str | None, int] = {}
        self._lock = Lock()

    def generate(self, request: GenerationRequest) -> GenerationResult:
        work_id = request.audit_work_id or request.operation_id
        with self._lock:
            sequence = self._next_sequence.get(work_id, 0)
            self._next_sequence[work_id] = sequence + 1
        try:
            result = self._delegate.generate(request)
        except ProviderRetriesExhaustedError as error:
            if self._reliability_tracker is not None:
                self._reliability_tracker.record_retry_exhaustion(
                    request,
                    attempt_count=error.attempts,
                    retry_attempts=error.retry_attempts,
                )
            raise
        with self._lock:
            self._generations.append(_RecordedGeneration(work_id, sequence, result))
        if self._reliability_tracker is not None:
            self._reliability_tracker.record_generation(request, result)
        return result

    @property
    def generations(self) -> tuple[GenerationResult, ...]:
        with self._lock:
            records = tuple(self._generations)
        if self._reliability_tracker is None:
            return tuple(record.result for record in records)
        positions = {
            work_id: index
            for index, work_id in enumerate(
                self._reliability_tracker.manifest_work_order()
            )
        }
        ordered = sorted(
            enumerate(records),
            key=lambda item: (
                positions.get(item[1].work_id, len(positions)),
                item[1].sequence,
                item[0],
            ),
        )
        return tuple(record.result for _, record in ordered)


def _hierarchical_capacity(
    report: BudgetReport,
    counter: TokenCounter,
    app: AppConfig,
    strategy: StrategyConfig,
    segmentation: SegmentationConfig,
) -> int:
    """Re-measure overlap-bearing leaves instead of trusting direct capacity."""
    if segmentation.overlap_tokens == 0:
        return report.usable_input_capacity
    window = resolve_context_window(
        provider=app.provider, model=app.model, explicit=strategy.context_window
    )
    return usable_input_capacity(
        window=window,
        overhead=measure_overhead(counter, with_overlap=True),
        config=strategy,
    )


def run_pipeline(
    document: SourceDocument,
    provider: ModelProvider,
    counter: TokenCounter,
    *,
    app: AppConfig,
    strategy: StrategyConfig,
    config: PipelineConfig = _DEFAULT_PIPELINE_CONFIG,
) -> PipelineResult:
    report = select_strategy(
        document, counter, provider=app.provider, model=app.model, config=strategy
    )
    if not config.cache.enabled:
        return _run_pipeline(
            document,
            provider,
            counter,
            app=app,
            strategy=strategy,
            config=config,
            report=report,
            coordinator=None,
        )
    if not config.reliability.run_id:
        raise ValueError("enabled cache requires a reliability run_id")
    seed = ("D000001",) if report.strategy == "direct" else ("segmentation",)
    effective_segmentation = config.segmentation or SegmentationConfig(
        max_tokens=report.usable_input_capacity
    )
    verification_runtime_descriptor: dict[str, object] | None = None
    if config.verification.enabled:
        runtime = config.verification_runtime
        if runtime is None:
            verification_runtime_descriptor = {
                "provider": app.provider,
                "model": app.model,
                "counter_identity": counter.identity,
                "counter_exact": counter.exact,
                "context_window_tokens": report.context_window_tokens,
                "timeout_seconds": app.timeout_seconds,
            }
        else:
            verification_runtime_descriptor = {
                "provider": runtime.provider_identity,
                "model": runtime.model,
                "counter_identity": runtime.counter.identity,
                "counter_exact": runtime.counter.exact,
                "context_window_tokens": runtime.context_window_tokens,
                "timeout_seconds": runtime.timeout_seconds,
            }
    descriptor = hashlib.sha256(
        json.dumps(
            {
                "app": {
                    "provider": app.provider,
                    "model": app.model,
                    "timeout_seconds": app.timeout_seconds,
                },
                "counter": {"identity": counter.identity, "exact": counter.exact},
                "strategy": asdict(strategy),
                "budget": asdict(report),
                "segmentation": asdict(effective_segmentation),
                "pipeline": {
                    "target_words": config.target_words,
                    "max_merge_children": config.max_merge_children,
                    "include_citations": config.include_citations,
                    "verification": asdict(config.verification),
                    "max_in_flight": config.reliability.max_in_flight,
                },
                "verification_runtime": verification_runtime_descriptor,
            },
            default=str,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    plan = RunPlan(
        run_id=config.reliability.run_id,
        descriptor_sha256=descriptor,
        source_sha256=document.source_id,
        work_ids=seed,
    )
    with CheckpointStore(config.cache.root).open(
        plan, resume=config.reliability.run_mode == "resume"
    ) as session:
        reliability_tracker = ReliabilityTracker(
            lambda: session.manifest.work_ids,
            resumed=config.reliability.run_mode == "resume",
        )
        coordinator = CacheCoordinator(
            store=CacheStore(config.cache.root),
            source_id=document.source_id,
            provider=app.provider,
            model=app.model,
            counter_identity=counter.identity,
            counter_exact=counter.exact,
            context_window_tokens=report.context_window_tokens,
            behavior={
                "strategy_config": {
                    "strategy": strategy.strategy,
                    "context_window": report.context_window_tokens,
                    "max_direct_tokens": strategy.max_direct_tokens,
                    "max_output_tokens": strategy.max_output_tokens,
                    "safety_margin_tokens": strategy.safety_margin_tokens,
                },
                "budget": {
                    "context_window": report.context_window_tokens,
                    "max_output_tokens": strategy.max_output_tokens,
                    "safety_margin_tokens": strategy.safety_margin_tokens,
                    "safety_margin_fraction": strategy.safety_margin_fraction,
                },
            },
            session=session,
            allow_unreferenced_cache=config.reliability.run_mode == "new",
            max_in_flight=config.reliability.max_in_flight,
            reliability_tracker=reliability_tracker,
        )
        return _run_pipeline(
            document,
            provider,
            counter,
            app=app,
            strategy=strategy,
            config=config,
            report=report,
            coordinator=coordinator,
        )


def _run_pipeline(
    document: SourceDocument,
    provider: ModelProvider,
    counter: TokenCounter,
    *,
    app: AppConfig,
    strategy: StrategyConfig,
    config: PipelineConfig,
    report: BudgetReport,
    coordinator: CacheCoordinator | None,
) -> PipelineResult:
    """Execute direct or hierarchical stages, then final editorial writing."""
    reliability_tracker = (
        coordinator.reliability_tracker if coordinator is not None else None
    )
    recording = _RecordingProvider(provider, coordinator, reliability_tracker)
    if report.strategy == "direct":
        segment = whole_document_segment(document, counter)
        summary = summarize_direct(
            document,
            recording,
            counter,
            model=app.model,
            timeout_seconds=app.timeout_seconds,
            coordinator=coordinator,
        )
        root = TreeNode(
            node_id="L0N0001",
            level=0,
            order=0,
            summary=summary,
            children=(),
            covered_segments=(segment.segment_id,),
        )
        nodes = (root,)
        segments = (segment,)
    else:
        requested_segmentation = config.segmentation or SegmentationConfig(
            max_tokens=max(
                1,
                report.usable_input_capacity
                // _DEFAULT_SEGMENT_CAPACITY_DIVISOR,
            )
        )
        capacity = _hierarchical_capacity(
            report, counter, app, strategy, requested_segmentation
        )
        if requested_segmentation.max_tokens > capacity:
            raise BudgetError(
                "segmentation max_tokens exceeds the safely measured leaf capacity"
            )
        segments = tuple(
            cached_segment_document(
                document, counter, requested_segmentation, coordinator=coordinator
            )
        )
        if coordinator is not None and coordinator.session is not None:
            coordinator.session.ensure_work_prefix(
                ("segmentation", *(segment.segment_id for segment in segments))
            )
        leaves = summarize_segments(
            segments,
            recording,
            model=app.model,
            timeout_seconds=app.timeout_seconds,
            coordinator=coordinator,
        )
        root, nodes, _ = build_hierarchy(
            leaves,
            recording,
            counter,
            source_id=document.source_id,
            covered=[(segment.segment_id,) for segment in segments],
            attributable={
                segment.segment_id: document.text[segment.core_start : segment.core_end]
                for segment in segments
            },
            usable_tokens=capacity,
            model=app.model,
            timeout_seconds=app.timeout_seconds,
            max_merge_children=config.max_merge_children,
            coordinator=coordinator,
        )

    completed_before_editorial = tuple(recording.generations)
    if coordinator is not None and coordinator.session is not None:
        coordinator.session.ensure_work_prefix(
            (
                *coordinator.session.manifest.work_ids[
                    : next(
                        (
                            index
                            for index, work_id in enumerate(
                                coordinator.session.manifest.work_ids
                            )
                            if work_id == "editorial-final"
                        ),
                        len(coordinator.session.manifest.work_ids),
                    )
                ],
                "editorial-final",
            )
        )
    verifier_runtime = config.verification_runtime
    if config.verification.enabled and verifier_runtime is None:
        verifier_runtime = VerificationRuntime(
            provider=recording,
            counter=counter,
            model=app.model,
            timeout_seconds=app.timeout_seconds,
            context_window_tokens=report.context_window_tokens,
            provider_identity=app.provider,
        )
    elif verifier_runtime is not None and reliability_tracker is not None:
        verifier_runtime = replace(
            verifier_runtime,
            provider=_RecordingProvider(
                verifier_runtime.provider,
                reliability_tracker=reliability_tracker,
            ),
        )
    verification_coordinator = None
    if config.verification.enabled and coordinator is not None:
        assert verifier_runtime is not None
        verification_coordinator = CacheCoordinator(
            store=coordinator.store,
            source_id=coordinator.source_id,
            provider=verifier_runtime.provider_identity,
            model=verifier_runtime.model,
            counter_identity=verifier_runtime.counter.identity,
            counter_exact=verifier_runtime.counter.exact,
            context_window_tokens=verifier_runtime.context_window_tokens,
            behavior={},
            session=coordinator.session,
            allow_unreferenced_cache=coordinator.allow_unreferenced_cache,
            max_in_flight=coordinator.max_in_flight,
            reliability_tracker=reliability_tracker,
        )
    if (
        config.verification.enabled
        and coordinator is not None
        and coordinator.session is not None
    ):
        coordinator.session.ensure_work_prefix(
            (
                *coordinator.session.manifest.work_ids[
                    : next(
                        (
                            index
                            for index, work_id in enumerate(
                                coordinator.session.manifest.work_ids
                            )
                            if work_id == "V01"
                        ),
                        len(coordinator.session.manifest.work_ids),
                    )
                ],
                "V01",
            )
        )
    final = _finalize_summary(
        root.summary,
        recording,
        source_id=document.source_id,
        model=app.model,
        timeout_seconds=app.timeout_seconds,
        target_words=config.target_words,
        strategy=report.strategy,
        segments=segments,
        nodes=nodes,
        root_node_id=root.node_id,
        include_citations=config.include_citations,
        audit_configuration={
            "app": asdict(app),
            "strategy": asdict(strategy),
            "pipeline": {
                "target_words": config.target_words,
                "max_merge_children": config.max_merge_children,
                "include_citations": config.include_citations,
            },
            "verification": asdict(config.verification),
            "budget": asdict(report),
        },
        audit_path=config.audit_path,
        generations=completed_before_editorial,
        counter=counter,
        source_cores={
            segment.segment_id: document.text[segment.core_start : segment.core_end]
            for segment in segments
        },
        verification=config.verification,
        verification_runtime=verifier_runtime,
        verification_context_window_tokens=report.context_window_tokens,
        verification_coordinator=verification_coordinator,
        reliability_tracker=reliability_tracker,
        materialize_audit=not (
            config.audit_path is not None
            and coordinator is not None
            and coordinator.session is not None
        ),
    )
    if (
        config.audit_path is not None
        and coordinator is not None
        and coordinator.session is not None
    ):
        publish_final_output(
            final,
            summary_path=app.output_path,
            audit_path=config.audit_path,
            session=coordinator.session,
        )
    return PipelineResult(final=final, strategy=report, root=root, nodes=nodes)
