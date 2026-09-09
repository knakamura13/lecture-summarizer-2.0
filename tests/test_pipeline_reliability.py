import json
import re

import pytest

from summarizer.checkpoint import (
    CheckpointError,
    CheckpointReason,
    CheckpointStore,
    RunPlan,
)
from summarizer.config import (
    AppConfig,
    CacheConfig,
    ReliabilityConfig,
    RetryPolicy,
    StrategyConfig,
)
from summarizer.finalization import (
    FinalizationVerificationError,
    read_published_summary,
)
from summarizer.ingestion import ingest_text
from summarizer.pipeline import PipelineConfig, run_pipeline
from summarizer.providers.base import (
    GenerationRequest,
    GenerationResult,
    ProviderTimeoutError,
)
from summarizer.providers.retrying import RetryingProvider
from summarizer.segmentation import SegmentationConfig
from summarizer.verification import VerificationConfig, VerificationRuntime


class Counter:
    identity = "test:characters"
    exact = True
    monotonic = True

    def count(self, text: str) -> int:
        return len(text)


class CountingProvider:
    def __init__(self) -> None:
        self.requests: list[GenerationRequest] = []

    def generate(self, request: GenerationRequest) -> GenerationResult:
        self.requests.append(request)
        if request.operation_id == "editorial-final":
            payload = {"text": "A cached final draft."}
        elif (request.operation_id or "").startswith(("S", "D")):
            payload = {
                "summary": "A grounded leaf.",
                "content_units": [],
                "entities": [],
                "qualifications": [],
                "contradictions": [],
                "quotations": [],
                "provenance": [request.operation_id],
                "level": 0,
            }
        else:
            payload = {
                "summary": "A grounded merge.",
                "content_units": [],
                "entities": [],
                "qualifications": [],
                "contradictions": [],
                "quotations": [],
                "provenance": [
                    re.findall(r'"segment_id":"(S\d+)"', request.input_text)[-1]
                ],
                "level": int((request.operation_id or "merge-L1").rsplit("L", 1)[1]),
            }
        return GenerationResult(json.dumps(payload), "fake", request.model)


def _app() -> AppConfig:
    return AppConfig(model="gpt-4o-mini", timeout_seconds=30)


def _strategy() -> StrategyConfig:
    return StrategyConfig(
        strategy="hierarchical",
        context_window=100_000,
        max_output_tokens=1,
        safety_margin_tokens=0,
        safety_margin_fraction=0,
    )


def _direct_strategy() -> StrategyConfig:
    return StrategyConfig(
        strategy="direct",
        context_window=100_000,
        max_output_tokens=1,
        safety_margin_tokens=0,
        safety_margin_fraction=0,
    )


class TimeoutOnceProvider(CountingProvider):
    def __init__(self) -> None:
        super().__init__()
        self._timed_out = False

    def generate(self, request: GenerationRequest) -> GenerationResult:
        if request.operation_id == "D000001" and not self._timed_out:
            self._timed_out = True
            raise ProviderTimeoutError("sensitive provider detail")
        return super().generate(request)


class MergeTimeoutOnceProvider(CountingProvider):
    def __init__(self) -> None:
        super().__init__()
        self._timed_out = False

    def generate(self, request: GenerationRequest) -> GenerationResult:
        if (request.audit_work_id or "").startswith("L") and not self._timed_out:
            self._timed_out = True
            raise ProviderTimeoutError("merge timeout detail")
        return super().generate(request)


class MalformedVerificationProvider(CountingProvider):
    def generate(self, request: GenerationRequest) -> GenerationResult:
        if (request.operation_id or "").startswith("verification-"):
            self.requests.append(request)
            return GenerationResult("not json", "fake", request.model)
        return super().generate(request)


class AlwaysTimeoutProvider:
    def generate(self, request: GenerationRequest) -> GenerationResult:
        raise ProviderTimeoutError("credential-like secret detail")


def test_published_audit_records_real_cache_outcomes_reuse_and_retry(tmp_path) -> None:
    document = ingest_text("A short source for the direct pipeline.")
    cache_root = tmp_path / "cache"
    audit_path = tmp_path / "audit.json"
    app = AppConfig(
        output_path=tmp_path / "summary.txt",
        model="gpt-4o-mini",
        timeout_seconds=30,
    )
    config = PipelineConfig(
        target_words=40,
        audit_path=audit_path,
        cache=CacheConfig(enabled=True, root=cache_root),
        reliability=ReliabilityConfig(run_id="observed-run"),
    )
    provider = RetryingProvider(
        TimeoutOnceProvider(),
        RetryPolicy(
            max_attempts=2,
            initial_delay_seconds=0.001,
            max_delay_seconds=0.001,
            jitter_fraction=0,
        ),
        sleeper=lambda _: None,
    )

    run_pipeline(
        document,
        provider,
        Counter(),
        app=app,
        strategy=_direct_strategy(),
        config=config,
    )

    first_audit = json.loads(audit_path.read_text())
    assert first_audit["schema_version"] == "audit/3"
    assert first_audit["reliability"]["cache"] == {
        "cache_hits": [],
        "cache_misses": ["missing", "missing"],
        "invalidation_reasons": [],
    }
    assert first_audit["reliability"]["attempts"] == [
        {
            "work_id": "D000001",
            "attempt_count": 2,
            "failure_reasons": ["timeout"],
        },
        {
            "work_id": "editorial-final",
            "attempt_count": 1,
            "failure_reasons": [],
        },
    ]
    assert "sensitive provider detail" not in audit_path.read_text()

    resumed_provider = CountingProvider()
    run_pipeline(
        document,
        resumed_provider,
        Counter(),
        app=app,
        strategy=_direct_strategy(),
        config=PipelineConfig(
            **{
                **config.__dict__,
                "reliability": ReliabilityConfig(
                    run_id="observed-run", run_mode="resume"
                ),
            }
        ),
    )

    resumed_audit = json.loads(audit_path.read_text())
    assert resumed_provider.requests == []
    assert resumed_audit["reliability"]["cache"] == {
        "cache_hits": ["hit", "hit"],
        "cache_misses": [],
        "invalidation_reasons": [],
    }
    assert resumed_audit["reliability"]["resumed"] is True
    assert resumed_audit["reliability"]["reused_count"] == 2
    assert resumed_audit["reliability"]["recomputed_count"] == 0
    assert resumed_audit["reliability"]["attempts"] == []


def test_concurrent_retry_audit_follows_manifest_work_order(tmp_path) -> None:
    cache_root = tmp_path / "cache"
    audit_path = tmp_path / "audit.json"
    provider = RetryingProvider(
        MergeTimeoutOnceProvider(),
        RetryPolicy(
            max_attempts=2,
            initial_delay_seconds=0.001,
            max_delay_seconds=0.001,
            jitter_fraction=0,
        ),
        sleeper=lambda _: None,
    )
    run_pipeline(
        ingest_text("one two three four five six seven eight nine ten " * 12),
        provider,
        Counter(),
        app=AppConfig(
            output_path=tmp_path / "summary.txt",
            model="gpt-4o-mini",
            timeout_seconds=30,
        ),
        strategy=_strategy(),
        config=PipelineConfig(
            target_words=40,
            segmentation=SegmentationConfig(max_tokens=35),
            max_merge_children=2,
            audit_path=audit_path,
            cache=CacheConfig(enabled=True, root=cache_root),
            reliability=ReliabilityConfig(
                run_id="concurrent-observed-run", max_in_flight=4
            ),
        ),
    )

    audit = json.loads(audit_path.read_text())
    manifest = json.loads(
        (cache_root / "runs" / "concurrent-observed-run.json").read_text()
    )
    attempts = audit["reliability"]["attempts"]
    attempted_ids = [attempt["work_id"] for attempt in attempts]
    assert attempted_ids == [
        work_id
        for work_id in manifest["work_ids"]
        if work_id not in {"segmentation", "V01"}
    ]
    retried = [attempt for attempt in attempts if attempt["attempt_count"] == 2]
    assert len(retried) == 1
    assert retried[0]["work_id"].startswith("L")
    assert retried[0]["failure_reasons"] == ["timeout"]


def test_failed_verification_audit_keeps_observability_safe_and_no_summary(
    tmp_path,
) -> None:
    audit_path = tmp_path / "audit.json"
    summary_path = tmp_path / "summary.txt"

    with pytest.raises(FinalizationVerificationError):
        run_pipeline(
            ingest_text("The source confirms the value is 41."),
            MalformedVerificationProvider(),
            Counter(),
            app=AppConfig(
                output_path=summary_path,
                model="gpt-4o-mini",
                timeout_seconds=30,
            ),
            strategy=_direct_strategy(),
            config=PipelineConfig(
                target_words=40,
                audit_path=audit_path,
                verification=VerificationConfig(enabled=True),
                cache=CacheConfig(enabled=True, root=tmp_path / "cache"),
                reliability=ReliabilityConfig(run_id="failed-observed-run"),
            ),
        )

    encoded_audit = audit_path.read_text()
    audit = json.loads(encoded_audit)
    assert audit["schema_version"] == "audit/3"
    assert audit["verification"]["failed"] is True
    assert audit["citations"] == []
    assert not summary_path.exists()
    assert [item["work_id"] for item in audit["reliability"]["attempts"]] == [
        "D000001",
        "editorial-final",
        "V01",
    ]
    assert "not json" not in encoded_audit


def test_exhausted_injected_verifier_attempts_are_audited_without_detail(
    tmp_path,
) -> None:
    audit_path = tmp_path / "audit.json"
    summary_path = tmp_path / "summary.txt"
    verifier = RetryingProvider(
        AlwaysTimeoutProvider(),
        RetryPolicy(
            max_attempts=2,
            initial_delay_seconds=0.001,
            max_delay_seconds=0.001,
            jitter_fraction=0,
        ),
        sleeper=lambda _: None,
    )

    with pytest.raises(FinalizationVerificationError):
        run_pipeline(
            ingest_text("The source confirms the value is 41."),
            CountingProvider(),
            Counter(),
            app=AppConfig(
                output_path=summary_path,
                model="gpt-4o-mini",
                timeout_seconds=30,
            ),
            strategy=_direct_strategy(),
            config=PipelineConfig(
                target_words=40,
                audit_path=audit_path,
                verification=VerificationConfig(enabled=True),
                verification_runtime=VerificationRuntime(
                    provider=verifier,
                    counter=Counter(),
                    model="gpt-4o-mini",
                    timeout_seconds=30,
                    context_window_tokens=100_000,
                ),
                cache=CacheConfig(enabled=True, root=tmp_path / "cache"),
                reliability=ReliabilityConfig(run_id="exhausted-verifier-run"),
            ),
        )

    encoded_audit = audit_path.read_text()
    audit = json.loads(encoded_audit)
    verifier_attempt = next(
        item
        for item in audit["reliability"]["attempts"]
        if item["work_id"] == "V01"
    )
    assert verifier_attempt == {
        "work_id": "V01",
        "attempt_count": 2,
        "failure_reasons": ["timeout"],
    }
    assert audit["verification"]["failed"] is True
    assert audit["citations"] == []
    assert not summary_path.exists()
    assert "credential-like secret detail" not in encoded_audit


def test_compatible_hierarchical_pipeline_reuses_segment_leaf_merge_and_editorial(
    tmp_path,
) -> None:
    document = ingest_text("one two three four five six seven eight nine ten " * 12)
    config = PipelineConfig(
        target_words=40,
        segmentation=SegmentationConfig(max_tokens=35),
        max_merge_children=2,
        cache=CacheConfig(enabled=True, root=tmp_path / "cache"),
        reliability=ReliabilityConfig(run_id="reliable-run"),
    )
    first = CountingProvider()
    run_pipeline(
        document, first, Counter(), app=_app(), strategy=_strategy(), config=config
    )

    second = CountingProvider()
    result = run_pipeline(
        document,
        second,
        Counter(),
        app=_app(),
        strategy=_strategy(),
        config=PipelineConfig(
            **{
                **config.__dict__,
                "reliability": ReliabilityConfig(
                    run_id="reliable-run", run_mode="resume"
                ),
            }
        ),
    )

    assert result.final.text == "A cached final draft."
    assert second.requests == []


def test_new_run_reuses_compatible_global_leaf_and_merge_objects(tmp_path) -> None:
    document = ingest_text("one two three four five six seven eight nine ten " * 12)
    cache_root = tmp_path / "cache"
    config = PipelineConfig(
        target_words=40,
        segmentation=SegmentationConfig(max_tokens=35),
        max_merge_children=2,
        cache=CacheConfig(enabled=True, root=cache_root),
        reliability=ReliabilityConfig(run_id="global-source"),
    )
    run_pipeline(
        document,
        CountingProvider(),
        Counter(),
        app=_app(),
        strategy=_strategy(),
        config=config,
    )
    source_manifest = json.loads(
        (cache_root / "runs" / "global-source.json").read_text()
    )
    expected_completed = source_manifest["completed"]

    reused = CountingProvider()
    result = run_pipeline(
        document,
        reused,
        Counter(),
        app=_app(),
        strategy=_strategy(),
        config=PipelineConfig(
            **{
                **config.__dict__,
                "reliability": ReliabilityConfig(run_id="global-consumer"),
            }
        ),
    )

    manifest = json.loads((cache_root / "runs" / "global-consumer.json").read_text())
    assert result.final.text == "A cached final draft."
    assert reused.requests == []
    assert manifest["completed"] == expected_completed


def test_resume_rejects_an_overlap_only_segmentation_change_before_provider_calls(
    tmp_path,
) -> None:
    document = ingest_text("one two three four five six seven eight nine ten " * 12)
    cache_root = tmp_path / "cache"
    config = PipelineConfig(
        target_words=40,
        segmentation=SegmentationConfig(max_tokens=35, overlap_tokens=0),
        max_merge_children=2,
        cache=CacheConfig(enabled=True, root=cache_root),
        reliability=ReliabilityConfig(run_id="overlap-change"),
    )
    run_pipeline(
        document,
        CountingProvider(),
        Counter(),
        app=_app(),
        strategy=_strategy(),
        config=config,
    )

    resumed = CountingProvider()
    with pytest.raises(CheckpointError) as raised:
        run_pipeline(
            document,
            resumed,
            Counter(),
            app=_app(),
            strategy=_strategy(),
            config=PipelineConfig(
                **{
                    **config.__dict__,
                    "segmentation": SegmentationConfig(max_tokens=35, overlap_tokens=1),
                    "reliability": ReliabilityConfig(
                        run_id="overlap-change", run_mode="resume"
                    ),
                }
            ),
        )

    assert raised.value.reason is CheckpointReason.INCOMPATIBLE
    assert resumed.requests == []


def test_resume_never_uses_unreferenced_global_leaf_or_merge_objects(tmp_path) -> None:
    document = ingest_text("one two three four five six seven eight nine ten " * 12)
    cache_root = tmp_path / "cache"
    config = PipelineConfig(
        target_words=40,
        segmentation=SegmentationConfig(max_tokens=35),
        max_merge_children=2,
        cache=CacheConfig(enabled=True, root=cache_root),
        reliability=ReliabilityConfig(run_id="global-source"),
    )
    run_pipeline(
        document,
        CountingProvider(),
        Counter(),
        app=_app(),
        strategy=_strategy(),
        config=config,
    )

    empty_run = PipelineConfig(
        **{
            **config.__dict__,
            "reliability": ReliabilityConfig(run_id="resume-no-global"),
        }
    )
    run_pipeline(
        document,
        CountingProvider(),
        Counter(),
        app=_app(),
        strategy=_strategy(),
        config=empty_run,
    )
    manifest_path = cache_root / "runs" / "resume-no-global.json"
    manifest = json.loads(manifest_path.read_text())
    plan = RunPlan(
        run_id="resume-no-global",
        descriptor_sha256=manifest["descriptor_sha256"],
        source_sha256=document.source_id,
        work_ids=("segmentation",),
    )
    with CheckpointStore(cache_root).open(plan, resume=True) as session:
        session.checkpoint(completed=(), descriptors={})

    resumed = CountingProvider()
    run_pipeline(
        document,
        resumed,
        Counter(),
        app=_app(),
        strategy=_strategy(),
        config=PipelineConfig(
            **{
                **empty_run.__dict__,
                "reliability": ReliabilityConfig(
                    run_id="resume-no-global", run_mode="resume"
                ),
            }
        ),
    )

    operation_ids = [request.operation_id or "" for request in resumed.requests]
    assert any(operation_id.startswith("S") for operation_id in operation_ids)
    assert any(operation_id.startswith("merge-L") for operation_id in operation_ids)


def test_pipeline_publishes_witnessed_pair_and_resume_repairs_tampering(
    tmp_path,
) -> None:
    document = ingest_text("one two three four five six seven eight nine ten " * 12)
    cache_root = tmp_path / "cache"
    audit_path = tmp_path / "audit.json"
    summary_path = tmp_path / "summary.txt"
    app = AppConfig(output_path=summary_path, model="gpt-4o-mini", timeout_seconds=30)
    config = PipelineConfig(
        target_words=40,
        segmentation=SegmentationConfig(max_tokens=35),
        max_merge_children=2,
        audit_path=audit_path,
        cache=CacheConfig(enabled=True, root=cache_root),
        reliability=ReliabilityConfig(run_id="published-run"),
    )
    run_pipeline(
        document,
        CountingProvider(),
        Counter(),
        app=app,
        strategy=_strategy(),
        config=config,
    )
    plan_manifest = json.loads((cache_root / "runs" / "published-run.json").read_text())
    plan = RunPlan(
        run_id="published-run",
        descriptor_sha256=plan_manifest["descriptor_sha256"],
        source_sha256=document.source_id,
        work_ids=("segmentation",),
    )
    with CheckpointStore(cache_root).open(plan, resume=True) as session:
        assert (
            read_published_summary(summary_path, audit_path, session.manifest)
            == "A cached final draft."
        )

    summary_path.write_text("tampered", encoding="utf-8")
    resumed_provider = CountingProvider()
    run_pipeline(
        document,
        resumed_provider,
        Counter(),
        app=app,
        strategy=_strategy(),
        config=PipelineConfig(
            **{
                **config.__dict__,
                "reliability": ReliabilityConfig(
                    run_id="published-run", run_mode="resume"
                ),
            }
        ),
    )
    assert resumed_provider.requests == []
    final_manifest = json.loads(
        (cache_root / "runs" / "published-run.json").read_text()
    )
    assert final_manifest["publication"] == "complete"
    assert summary_path.read_text() == "A cached final draft."


def test_pipeline_without_audit_path_returns_text_without_publishing(
    tmp_path,
) -> None:
    document = ingest_text("one two three four five six seven eight nine ten " * 12)
    summary_path = tmp_path / "summary.txt"
    result = run_pipeline(
        document,
        CountingProvider(),
        Counter(),
        app=AppConfig(
            output_path=summary_path, model="gpt-4o-mini", timeout_seconds=30
        ),
        strategy=_strategy(),
        config=PipelineConfig(
            target_words=40,
            segmentation=SegmentationConfig(max_tokens=35),
            max_merge_children=2,
            cache=CacheConfig(enabled=True, root=tmp_path / "cache"),
            reliability=ReliabilityConfig(run_id="no-audit"),
        ),
    )
    assert result.final.text == "A cached final draft."
    assert result.final.audit is None
    assert not summary_path.exists()
