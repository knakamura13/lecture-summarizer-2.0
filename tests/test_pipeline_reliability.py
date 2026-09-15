import json
import re
from concurrent.futures import ThreadPoolExecutor
from threading import Event

import pytest

from summarizer.checkpoint import (
    CheckpointError,
    CheckpointReason,
    CheckpointStore,
    RunPlan,
)
from summarizer.cache import CacheStore
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
from summarizer.grounding import GroundingPolicy
from summarizer.ingestion import ingest_text
from summarizer.pipeline import PipelineConfig, run_pipeline
from summarizer.providers.base import (
    GenerationRequest,
    GenerationResult,
    ProviderTimeoutError,
    RetryAttempt,
    RetryErrorCategory,
)
from summarizer.providers.retrying import RetryingProvider
from summarizer.segmentation import CacheCoordinator, SegmentationConfig
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


class ModelSensitiveLeafProvider(CountingProvider):
    def __init__(self) -> None:
        super().__init__()
        self.leaf_summaries: list[str] = []

    def generate(self, request: GenerationRequest) -> GenerationResult:
        if (request.operation_id or "").startswith(("S", "D")):
            self.requests.append(request)
            summary = f"A grounded leaf for {request.model}."
            self.leaf_summaries.append(summary)
            payload = {
                "summary": summary,
                "content_units": [],
                "entities": [],
                "qualifications": [],
                "contradictions": [],
                "quotations": [],
                "provenance": [request.operation_id],
                "level": 0,
            }
            return GenerationResult(json.dumps(payload), "fake", request.model)
        return super().generate(request)


class EditorialMetadataProvider(CountingProvider):
    def generate(self, request: GenerationRequest) -> GenerationResult:
        if request.operation_id != "editorial-final":
            return super().generate(request)
        self.requests.append(request)
        return GenerationResult(
            text=json.dumps({"text": "A metadata-bearing final draft."}),
            provider="editorial-provider",
            model="editorial-model",
            input_tokens=17,
            output_tokens=9,
            finish_status="stop",
            retry_attempts=(
                RetryAttempt(
                    attempt=1,
                    error_category=RetryErrorCategory.TIMEOUT,
                    planned_delay_seconds=0.1,
                    exhausted=False,
                    recorded_at_seconds=0,
                ),
            ),
        )


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


class RepairingVerifierTimeoutOnceProvider(CountingProvider):
    def __init__(self) -> None:
        super().__init__()
        self._timed_out = False

    def generate(self, request: GenerationRequest) -> GenerationResult:
        operation = request.operation_id or ""
        if not operation.startswith("verification-"):
            if operation == "editorial-final":
                self.requests.append(request)
                return GenerationResult(
                    json.dumps({"text": "42."}), "fake", request.model
                )
            return super().generate(request)
        self.requests.append(request)
        if operation == "verification-decompose:V02" and not self._timed_out:
            self._timed_out = True
            raise ProviderTimeoutError("secret post-repair timeout detail")
        if operation == "verification-decompose:V01":
            payload = {"spans": [{"span_id": "V01S000001", "anchors": ["42"]}]}
        elif operation == "verification-classify:V01":
            payload = {
                "findings": [
                    {
                        "claim_id": claim_id,
                        "verdict": "contradicted",
                        "evidence": [{"segment_id": "D000001", "exact_quote": "41."}],
                    }
                    for claim_id in ("V01C000001", "V01C000002")
                ]
            }
        elif operation == "verification-repair:V01":
            original_hash = re.search(
                r'"original_hash":"([0-9a-f]{64})"', request.input_text
            )
            assert original_hash is not None
            payload = {
                "repairs": [
                    {
                        "span_id": "V01S000001",
                        "original_hash": original_hash.group(1),
                        "action": "replace",
                        "replacement": "41.",
                    }
                ]
            }
        elif operation == "verification-decompose:V02":
            payload = {"spans": [{"span_id": "V02S000001", "anchors": []}]}
        elif operation == "verification-classify:V02":
            payload = {
                "findings": [
                    {
                        "claim_id": "V02C000001",
                        "verdict": "supported",
                        "evidence": [{"segment_id": "D000001", "exact_quote": "41."}],
                    }
                ]
            }
        else:  # pragma: no cover - unexpected requests should stay visible
            raise AssertionError(f"unexpected operation {operation}")
        return GenerationResult(json.dumps(payload), "fake", request.model, 1, 1)


class ReverseLeafCompletionProvider(CountingProvider):
    def __init__(self) -> None:
        super().__init__()
        self.started = {
            work_id: Event() for work_id in ("S000001", "S000002", "S000003")
        }
        self.release = {work_id: Event() for work_id in self.started}

    def generate(self, request: GenerationRequest) -> GenerationResult:
        operation = request.operation_id or ""
        if operation in self.started:
            self.started[operation].set()
            assert self.release[operation].wait(timeout=5)
            self.requests.append(request)
            payload = {
                "summary": "A grounded leaf.",
                "content_units": [],
                "entities": [],
                "qualifications": [],
                "contradictions": [],
                "quotations": [],
                "provenance": [operation],
                "level": 0,
            }
            return GenerationResult(
                json.dumps(payload), "fake", request.model, int(operation[-1]), 1
            )
        result = super().generate(request)
        marker = 200 if operation == "editorial-final" else 100
        return GenerationResult(result.text, result.provider, result.model, marker, 1)


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


def test_editorial_cache_miss_keeps_real_generation_metadata_and_hit_is_synthetic(
    tmp_path,
) -> None:
    cache_root = tmp_path / "cache"
    audit_path = tmp_path / "audit.json"
    document = ingest_text("A short source for editorial metadata.")
    app = AppConfig(
        output_path=tmp_path / "summary.txt",
        model="gpt-4o-mini",
        timeout_seconds=30,
    )
    config = PipelineConfig(
        target_words=40,
        audit_path=audit_path,
        cache=CacheConfig(enabled=True, root=cache_root),
        reliability=ReliabilityConfig(run_id="editorial-metadata"),
    )

    run_pipeline(
        document,
        EditorialMetadataProvider(),
        Counter(),
        app=app,
        strategy=_direct_strategy(),
        config=config,
    )

    first_audit = json.loads(audit_path.read_text())
    assert first_audit["usage"][-1] == {
        "provider": "editorial-provider",
        "model": "editorial-model",
        "input_tokens": 17,
        "output_tokens": 9,
        "finish_status": "stop",
    }
    assert first_audit["reliability"]["attempts"][-1] == {
        "work_id": "editorial-final",
        "attempt_count": 2,
        "failure_reasons": ["timeout"],
    }

    run_pipeline(
        document,
        CountingProvider(),
        Counter(),
        app=app,
        strategy=_direct_strategy(),
        config=PipelineConfig(
            **{
                **config.__dict__,
                "reliability": ReliabilityConfig(
                    run_id="editorial-metadata", run_mode="resume"
                ),
            }
        ),
    )

    resumed_audit = json.loads(audit_path.read_text())
    assert resumed_audit["usage"][-1] == {
        "provider": "cache",
        "model": "gpt-4o-mini",
        "input_tokens": None,
        "output_tokens": None,
        "finish_status": None,
    }


def _run_direct_with_audit(
    *,
    document,
    cache_root,
    audit_path,
    output_path,
    run_id: str,
    app: AppConfig | None = None,
    strategy: StrategyConfig | None = None,
) -> None:
    run_pipeline(
        document,
        CountingProvider(),
        Counter(),
        app=app
        or AppConfig(
            output_path=output_path, model="gpt-4o-mini", timeout_seconds=30
        ),
        strategy=strategy or _direct_strategy(),
        config=PipelineConfig(
            target_words=40,
            audit_path=audit_path,
            cache=CacheConfig(enabled=True, root=cache_root),
            reliability=ReliabilityConfig(run_id=run_id),
        ),
    )


def _invalidation_reasons(audit_path) -> list[str]:
    return json.loads(audit_path.read_text())["reliability"]["cache"][
        "invalidation_reasons"
    ]


def _run_hierarchical_with_audit(
    *,
    document,
    cache_root,
    audit_path,
    output_path,
    run_id: str,
) -> None:
    run_pipeline(
        document,
        CountingProvider(),
        Counter(),
        app=AppConfig(output_path=output_path, model="gpt-4o-mini", timeout_seconds=30),
        strategy=_strategy(),
        config=PipelineConfig(
            target_words=40,
            segmentation=SegmentationConfig(max_tokens=35),
            max_merge_children=2,
            audit_path=audit_path,
            cache=CacheConfig(enabled=True, root=cache_root),
            reliability=ReliabilityConfig(run_id=run_id),
        ),
    )


def test_cache_coordinator_returns_the_locked_first_writer_payload_for_concurrent_runs(
    tmp_path,
) -> None:
    cache_root = tmp_path / "cache"
    document = ingest_text("concurrent source")
    checkpoint_store = CheckpointStore(cache_root)
    descriptor_sha256 = "a" * 64
    started = Event()
    release = Event()

    def decode(payload: object) -> dict[str, str]:
        if not isinstance(payload, dict) or set(payload) != {"summary"}:
            raise ValueError("expected one summary")
        summary = payload["summary"]
        if not isinstance(summary, str) or not summary:
            raise ValueError("summary must be nonblank")
        return {"summary": summary}

    def coordinator(session) -> CacheCoordinator:
        return CacheCoordinator(
            store=CacheStore(cache_root),
            source_id=document.source_id,
            provider="openai",
            model="gpt-4o-mini",
            counter_identity="test:characters",
            counter_exact=True,
            context_window_tokens=100_000,
            behavior={},
            session=session,
        )

    def resolve(instance: CacheCoordinator, compute):
        return instance.resolve(
            stage="direct",
            work_id="D000001",
            prompt_version="leaf-prompt/1",
            schema_version="leaf-schema/1",
            input_value={"source_id": document.source_id},
            behavior={"max_output_tokens": 1},
            decode=decode,
            encode=lambda payload: payload,
            compute=compute,
        )

    def delayed_loser() -> dict[str, str]:
        started.set()
        assert release.wait(timeout=2)
        return {"summary": "loser"}

    first_plan = RunPlan(
        run_id="cache-race-first",
        descriptor_sha256=descriptor_sha256,
        source_sha256=document.source_id,
        work_ids=("D000001",),
    )
    second_plan = RunPlan(
        run_id="cache-race-second",
        descriptor_sha256=descriptor_sha256,
        source_sha256=document.source_id,
        work_ids=("D000001",),
    )
    with checkpoint_store.open(first_plan, resume=False) as first_session:
        with checkpoint_store.open(second_plan, resume=False) as second_session:
            first = coordinator(first_session)
            second = coordinator(second_session)
            descriptor = first.descriptor_for(
                stage="direct",
                work_id="D000001",
                prompt_version="leaf-prompt/1",
                schema_version="leaf-schema/1",
                input_value={"source_id": document.source_id},
                behavior={"max_output_tokens": 1},
            )
            with ThreadPoolExecutor(max_workers=1) as executor:
                first_result = executor.submit(resolve, first, delayed_loser)
                assert started.wait(timeout=2)
                second_result = resolve(second, lambda: {"summary": "winner"})
                release.set()
                first_result_value = first_result.result(timeout=2)

            winner = CacheStore(cache_root).load(descriptor, decode).payload
            assert winner == {"summary": "winner"}
            assert first_result_value == winner
            assert second_result == winner
            for session in (first_session, second_session):
                assert session.reusable_for(
                    work_ids=("D000001",),
                    descriptors={"D000001": descriptor},
                    validators={"D000001": decode},
                )[0].payload == winner


def test_pipeline_audit_reports_source_descriptor_invalidation(tmp_path) -> None:
    cache_root = tmp_path / "cache"
    output_path = tmp_path / "summary.txt"
    _run_direct_with_audit(
        document=ingest_text("first source"),
        cache_root=cache_root,
        audit_path=tmp_path / "first-audit.json",
        output_path=output_path,
        run_id="source-before",
    )
    changed_audit = tmp_path / "changed-audit.json"

    _run_direct_with_audit(
        document=ingest_text("second source"),
        cache_root=cache_root,
        audit_path=changed_audit,
        output_path=output_path,
        run_id="source-after",
    )

    assert _invalidation_reasons(changed_audit) == ["source_changed"]


def test_pipeline_audit_reports_prompt_descriptor_invalidation(tmp_path, monkeypatch) -> None:
    cache_root = tmp_path / "cache"
    output_path = tmp_path / "summary.txt"
    document = ingest_text("stable source")
    _run_direct_with_audit(
        document=document,
        cache_root=cache_root,
        audit_path=tmp_path / "first-audit.json",
        output_path=output_path,
        run_id="prompt-before",
    )
    monkeypatch.setattr("summarizer.direct.LEAF_PROMPT_VERSION", "leaf-prompt/4")
    changed_audit = tmp_path / "changed-audit.json"

    _run_direct_with_audit(
        document=document,
        cache_root=cache_root,
        audit_path=changed_audit,
        output_path=output_path,
        run_id="prompt-after",
    )

    assert _invalidation_reasons(changed_audit) == ["prompt_changed"]


def test_pipeline_audit_reports_schema_descriptor_invalidation(tmp_path, monkeypatch) -> None:
    cache_root = tmp_path / "cache"
    output_path = tmp_path / "summary.txt"
    document = ingest_text("stable source")
    _run_direct_with_audit(
        document=document,
        cache_root=cache_root,
        audit_path=tmp_path / "first-audit.json",
        output_path=output_path,
        run_id="schema-before",
    )
    monkeypatch.setattr("summarizer.direct.LEAF_SCHEMA_VERSION", "leaf-schema/2")
    changed_audit = tmp_path / "changed-audit.json"

    _run_direct_with_audit(
        document=document,
        cache_root=cache_root,
        audit_path=changed_audit,
        output_path=output_path,
        run_id="schema-after",
    )

    assert _invalidation_reasons(changed_audit) == ["schema_changed"]


def test_pipeline_audit_reports_model_descriptor_invalidation(tmp_path) -> None:
    cache_root = tmp_path / "cache"
    output_path = tmp_path / "summary.txt"
    document = ingest_text("stable source")
    _run_direct_with_audit(
        document=document,
        cache_root=cache_root,
        audit_path=tmp_path / "first-audit.json",
        output_path=output_path,
        run_id="model-before",
    )
    changed_audit = tmp_path / "changed-audit.json"

    _run_direct_with_audit(
        document=document,
        cache_root=cache_root,
        audit_path=changed_audit,
        output_path=output_path,
        run_id="model-after",
        app=AppConfig(output_path=output_path, model="gpt-4o", timeout_seconds=30),
    )

    assert _invalidation_reasons(changed_audit) == ["model_changed"]


def test_pipeline_audit_reports_behavior_descriptor_invalidation(tmp_path) -> None:
    cache_root = tmp_path / "cache"
    output_path = tmp_path / "summary.txt"
    document = ingest_text("stable source")
    _run_direct_with_audit(
        document=document,
        cache_root=cache_root,
        audit_path=tmp_path / "first-audit.json",
        output_path=output_path,
        run_id="behavior-before",
    )
    changed_audit = tmp_path / "changed-audit.json"

    _run_direct_with_audit(
        document=document,
        cache_root=cache_root,
        audit_path=changed_audit,
        output_path=output_path,
        run_id="behavior-after",
        strategy=StrategyConfig(
            strategy="direct",
            context_window=100_000,
            max_output_tokens=2,
            safety_margin_tokens=0,
            safety_margin_fraction=0,
        ),
    )

    assert _invalidation_reasons(changed_audit) == ["behavior_changed"]


def test_hierarchical_leaf_batch_invalidation_uses_the_successful_next_baseline(
    tmp_path, monkeypatch
) -> None:
    cache_root = tmp_path / "cache"
    output_path = tmp_path / "summary.txt"
    document = ingest_text("one two three four five six seven eight nine ten " * 12)
    _run_hierarchical_with_audit(
        document=document,
        cache_root=cache_root,
        audit_path=tmp_path / "initial-audit.json",
        output_path=output_path,
        run_id="hierarchical-initial",
    )
    monkeypatch.setattr("summarizer.leaf.LEAF_PROMPT_VERSION", "leaf-prompt/4")
    prompt_audit = tmp_path / "prompt-audit.json"

    _run_hierarchical_with_audit(
        document=document,
        cache_root=cache_root,
        audit_path=prompt_audit,
        output_path=output_path,
        run_id="hierarchical-prompt",
    )

    assert _invalidation_reasons(prompt_audit) == ["prompt_changed"]

    monkeypatch.setattr("summarizer.leaf.LEAF_SCHEMA_VERSION", "leaf-schema/2")
    schema_audit = tmp_path / "schema-audit.json"
    _run_hierarchical_with_audit(
        document=document,
        cache_root=cache_root,
        audit_path=schema_audit,
        output_path=output_path,
        run_id="hierarchical-schema",
    )

    assert _invalidation_reasons(schema_audit) == ["schema_changed"]


def test_hierarchical_model_change_with_changed_leaf_output_reports_only_model_invalidation(
    tmp_path,
) -> None:
    cache_root = tmp_path / "cache"
    output_path = tmp_path / "summary.txt"
    document = ingest_text("one two three four five six seven eight nine ten " * 12)
    initial_provider = ModelSensitiveLeafProvider()
    run_pipeline(
        document,
        initial_provider,
        Counter(),
        app=AppConfig(output_path=output_path, model="gpt-4o-mini", timeout_seconds=30),
        strategy=_strategy(),
        config=PipelineConfig(
            target_words=40,
            segmentation=SegmentationConfig(max_tokens=35),
            max_merge_children=2,
            audit_path=tmp_path / "initial-audit.json",
            cache=CacheConfig(enabled=True, root=cache_root),
            reliability=ReliabilityConfig(run_id="hierarchical-model-initial"),
        ),
    )
    changed_provider = ModelSensitiveLeafProvider()
    changed_audit = tmp_path / "changed-audit.json"

    run_pipeline(
        document,
        changed_provider,
        Counter(),
        app=AppConfig(output_path=output_path, model="gpt-4o", timeout_seconds=30),
        strategy=_strategy(),
        config=PipelineConfig(
            target_words=40,
            segmentation=SegmentationConfig(max_tokens=35),
            max_merge_children=2,
            audit_path=changed_audit,
            cache=CacheConfig(enabled=True, root=cache_root),
            reliability=ReliabilityConfig(run_id="hierarchical-model-changed"),
        ),
    )

    assert initial_provider.leaf_summaries
    assert changed_provider.leaf_summaries
    assert initial_provider.leaf_summaries != changed_provider.leaf_summaries
    assert _invalidation_reasons(changed_audit) == ["model_changed"]


def test_hierarchical_grounding_budget_change_reports_behavior_invalidation(
    tmp_path, monkeypatch
) -> None:
    cache_root = tmp_path / "cache"
    output_path = tmp_path / "summary.txt"
    document = ingest_text("one two three four five six seven eight nine ten " * 12)
    _run_hierarchical_with_audit(
        document=document,
        cache_root=cache_root,
        audit_path=tmp_path / "initial-audit.json",
        output_path=output_path,
        run_id="hierarchical-grounding-initial",
    )
    monkeypatch.setattr(
        "summarizer.hierarchy.DEFAULT_GROUNDING_POLICY",
        GroundingPolicy(max_tokens=512),
    )
    changed_audit = tmp_path / "changed-audit.json"

    _run_hierarchical_with_audit(
        document=document,
        cache_root=cache_root,
        audit_path=changed_audit,
        output_path=output_path,
        run_id="hierarchical-grounding-changed",
    )

    assert _invalidation_reasons(changed_audit) == ["behavior_changed"]


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


def test_reverification_retries_share_the_manifest_verification_work_id(
    tmp_path,
) -> None:
    audit_path = tmp_path / "audit.json"
    delegate = RepairingVerifierTimeoutOnceProvider()
    provider = RetryingProvider(
        delegate,
        RetryPolicy(
            max_attempts=2,
            initial_delay_seconds=0.001,
            max_delay_seconds=0.001,
            jitter_fraction=0,
        ),
        sleeper=lambda _: None,
    )

    result = run_pipeline(
        ingest_text("The source confirms the value is 41."),
        provider,
        Counter(),
        app=AppConfig(
            output_path=tmp_path / "summary.txt",
            model="gpt-4o-mini",
            timeout_seconds=30,
        ),
        strategy=_direct_strategy(),
        config=PipelineConfig(
            target_words=40,
            audit_path=audit_path,
            verification=VerificationConfig(enabled=True),
            cache=CacheConfig(enabled=True, root=tmp_path / "cache"),
            reliability=ReliabilityConfig(run_id="reverification-retry"),
        ),
    )

    verification_requests = [
        request
        for request in delegate.requests
        if (request.operation_id or "").startswith("verification-")
    ]
    assert result.final.text == "41."
    assert {request.audit_work_id for request in verification_requests} == {"V01"}
    assert [request.operation_id for request in verification_requests] == [
        "verification-decompose:V01",
        "verification-classify:V01",
        "verification-repair:V01",
        "verification-decompose:V02",
        "verification-decompose:V02",
        "verification-classify:V02",
    ]
    encoded_audit = audit_path.read_text()
    audit = json.loads(encoded_audit)
    assert [
        attempt
        for attempt in audit["reliability"]["attempts"]
        if attempt["work_id"] == "V01"
    ] == [
        {
            "work_id": "V01",
            "attempt_count": 6,
            "failure_reasons": ["timeout"],
        }
    ]
    assert "secret post-repair timeout detail" not in encoded_audit


def test_concurrent_generation_usage_follows_manifest_order_across_runs(
    tmp_path,
) -> None:
    observed_usage = []
    for run_index in range(2):
        provider = ReverseLeafCompletionProvider()
        audit_path = tmp_path / f"audit-{run_index}.json"
        config = PipelineConfig(
            target_words=40,
            segmentation=SegmentationConfig(max_tokens=35),
            max_merge_children=3,
            audit_path=audit_path,
            cache=CacheConfig(enabled=True, root=tmp_path / f"cache-{run_index}"),
            reliability=ReliabilityConfig(
                run_id=f"reverse-completion-{run_index}", max_in_flight=2
            ),
        )
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(
                run_pipeline,
                ingest_text("one two three four five six seven eight nine ten " * 2),
                provider,
                Counter(),
                app=AppConfig(
                    output_path=tmp_path / f"summary-{run_index}.txt",
                    model="gpt-4o-mini",
                    timeout_seconds=30,
                ),
                strategy=_strategy(),
                config=config,
            )
            assert provider.started["S000001"].wait(timeout=5)
            assert provider.started["S000002"].wait(timeout=5)
            provider.release["S000002"].set()
            assert provider.started["S000003"].wait(timeout=5)
            provider.release["S000001"].set()
            provider.release["S000003"].set()
            future.result(timeout=10)

        manifest = json.loads(
            (
                tmp_path
                / f"cache-{run_index}"
                / "runs"
                / f"reverse-completion-{run_index}.json"
            ).read_text()
        )
        assert manifest["work_ids"][:4] == [
            "segmentation",
            "S000001",
            "S000002",
            "S000003",
        ]
        usage = [
            item["input_tokens"] for item in json.loads(audit_path.read_text())["usage"]
        ]
        observed_usage.append(usage)

    assert observed_usage == [[1, 2, 3, 100, 200], [1, 2, 3, 100, 200]]


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
