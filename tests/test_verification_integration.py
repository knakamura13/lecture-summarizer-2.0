import json
import re

import pytest

from summarizer.cache import CacheStore
from summarizer.checkpoint import (
    CheckpointError,
    CheckpointReason,
    CheckpointStore,
    RunPlan,
)
from summarizer.config import AppConfig, CacheConfig, ReliabilityConfig, StrategyConfig
from summarizer.finalization import FinalizationVerificationError
from summarizer.ingestion import ingest_text
from summarizer.pipeline import PipelineConfig, run_pipeline
from summarizer.providers.base import GenerationRequest, GenerationResult
from summarizer.segmentation import CacheCoordinator, SegmentationConfig
from summarizer.verification import (
    SourceLexicalEntry,
    SourceLexicalIndex,
    VerificationConfig,
    VerificationRuntime,
    build_source_lexical_index,
    verify_and_repair,
)


class CharacterCounter:
    identity = "test:characters"
    exact = True
    monotonic = True

    def count(self, text: str) -> int:
        return len(text)


class VerifierCounter(CharacterCounter):
    identity = "test:verifier-characters"
    exact = False


class VerificationPipelineProvider:
    """Network-free complete pipeline provider with deterministic verifier replies."""

    def __init__(self, *, verification: str) -> None:
        self.requests: list[GenerationRequest] = []
        self.verification = verification

    def generate(self, request: GenerationRequest) -> GenerationResult:
        self.requests.append(request)
        operation = request.operation_id or ""
        if operation == "editorial-final":
            response = {"text": "42."}
        elif operation == "D000001" or operation.startswith("S"):
            response = self._node(0, operation or "S000001")
        elif operation.startswith("merge-"):
            level = int(operation.rsplit("L", 1)[1])
            identifiers = re.findall(r'"segment_id":"([SD]\d+)"', request.input_text)
            response = self._node(level, identifiers[-1])
        elif operation == "verification-decompose:V01":
            response = self._decomposition("V01")
        elif operation == "verification-classify:V01":
            response = self._classification("V01", request.input_text)
        elif operation == "verification-repair:V01":
            original_hash = re.search(r'"original_hash":"([0-9a-f]{64})"', request.input_text)
            assert original_hash is not None
            response = {
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
            response = {"spans": [{"span_id": "V02S000001", "anchors": []}]}
        elif operation == "verification-classify:V02":
            response = self._findings("V02", "supported", request.input_text)
        else:  # pragma: no cover - makes unanticipated pipeline calls visible
            raise AssertionError(f"unexpected operation {operation}")
        if self.verification == "malformed" and operation.startswith("verification-"):
            return GenerationResult("not-json", "fake", request.model, 1, 1, "completed")
        return GenerationResult(json.dumps(response), "fake", request.model, 1, 1, "completed")

    def _decomposition(self, prefix: str) -> dict[str, object]:
        if self.verification == "supported":
            return {"spans": [{"span_id": f"{prefix}S000001", "anchors": []}]}
        return {"spans": [{"span_id": f"{prefix}S000001", "anchors": ["42"]}]}

    def _classification(self, prefix: str, request_text: str) -> dict[str, object]:
        if self.verification == "supported":
            return self._findings(prefix, "supported", request_text)
        return self._findings(prefix, "contradicted", request_text, include_fallback=True)

    @staticmethod
    def _findings(
        prefix: str, verdict: str, request_text: str, *, include_fallback: bool = False
    ) -> dict[str, object]:
        evidence_id = re.search(r'"segment_id":"([DS]\d+)"', request_text)
        assert evidence_id is not None
        identifiers = [f"{prefix}C000001"]
        if include_fallback:
            identifiers.append(f"{prefix}C000002")
        return {
            "findings": [
                {
                    "claim_id": identifier,
                    "verdict": verdict,
                    "evidence": [
                        {"segment_id": evidence_id.group(1), "exact_quote": "41."}
                    ],
                }
                for identifier in identifiers
            ]
        }

    @staticmethod
    def _node(level: int, identifier: str) -> dict[str, object]:
        return {
            "summary": f"Grounded level {level}.",
            "content_units": [],
            "entities": [],
            "qualifications": [],
            "contradictions": [],
            "quotations": [],
            "provenance": [identifier],
            "level": level,
        }


def _verify_with_global_cache(
    *,
    cache_root,
    source_id: str,
    provider: VerificationPipelineProvider,
    source_index: SourceLexicalIndex,
) -> None:
    verify_and_repair(
        "42.",
        source_id=source_id,
        source_index=source_index,
        runtime=VerificationRuntime(
            provider=provider,
            counter=CharacterCounter(),
            model="verifier-model",
            timeout_seconds=10,
            context_window_tokens=100_000,
        ),
        config=VerificationConfig(enabled=True),
        coordinator=CacheCoordinator(
            store=CacheStore(cache_root),
            source_id=source_id,
            provider="openai",
            model="verifier-model",
            counter_identity="test:characters",
            counter_exact=True,
            context_window_tokens=100_000,
            behavior={},
        ),
    )


def app() -> AppConfig:
    return AppConfig(model="gpt-4o-mini", timeout_seconds=30)


def strategy(*, hierarchical: bool = False) -> StrategyConfig:
    return StrategyConfig(
        strategy="hierarchical" if hierarchical else "direct",
        context_window=100_000,
        max_output_tokens=1,
        safety_margin_tokens=0,
        safety_margin_fraction=0,
    )


def test_enabled_direct_verification_repairs_editorial_before_citations_and_audit(tmp_path) -> None:
    provider = VerificationPipelineProvider(verification="repair")

    result = run_pipeline(
        ingest_text("The source confirms the value is 41."),
        provider,
        CharacterCounter(),
        app=app(),
        strategy=strategy(),
        config=PipelineConfig(
            target_words=40,
            include_citations=True,
            audit_path=tmp_path / "audit.json",
            verification=VerificationConfig(enabled=True),
        ),
    )

    assert result.final.text == "41.\n\nSources: D000001"
    assert [request.operation_id for request in provider.requests] == [
        "D000001",
        "editorial-final",
        "verification-decompose:V01",
        "verification-classify:V01",
        "verification-repair:V01",
        "verification-decompose:V02",
        "verification-classify:V02",
    ]
    assert result.final.audit is not None
    assert [item.phase for item in result.final.audit.verification.usage] == [
        "decomposition", "classification", "repair", "decomposition", "classification"
    ]
    assert result.final.audit.verification.enabled
    assert result.final.audit.verification.repairs[0].action == "replace"
    assert result.final.audit.configuration["verification"]["enabled"] is True


def test_enabled_hierarchical_verification_uses_default_complete_runtime(tmp_path) -> None:
    provider = VerificationPipelineProvider(verification="supported")
    counter = CharacterCounter()

    result = run_pipeline(
        ingest_text("The source confirms the value is 41. " * 12),
        provider,
        counter,
        app=app(),
        strategy=strategy(hierarchical=True),
        config=PipelineConfig(
            target_words=40,
            segmentation=SegmentationConfig(max_tokens=35),
            max_merge_children=2,
            audit_path=tmp_path / "audit.json",
            verification=VerificationConfig(enabled=True),
        ),
    )

    assert result.root.level >= 2
    verification_requests = [
        request for request in provider.requests if (request.operation_id or "").startswith("verification-")
    ]
    assert [request.operation_id for request in verification_requests] == [
        "verification-decompose:V01", "verification-classify:V01"
    ]
    assert all(request.model == "gpt-4o-mini" for request in verification_requests)
    assert all(request.timeout_seconds == 30 for request in verification_requests)
    assert result.final.audit is not None
    assert result.final.audit.verification.enabled


def test_pipeline_uses_an_injected_complete_verifier_runtime() -> None:
    summary_provider = VerificationPipelineProvider(verification="supported")
    verifier_provider = VerificationPipelineProvider(verification="supported")
    verifier_counter = CharacterCounter()
    runtime = VerificationRuntime(
        provider=verifier_provider,
        counter=verifier_counter,
        model="verifier-model",
        timeout_seconds=10,
        context_window_tokens=100_000,
    )

    result = run_pipeline(
        ingest_text("The source confirms the value is 41."),
        summary_provider,
        CharacterCounter(),
        app=app(),
        strategy=strategy(),
        config=PipelineConfig(
            target_words=40,
            verification=VerificationConfig(enabled=True),
            verification_runtime=runtime,
        ),
    )

    assert [request.operation_id for request in summary_provider.requests] == [
        "D000001", "editorial-final"
    ]
    assert [request.operation_id for request in verifier_provider.requests] == [
        "verification-decompose:V01", "verification-classify:V01"
    ]
    assert all(request.model == "verifier-model" for request in verifier_provider.requests)
    assert all(request.timeout_seconds == 10 for request in verifier_provider.requests)
    assert result.final.audit is None


def test_provider_cache_coordinator_attribute_cannot_enable_verification_caching(
    tmp_path,
) -> None:
    source_id = "a" * 64
    cache_root = tmp_path / "cache"
    plan = RunPlan(
        run_id="explicit-verification",
        descriptor_sha256="b" * 64,
        source_sha256=source_id,
        work_ids=("V01",),
    )
    source_index = build_source_lexical_index(
        provenance_ids=("S000001",),
        source={"S000001": "The source confirms the value is 41."},
    )
    with CheckpointStore(cache_root).open(plan, resume=False) as session:
        coordinator = CacheCoordinator(
            store=CacheStore(cache_root),
            source_id=source_id,
            provider="openai",
            model="verifier-model",
            counter_identity="test:characters",
            counter_exact=True,
            context_window_tokens=100_000,
            behavior={},
            session=session,
        )
        implicit_provider = VerificationPipelineProvider(verification="supported")
        implicit_provider.cache_coordinator = coordinator
        implicit_result = verify_and_repair(
            "42.",
            source_id=source_id,
            source_index=source_index,
            runtime=VerificationRuntime(
                provider=implicit_provider,
                counter=CharacterCounter(),
                model="verifier-model",
                timeout_seconds=10,
                context_window_tokens=100_000,
            ),
            config=VerificationConfig(enabled=True),
        )

        assert not implicit_result.failed
        assert session.manifest.completed == ()
        assert not list((cache_root / "objects").rglob("*.json"))

        explicit_provider = VerificationPipelineProvider(verification="supported")
        explicit_result = verify_and_repair(
            "42.",
            source_id=source_id,
            source_index=source_index,
            runtime=VerificationRuntime(
                provider=explicit_provider,
                counter=CharacterCounter(),
                model="verifier-model",
                timeout_seconds=10,
                context_window_tokens=100_000,
            ),
            config=VerificationConfig(enabled=True),
            coordinator=coordinator,
        )

        assert not explicit_result.failed
        assert [ref.work_id for ref in session.manifest.completed] == ["V01"]
        assert list((cache_root / "objects").rglob("*.json"))

        reused_provider = VerificationPipelineProvider(verification="supported")
        reused_result = verify_and_repair(
            "42.",
            source_id=source_id,
            source_index=source_index,
            runtime=VerificationRuntime(
                provider=reused_provider,
                counter=CharacterCounter(),
                model="verifier-model",
                timeout_seconds=10,
                context_window_tokens=100_000,
            ),
            config=VerificationConfig(enabled=True),
            coordinator=coordinator,
        )

        assert reused_result == explicit_result
        assert reused_provider.requests == []


def test_global_verification_cache_recomputes_when_injected_terms_change(tmp_path) -> None:
    source_id = "a" * 64
    cache_root = tmp_path / "cache"

    original = SourceLexicalIndex(
        entries=(
            SourceLexicalEntry(
                segment_id="S000001",
                text="The source confirms the value is 41.",
                source_order=0,
                terms=frozenset({"42"}),
            ),
        )
    )
    changed_terms = SourceLexicalIndex(
        entries=(
            SourceLexicalEntry(
                segment_id="S000001",
                text="The source confirms the value is 41.",
                source_order=0,
                terms=frozenset({"unrelated"}),
            ),
        )
    )

    first = VerificationPipelineProvider(verification="supported")
    _verify_with_global_cache(
        cache_root=cache_root, source_id=source_id, provider=first, source_index=original
    )
    changed = VerificationPipelineProvider(verification="supported")
    _verify_with_global_cache(
        cache_root=cache_root,
        source_id=source_id,
        provider=changed,
        source_index=changed_terms,
    )
    identical = VerificationPipelineProvider(verification="supported")
    _verify_with_global_cache(
        cache_root=cache_root,
        source_id=source_id,
        provider=identical,
        source_index=changed_terms,
    )

    assert [request.operation_id for request in first.requests] == [
        "verification-decompose:V01",
        "verification-classify:V01",
    ]
    assert [request.operation_id for request in changed.requests] == [
        "verification-decompose:V01",
        "verification-classify:V01",
    ]
    assert identical.requests == []


def test_global_verification_cache_recomputes_when_source_order_changes(tmp_path) -> None:
    source_id = "a" * 64
    cache_root = tmp_path / "cache"

    def index(*, first_order: int, second_order: int) -> SourceLexicalIndex:
        return SourceLexicalIndex(
            entries=(
                SourceLexicalEntry(
                    segment_id="S000001",
                    text="The source confirms the value is 41.",
                    source_order=first_order,
                    terms=frozenset({"42"}),
                ),
                SourceLexicalEntry(
                    segment_id="S000002",
                    text="The source confirms the value is 41.",
                    source_order=second_order,
                    terms=frozenset({"42"}),
                ),
            )
        )

    first = VerificationPipelineProvider(verification="supported")
    _verify_with_global_cache(
        cache_root=cache_root,
        source_id=source_id,
        provider=first,
        source_index=index(first_order=0, second_order=1),
    )
    reordered = VerificationPipelineProvider(verification="supported")
    _verify_with_global_cache(
        cache_root=cache_root,
        source_id=source_id,
        provider=reordered,
        source_index=index(first_order=1, second_order=0),
    )

    assert [request.operation_id for request in first.requests] == [
        "verification-decompose:V01",
        "verification-classify:V01",
    ]
    assert [request.operation_id for request in reordered.requests] == [
        "verification-decompose:V01",
        "verification-classify:V01",
    ]
    first_evidence_id = re.search(
        r'"segment_id":"(S\d+)"', reordered.requests[1].input_text
    )
    assert first_evidence_id is not None
    assert first_evidence_id.group(1) == "S000002"


def test_global_verification_cache_binds_the_validated_provenance_index(tmp_path) -> None:
    cache_root = tmp_path / "cache"
    document = ingest_text("The source confirms the value is 41. " * 12)
    direct_config = PipelineConfig(
        target_words=40,
        cache=CacheConfig(enabled=True, root=cache_root),
        reliability=ReliabilityConfig(run_id="verification-direct"),
        verification=VerificationConfig(enabled=True),
    )
    run_pipeline(
        document,
        VerificationPipelineProvider(verification="supported"),
        CharacterCounter(),
        app=app(),
        strategy=strategy(),
        config=direct_config,
    )

    hierarchical = VerificationPipelineProvider(verification="supported")
    run_pipeline(
        document,
        hierarchical,
        CharacterCounter(),
        app=app(),
        strategy=strategy(hierarchical=True),
        config=PipelineConfig(
            **{
                **direct_config.__dict__,
                "segmentation": SegmentationConfig(max_tokens=35),
                "max_merge_children": 2,
                "reliability": ReliabilityConfig(run_id="verification-hierarchical"),
            }
        ),
    )

    assert [
        request.operation_id
        for request in hierarchical.requests
        if (request.operation_id or "").startswith("verification-")
    ] == ["verification-decompose:V01", "verification-classify:V01"]

    compatible = VerificationPipelineProvider(verification="supported")
    run_pipeline(
        document,
        compatible,
        CharacterCounter(),
        app=app(),
        strategy=strategy(),
        config=PipelineConfig(
            **{
                **direct_config.__dict__,
                "reliability": ReliabilityConfig(run_id="verification-direct-copy"),
            }
        ),
    )

    assert compatible.requests == []


def test_injected_verifier_runtime_reuses_its_own_cached_terminal_result(tmp_path) -> None:
    cache_root = tmp_path / "cache"
    document = ingest_text("The source confirms the value is 41.")
    config = PipelineConfig(
        target_words=40,
        cache=CacheConfig(enabled=True, root=cache_root),
        reliability=ReliabilityConfig(run_id="injected-verifier"),
        verification=VerificationConfig(enabled=True),
    )
    first_summary = VerificationPipelineProvider(verification="supported")
    first_verifier = VerificationPipelineProvider(verification="supported")
    runtime = VerificationRuntime(
        provider=first_verifier,
        counter=VerifierCounter(),
        model="verifier-model",
        timeout_seconds=10,
        context_window_tokens=100_000,
        provider_identity="ollama",
    )

    run_pipeline(
        document,
        first_summary,
        CharacterCounter(),
        app=app(),
        strategy=strategy(),
        config=PipelineConfig(**{**config.__dict__, "verification_runtime": runtime}),
    )
    manifest = json.loads((cache_root / "runs" / "injected-verifier.json").read_text())
    verification_ref = next(
        ref for ref in manifest["completed"] if ref["work_id"] == "V01"
    )
    envelope = json.loads(
        (
            cache_root
            / "objects"
            / verification_ref["cache_key"][:2]
            / f'{verification_ref["cache_key"]}.json'
        ).read_text()
    )
    assert envelope["descriptor"]["provider"] == "ollama"
    assert envelope["descriptor"]["model"] == "verifier-model"
    assert envelope["descriptor"]["counter_identity"] == "test:verifier-characters"
    assert envelope["descriptor"]["counter_exact"] is False
    assert envelope["descriptor"]["context_window_tokens"] == 100_000
    assert envelope["descriptor"]["behavior"]["verification"] == {
        "evidence_tokens": 4096,
        "request_tokens": 8192,
        "output_reserve_tokens": 1024,
        "safety_margin_tokens": 256,
        "max_repair_passes": 1,
        "verification_enabled": True,
    }

    resumed_summary = VerificationPipelineProvider(verification="supported")
    resumed_verifier = VerificationPipelineProvider(verification="supported")
    resumed_runtime = VerificationRuntime(
        provider=resumed_verifier,
        counter=VerifierCounter(),
        model="verifier-model",
        timeout_seconds=10,
        context_window_tokens=100_000,
        provider_identity="ollama",
    )
    resumed = run_pipeline(
        document,
        resumed_summary,
        CharacterCounter(),
        app=app(),
        strategy=strategy(),
        config=PipelineConfig(
            **{
                **config.__dict__,
                "verification_runtime": resumed_runtime,
                "reliability": ReliabilityConfig(
                    run_id="injected-verifier", run_mode="resume"
                ),
            }
        ),
    )

    assert resumed.final.text == "42."
    assert resumed_summary.requests == []
    assert resumed_verifier.requests == []

    for changed_runtime, changed_verification in (
        (
            VerificationRuntime(
                provider=VerificationPipelineProvider(verification="supported"),
                counter=VerifierCounter(),
                model="verifier-model",
                timeout_seconds=10,
                context_window_tokens=100_000,
                provider_identity="openai",
            ),
            config.verification,
        ),
        (resumed_runtime, VerificationConfig(enabled=True, evidence_tokens=128)),
    ):
        attempted_summary = VerificationPipelineProvider(verification="supported")
        with pytest.raises(CheckpointError) as raised:
            run_pipeline(
                document,
                attempted_summary,
                CharacterCounter(),
                app=app(),
                strategy=strategy(),
                config=PipelineConfig(
                    **{
                        **config.__dict__,
                        "verification_runtime": changed_runtime,
                        "verification": changed_verification,
                        "reliability": ReliabilityConfig(
                            run_id="injected-verifier", run_mode="resume"
                        ),
                    }
                ),
            )
        assert raised.value.reason is CheckpointReason.INCOMPATIBLE
        assert attempted_summary.requests == []


def test_pipeline_rejects_an_injected_runtime_when_verification_is_disabled() -> None:
    runtime = VerificationRuntime(
        provider=VerificationPipelineProvider(verification="supported"),
        counter=CharacterCounter(),
        model="verifier-model",
        timeout_seconds=10,
        context_window_tokens=100,
    )

    with pytest.raises(ValueError, match="requires enabled"):
        PipelineConfig(verification_runtime=runtime)


def test_terminal_verification_failure_writes_audit_before_reader_output(tmp_path) -> None:
    provider = VerificationPipelineProvider(verification="malformed")
    audit_path = tmp_path / "audit.json"

    with pytest.raises(FinalizationVerificationError, match="verification"):
        run_pipeline(
            ingest_text("The source confirms the value is 41."),
            provider,
            CharacterCounter(),
            app=app(),
            strategy=strategy(),
            config=PipelineConfig(
                target_words=40,
                include_citations=True,
                audit_path=audit_path,
                verification=VerificationConfig(enabled=True),
            ),
        )

    body = json.loads(audit_path.read_text())
    assert body["verification"]["failed"] is True
    assert body["verification"]["failure_codes"] == ["decomposition_failed"]
    assert body["citations"] == []
    assert "Sources:" not in audit_path.read_text()


def test_exhausted_contradiction_writes_terminal_audit_before_raising(tmp_path) -> None:
    provider = VerificationPipelineProvider(verification="repair")
    audit_path = tmp_path / "audit.json"

    with pytest.raises(FinalizationVerificationError, match="verification"):
        run_pipeline(
            ingest_text("The source confirms the value is 41."),
            provider,
            CharacterCounter(),
            app=app(),
            strategy=strategy(),
            config=PipelineConfig(
                target_words=40,
                audit_path=audit_path,
                verification=VerificationConfig(enabled=True, max_repair_passes=0),
            ),
        )

    body = json.loads(audit_path.read_text())
    assert body["verification"]["failed"] is True
    assert body["verification"]["exhausted"] is True
    assert body["verification"]["failure_codes"] == ["material_contradiction"]
    assert body["citations"] == []


def test_terminal_verification_failure_is_not_cached_and_retries_on_resume(tmp_path) -> None:
    config = PipelineConfig(
        target_words=40,
        cache=CacheConfig(enabled=True, root=tmp_path / "cache"),
        reliability=ReliabilityConfig(run_id="verification-failure"),
        verification=VerificationConfig(enabled=True, max_repair_passes=0),
    )
    document = ingest_text("The source confirms the value is 41.")
    first = VerificationPipelineProvider(verification="repair")
    second = VerificationPipelineProvider(verification="repair")

    with pytest.raises(FinalizationVerificationError):
        run_pipeline(document, first, CharacterCounter(), app=app(), strategy=strategy(), config=config)
    objects_after_first_failure = sorted(
        (tmp_path / "cache" / "objects").rglob("*.json")
    )
    manifest = json.loads(
        (tmp_path / "cache" / "runs" / "verification-failure.json").read_text()
    )
    assert all(ref["work_id"] != "V01" for ref in manifest["completed"])
    assert all(
        json.loads(path.read_text())["descriptor"]["stage"] != "verification"
        for path in objects_after_first_failure
    )
    with pytest.raises(FinalizationVerificationError):
        run_pipeline(document, second, CharacterCounter(), app=app(), strategy=strategy(), config=PipelineConfig(**{**config.__dict__, "reliability": ReliabilityConfig(run_id="verification-failure", run_mode="resume")}))

    assert [request.operation_id for request in second.requests] == [
        "verification-decompose:V01", "verification-classify:V01"
    ]
    assert sorted((tmp_path / "cache" / "objects").rglob("*.json")) == objects_after_first_failure


def test_successful_frozen_verification_reuses_its_terminal_result(tmp_path) -> None:
    config = PipelineConfig(
        target_words=40,
        cache=CacheConfig(enabled=True, root=tmp_path / "cache"),
        reliability=ReliabilityConfig(run_id="verification-success"),
        verification=VerificationConfig(enabled=True),
    )
    document = ingest_text("The source confirms the value is 41.")
    first = VerificationPipelineProvider(verification="supported")
    second = VerificationPipelineProvider(verification="supported")

    run_pipeline(document, first, CharacterCounter(), app=app(), strategy=strategy(), config=config)
    result = run_pipeline(document, second, CharacterCounter(), app=app(), strategy=strategy(), config=PipelineConfig(**{**config.__dict__, "reliability": ReliabilityConfig(run_id="verification-success", run_mode="resume")}))

    assert result.final.text == "42."
    assert second.requests == []


def test_successfully_repaired_verification_reuses_its_terminal_result(tmp_path) -> None:
    cache_root = tmp_path / "cache"
    config = PipelineConfig(
        target_words=40,
        cache=CacheConfig(enabled=True, root=cache_root),
        reliability=ReliabilityConfig(run_id="verification-repaired"),
        verification=VerificationConfig(enabled=True),
    )
    document = ingest_text("The source confirms the value is 41.")
    first = VerificationPipelineProvider(verification="repair")
    second = VerificationPipelineProvider(verification="repair")

    first_result = run_pipeline(
        document, first, CharacterCounter(), app=app(), strategy=strategy(), config=config
    )
    manifest = json.loads(
        (cache_root / "runs" / "verification-repaired.json").read_text()
    )
    verification_ref = next(
        ref for ref in manifest["completed"] if ref["work_id"] == "V01"
    )
    assert (
        cache_root
        / "objects"
        / verification_ref["cache_key"][:2]
        / f'{verification_ref["cache_key"]}.json'
    ).is_file()

    resumed = run_pipeline(
        document,
        second,
        CharacterCounter(),
        app=app(),
        strategy=strategy(),
        config=PipelineConfig(
            **{
                **config.__dict__,
                "reliability": ReliabilityConfig(
                    run_id="verification-repaired", run_mode="resume"
                ),
            }
        ),
    )

    assert [request.operation_id for request in first.requests][-5:] == [
        "verification-decompose:V01",
        "verification-classify:V01",
        "verification-repair:V01",
        "verification-decompose:V02",
        "verification-classify:V02",
    ]
    assert resumed.final.text == first_result.final.text == "41."
    assert second.requests == []


def test_runtime_budget_exhaustion_writes_terminal_audit_before_raising(tmp_path) -> None:
    provider = VerificationPipelineProvider(verification="supported")
    audit_path = tmp_path / "audit.json"
    runtime = VerificationRuntime(
        provider=provider,
        counter=CharacterCounter(),
        model="verifier-model",
        timeout_seconds=10,
        context_window_tokens=100,
    )

    with pytest.raises(FinalizationVerificationError, match="verification"):
        run_pipeline(
            ingest_text("The source confirms the value is 41."),
            VerificationPipelineProvider(verification="supported"),
            CharacterCounter(),
            app=app(),
            strategy=strategy(),
            config=PipelineConfig(
                target_words=40,
                audit_path=audit_path,
                verification=VerificationConfig(
                    enabled=True,
                    output_reserve_tokens=100,
                    safety_margin_tokens=0,
                ),
                verification_runtime=runtime,
            ),
        )

    body = json.loads(audit_path.read_text())
    assert body["verification"]["failed"] is True
    assert body["verification"]["failure_codes"] == ["decomposition_capacity_failed"]
    assert body["citations"] == []
