import json
import re

import pytest

from summarizer.config import AppConfig, StrategyConfig
from summarizer.finalization import FinalizationVerificationError
from summarizer.ingestion import ingest_text
from summarizer.pipeline import PipelineConfig, run_pipeline
from summarizer.providers.base import GenerationRequest, GenerationResult
from summarizer.segmentation import SegmentationConfig
from summarizer.verification import VerificationConfig, VerificationRuntime


class CharacterCounter:
    identity = "test:characters"
    exact = True
    monotonic = True

    def count(self, text: str) -> int:
        return len(text)


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
