import json
import re

from summarizer.config import AppConfig, StrategyConfig
from summarizer.ingestion import ingest_text
from summarizer.pipeline import PipelineConfig, run_pipeline
from summarizer.providers.base import GenerationRequest, GenerationResult
from summarizer.segmentation import SegmentationConfig
from summarizer.verification import VerificationConfig


class CharacterCounter:
    identity = "test:characters"
    exact = True
    monotonic = True

    def count(self, text: str) -> int:
        return len(text)


class PipelineProvider:
    def __init__(self) -> None:
        self.requests: list[GenerationRequest] = []

    def generate(self, request: GenerationRequest) -> GenerationResult:
        self.requests.append(request)
        if request.operation_id == "editorial-final":
            payload = {"text": "A concise, coherent final summary."}
        elif request.operation_id == "D000001":
            payload = self._node(0, "D000001")
        elif (request.operation_id or "").startswith("S"):
            payload = self._node(0, request.operation_id or "S000001")
        else:
            level = int((request.operation_id or "merge-L1").rsplit("L", 1)[1])
            identifiers = re.findall(r'"segment_id":"([SD]\d+)"', request.input_text)
            payload = self._node(level, identifiers[-1] if identifiers else "S000001")
        return GenerationResult(json.dumps(payload), "fake", request.model, 1, 1, "stop")

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


def test_direct_pipeline_runs_a_final_call_and_keeps_default_output_plain(tmp_path) -> None:
    provider = PipelineProvider()
    result = run_pipeline(
        ingest_text("A short source."),
        provider,
        CharacterCounter(),
        app=app(),
        strategy=StrategyConfig(context_window=100_000, max_output_tokens=1, safety_margin_tokens=0, safety_margin_fraction=0),
        config=PipelineConfig(target_words=40, audit_path=tmp_path / "audit.json"),
    )

    assert result.strategy.strategy == "direct"
    assert result.final.text == "A concise, coherent final summary."
    assert "Sources:" not in result.final.text
    assert result.final.citations[0].segment_id == "D000001"
    assert result.final.audit is not None
    assert result.final.audit.verification.enabled is False
    assert [request.operation_id for request in provider.requests] == ["D000001", "editorial-final"]


def test_pipeline_verification_is_explicitly_disabled_by_default() -> None:
    assert PipelineConfig().verification == VerificationConfig()


def test_hierarchical_pipeline_forces_multiple_levels_then_edits_and_cites(tmp_path) -> None:
    provider = PipelineProvider()
    result = run_pipeline(
        ingest_text("one two three four five six seven eight nine ten " * 12),
        provider,
        CharacterCounter(),
        app=app(),
        strategy=StrategyConfig(strategy="hierarchical", context_window=100_000, max_output_tokens=1, safety_margin_tokens=0, safety_margin_fraction=0),
        config=PipelineConfig(
            target_words=40,
            segmentation=SegmentationConfig(max_tokens=35),
            max_merge_children=2,
            include_citations=True,
            audit_path=tmp_path / "audit.json",
        ),
    )

    assert result.strategy.strategy == "hierarchical"
    assert result.root.level >= 2
    assert result.final.text.endswith(
        f"Sources: {result.final.citations[0].segment_id}"
    )
    assert result.final.citations[0].segment_id in result.root.covered_segments
    assert provider.requests[-1].operation_id == "editorial-final"
    assert result.final.audit is not None
    assert len(result.final.audit.tree_nodes) == len(result.nodes)
