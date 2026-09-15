import json
import re

import pytest
from tiktoken.core import Encoding

import summarizer.pipeline as pipeline
from summarizer.budget import select_strategy
from summarizer.config import AppConfig, StrategyConfig
from summarizer.ingestion import ingest_text
from summarizer.pipeline import PipelineConfig, run_pipeline
from summarizer.providers.base import GenerationRequest, GenerationResult
from summarizer.segmentation import SegmentationConfig
from summarizer.tokenization import TiktokenCounter, resolve_token_counter
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


class GroundedPipelineProvider(PipelineProvider):
    """Return every reference the merge prompt supplies to exercise grounding."""

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
            identifiers = tuple(dict.fromkeys(
                re.findall(r'"segment_id":"([SD]\d+)"', request.input_text)
            ))
            payload = self._node(level, identifiers[0])
            payload["provenance"] = list(identifiers)
        return GenerationResult(json.dumps(payload), "fake", request.model, 1, 1, "stop")


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


def test_hierarchical_pipeline_runs_offline_with_ollama_defaults_and_explicit_window(
    tmp_path,
) -> None:
    provider = PipelineProvider()
    app_config = AppConfig(provider="ollama", model="qwen3.8")
    counter = resolve_token_counter(
        provider=app_config.provider,
        model=app_config.model,
    )

    result = run_pipeline(
        ingest_text("A short source."),
        provider,
        counter,
        app=app_config,
        strategy=StrategyConfig(
            strategy="hierarchical",
            context_window=32_768,
        ),
        config=PipelineConfig(audit_path=tmp_path / "audit.json"),
    )

    assert counter.identity == "estimate:utf8-bytes"
    assert result.strategy.strategy == "hierarchical"
    assert [request.operation_id for request in provider.requests] == [
        "S000001",
        "editorial-final",
    ]


def test_default_pipeline_merges_full_capacity_segments_with_a_real_tokenizer() -> None:
    encoding = Encoding(
        name="offline-byte-bpe",
        pat_str=r"(?s).",
        mergeable_ranks={bytes([value]): value for value in range(256)},
        special_tokens={},
    )
    counter = TiktokenCounter(encoding)

    app_config = AppConfig(model="gpt-4", timeout_seconds=30)
    strategy = StrategyConfig(context_window=32_768)
    capacity = select_strategy(
        ingest_text("alpha"),
        counter,
        provider=app_config.provider,
        model=app_config.model,
        config=strategy,
    ).usable_input_capacity
    document = ingest_text("alpha " * (2 * capacity - 1))
    provider = PipelineProvider()

    result = run_pipeline(
        document,
        provider,
        counter,
        app=app_config,
        strategy=strategy,
        config=PipelineConfig(),
    )

    assert counter.count(document.text) > capacity
    assert result.strategy.strategy == "hierarchical"
    assert result.root.level >= 1
    assert any(
        (request.operation_id or "").startswith("merge-L")
        for request in provider.requests
    )


def test_default_pipeline_hierarchy_with_a_real_model_tokenizer_keeps_references(tmp_path) -> None:
    import tiktoken

    try:
        tiktoken.encoding_for_model("gpt-4o-mini")
    except Exception as error:  # pragma: no cover - depends on the local cache
        import pytest

        pytest.skip(f"tiktoken vocabulary is unavailable offline: {error}")

    counter = TiktokenCounter.for_model("gpt-4o-mini")
    app_config = AppConfig(model="gpt-4o-mini", timeout_seconds=30)
    strategy = StrategyConfig(context_window=8_192)
    capacity = select_strategy(
        ingest_text("alpha"),
        counter,
        provider=app_config.provider,
        model=app_config.model,
        config=strategy,
    ).usable_input_capacity
    # Two full-capacity leaves, not one full leaf plus a near-empty tail:
    # a tail segment's tiny source passage always fits the (pre-#26) flat
    # 1,024-token grounding reserve on its own, masking the reserve-sizing
    # defect. Both leaves here exceed the reserve, so the merge only
    # succeeds once the reserve scales with merge capacity (#26) and the
    # keep-every-reference merge output is accepted as grounded (#27).
    document = ingest_text("alpha " * int(capacity * 1.5))
    provider = GroundedPipelineProvider()

    result = run_pipeline(
        document,
        provider,
        counter,
        app=app_config,
        strategy=strategy,
        config=PipelineConfig(audit_path=tmp_path / "audit.json"),
    )

    assert result.strategy.strategy == "hierarchical"
    assert set(result.root.covered_segments) == {
        segment.segment_id for segment in result.final.audit.source_segments
    }


def test_pipeline_recomputes_hierarchical_capacity_when_segments_overlap() -> None:
    counter = CharacterCounter()
    app_config = AppConfig(model="gpt-4o-mini", timeout_seconds=30)
    strategy = StrategyConfig(
        strategy="hierarchical",
        context_window=10_000,
        max_output_tokens=1,
        safety_margin_tokens=0,
        safety_margin_fraction=0,
    )
    document = ingest_text("alpha " * 800)
    report = select_strategy(
        document,
        counter,
        provider=app_config.provider,
        model=app_config.model,
        config=strategy,
    )
    overlap = SegmentationConfig(
        max_tokens=1, overlap_tokens=50
    )
    overlap_capacity = pipeline._hierarchical_capacity(
        report, counter, app_config, strategy, overlap
    )

    assert overlap_capacity < report.usable_input_capacity
    with pytest.raises(Exception, match="safely measured leaf capacity"):
        run_pipeline(
            document,
            PipelineProvider(),
            counter,
            app=app_config,
            strategy=strategy,
            config=PipelineConfig(
                segmentation=SegmentationConfig(
                    max_tokens=report.usable_input_capacity, overlap_tokens=50
                )
            ),
        )

    result = run_pipeline(
        document,
        GroundedPipelineProvider(),
        counter,
        app=app_config,
        strategy=strategy,
        config=PipelineConfig(
            segmentation=SegmentationConfig(
                max_tokens=overlap_capacity, overlap_tokens=50
            )
        ),
    )

    assert result.strategy.strategy == "hierarchical"
