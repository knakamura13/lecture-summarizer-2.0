import json
import re

import pytest

from summarizer.checkpoint import (
    CheckpointError,
    CheckpointReason,
    CheckpointStore,
    RunPlan,
)
from summarizer.config import AppConfig, CacheConfig, ReliabilityConfig, StrategyConfig
from summarizer.ingestion import ingest_text
from summarizer.pipeline import PipelineConfig, run_pipeline
from summarizer.providers.base import GenerationRequest, GenerationResult
from summarizer.segmentation import SegmentationConfig


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
                "summary": "A grounded leaf.", "content_units": [], "entities": [],
                "qualifications": [], "contradictions": [], "quotations": [],
                "provenance": [request.operation_id], "level": 0,
            }
        else:
            payload = {
                "summary": "A grounded merge.", "content_units": [], "entities": [],
                "qualifications": [], "contradictions": [], "quotations": [],
                "provenance": [re.findall(r'"segment_id":"(S\d+)"', request.input_text)[-1]], "level": int((request.operation_id or "merge-L1").rsplit("L", 1)[1]),
            }
        return GenerationResult(json.dumps(payload), "fake", request.model)


def _app() -> AppConfig:
    return AppConfig(model="gpt-4o-mini", timeout_seconds=30)


def _strategy() -> StrategyConfig:
    return StrategyConfig(
        strategy="hierarchical", context_window=100_000, max_output_tokens=1,
        safety_margin_tokens=0, safety_margin_fraction=0,
    )


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
    run_pipeline(document, first, Counter(), app=_app(), strategy=_strategy(), config=config)

    second = CountingProvider()
    result = run_pipeline(
        document,
        second,
        Counter(),
        app=_app(),
        strategy=_strategy(),
        config=PipelineConfig(
            **{**config.__dict__, "reliability": ReliabilityConfig(run_id="reliable-run", run_mode="resume")}
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
        document, CountingProvider(), Counter(), app=_app(), strategy=_strategy(), config=config
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
        document, CountingProvider(), Counter(), app=_app(), strategy=_strategy(), config=config
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
                    "segmentation": SegmentationConfig(
                        max_tokens=35, overlap_tokens=1
                    ),
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
    run_pipeline(document, CountingProvider(), Counter(), app=_app(), strategy=_strategy(), config=config)

    empty_run = PipelineConfig(
        **{
            **config.__dict__,
            "reliability": ReliabilityConfig(run_id="resume-no-global"),
        }
    )
    run_pipeline(
        document, CountingProvider(), Counter(), app=_app(), strategy=_strategy(), config=empty_run
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
