import pytest

from summarizer.grounding import SourcePassage
from summarizer.tokenization import ConservativeUtf8TokenCounter
from summarizer.verification import (
    Claim,
    SourceLexicalIndex,
    VerificationConfig,
    VerificationRuntime,
    build_source_lexical_index,
    pack_work_items,
    select_claim_evidence,
)


class Provider:
    def generate(self, request):  # pragma: no cover - protocol fixture
        raise AssertionError("not called")


def claim(anchor: str = "Alpha café growth") -> Claim:
    return Claim(
        claim_id="V01C000001",
        span_id="V01S000001",
        ordinal=1,
        anchor=anchor,
        is_fallback=True,
    )


def test_evidence_ranking_uses_overlap_then_source_order() -> None:
    source = {
        "S000001": "Unrelated introduction.",
        "S000002": "The ALPHA café reported growth.",
        "S000003": "Alpha growth continued at the café.",
    }

    index = build_source_lexical_index(
        provenance_ids=("S000001", "S000003", "S000002"), source=source
    )
    bundle = select_claim_evidence(
        claim(),
        source_index=index,
        counter=ConservativeUtf8TokenCounter(),
        max_tokens=1000,
    )

    assert [passage.segment_id for passage in bundle.passages] == [
        "S000003",
        "S000002",
        "S000001",
    ]
    assert bundle.selection.selected_ids == ("S000003", "S000002", "S000001")
    assert bundle.selection.examined_ids == bundle.selection.selected_ids
    assert bundle.selection.omitted_ids == ()
    assert bundle.selection.retrieval_complete


def test_evidence_selection_packs_only_complete_passages_and_records_omissions() -> None:
    source = {
        "S000001": "Alpha evidence.",
        "S000002": "Alpha " + "large " * 30,
        "S000003": "Other.",
    }

    index = build_source_lexical_index(
        provenance_ids=("S000001", "S000002", "S000003"), source=source
    )
    bundle = select_claim_evidence(
        claim("Alpha"),
        source_index=index,
        counter=ConservativeUtf8TokenCounter(),
        max_tokens=70,
    )

    assert bundle.selection.selected_ids == ("S000001",)
    assert bundle.selection.examined_ids == ("S000001",)
    assert bundle.selection.omitted_ids == ("S000002", "S000003")
    assert not bundle.selection.retrieval_complete
    assert bundle.passages == (SourcePassage("S000001", "Alpha evidence."),)


def test_evidence_selection_deduplicates_provenance_but_preserves_distinct_ids() -> None:
    source = {"S000001": "Same text.", "S000002": "Same text."}

    index = build_source_lexical_index(
        provenance_ids=("S000001", "S000001", "S000002"), source=source
    )
    bundle = select_claim_evidence(
        claim("Same"),
        source_index=index,
        counter=ConservativeUtf8TokenCounter(),
        max_tokens=1000,
    )

    assert bundle.selection.selected_ids == ("S000001", "S000002")


def test_evidence_selection_rejects_unknown_or_unfittable_provenance() -> None:
    with pytest.raises(ValueError, match="missing"):
        build_source_lexical_index(
            provenance_ids=("S999999",),
            source={},
        )

    index = build_source_lexical_index(
        provenance_ids=("S000001",), source={"S000001": "too large"}
    )
    with pytest.raises(ValueError, match="cannot hold"):
        select_claim_evidence(
            claim(),
            source_index=index,
            counter=ConservativeUtf8TokenCounter(),
            max_tokens=1,
        )


def test_index_normalizes_composed_and_decomposed_unicode_before_ranking() -> None:
    index = build_source_lexical_index(
        provenance_ids=("S000001", "S000002"),
        source={
            "S000001": "café unrelated",
            "S000002": "cafe\u0301 growth",
        },
    )

    bundle = select_claim_evidence(
        claim("café growth"),
        source_index=index,
        counter=ConservativeUtf8TokenCounter(),
        max_tokens=53,
    )

    assert bundle.selection.selected_ids == ("S000002",)


def test_index_is_immutable_and_reused_for_multiple_claims() -> None:
    index = build_source_lexical_index(
        provenance_ids=("S000001",), source={"S000001": "Alpha evidence."}
    )

    assert isinstance(index, SourceLexicalIndex)
    assert index.entries[0].terms == frozenset({"alpha", "evidence"})
    first = select_claim_evidence(
        claim("Alpha"),
        source_index=index,
        counter=ConservativeUtf8TokenCounter(),
        max_tokens=1000,
    )
    second = select_claim_evidence(
        claim("evidence"),
        source_index=index,
        counter=ConservativeUtf8TokenCounter(),
        max_tokens=1000,
    )

    assert first.passages == second.passages
    assert first.selection.retrieval_method == "lexical-overlap/1"


def runtime(*, context_window_tokens: int = 100) -> VerificationRuntime:
    return VerificationRuntime(
        provider=Provider(),
        counter=ConservativeUtf8TokenCounter(),
        model="model",
        timeout_seconds=30,
        context_window_tokens=context_window_tokens,
    )


def render(items: tuple[str, ...]) -> str:
    return "header:" + ",".join(items)


def test_work_item_packing_is_stable_complete_and_nonduplicating() -> None:
    config = VerificationConfig(
        request_tokens=12,
        output_reserve_tokens=4,
        safety_margin_tokens=2,
    )

    batches = pack_work_items(
        ("aa", "bb", "cc"),
        render_request=render,
        runtime=runtime(),
        config=config,
    )

    assert batches == (("aa", "bb"), ("cc",))
    assert tuple(item for batch in batches for item in batch) == ("aa", "bb", "cc")


def test_work_item_packing_obeys_context_capacity_below_request_budget() -> None:
    config = VerificationConfig(
        request_tokens=100,
        output_reserve_tokens=5,
        safety_margin_tokens=5,
    )

    batches = pack_work_items(
        ("aaa", "bbb"),
        render_request=render,
        runtime=runtime(context_window_tokens=20),
        config=config,
    )

    assert batches == (("aaa",), ("bbb",))


def test_work_item_packing_counts_complete_structured_requests() -> None:
    config = VerificationConfig(
        request_tokens=100,
        output_reserve_tokens=4,
        safety_margin_tokens=2,
    )

    def structured(items: tuple[str, ...]) -> str:
        return '{"instructions":"verify","schema":{"type":"object"},"data":[' + ",".join(items) + "]}"

    batches = pack_work_items(
        ('"decomposition"', '"classification"', '"repair"'),
        render_request=structured,
        runtime=runtime(),
        config=config,
    )

    assert batches == (("\"decomposition\"", "\"classification\""), ("\"repair\"",))


def test_evidence_selection_records_exact_serialized_token_cost() -> None:
    counter = ConservativeUtf8TokenCounter()
    index = build_source_lexical_index(
        provenance_ids=("S000001",), source={"S000001": "Alpha evidence."}
    )

    bundle = select_claim_evidence(
        claim("Alpha"), source_index=index, counter=counter, max_tokens=1000
    )

    assert bundle.selection.token_cost == counter.count(
        '{"segment_id":"S000001","text":"Alpha evidence."}'
    )


def test_work_item_packing_rejects_one_oversized_item_before_provider_use() -> None:
    config = VerificationConfig(
        request_tokens=8,
        output_reserve_tokens=4,
        safety_margin_tokens=2,
    )

    with pytest.raises(ValueError, match="single work item"):
        pack_work_items(
            ("oversized",),
            render_request=render,
            runtime=runtime(),
            config=config,
        )
