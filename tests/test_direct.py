import json
from dataclasses import dataclass

import pytest

from summarizer.cache import CacheStore
from summarizer.direct import (
    DOCUMENT_SEGMENT_ID,
    summarize_direct,
    whole_document_segment,
)
from summarizer.ingestion import ingest_text
from summarizer.leaf import LeafSummaryError, build_leaf_request
from summarizer.providers.base import (
    GenerationRequest,
    GenerationResult,
    ProviderConnectionError,
)
from summarizer.segmentation import BoundaryKind, CacheCoordinator

DOCUMENT = (
    "# Archive migration\n\n"
    "The archive moved in March. The index was rebuilt afterwards.\n\n"
    "Staffing was unchanged."
)


@dataclass(frozen=True)
class CharacterCounter:
    identity: str = "test:characters"
    exact: bool = True
    monotonic: bool = True

    def count(self, text: str) -> int:
        return len(text)


def payload(**overrides: object) -> str:
    body: dict[str, object] = {
        "summary": "The archive moved and the index was rebuilt.",
        "content_units": [],
        "entities": [],
        "qualifications": [],
        "contradictions": [],
        "quotations": [],
        "provenance": [DOCUMENT_SEGMENT_ID],
        "level": 0,
    }
    body.update(overrides)
    return json.dumps(body)


class RecordingProvider:
    def __init__(self, text: str | None = None) -> None:
        self.requests: list[GenerationRequest] = []
        self.text = text if text is not None else payload()

    def generate(self, request: GenerationRequest) -> GenerationResult:
        self.requests.append(request)
        return GenerationResult(text=self.text, provider="fake", model=request.model)


def coordinator(tmp_path, document) -> CacheCoordinator:
    counter = CharacterCounter()
    return CacheCoordinator(
        store=CacheStore(tmp_path / "cache"), source_id=document.source_id,
        provider="openai", model="m", counter_identity=counter.identity,
        counter_exact=counter.exact, context_window_tokens=100,
        behavior={"strategy_config": {"strategy": "direct", "context_window": 100, "max_direct_tokens": None, "max_output_tokens": 1, "safety_margin_tokens": 0}},
    )


def test_whole_document_segment_spans_the_canonical_text() -> None:
    document = ingest_text(DOCUMENT)

    segment = whole_document_segment(document, CharacterCounter())

    assert segment.text == document.text
    assert (segment.core_start, segment.core_end) == (0, len(document.text))
    assert (segment.context_start, segment.context_end) == (0, len(document.text))
    assert segment.leading_overlap_tokens == 0
    assert segment.trailing_overlap_tokens == 0
    assert segment.segment_id == DOCUMENT_SEGMENT_ID
    assert segment.source_id == document.source_id


def test_whole_document_segment_is_labelled_as_a_whole_document() -> None:
    """No boundary kind describing a break *within* a document would be true.

    Segmentation never produces this kind, so it also tells a later stage that
    provenance covers everything rather than a region.
    """
    segment = whole_document_segment(ingest_text(DOCUMENT), CharacterCounter())

    assert segment.boundary_kind is BoundaryKind.DOCUMENT


def test_the_prompt_does_not_describe_a_whole_document_as_a_fragment() -> None:
    """Telling a model it holds a fragment invites it to hedge about context.

    That is the opposite of the cohesive result a direct run is meant to give.
    """
    document = ingest_text(DOCUMENT)
    whole = build_leaf_request(
        whole_document_segment(document, CharacterCounter()),
        model="m",
        timeout_seconds=30,
    )

    # Assert the property, not just the sentence that was edited: the rules
    # below the framing must not keep calling the input a region either.
    assert "region" not in whole.instructions
    assert "an entire document" in whole.instructions


def test_a_leaf_request_still_describes_a_region() -> None:
    from summarizer.segmentation import SegmentationConfig, segment_document

    segments = segment_document(
        ingest_text(DOCUMENT), CharacterCounter(), SegmentationConfig(max_tokens=40)
    )
    leaf = build_leaf_request(segments[0], model="m", timeout_seconds=30)

    assert "one region of a longer document" in leaf.instructions
    assert "the region states" in leaf.instructions


def test_summarizes_a_whole_document_in_one_call() -> None:
    document = ingest_text(DOCUMENT)
    provider = RecordingProvider()

    node = summarize_direct(
        document, provider, CharacterCounter(), model="m", timeout_seconds=30
    )

    assert node.level == 0
    assert node.provenance == (DOCUMENT_SEGMENT_ID,)
    assert len(provider.requests) == 1
    assert document.text in provider.requests[0].input_text


def test_compatible_direct_result_reuses_the_validated_summary(tmp_path) -> None:
    document = ingest_text(DOCUMENT)
    first, second = RecordingProvider(), RecordingProvider()

    summarize_direct(document, first, CharacterCounter(), model="m", timeout_seconds=30, coordinator=coordinator(tmp_path, document))
    summary = summarize_direct(document, second, CharacterCounter(), model="m", timeout_seconds=30, coordinator=coordinator(tmp_path, document))

    assert summary.provenance == (DOCUMENT_SEGMENT_ID,)
    assert len(first.requests) == 1
    assert second.requests == []


def test_malformed_direct_response_is_not_cached(tmp_path) -> None:
    document = ingest_text(DOCUMENT)
    first = RecordingProvider("not json")
    second = RecordingProvider()

    with pytest.raises(LeafSummaryError):
        summarize_direct(document, first, CharacterCounter(), model="m", timeout_seconds=30, coordinator=coordinator(tmp_path, document))
    summarize_direct(document, second, CharacterCounter(), model="m", timeout_seconds=30, coordinator=coordinator(tmp_path, document))

    assert len(first.requests) == 1
    assert len(second.requests) == 1


def test_a_quotation_from_anywhere_in_the_document_validates() -> None:
    """The whole document is attributable, unlike a segment's overlap context."""
    document = ingest_text(DOCUMENT)
    provider = RecordingProvider(
        payload(
            quotations=[
                {"segment_id": DOCUMENT_SEGMENT_ID, "quote": "Staffing was unchanged."}
            ]
        )
    )

    node = summarize_direct(
        document, provider, CharacterCounter(), model="m", timeout_seconds=30
    )

    assert node.quotations[0].quote == "Staffing was unchanged."


def test_rejects_a_fabricated_quotation() -> None:
    provider = RecordingProvider(
        payload(
            quotations=[{"segment_id": DOCUMENT_SEGMENT_ID, "quote": "the archive burned"}]
        )
    )

    with pytest.raises(LeafSummaryError, match="quotation"):
        summarize_direct(
            ingest_text(DOCUMENT),
            provider,
            CharacterCounter(),
            model="m",
            timeout_seconds=30,
        )


def test_malformed_output_fails_naming_the_document_identifier() -> None:
    provider = RecordingProvider("I would rather not.")

    with pytest.raises(LeafSummaryError, match=DOCUMENT_SEGMENT_ID):
        summarize_direct(
            ingest_text(DOCUMENT),
            provider,
            CharacterCounter(),
            model="m",
            timeout_seconds=30,
        )


def test_provider_failures_propagate_unchanged() -> None:
    class FailingProvider:
        def generate(self, request: GenerationRequest) -> GenerationResult:
            raise ProviderConnectionError("service is unreachable")

    with pytest.raises(ProviderConnectionError):
        summarize_direct(
            ingest_text(DOCUMENT),
            FailingProvider(),
            CharacterCounter(),
            model="m",
            timeout_seconds=30,
        )


def test_direct_requests_are_deterministic() -> None:
    document = ingest_text(DOCUMENT)
    first, second = RecordingProvider(), RecordingProvider()

    node_one = summarize_direct(
        document, first, CharacterCounter(), model="m", timeout_seconds=30
    )
    node_two = summarize_direct(
        document, second, CharacterCounter(), model="m", timeout_seconds=30
    )

    assert node_one == node_two
    assert first.requests == second.requests


def test_document_segment_records_its_measured_token_counts() -> None:
    document = ingest_text(DOCUMENT)

    segment = whole_document_segment(document, CharacterCounter())

    assert segment.core_token_count == len(document.text)
    assert segment.token_count == len(document.text)


def test_explicit_direct_refuses_an_assumed_context_window() -> None:
    """A guessed window cannot establish a fit.

    This is the dangerous combination rather than a pedantic one: the local
    provider truncates an oversized prompt silently instead of rejecting it,
    so a fit checked against a guess yields a confidently ungrounded summary.
    """
    from summarizer.budget import BudgetError, select_strategy
    from summarizer.config import StrategyConfig

    with pytest.raises(BudgetError, match="not known"):
        select_strategy(
            ingest_text(DOCUMENT),
            CharacterCounter(),
            provider="ollama",
            model="qwen3.8",
            config=StrategyConfig(strategy="direct", context_window=None),
        )
