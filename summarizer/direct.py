from __future__ import annotations

import json

from summarizer.ingestion import SourceDocument
from summarizer.leaf import LEAF_PROMPT_VERSION, build_leaf_request, parse_leaf_summary
from summarizer.providers.base import ModelProvider
from summarizer.segmentation import BoundaryKind, CacheCoordinator, SourceSegment
from summarizer.summaries import LEAF_SCHEMA_VERSION, SummaryNode
from summarizer.tokenization import TokenCounter

# Deliberately distinct from segmentation's identifiers, which are
# f"S{order+1:06d}". Reusing "S000001" would collide with the first segment of
# a hierarchical run while covering a different extent, and provenance is a
# bare tuple of identifiers with no boundary kind attached - so the two
# meanings would be unrecoverable downstream, exactly where later stages need
# to trace a root back to source.
DOCUMENT_SEGMENT_ID = "D000001"


def whole_document_segment(
    document: SourceDocument,
    counter: TokenCounter,
) -> SourceSegment:
    """Represent an entire document as one segment.

    Segmentation cannot be reused to produce this. A heading forces a hard
    break during packing, so a document with more than one heading yields
    several segments no matter how large the budget - measured at three
    segments for a three-heading document at double the required budget. The
    record is therefore constructed directly.

    Because its core spans the whole document, the provenance rules the leaf
    validator already enforces do the right thing unchanged: the single legal
    identifier is this one, and every quotation is checked against the whole
    text rather than a region of it.
    """
    tokens = counter.count(document.text)
    return SourceSegment(
        segment_id=DOCUMENT_SEGMENT_ID,
        source_id=document.source_id,
        order=0,
        text=document.text,
        core_start=0,
        core_end=len(document.text),
        context_start=0,
        context_end=len(document.text),
        core_token_count=tokens,
        token_count=tokens,
        leading_overlap_tokens=0,
        trailing_overlap_tokens=0,
        boundary_kind=BoundaryKind.DOCUMENT,
    )


def summarize_direct(
    document: SourceDocument,
    provider: ModelProvider,
    counter: TokenCounter,
    *,
    model: str,
    timeout_seconds: float,
    coordinator: CacheCoordinator | None = None,
) -> SummaryNode:
    """Summarize a whole document in a single call.

    Reuses the leaf request builder, parser, and provenance validator, so a
    direct result is the same record the rest of the hierarchy consumes. The
    caller is responsible for having established that the document fits; this
    function does not re-check the budget.
    """
    segment = whole_document_segment(document, counter)
    request = build_leaf_request(
        segment, model=model, timeout_seconds=timeout_seconds
    )
    if coordinator is None:
        result = provider.generate(request)
        return parse_leaf_summary(result.text, segment=segment)

    def decode(payload: object) -> SummaryNode:
        node = SummaryNode.model_validate(payload)
        return parse_leaf_summary(json.dumps(node.model_dump(mode="json")), segment=segment)

    return coordinator.resolve(
        stage="direct",
        work_id=DOCUMENT_SEGMENT_ID,
        prompt_version=LEAF_PROMPT_VERSION,
        schema_version=LEAF_SCHEMA_VERSION,
        input_value={
            "instructions": request.instructions,
            "input_text": request.input_text,
            "schema": request.response_schema,
        },
        behavior={},
        decode=decode,
        encode=lambda node: node.model_dump(mode="json"),
        compute=lambda: parse_leaf_summary(provider.generate(request).text, segment=segment),
    )
