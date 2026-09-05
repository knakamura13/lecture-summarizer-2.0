import json

import pytest

from summarizer.verification import (
    ClaimVerdict,
    VerificationResponseError,
    parse_claim_anchors,
    parse_claim_findings,
    split_draft_spans,
)


def test_claim_parser_assigns_ids_and_reuses_full_span_fallback() -> None:
    spans = split_draft_spans("Alice bought and sold shares.", pass_index=1)
    response = json.dumps(
        {
            "spans": [
                {
                    "span_id": "V01S000001",
                    "anchors": ["Alice bought", "Alice bought and sold shares."],
                }
            ]
        }
    )

    claims = parse_claim_anchors(response, spans=spans, pass_index=1)

    assert [claim.claim_id for claim in claims] == ["V01C000001", "V01C000002"]
    assert [claim.anchor for claim in claims] == [
        "Alice bought",
        "Alice bought and sold shares.",
    ]
    assert [claim.is_fallback for claim in claims] == [False, True]


def test_claim_parser_reuses_full_span_fallback_with_trailing_whitespace() -> None:
    spans = split_draft_spans("A fact. ", pass_index=1)
    response = json.dumps(
        {
            "spans": [
                {
                    "span_id": "V01S000001",
                    "anchors": ["A fact. "],
                }
            ]
        }
    )

    claims = parse_claim_anchors(response, spans=spans, pass_index=1)

    assert [(claim.anchor, claim.is_fallback) for claim in claims] == [
        ("A fact. ", True)
    ]


def test_claim_parser_adds_a_local_fallback_and_allows_overlapping_anchors() -> None:
    spans = split_draft_spans("Alice bought and sold shares.", pass_index=2)
    response = json.dumps(
        {
            "spans": [
                {
                    "span_id": "V02S000001",
                    "anchors": ["Alice bought", "bought and sold"],
                }
            ]
        }
    )

    claims = parse_claim_anchors(response, spans=spans, pass_index=2)

    assert [claim.anchor for claim in claims] == [
        "Alice bought",
        "bought and sold",
        "Alice bought and sold shares.",
    ]
    assert claims[-1].is_fallback


def test_claim_ids_follow_local_span_order_not_response_order() -> None:
    spans = split_draft_spans("First. Second.", pass_index=1)
    response = json.dumps(
        {
            "spans": [
                {"span_id": "V01S000002", "anchors": []},
                {"span_id": "V01S000001", "anchors": []},
            ]
        }
    )

    claims = parse_claim_anchors(response, spans=spans, pass_index=1)

    assert [(claim.claim_id, claim.span_id) for claim in claims] == [
        ("V01C000001", "V01S000001"),
        ("V01C000002", "V01S000002"),
    ]


@pytest.mark.parametrize(
    "payload",
    (
        {"spans": []},
        {"spans": [{"span_id": "unknown", "anchors": ["Fact"]}]},
        {"spans": [{"span_id": "V01S000001", "anchors": ["invented"]}]},
        {"spans": [{"span_id": "V01S000001", "anchors": ["Fact", "Fact"]}]},
        {"spans": [{"span_id": "V01S000001", "anchors": [], "extra": 1}]},
    ),
)
def test_claim_parser_rejects_missing_unknown_invented_or_duplicate_data(payload) -> None:
    spans = split_draft_spans("Fact.", pass_index=1)

    with pytest.raises(VerificationResponseError):
        parse_claim_anchors(json.dumps(payload), spans=spans, pass_index=1)


def test_finding_parser_requires_one_result_per_claim_and_exact_quotes() -> None:
    spans = split_draft_spans("The value is 42.", pass_index=1)
    claims = parse_claim_anchors(
        '{"spans":[{"span_id":"V01S000001","anchors":[]}]}',
        spans=spans,
        pass_index=1,
    )
    selected = {claims[0].claim_id: {"S000001": "The measured value is 42."}}
    response = json.dumps(
        {
            "findings": [
                {
                    "claim_id": claims[0].claim_id,
                    "verdict": "supported",
                    "evidence": [
                        {"segment_id": "S000001", "exact_quote": "value is 42"}
                    ],
                }
            ]
        }
    )

    findings = parse_claim_findings(response, claims=claims, selected=selected)

    assert findings[0].verdict is ClaimVerdict.SUPPORTED
    assert findings[0].evidence_ids == ("S000001",)


@pytest.mark.parametrize(
    "response",
    (
        "not json",
        "{} {}",
        '{"findings":[]}',
        '{"findings":[{"claim_id":"unknown","verdict":"supported","evidence":[]}]}'
    ),
)
def test_finding_parser_rejects_malformed_missing_and_unknown_results(response: str) -> None:
    spans = split_draft_spans("Fact.", pass_index=1)
    claims = parse_claim_anchors(
        '{"spans":[{"span_id":"V01S000001","anchors":[]}]}',
        spans=spans,
        pass_index=1,
    )

    with pytest.raises(VerificationResponseError) as error:
        parse_claim_findings(
            response,
            claims=claims,
            selected={claims[0].claim_id: {"S000001": "Fact."}},
        )

    assert response not in str(error.value)


def test_finding_parser_rejects_unselected_evidence_and_quote_mismatch() -> None:
    spans = split_draft_spans("Fact.", pass_index=1)
    claims = parse_claim_anchors(
        '{"spans":[{"span_id":"V01S000001","anchors":[]}]}',
        spans=spans,
        pass_index=1,
    )
    response = json.dumps(
        {
            "findings": [
                {
                    "claim_id": claims[0].claim_id,
                    "verdict": "contradicted",
                    "evidence": [
                        {"segment_id": "S000002", "exact_quote": "not present"}
                    ],
                }
            ]
        }
    )

    with pytest.raises(VerificationResponseError):
        parse_claim_findings(
            response,
            claims=claims,
            selected={claims[0].claim_id: {"S000001": "Fact."}},
        )


@pytest.mark.parametrize(
    "evidence",
    (
        [{"segment_id": "S000001", "exact_quote": " "}],
        [
            {"segment_id": "S000001", "exact_quote": "Fact"},
            {"segment_id": "S000001", "exact_quote": "Fact"},
        ],
    ),
)
def test_finding_parser_rejects_blank_quotes_and_duplicate_evidence(evidence) -> None:
    spans = split_draft_spans("Claim.", pass_index=1)
    claims = parse_claim_anchors(
        '{"spans":[{"span_id":"V01S000001","anchors":[]}]}',
        spans=spans,
        pass_index=1,
    )
    response = json.dumps(
        {
            "findings": [
                {
                    "claim_id": claims[0].claim_id,
                    "verdict": "supported",
                    "evidence": evidence,
                }
            ]
        }
    )

    with pytest.raises(VerificationResponseError):
        parse_claim_findings(
            response,
            claims=claims,
            selected={claims[0].claim_id: {"S000001": "irrelevant source passage Fact"}},
        )
