from dataclasses import dataclass

import pytest

from summarizer.budget import BudgetError
from summarizer.grounding import GroundingPolicy, select_source_passages
from summarizer.summaries import SummaryNode


@dataclass(frozen=True)
class CharacterCounter:
    identity: str = "test:characters"
    exact: bool = True
    monotonic: bool = True

    def count(self, text: str) -> int:
        return len(text)


SOURCE = {
    "S000001": "ordinary source",
    "S000002": "quoted source",
    "S000003": "uncertain source",
    "S000004": "contradictory source",
}


def child() -> SummaryNode:
    return SummaryNode.model_validate(
        {
            "summary": "A generated child.",
            "content_units": [
                {
                    "text": "An ordinary claim.",
                    "kind": "claim",
                    "evidence": [{"segment_id": "S000001", "quote": None}],
                    "qualification": None,
                    "uncertain": False,
                },
                {
                    "text": "An uncertain claim.",
                    "kind": "claim",
                    "evidence": [{"segment_id": "S000003", "quote": None}],
                    "qualification": "The source hedges this.",
                    "uncertain": True,
                },
            ],
            "entities": [],
            "qualifications": [],
            "contradictions": [
                {
                    "text": "A source disagrees.",
                    "evidence": [{"segment_id": "S000004", "quote": None}],
                }
            ],
            "quotations": [{"segment_id": "S000002", "quote": "quoted"}],
            "provenance": ["S000001", "S000002", "S000003", "S000004"],
            "level": 1,
        }
    )


def test_prioritizes_ambiguous_evidence_before_other_retained_claims() -> None:
    selection = select_source_passages(
        (child(),),
        source=SOURCE,
        counter=CharacterCounter(),
        policy=GroundingPolicy(max_tokens=10_000),
    )

    assert selection.selected_ids == (
        "S000004",
        "S000003",
        "S000002",
        "S000001",
    )


def test_refuses_to_drop_a_mandatory_ambiguous_source() -> None:
    with pytest.raises(ValueError, match="mandatory"):
        select_source_passages(
            (child(),),
            source=SOURCE,
            counter=CharacterCounter(),
            policy=GroundingPolicy(max_tokens=1),
        )


def test_counts_the_complete_source_section_inside_the_grounding_reserve() -> None:
    with pytest.raises(ValueError, match="mandatory"):
        select_source_passages(
            (child(),),
            source=SOURCE,
            counter=CharacterCounter(),
            policy=GroundingPolicy(max_tokens=99),
            selection_cost=lambda _passages: 100,
        )


def test_counts_separators_between_selected_source_blocks() -> None:
    def section_cost(passages: tuple[object, ...]) -> int:
        return 10 * len(passages) + max(len(passages) - 1, 0)

    with pytest.raises(ValueError, match="mandatory"):
        select_source_passages(
            (child(),),
            source=SOURCE,
            counter=CharacterCounter(),
            policy=GroundingPolicy(max_tokens=20),
            selection_cost=section_cost,
        )


def test_empty_selection_reports_reserve_and_smallest_candidate() -> None:
    source = {
        "S000001": "x" * 30,
        "S000002": "y" * 20,
    }
    children = (
        SummaryNode.model_validate(
            {
                "summary": "Two sources.",
                "content_units": [],
                "entities": [],
                "qualifications": [],
                "contradictions": [],
                "quotations": [],
                "provenance": ["S000001", "S000002"],
                "level": 0,
            }
        ),
    )

    with pytest.raises(
        BudgetError,
        match=(
            r"grounding reserve of 10 tokens cannot hold source passage "
            r"S000002 costing 54 tokens"
        ),
    ):
        select_source_passages(
            children,
            source=source,
            counter=CharacterCounter(),
            policy=GroundingPolicy(max_tokens=10),
        )
