"""Deterministic selection of authoritative source passages for a merge."""

from __future__ import annotations

import json
from dataclasses import dataclass
from collections.abc import Callable, Mapping, Sequence

from summarizer.budget import BudgetError
from summarizer.summaries import EvidenceItem, SummaryNode
from summarizer.tokenization import TokenCounter


@dataclass(frozen=True)
class GroundingPolicy:
    """The input space reserved for original source text at each merge."""

    max_tokens: int

    def __post_init__(self) -> None:
        if self.max_tokens <= 0:
            raise ValueError("grounding max_tokens must be positive")


@dataclass(frozen=True)
class SourcePassage:
    """One complete citable source core selected for a merge request."""

    segment_id: str
    text: str

    def __post_init__(self) -> None:
        if not self.segment_id.strip() or not self.text:
            raise ValueError("a source passage needs a segment identifier and text")


@dataclass(frozen=True)
class GroundingSelection:
    passages: tuple[SourcePassage, ...]
    selected_ids: tuple[str, ...]
    omitted_ids: tuple[str, ...]


def serialize_source_passage(passage: SourcePassage) -> str:
    """Use one compact, deterministic representation for budget and payload."""
    return json.dumps(
        {"segment_id": passage.segment_id, "text": passage.text},
        separators=(",", ":"),
        sort_keys=True,
    )


def _evidence_ids(items: Sequence[EvidenceItem]) -> tuple[str, ...]:
    return tuple(item.segment_id for item in items)


def _candidates(children: Sequence[SummaryNode]) -> tuple[tuple[str, bool], ...]:
    """Return unique source IDs in priority order, with mandatory support first."""
    ranked: list[tuple[str, bool]] = []
    for child in children:
        for annotation in child.contradictions:
            ranked.extend((identifier, True) for identifier in _evidence_ids(annotation.evidence))
    for child in children:
        for annotation in child.qualifications:
            ranked.extend((identifier, True) for identifier in _evidence_ids(annotation.evidence))
        for unit in child.content_units:
            if unit.uncertain or unit.qualification is not None:
                ranked.extend((identifier, True) for identifier in _evidence_ids(unit.evidence))
    for child in children:
        ranked.extend((identifier, False) for identifier in _evidence_ids(child.quotations))
    for child in children:
        for unit in child.content_units:
            ranked.extend((identifier, False) for identifier in _evidence_ids(unit.evidence))
    for child in children:
        ranked.extend((identifier, False) for identifier in child.provenance)

    selected: list[tuple[str, bool]] = []
    seen: set[str] = set()
    for identifier, mandatory in ranked:
        if identifier not in seen:
            selected.append((identifier, mandatory))
            seen.add(identifier)
    return tuple(selected)


def select_source_passages(
    children: Sequence[SummaryNode],
    *,
    source: Mapping[str, str],
    counter: TokenCounter,
    policy: GroundingPolicy,
    selection_cost: Callable[[tuple[SourcePassage, ...]], int] | None = None,
) -> GroundingSelection:
    """Pack complete authoritative source passages into a fixed token reserve.

    Model output supplies only candidate identifiers. The source mapping is the
    authority for both text and eligibility, so an unknown candidate fails
    before a request is built. Ambiguous or qualified support is mandatory;
    ordinary support may be omitted only after the reserve is exhausted.
    """
    if not children:
        raise ValueError("source grounding requires at least one child")

    candidates = _candidates(children)
    passages: list[SourcePassage] = []
    selected_ids: list[str] = []
    rejected: list[tuple[int, str]] = []
    for identifier, mandatory in candidates:
        if identifier not in source:
            raise ValueError(f"source text is missing for segment {identifier}")
        passage = SourcePassage(identifier, source[identifier])
        tentative = tuple((*passages, passage))
        cost = (
            selection_cost(tentative)
            if selection_cost is not None
            else counter.count("\n".join(serialize_source_passage(item) for item in tentative))
        )
        if cost <= policy.max_tokens:
            passages.append(passage)
            selected_ids.append(identifier)
        else:
            rejected.append((cost, identifier))
            if mandatory:
                raise BudgetError(
                    f"grounding reserve of {policy.max_tokens} tokens cannot hold "
                    f"mandatory source passage {identifier} costing {cost} tokens"
                )

    if not passages:
        smallest_cost, smallest_id = min(rejected)
        raise BudgetError(
            f"grounding reserve of {policy.max_tokens} tokens cannot hold source "
            f"passage {smallest_id} costing {smallest_cost} tokens"
        )
    omitted_ids = tuple(identifier for identifier, _ in candidates if identifier not in selected_ids)
    return GroundingSelection(
        passages=tuple(passages),
        selected_ids=tuple(selected_ids),
        omitted_ids=omitted_ids,
    )
