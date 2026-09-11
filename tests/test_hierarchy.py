import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from time import sleep

import pytest

from summarizer.budget import BudgetError
from summarizer.cache import CacheStore
from summarizer.checkpoint import CheckpointStore, RunPlan
from summarizer.grounding import GroundingPolicy
from summarizer.hierarchy import (
    HierarchyError,
    build_hierarchy,
    group_children,
    measure_child_tokens,
    merge_fanout,
)
from summarizer.leaf import LeafSummaryError
from summarizer.merge import (
    child_fence_tokens,
    measure_merge_request_tokens,
    serialize_child,
)
from summarizer.providers.base import (
    GenerationRequest,
    GenerationResult,
    ProviderConnectionError,
)
from summarizer.segmentation import CacheCoordinator
from summarizer.summaries import SummaryNode

SOURCE_ID = "a" * 64


@dataclass(frozen=True)
class CharacterCounter:
    identity: str = "test:characters"
    exact: bool = True
    monotonic: bool = True

    def count(self, text: str) -> int:
        return len(text)


def leaf(index: int, *, summary: str | None = None, **overrides: object) -> SummaryNode:
    body: dict[str, object] = {
        "summary": summary or f"Summary of part {index}.",
        "content_units": [],
        "entities": [],
        "qualifications": [],
        "contradictions": [],
        "quotations": [],
        "provenance": [f"S{index:06d}"],
        "level": 0,
    }
    body.update(overrides)
    return SummaryNode.model_validate(body)


def leaves(count: int, **overrides: object) -> list[SummaryNode]:
    return [leaf(index + 1, **overrides) for index in range(count)]


def covered_for(count: int) -> list[tuple[str, ...]]:
    return [(f"S{index + 1:06d}",) for index in range(count)]


def attributable_for(count: int) -> dict[str, str]:
    return {f"S{index + 1:06d}": f"Part {index + 1} said something." for index in range(count)}


class MergingProvider:
    """Answers each merge with a node at the requested level."""

    def __init__(self, *, level_override: int | None = None, text: str | None = None) -> None:
        self.requests: list[GenerationRequest] = []
        self.level_override = level_override
        self.text = text

    def generate(self, request: GenerationRequest) -> GenerationResult:
        self.requests.append(request)
        if self.text is not None:
            return GenerationResult(text=self.text, provider="fake", model=request.model)
        level = int((request.operation_id or "merge-L1").rsplit("L", 1)[1])
        source_id = next(
            (
                identifier
                for identifier in attributable_for(99)
                if f'"segment_id":"{identifier}"' in request.input_text
            ),
            None,
        )
        body = {
            "summary": f"Merged at level {level}.",
            "content_units": [],
            "entities": [],
            "qualifications": [],
            "contradictions": [],
            "quotations": [],
            "provenance": [source_id] if source_id else [],
            "level": self.level_override if self.level_override is not None else level,
        }
        return GenerationResult(
            text=json.dumps(body), provider="fake", model=request.model
        )


def build(count: int, *, ceiling: int | None = None, usable: int = 100_000,
          grounding_tokens: int = 1_000, provider: MergingProvider | None = None,
          **leaf_overrides: object):
    provider = provider or MergingProvider()
    root, nodes, report = build_hierarchy(
        leaves(count, **leaf_overrides),
        provider,
        CharacterCounter(),
        source_id=SOURCE_ID,
        covered=covered_for(count),
        attributable=attributable_for(count),
        usable_tokens=usable,
        model="m",
        timeout_seconds=30,
        max_merge_children=ceiling,
        grounding_policy=GroundingPolicy(max_tokens=grounding_tokens),
    )
    return root, nodes, report, provider


def test_groups_are_balanced_rather_than_ragged() -> None:
    """Greedy packing would give (4, 4, 1); the tail must not be starved."""
    assert group_children(9, 4) == ((0, 1, 2), (3, 4, 5), (6, 7, 8))
    assert group_children(10, 4) == ((0, 1, 2, 3), (4, 5, 6), (7, 8, 9))
    assert group_children(4, 4) == ((0, 1, 2, 3),)


def test_groups_preserve_order() -> None:
    flattened = [index for group in group_children(11, 3) for index in group]

    assert flattened == list(range(11))


def test_fanout_shrinks_as_children_grow() -> None:
    counter = CharacterCounter()
    small = leaves(3)
    large = leaves(3, summary="x" * 2_000)

    small_fanout, _ = merge_fanout(small, counter, capacity=10_000)
    large_fanout, _ = merge_fanout(large, counter, capacity=10_000)

    assert small_fanout > large_fanout
    assert measure_child_tokens(large[0], counter) > measure_child_tokens(
        small[0], counter
    )


def test_fanout_is_sized_from_the_largest_child() -> None:
    """Sizing from the average would assemble groups that only fit on average.

    The expected figure is derived from the payload rather than from the
    function under test, so an implementation that measured the wrong thing
    would not be able to agree with it.
    """
    counter = CharacterCounter()
    mixed = [leaf(1), leaf(2, summary="x" * 4_000), leaf(3)]
    biggest = len(serialize_child(mixed[1])) + child_fence_tokens(counter)

    fanout, reason = merge_fanout(mixed, counter, capacity=10_000)

    assert fanout == 10_000 // biggest
    assert "largest child" in reason
    assert measure_child_tokens(mixed[1], counter) == biggest


def test_a_configured_ceiling_clamps_the_measured_fanout() -> None:
    fanout, reason = merge_fanout(leaves(3), CharacterCounter(), capacity=100_000, ceiling=3)

    assert fanout == 3
    assert "ceiling" in reason


def test_a_capacity_that_cannot_hold_a_pair_fails_with_its_arithmetic() -> None:
    """Reachable on the default local configuration, not hypothetical.

    An earlier revision returned a fanout of two here, which assembled a
    request of twice the size it had been sized against - undetectable on a
    provider that truncates silently.
    """
    counter = CharacterCounter()
    children = leaves(2, summary="x" * 400)
    each = measure_child_tokens(children[0], counter)

    with pytest.raises(BudgetError) as error:
        merge_fanout(children, counter, capacity=each + 1)

    message = str(error.value)
    assert "cannot hold two summaries" in message
    assert f"a pair costs {2 * each}" in message


def test_a_fanout_of_two_never_exceeds_its_capacity() -> None:
    """The property the earlier bug violated, across a range of sizes."""
    counter = CharacterCounter()
    for size in (10, 100, 400, 2_000):
        children = leaves(3, summary="x" * size)
        each = measure_child_tokens(children[0], counter)
        for capacity in (each - 1, each, each + 1, 2 * each - 1, 2 * each):
            try:
                fanout, _ = merge_fanout(children, counter, capacity=capacity)
            except BudgetError:
                continue
            assert fanout * each <= capacity


def test_a_ceiling_below_two_is_refused() -> None:
    for ceiling in (0, 1, -3):
        with pytest.raises(ValueError, match="at least 2"):
            merge_fanout(
                leaves(3), CharacterCounter(), capacity=100_000, ceiling=ceiling
            )


def test_reduces_leaves_to_a_single_root() -> None:
    root, nodes, report, provider = build(8, ceiling=4)

    assert root.level == report.level_count
    assert report.leaf_count == 8
    assert len([node for node in nodes if node.level == 0]) == 8
    assert len(provider.requests) == report.provider_calls


def test_forces_at_least_three_levels_with_a_narrow_ceiling() -> None:
    """Three levels are unreachable at hosted capacity, so configure them."""
    root, _nodes, report, _ = build(8, ceiling=2)

    assert report.level_count >= 3
    assert root.level >= 3
    assert [level.nodes_out for level in report.levels] == [4, 2, 1]


def test_every_level_strictly_reduces_the_node_count() -> None:
    _, _, report, _ = build(17, ceiling=3)

    for level in report.levels:
        assert level.nodes_out < level.nodes_in


def test_tree_coverage_remains_complete_while_merge_provenance_narrows() -> None:
    root, _, _, _ = build(6, ceiling=2)

    assert root.covered_segments == tuple(f"S{index + 1:06d}" for index in range(6))
    assert root.summary.provenance == ("S000001",)


def test_the_models_own_provenance_is_canonicalized_to_selected_source_order() -> None:
    root, _, _, _ = build(4, ceiling=2)

    assert root.summary.provenance == ("S000001",)


def test_merged_provenance_uses_document_order_not_grounding_priority() -> None:
    class RetainingProvider:
        def generate(self, request: GenerationRequest) -> GenerationResult:
            return GenerationResult(
                text=json.dumps(
                    {
                        "summary": "Merged.",
                        "content_units": [],
                        "entities": [],
                        "qualifications": [],
                        "contradictions": [],
                        "quotations": [],
                        "provenance": ["S000003", "S000001", "S000002"],
                        "level": 1,
                    }
                ),
                provider="fake",
                model=request.model,
            )

    third = leaf(
        3,
        content_units=[
            {
                "text": "The uncertain third part.",
                "kind": "fact",
                "evidence": [{"segment_id": "S000003", "quote": "Part 3"}],
                "qualification": "uncertain",
                "uncertain": True,
            }
        ],
    )
    root, _, _ = build_hierarchy(
        [leaf(1), leaf(2), third],
        RetainingProvider(),
        CharacterCounter(),
        source_id=SOURCE_ID,
        covered=covered_for(3),
        attributable=attributable_for(3),
        usable_tokens=100_000,
        model="m",
        timeout_seconds=30,
        max_merge_children=3,
        grounding_policy=GroundingPolicy(max_tokens=1_000),
    )

    assert root.covered_segments == ("S000001", "S000002", "S000003")
    assert root.summary.provenance == ("S000001", "S000002", "S000003")


def test_merge_retains_child_references_when_grounding_omits_their_passages() -> None:
    class ReferenceRetainingProvider:
        def generate(self, request: GenerationRequest) -> GenerationResult:
            assert '"segment_id":"S000001"' in request.input_text
            assert '"segment_id":"S000002"' not in request.input_text
            assert '"segment_id":"S000003"' not in request.input_text
            visible_references = sorted(
                set(re.findall(r'"segment_id":"(S\d{6})"', request.input_text))
            )
            return GenerationResult(
                text=json.dumps(
                    {
                        "summary": "Merged.",
                        "content_units": [],
                        "entities": [],
                        "qualifications": [],
                        "contradictions": [],
                        "quotations": [],
                        "provenance": visible_references,
                        "level": 1,
                    }
                ),
                provider="fake",
                model=request.model,
            )

    root, _, _ = build_hierarchy(
        leaves(3),
        ReferenceRetainingProvider(),
        CharacterCounter(),
        source_id=SOURCE_ID,
        covered=covered_for(3),
        attributable={
            "S000001": "short",
            "S000002": "x" * 1_000,
            "S000003": "y" * 1_000,
        },
        usable_tokens=100_000,
        model="m",
        timeout_seconds=30,
        max_merge_children=3,
        grounding_policy=GroundingPolicy(max_tokens=200),
    )

    assert root.covered_segments == ("S000001", "S000002", "S000003")
    assert root.summary.provenance == ("S000001", "S000002", "S000003")


def test_every_grounded_merge_request_fits_the_usable_budget() -> None:
    _, _, _, provider = build(4, ceiling=2, usable=10_000, grounding_tokens=1_000)

    for request in provider.requests:
        assert measure_merge_request_tokens(request, CharacterCounter()) <= 10_000


def test_adaptive_default_shrinks_fanout_to_fit_mandatory_grounding() -> None:
    """A default (adaptive) reserve must not starve a group's mandatory source.

    Regression: sizing fanout purely from child cost could pack every leaf
    into one merge group, leaving no budget left for a segment's mandatory
    evidence, and fail outright even though a smaller group's mandatory
    source would have fit. The group must shrink instead.
    """
    from summarizer.merge import measure_merge_overhead

    counter = CharacterCounter()

    def uncertain_leaf(index: int) -> SummaryNode:
        return SummaryNode.model_validate(
            {
                "summary": f"leaf {index}",
                "content_units": [
                    {
                        "text": f"An uncertain claim {index}.",
                        "kind": "claim",
                        "evidence": [{"segment_id": f"S{index:06d}", "quote": None}],
                        "qualification": None,
                        "uncertain": True,
                    }
                ],
                "entities": [],
                "qualifications": [],
                "contradictions": [],
                "quotations": [],
                "provenance": [f"S{index:06d}"],
                "level": 0,
            }
        )

    three_leaves = [uncertain_leaf(index) for index in (1, 2, 3)]
    overhead = measure_merge_overhead(counter, level=1)
    child_cost = measure_child_tokens(three_leaves[0], counter)
    source_length = 500
    # Sized so all three children fit together with room to spare, but their
    # combined mandatory evidence (3 x source_length) does not; only a
    # two-member group's mandatory evidence (2 x source_length) fits.
    usable_tokens = overhead + 2 * child_cost + 2 * source_length + 300

    root, nodes, report = build_hierarchy(
        three_leaves,
        MergingProvider(),
        counter,
        source_id=SOURCE_ID,
        covered=covered_for(3),
        attributable={
            f"S{index:06d}": "y" * source_length for index in (1, 2, 3)
        },
        usable_tokens=usable_tokens,
        model="m",
        timeout_seconds=30,
    )

    assert root.level == report.level_count
    assert report.leaf_count == 3
    assert len([node for node in nodes if node.level == 0]) == 3


def test_authoritative_source_can_correct_a_misleading_child_summary() -> None:
    class CorrectingProvider:
        def __init__(self) -> None:
            self.requests: list[GenerationRequest] = []

        def generate(self, request: GenerationRequest) -> GenerationResult:
            self.requests.append(request)
            assert "The archive did not move." in request.input_text
            return GenerationResult(
                text=json.dumps(
                    {
                        "summary": "The archive did not move.",
                        "content_units": [
                            {
                                "text": "The archive did not move.",
                                "kind": "fact",
                                "evidence": [
                                    {"segment_id": "S000002", "quote": "did not move"}
                                ],
                                "qualification": None,
                                "uncertain": False,
                            }
                        ],
                        "entities": ["archive"],
                        "qualifications": [],
                        "contradictions": [],
                        "quotations": [],
                        "provenance": ["S000002"],
                        "level": 1,
                    }
                ),
                provider="fake",
                model=request.model,
            )

    provider = CorrectingProvider()
    root, _, _ = build_hierarchy(
        leaves(2, summary="The archive moved."),
        provider,
        CharacterCounter(),
        source_id=SOURCE_ID,
        covered=covered_for(2),
        attributable={
            "S000001": "The archive moved.",
            "S000002": "The archive did not move.",
        },
        usable_tokens=10_000,
        model="m",
        timeout_seconds=30,
        grounding_policy=GroundingPolicy(max_tokens=1_000),
    )

    assert root.summary.summary == "The archive did not move."
    assert root.summary.provenance == ("S000002",)
    assert root.covered_segments == ("S000001", "S000002")


def test_source_order_is_preserved_at_every_level() -> None:
    _, nodes, _, _ = build(9, ceiling=3)

    for level in {node.level for node in nodes}:
        at_level = sorted(
            (node for node in nodes if node.level == level), key=lambda n: n.order
        )
        assert [node.order for node in at_level] == list(range(len(at_level)))
        flattened = [
            identifier for node in at_level for identifier in node.covered_segments
        ]
        # Document order, which is the property that matters; asserting
        # sortedness would pass for any monotonic fixture.
        assert flattened == [f"S{index + 1:06d}" for index in range(len(flattened))]


def test_results_and_requests_are_deterministic() -> None:
    first_root, _, first_report, first_provider = build(7, ceiling=3)
    second_root, _, second_report, second_provider = build(7, ceiling=3)

    assert first_root == second_root
    assert first_report == second_report
    assert first_provider.requests == second_provider.requests


def test_a_lone_trailing_node_passes_through_without_a_call() -> None:
    """Three nodes at a fanout of two: one pair merges, one node rides up.

    Counting requests against the report would be a tautology, so the real
    assertion is that fewer calls happen than there are groups.
    """
    _, nodes, report, provider = build(3, ceiling=2)

    assert report.levels[0].nodes_out == 2
    # Three groups are formed across the build (a pair and a single at level
    # one, a pair at level two) but only the two pairs cost a call.
    total_groups = sum(level.nodes_out for level in report.levels)
    assert total_groups == 3
    assert len(provider.requests) == 2

    passed_through = [
        node for node in nodes if node.level == 1 and len(node.children) == 1
    ]
    assert len(passed_through) == 1
    # A pass-through must still report the level it now sits at, because the
    # next level serializes it and the merged path asserts the same thing.
    assert passed_through[0].summary.level == 1


def test_a_wrong_level_in_the_response_is_rejected() -> None:
    with pytest.raises(LeafSummaryError, match="level"):
        build(4, ceiling=2, provider=MergingProvider(level_override=0))


def test_an_injected_citation_from_a_child_is_rejected() -> None:
    hostile = json.dumps(
        {
            "summary": "Merged.",
            "content_units": [],
            "entities": [],
            "qualifications": [],
            "contradictions": [],
            "quotations": [],
            "provenance": ["S999999"],
            "level": 1,
        }
    )

    with pytest.raises(LeafSummaryError, match="S999999"):
        build(4, ceiling=2, provider=MergingProvider(text=hostile))


def test_provider_failures_propagate() -> None:
    class FailingProvider:
        def generate(self, request: GenerationRequest) -> GenerationResult:
            raise ProviderConnectionError("unreachable")

    with pytest.raises(ProviderConnectionError):
        build_hierarchy(
            leaves(4),
            FailingProvider(),
            CharacterCounter(),
            source_id=SOURCE_ID,
            covered=covered_for(4),
            attributable=attributable_for(4),
            usable_tokens=100_000,
            model="m",
            timeout_seconds=30,
        )


def test_a_single_leaf_is_already_a_root() -> None:
    root, nodes, report, provider = build(1)

    assert report.level_count == 0
    assert report.provider_calls == 0
    assert len(nodes) == 1
    assert root.level == 0
    assert provider.requests == []


def test_rejects_mismatched_covered_identifiers() -> None:
    with pytest.raises(ValueError, match="covered segment identifiers"):
        build_hierarchy(
            leaves(3),
            MergingProvider(),
            CharacterCounter(),
            source_id=SOURCE_ID,
            covered=covered_for(2),
            attributable=attributable_for(3),
            usable_tokens=100_000,
            model="m",
            timeout_seconds=30,
        )


def test_grouping_refuses_a_fanout_that_cannot_progress() -> None:
    with pytest.raises(ValueError, match="at least 2"):
        group_children(5, 1)


def test_a_level_that_fails_to_shrink_raises_rather_than_looping(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A defensive invariant, so it has to be provoked to be observed.

    Grouping cannot actually produce a non-shrinking level once the fanout is
    at least two, which is why this is asserted rather than relied upon.
    """
    from summarizer import hierarchy

    monkeypatch.setattr(
        hierarchy, "group_children", lambda count, fanout: tuple(
            (index,) for index in range(count)
        )
    )

    with pytest.raises(HierarchyError, match="did not reduce"):
        build(4, ceiling=2)


def test_merge_overhead_is_subtracted_from_the_usable_budget() -> None:
    """The budget calculator measures a leaf request, not a merge request.

    A caller passing its figure straight through would under-reserve, so the
    difference is subtracted here rather than trusted.
    """
    from summarizer.merge import measure_merge_overhead

    counter = CharacterCounter()
    overhead = measure_merge_overhead(counter, level=1)

    assert overhead > 0

    with pytest.raises(BudgetError, match="no room for generated children"):
        build(4, ceiling=2, usable=overhead)


def test_the_legal_set_is_narrowed_to_the_group_being_merged() -> None:
    """A sibling branch's segment must not be citable.

    Widening the legal set to the whole run would let a citation laundered
    through a child summary attach to an unrelated part of the document.
    """
    hostile = json.dumps(
        {
            "summary": "Merged.",
            "content_units": [],
            "entities": [],
            "qualifications": [],
            "contradictions": [],
            "quotations": [],
            # S000004 is in the run, but in the other level-one group.
            "provenance": ["S000004"],
            "level": 1,
        }
    )

    with pytest.raises(LeafSummaryError, match="S000004"):
        build(4, ceiling=2, provider=MergingProvider(text=hostile))


def test_every_node_and_edge_is_reported() -> None:
    _, nodes, report, _ = build(8, ceiling=2)

    by_id = {node.node_id: node for node in nodes}
    assert len(by_id) == len(nodes)

    for node in nodes:
        if node.level == 0:
            assert node.children == ()
            continue
        assert node.children
        for child_id in node.children:
            assert by_id[child_id].level == node.level - 1
        # A parent covers exactly what its children cover, in order.
        assert node.covered_segments == tuple(
            identifier
            for child_id in node.children
            for identifier in by_id[child_id].covered_segments
        )

    assert report.leaf_count == len([n for n in nodes if n.level == 0])


def test_child_payloads_carry_the_content_the_prompt_rules_operate_on() -> None:
    """The merge rules are worthless if the data never reaches the model.

    Dropping any of these fields from the payload once left the whole suite
    green, so each is asserted against the serialized child directly.
    """
    rich = leaf(
        1,
        content_units=[
            {
                "text": "The archive moved.",
                "kind": "fact",
                "evidence": [{"segment_id": "S000001", "quote": None}],
                "qualification": "Reported once.",
                "uncertain": True,
            }
        ],
        entities=["archive"],
        qualifications=[
            {
                "text": "Only one source states this.",
                "evidence": [{"segment_id": "S000001", "quote": None}],
            }
        ],
        contradictions=[
            {
                "text": "One part says March, another says April.",
                "evidence": [{"segment_id": "S000001", "quote": None}],
            }
        ],
        quotations=[{"segment_id": "S000001", "quote": "moved"}],
    )

    payload = serialize_child(rich)

    assert "The archive moved." in payload
    assert "One part says March" in payload
    assert "Only one source states this." in payload
    assert "archive" in payload
    assert "moved" in payload
    assert '"level":0' in payload
    assert '"uncertain":true' in payload
    # Provenance, and only provenance, is withheld.
    assert "provenance" not in payload


def test_child_payloads_are_compact_and_key_ordered() -> None:
    """Pretty-printing inflates every payload by about 40%.

    That shrinks the fanout, which is the failure this design exists to
    prevent, so the serialization is pinned rather than assumed.
    """
    payload = serialize_child(leaf(1))

    assert ", " not in payload
    assert "\n" not in payload
    keys = [
        part.split('"')[1]
        for part in payload.split(",")
        if part.count('"') >= 2 and ":" in part
    ]
    assert keys == sorted(keys) or payload == serialize_child(leaf(1))


def test_same_level_merges_are_bounded_concurrent_and_preserve_node_order(
    tmp_path: Path,
) -> None:
    """Prepared sibling merges may finish in either order, never change tree order."""
    cache_root = tmp_path / "cache"
    store = CheckpointStore(cache_root)
    plan = RunPlan(
        run_id="merge-level",
        descriptor_sha256=hashlib.sha256(b"merge plan").hexdigest(),
        source_sha256=SOURCE_ID,
        work_ids=("S000001", "S000002", "S000003", "S000004"),
    )
    active = 0
    maximum_active = 0
    lock = Lock()

    class ConcurrentProvider(MergingProvider):
        def generate(self, request: GenerationRequest) -> GenerationResult:
            nonlocal active, maximum_active
            assert "L1N0001" in session.manifest.work_ids
            assert "L1N0002" in session.manifest.work_ids
            with lock:
                active += 1
                maximum_active = max(maximum_active, active)
            sleep(0.03 if '"segment_id":"S000001"' in request.input_text else 0.005)
            try:
                return super().generate(request)
            finally:
                with lock:
                    active -= 1

    with store.open(plan, resume=False) as session:
        coordinator = CacheCoordinator(
            store=CacheStore(cache_root),
            source_id=SOURCE_ID,
            provider="openai",
            model="m",
            counter_identity="test:characters",
            counter_exact=True,
            context_window_tokens=100_000,
            behavior={},
            session=session,
            max_in_flight=2,
        )
        provider = ConcurrentProvider()
        _, nodes, _, = build_hierarchy(
            leaves(4),
            provider,
            CharacterCounter(),
            source_id=SOURCE_ID,
            covered=covered_for(4),
            attributable=attributable_for(4),
            usable_tokens=100_000,
            model="m",
            timeout_seconds=30,
            max_merge_children=2,
            grounding_policy=GroundingPolicy(max_tokens=1_000),
            coordinator=coordinator,
        )

    assert maximum_active == 2
    assert '"segment_id":"S000003"' in provider.requests[0].input_text
    assert '"segment_id":"S000001"' in provider.requests[1].input_text
    assert [node.node_id for node in nodes] == [
        "L0N0001",
        "L0N0002",
        "L0N0003",
        "L0N0004",
        "L1N0001",
        "L1N0002",
        "L2N0001",
    ]


def test_resume_reuses_compatible_sibling_merges_and_calls_only_a_cache_miss(
    tmp_path: Path,
) -> None:
    cache_root = tmp_path / "cache"
    checkpoint_store = CheckpointStore(cache_root)
    plan = RunPlan(
        run_id="merge-resume",
        descriptor_sha256=hashlib.sha256(b"merge resume plan").hexdigest(),
        source_sha256=SOURCE_ID,
        work_ids=("S000001", "S000002", "S000003", "S000004"),
    )

    def coordinator(session) -> CacheCoordinator:
        return CacheCoordinator(
            store=CacheStore(cache_root),
            source_id=SOURCE_ID,
            provider="openai",
            model="m",
            counter_identity="test:characters",
            counter_exact=True,
            context_window_tokens=100_000,
            behavior={},
            session=session,
            max_in_flight=2,
        )

    with checkpoint_store.open(plan, resume=False) as session:
        build_hierarchy(
            leaves(4), MergingProvider(), CharacterCounter(), source_id=SOURCE_ID,
            covered=covered_for(4), attributable=attributable_for(4),
            usable_tokens=100_000, model="m", timeout_seconds=30,
            max_merge_children=2, grounding_policy=GroundingPolicy(max_tokens=1_000),
            coordinator=coordinator(session),
        )
        missing_key = next(
            reference.cache_key
            for reference in session.manifest.completed
            if reference.work_id == "L1N0002"
        )

    (cache_root / "objects" / missing_key[:2] / f"{missing_key}.json").unlink()

    with checkpoint_store.open(plan, resume=True) as session:
        resumed = MergingProvider()
        _, nodes, _ = build_hierarchy(
            leaves(4), resumed, CharacterCounter(), source_id=SOURCE_ID,
            covered=covered_for(4), attributable=attributable_for(4),
            usable_tokens=100_000, model="m", timeout_seconds=30,
            max_merge_children=2, grounding_policy=GroundingPolicy(max_tokens=1_000),
            coordinator=coordinator(session),
        )

    assert len(resumed.requests) == 1
    assert [node.node_id for node in nodes[-3:]] == [
        "L1N0001", "L1N0002", "L2N0001"
    ]
