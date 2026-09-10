from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace

from summarizer.budget import BudgetError
from summarizer.cache import CacheDescriptor
from summarizer.grounding import GroundingPolicy, select_source_passages
from summarizer.leaf import derive_provenance, validate_provenance
from summarizer.merge import (
    MERGE_PROMPT_VERSION,
    build_merge_request,
    child_fence_tokens,
    measure_merge_overhead,
    measure_merge_request_tokens,
    parse_merged_summary,
    serialize_child,
    serialize_source_passage_block,
)
from summarizer.providers.base import GenerationRequest, ModelProvider
from summarizer.scheduler import BoundedScheduler, ScheduledWork
from summarizer.segmentation import CacheCoordinator
from summarizer.summaries import LEAF_SCHEMA_VERSION, SummaryNode
from summarizer.tokenization import TokenCounter


class HierarchyError(ValueError):
    """The summary tree could not be reduced to a single root."""


@dataclass(frozen=True)
class _PreparedMerge:
    """Frozen, independently executable work for one non-singleton merge."""

    node_id: str
    level: int
    order: int
    children: tuple[str, ...]
    covered_segments: tuple[str, ...]
    request: GenerationRequest
    legal: Mapping[str, str]
    descriptor: CacheDescriptor | None

    def decode(self, payload: object) -> SummaryNode:
        return parse_merged_summary(
            json.dumps(payload),
            legal=self.legal,
            subject=self.node_id,
            level=self.level,
        )

    def node(self, payload: object) -> TreeNode:
        return TreeNode(
            node_id=self.node_id,
            level=self.level,
            order=self.order,
            summary=self.decode(payload),
            children=self.children,
            covered_segments=self.covered_segments,
        )


DEFAULT_GROUNDING_POLICY = GroundingPolicy(max_tokens=1_024)


@dataclass(frozen=True)
class TreeNode:
    """One node of the summary tree, with the structure `SummaryNode` lacks.

    `SummaryNode` carries no identifier, no children, and no order, so the
    tree lives here. Order is explicit rather than implied by list position,
    which keeps ordering deterministic if merges later run concurrently.
    """

    node_id: str
    level: int
    order: int
    summary: SummaryNode
    children: tuple[str, ...]
    covered_segments: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.node_id.strip():
            raise ValueError("node_id must not be blank")
        if self.level < 0:
            raise ValueError("level must not be negative")
        if self.order < 0:
            raise ValueError("order must not be negative")
        if not self.covered_segments:
            raise ValueError("a node must cover at least one segment")


@dataclass(frozen=True)
class LevelReport:
    """What happened at one level of reduction."""

    level: int
    nodes_in: int
    nodes_out: int
    fanout: int
    reason: str


@dataclass(frozen=True)
class HierarchyReport:
    """The shape of the tree, and why it has that shape."""

    leaf_count: int
    level_count: int
    provider_calls: int
    levels: tuple[LevelReport, ...] = field(default_factory=tuple)


def measure_child_tokens(node: SummaryNode, counter: TokenCounter) -> int:
    """Measure what one child costs inside a merge request, delimiters included."""
    return counter.count(serialize_child(node)) + child_fence_tokens(counter)


def merge_fanout(
    children: Sequence[SummaryNode],
    counter: TokenCounter,
    *,
    capacity: int,
    ceiling: int | None = None,
) -> tuple[int, str]:
    """Derive how many children fit one merge request, and say why.

    Sized from the largest child rather than the average, so a group is never
    assembled that only fits on average.

    Raises `BudgetError` when the capacity cannot hold a *pair*, rather than
    returning two anyway: a merge of one is not a merge, and reporting two
    would assemble a request of twice the size it was sized against. That is
    reachable on the default local configuration, and on a provider that
    truncates an oversized prompt silently it would be undetectable content
    loss rather than an error.
    """
    if not children:
        raise ValueError("fanout requires at least one child")

    if ceiling is not None and ceiling < 2:
        raise ValueError("max_merge_children must be at least 2 to make progress")

    costs = [measure_child_tokens(child, counter) for child in children]
    largest = max(costs)
    measured = capacity // largest if largest else len(children)

    # A merge level cannot be narrower than a pair, so a capacity that admits
    # only one child admits no merge at all. Reporting a fanout of 2 here -
    # which an earlier revision did - assembles a request of twice the size it
    # was sized against, and on a provider that truncates silently that is
    # undetectable content loss rather than an error.
    if measured < 2:
        worst = costs.index(largest)
        raise BudgetError(
            f"a merge request cannot hold two summaries: child {worst} costs "
            f"{largest} tokens and a pair costs {2 * largest} against a "
            f"capacity of {capacity}"
        )

    if ceiling is not None and ceiling < measured:
        return ceiling, (
            f"the configured ceiling of {ceiling} is below the measured "
            f"fanout of {measured}"
        )
    return measured, (
        f"a largest child of {largest} tokens fits {measured} times in a "
        f"capacity of {capacity}"
    )


def group_children(count: int, fanout: int) -> tuple[tuple[int, ...], ...]:
    """Split `count` children into balanced, order-preserving groups.

    Balanced rather than greedily packed: greedy filling leaves a ragged final
    group, which compresses the end of a document less than the beginning -
    the uneven compression a balanced hierarchy is meant to avoid.
    """
    if count <= 0:
        raise ValueError("cannot group zero children")
    if fanout < 2:
        raise ValueError("fanout must be at least 2 to make progress")

    groups = -(-count // fanout)
    base, remainder = divmod(count, groups)
    indices = []
    start = 0
    for position in range(groups):
        size = base + (1 if position < remainder else 0)
        indices.append(tuple(range(start, start + size)))
        start += size
    return tuple(indices)


def build_hierarchy(
    leaves: Sequence[SummaryNode],
    provider: ModelProvider,
    counter: TokenCounter,
    *,
    source_id: str,
    covered: Sequence[Sequence[str]],
    attributable: Mapping[str, str],
    usable_tokens: int,
    model: str,
    timeout_seconds: float,
    max_merge_children: int | None = None,
    grounding_policy: GroundingPolicy | None = None,
    coordinator: CacheCoordinator | None = None,
) -> tuple[TreeNode, tuple[TreeNode, ...], HierarchyReport]:
    """Reduce ordered leaves to a single root through as many levels as needed.

    `covered` gives the segment identifiers each leaf covers, in the same
    order as `leaves`, and `attributable` maps each identifier to the text a
    quotation from it may be drawn from - injected rather than held in module
    state, so a run carries its own material and nothing leaks between runs.

    `usable_tokens` is the whole input budget for a request. The merge
    instructions, schema, and outer delimiters are subtracted here rather than
    by the caller, because the budget calculator measures a *leaf* request and
    a caller passing that figure straight through would under-reserve.

    Termination rests on a fanout of at least two, which `merge_fanout`
    guarantees by refusing a capacity that cannot hold a pair. The node count
    is additionally asserted to fall at each level, as a defensive invariant
    rather than the proof.
    """
    if not leaves:
        raise ValueError("a hierarchy requires at least one leaf")
    if len(covered) != len(leaves):
        raise ValueError("each leaf needs its covered segment identifiers")

    policy = grounding_policy or DEFAULT_GROUNDING_POLICY
    prepared_leaves: list[SummaryNode] = []
    prepared_covered: list[tuple[str, ...]] = []
    for index, (leaf, identifiers) in enumerate(zip(leaves, covered), start=1):
        local_covered = tuple(identifiers)
        missing = [
            identifier for identifier in local_covered if identifier not in attributable
        ]
        if missing:
            raise ValueError(
                "attributable text is missing for segments " + ", ".join(missing)
            )
        legal = {identifier: attributable[identifier] for identifier in local_covered}
        subject = f"L0N{index:04d}"
        validate_provenance(leaf, legal=legal, subject=subject)
        prepared_leaves.append(
            leaf.model_copy(
                update={"provenance": derive_provenance(leaf, source_order=local_covered)}
            )
        )
        prepared_covered.append(local_covered)

    all_nodes: list[TreeNode] = []
    current = [
        TreeNode(
            node_id=f"L0N{index + 1:04d}",
            level=0,
            order=index,
            summary=leaf,
            children=(),
            covered_segments=tuple(segments),
        )
        for index, (leaf, segments) in enumerate(
            zip(prepared_leaves, prepared_covered)
        )
    ]
    all_nodes.extend(current)

    levels: list[LevelReport] = []
    provider_calls = 0
    level = 0
    planned_merge_ids: list[str] = []
    if coordinator is not None and coordinator.session is not None:
        plan = coordinator.session.manifest.work_ids
        first_merge = next(
            (index for index, work_id in enumerate(plan) if work_id.startswith("L")),
            len(plan),
        )
        merge_prefix = plan[:first_merge]
    else:
        merge_prefix = ()

    while len(current) > 1:
        level += 1
        overhead = measure_merge_overhead(counter, level=level)
        child_capacity = usable_tokens - overhead - policy.max_tokens
        if child_capacity <= 0:
            raise BudgetError(
                f"no room for generated children in a grounded merge request at "
                f"level {level}: {usable_tokens} usable tokens leave no room after "
                f"{overhead} tokens of merge overhead and {policy.max_tokens} "
                "tokens reserved for source grounding"
            )
        fanout, reason = merge_fanout(
            [node.summary for node in current],
            counter,
            capacity=child_capacity,
            ceiling=max_merge_children,
        )
        groups = group_children(len(current), fanout)
        prepared: list[_PreparedMerge] = []
        passthrough: dict[int, TreeNode] = {}
        for order, indices in enumerate(groups):
            members = [current[index] for index in indices]
            if len(members) == 1:
                # Pass a lone node upward without a call; the count still
                # falls because other groups merged.
                only = members[0]
                passthrough[order] = TreeNode(
                    node_id=f"L{level}N{order + 1:04d}",
                    level=level,
                    order=order,
                    # Restamped: the merged path asserts that a node's
                    # summary reports its own level, and this path is fed
                    # into the next level's payload, so a stale value
                    # would show the model children at mixed levels.
                    summary=only.summary.model_copy(update={"level": level}),
                    children=(only.node_id,),
                    covered_segments=only.covered_segments,
                )
                continue
            prepared.append(
                _prepare_merge(
                    members,
                    attributable=attributable,
                    level=level,
                    order=order,
                    source_id=source_id,
                    model=model,
                    timeout_seconds=timeout_seconds,
                    counter=counter,
                    usable_tokens=usable_tokens,
                    grounding_policy=policy,
                    coordinator=coordinator,
                )
            )

        # Every request, descriptor, and legal grounding scope is frozen before
        # a sibling can call the provider. The manifest therefore witnesses the
        # entire level before its first externally visible side effect.
        if coordinator is not None and coordinator.session is not None:
            planned_merge_ids.extend(item.node_id for item in prepared)
            coordinator.session.ensure_work_prefix((*merge_prefix, *planned_merge_ids))
            descriptors = {item.node_id: item.descriptor for item in prepared}
            assert all(descriptor is not None for descriptor in descriptors.values())
            values = coordinator.reusable_batch(
                work_ids=tuple(item.node_id for item in prepared),
                descriptors=descriptors,
                validators={
                    item.node_id: (
                        lambda payload, item=item: item.decode(payload).model_dump(
                            mode="json"
                        )
                    )
                    for item in prepared
                },
            )
            misses = tuple(item for item in prepared if item.node_id not in values)
            scheduled = BoundedScheduler(
                max_in_flight=coordinator.max_in_flight,
                cache=coordinator.store,
            ).run(
                tuple(
                    ScheduledWork(
                        descriptor=item.descriptor,
                        operation=lambda item=item: _execute_prepared_merge(
                            item, provider
                        ),
                        validate=lambda payload, item=item: item.decode(
                            payload
                        ).model_dump(mode="json"),
                    )
                    for item in misses
                ),
                coordinator.session,
            )
            values.update({result.work_id: result.payload for result in scheduled})
            provider_calls += len(misses)
        else:
            values = {}
            for item in prepared:
                values[item.node_id] = _execute_prepared_merge(item, provider)
            provider_calls += len(prepared)

        produced = []
        by_order = {item.order: item for item in prepared}
        for order in range(len(groups)):
            if order in passthrough:
                produced.append(passthrough[order])
            else:
                item = by_order[order]
                produced.append(item.node(values[item.node_id]))

        if len(produced) >= len(current):
            raise HierarchyError(
                f"level {level} did not reduce the tree: {len(current)} nodes "
                f"produced {len(produced)} with a fanout of {fanout}"
            )

        levels.append(
            LevelReport(
                level=level,
                nodes_in=len(current),
                nodes_out=len(produced),
                fanout=fanout,
                reason=reason,
            )
        )
        all_nodes.extend(produced)
        current = produced

    report = HierarchyReport(
        leaf_count=len(leaves),
        level_count=level,
        provider_calls=provider_calls,
        levels=tuple(levels),
    )
    return current[0], tuple(all_nodes), report


def _prepare_merge(
    members: Sequence[TreeNode],
    *,
    attributable: Mapping[str, str],
    level: int,
    order: int,
    source_id: str,
    model: str,
    timeout_seconds: float,
    counter: TokenCounter,
    usable_tokens: int,
    grounding_policy: GroundingPolicy,
    coordinator: CacheCoordinator | None,
) -> _PreparedMerge:
    node_id = f"L{level}N{order + 1:04d}"
    # A union in document order: deduplicated, first occurrence wins. Three
    # documents call this a union, and a caller supplying overlapping coverage
    # would otherwise store duplicates for issue #8 to narrow.
    covered = tuple(
        dict.fromkeys(
            identifier
            for member in members
            for identifier in member.covered_segments
        )
    )
    missing = sorted(
        identifier for identifier in covered if identifier not in attributable
    )
    if missing:
        raise ValueError(
            "attributable text is missing for segments "
            f"{', '.join(missing)}"
        )
    legal = {identifier: attributable[identifier] for identifier in covered}

    selection = select_source_passages(
        [member.summary for member in members],
        source=legal,
        counter=counter,
        policy=grounding_policy,
        selection_cost=lambda passages: counter.count(
            "\n".join(
                serialize_source_passage_block(
                    passage,
                    source_id=source_id,
                    level=level,
                    ordinal=ordinal,
                )
                for ordinal, passage in enumerate(passages)
            )
        ),
    )
    grounded = {passage.segment_id: passage.text for passage in selection.passages}
    request = build_merge_request(
        [member.summary for member in members],
        passages=selection.passages,
        level=level,
        source_id=source_id,
        model=model,
        timeout_seconds=timeout_seconds,
    )
    request = replace(request, audit_work_id=node_id)
    request_tokens = measure_merge_request_tokens(request, counter)
    if request_tokens > usable_tokens:
        raise BudgetError(
            f"grounded merge request at {node_id} costs {request_tokens} tokens "
            f"against a usable capacity of {usable_tokens}"
        )
    descriptor = None
    if coordinator is not None and coordinator.session is not None:
        descriptor = coordinator.descriptor_for(
            stage="merge",
            work_id=node_id,
            prompt_version=MERGE_PROMPT_VERSION,
            schema_version=LEAF_SCHEMA_VERSION,
            input_value={
                "child_node_ids": [member.node_id for member in members],
                "covered_segment_ids": list(covered),
                "grounding_source_ids": [
                    passage.segment_id for passage in selection.passages
                ],
                "instructions": request.instructions,
                "input_text": request.input_text,
                "schema": request.response_schema,
                "grounding_max_tokens": grounding_policy.max_tokens,
                "usable_tokens": usable_tokens,
            },
            behavior={
                "grounding": {"max_tokens": grounding_policy.max_tokens},
                "grounding_policy": "grounding/1",
            },
        )
    return _PreparedMerge(
        node_id=node_id,
        level=level,
        order=order,
        children=tuple(member.node_id for member in members),
        covered_segments=covered,
        request=request,
        legal=grounded,
        descriptor=descriptor,
    )


def _execute_prepared_merge(
    prepared: _PreparedMerge, provider: ModelProvider
) -> object:
    """Perform only the provider call and pre-existing response validation."""
    result = provider.generate(prepared.request)
    return parse_merged_summary(
        result.text,
        legal=prepared.legal,
        subject=prepared.node_id,
        level=prepared.level,
    ).model_dump(mode="json")
