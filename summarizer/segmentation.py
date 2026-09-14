"""Structure-aware source segmentation models."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import Enum
from typing import TypeVar

from nltk.tokenize import PunktSentenceTokenizer

from summarizer.cache import CacheDescriptor, CacheStore
from summarizer.checkpoint import CheckpointSession, CompletedRef
from summarizer.ingestion import SourceDocument
from summarizer.reliability import ReliabilityTracker
from summarizer.tokenization import (
    PrefixTokenCounter,
    SuffixTokenCounter,
    TokenAccountingError,
    TokenCounter,
)


class BoundaryKind(str, Enum):
    HEADING = "heading"
    PARAGRAPH = "paragraph"
    LIST = "list"
    SENTENCE = "sentence"
    HARD = "hard"
    # A unit that is the entire document has no boundary *within* a document.
    # Segmentation never produces this kind; the direct path does, and it tells
    # a later stage that provenance covers everything rather than a region.
    DOCUMENT = "document"
    # An opaque fenced code block (``` ... ```); heading/list detection is
    # suppressed for lines inside a fence.
    CODE_FENCE = "code_fence"


class SegmentationError(ValueError):
    """Raised when segmentation cannot make budget-compliant progress."""


_Cached = TypeVar("_Cached")
SEGMENTATION_CACHE_VERSION = "segmentation/1"


@dataclass(frozen=True)
class CacheCoordinator:
    """Keep cache mechanics out of prompt and provider boundaries."""

    store: CacheStore
    source_id: str
    provider: str
    model: str
    counter_identity: str
    counter_exact: bool
    context_window_tokens: int
    behavior: Mapping[str, object]
    session: CheckpointSession | None = None
    allow_unreferenced_cache: bool = True
    max_in_flight: int = 1
    reliability_tracker: ReliabilityTracker | None = None

    def descriptor_for(
        self, *, stage: str, work_id: str, prompt_version: str, schema_version: str,
        input_value: object, behavior: Mapping[str, object],
    ) -> CacheDescriptor:
        encoded_input = json.dumps(input_value, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode("utf-8")
        return CacheDescriptor(
            source_id=self.source_id, input_hash=hashlib.sha256(encoded_input).hexdigest(),
            stage=stage, work_id=work_id, prompt_version=prompt_version,
            schema_version=schema_version, provider=self.provider, model=self.model,
            counter_identity=self.counter_identity, counter_exact=self.counter_exact,
            context_window_tokens=self.context_window_tokens,
            behavior={**self.behavior, **behavior},
        )

    def resolve(
        self,
        *,
        stage: str,
        work_id: str,
        prompt_version: str,
        schema_version: str,
        input_value: object,
        behavior: Mapping[str, object],
        decode: Callable[[object], _Cached],
        encode: Callable[[_Cached], object],
        compute: Callable[[], _Cached],
        cache_if: Callable[[_Cached], bool] = lambda _: True,
    ) -> _Cached:
        descriptor = self.descriptor_for(stage=stage, work_id=work_id, prompt_version=prompt_version, schema_version=schema_version, input_value=input_value, behavior=behavior)
        invalidation_reasons = self.store.invalidation_reasons(descriptor)

        def validate(payload: object) -> object:
            return encode(decode(payload))

        if self.session is not None:
            reusable = self.session.reusable_for(
                work_ids=(work_id,),
                descriptors={work_id: descriptor},
                validators={work_id: validate},
            )[0]
            if reusable.payload is not None:
                self._record_cache_hit(work_id)
                self.store.record_descriptor_projection(descriptor)
                return decode(reusable.payload)
            miss_reason = (
                reusable.reason.value if reusable.reason is not None else "missing"
            )
        else:
            miss_reason = "missing"
        if self.allow_unreferenced_cache:
            cached = self.store.load(descriptor, validate)
            if cached.hit:
                result = decode(cached.payload)
                self._checkpoint(descriptor)
                self._record_cache_hit(work_id)
                self.store.record_descriptor_projection(descriptor)
                return result
            if cached.miss_reason is not None:
                miss_reason = cached.miss_reason.value
        self._record_cache_miss(work_id, miss_reason)
        self._record_invalidation_reasons(invalidation_reasons)
        result = compute()
        if cache_if(result):
            result = decode(
                self.store.store_winner(descriptor, encode(result), validate)
            )
            self.store.record_descriptor_projection(descriptor)
            self._checkpoint(descriptor)
        return result

    def reusable_batch(
        self,
        *,
        work_ids: tuple[str, ...],
        descriptors: Mapping[str, CacheDescriptor],
        validators: Mapping[str, Callable[[object], object]],
    ) -> dict[str, object]:
        """Return validated batch hits and make new-run hits resumable first.

        A run manifest is the only reuse authority while resuming.  A new run
        may additionally adopt compatible cache objects, but adopts them into
        its manifest before any sibling request is submitted.
        """
        hits: dict[str, object] = {}
        misses: dict[str, str] = {}
        if self.session is not None:
            reusable_results = self.session.reusable_for(
                work_ids=work_ids,
                descriptors=descriptors,
                validators=validators,
            )
            for work_id, reusable in zip(work_ids, reusable_results):
                if reusable.reference is not None and reusable.payload is not None:
                    hits[reusable.reference.work_id] = reusable.payload
                    self._record_cache_hit(work_id)
                    self.store.record_descriptor_projection(descriptors[work_id])
                elif reusable.reason is not None:
                    misses[work_id] = reusable.reason.value

        if not self.allow_unreferenced_cache:
            for work_id in work_ids:
                if work_id not in hits:
                    self._record_cache_miss(work_id, misses.get(work_id, "missing"))
                    self._record_invalidation_reasons(
                        self.store.invalidation_reasons(descriptors[work_id])
                    )
            return hits

        adopted: list[CompletedRef] = []
        for work_id in work_ids:
            if work_id in hits:
                continue
            descriptor = descriptors[work_id]
            cached = self.store.load(descriptor, validators[work_id])
            if cached.hit:
                hits[work_id] = cached.payload
                self._record_cache_hit(work_id)
                self.store.record_descriptor_projection(descriptor)
                adopted.append(
                    CompletedRef(work_id=work_id, cache_key=descriptor.key)
                )
            elif cached.miss_reason is not None:
                misses[work_id] = cached.miss_reason.value

        if adopted and self.session is not None:
            self.session.checkpoint_scheduler_state(
                completed=tuple(adopted),
                descriptors=descriptors,
                clear_terminal_failure=True,
            )
        for work_id in work_ids:
            if work_id not in hits:
                self._record_cache_miss(work_id, misses.get(work_id, "missing"))
                self._record_invalidation_reasons(
                    self.store.invalidation_reasons(descriptors[work_id])
                )
        return hits

    def _record_cache_hit(self, work_id: str) -> None:
        if self.reliability_tracker is not None:
            self.reliability_tracker.record_cache_hit(work_id)

    def _record_cache_miss(self, work_id: str, reason: str) -> None:
        if self.reliability_tracker is not None:
            self.reliability_tracker.record_cache_miss(work_id, reason)

    def _record_invalidation_reasons(self, reasons: tuple[str, ...]) -> None:
        if self.reliability_tracker is not None:
            self.reliability_tracker.record_invalidation_reasons(reasons)

    def _checkpoint(self, descriptor: CacheDescriptor) -> None:
        if self.session is not None:
            self.session.checkpoint_scheduler_state(
                completed=(
                    CompletedRef(work_id=descriptor.work_id, cache_key=descriptor.key),
                ),
                descriptors={descriptor.work_id: descriptor},
                clear_terminal_failure=True,
            )


@dataclass(frozen=True)
class StructuralBlock:
    start: int
    end: int
    boundary_kind: BoundaryKind

    def __post_init__(self) -> None:
        if self.start < 0 or self.end <= self.start:
            raise ValueError("structural block range must be nonempty and ordered")


@dataclass(frozen=True)
class SegmentationConfig:
    max_tokens: int
    overlap_tokens: int = 0

    def __post_init__(self) -> None:
        if self.max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        if self.overlap_tokens < 0:
            raise ValueError("overlap_tokens cannot be negative")


@dataclass(frozen=True)
class SourceSegment:
    segment_id: str
    source_id: str
    order: int
    text: str
    core_start: int
    core_end: int
    context_start: int
    context_end: int
    core_token_count: int
    token_count: int
    leading_overlap_tokens: int
    trailing_overlap_tokens: int
    boundary_kind: BoundaryKind

    def __post_init__(self) -> None:
        for field_name in ("segment_id", "source_id", "text"):
            if not getattr(self, field_name):
                raise ValueError(f"{field_name} cannot be empty")
        if self.order < 0:
            raise ValueError("order cannot be negative")
        for field_name in (
            "core_token_count",
            "token_count",
            "leading_overlap_tokens",
            "trailing_overlap_tokens",
        ):
            if getattr(self, field_name) < 0:
                raise ValueError(f"{field_name} cannot be negative")
        if self.core_start < 0 or self.core_end <= self.core_start:
            raise ValueError("core range must be nonempty and ordered")
        if self.context_start < 0 or self.context_end <= self.context_start:
            raise ValueError("context range must be nonempty and ordered")
        if self.context_start > self.core_start or self.context_end < self.core_end:
            raise ValueError("context range must contain core range")
        if len(self.text) != self.context_end - self.context_start:
            raise ValueError("text length must match context range")


_ATX_HEADING = re.compile(r" {0,3}#{1,6}(?:[ \t]+|$)")
_SETEXT_UNDERLINE = re.compile(r" {0,3}(?:=+|-+)[ \t]*$")
_LIST_ITEM = re.compile(r" {0,3}(?:[-+*][ \t]+|\d+[.)][ \t]+)")
_FENCE_OPEN = re.compile(r"(`{3,}|~{3,})")


@dataclass(frozen=True)
class _LineSpan:
    start: int
    end: int
    content: str


@dataclass(frozen=True)
class _CoreUnit:
    start: int
    end: int
    boundary_kind: BoundaryKind


def _line_spans(text: str) -> list[_LineSpan]:
    return [
        _LineSpan(match.start(), match.end(), match.group().removesuffix("\n"))
        for match in re.finditer(r"[^\n]*(?:\n|$)", text)
        if match.end() > match.start()
    ]


def _is_setext_heading(lines: list[_LineSpan], index: int) -> bool:
    return (
        bool(lines[index].content)
        and index + 1 < len(lines)
        and _SETEXT_UNDERLINE.fullmatch(lines[index + 1].content) is not None
    )


def _consume_blank_lines(lines: list[_LineSpan], index: int) -> int:
    while index < len(lines) and not lines[index].content:
        index += 1
    return index


def detect_structural_blocks(text: str) -> list[StructuralBlock]:
    """Return contiguous heading, paragraph, list, and fenced-code ranges."""
    lines = _line_spans(text)
    blocks: list[StructuralBlock] = []
    index = 0
    while index < len(lines):
        start_index = index
        content = lines[index].content
        # Detect the opening of a fenced code block (``` or ~~~, 3+ chars).
        fence_match = _FENCE_OPEN.match(content)
        if fence_match:
            fence_marker = fence_match.group(1)
            close_pattern = re.compile(r"^\s*" + re.escape(fence_marker[0] * len(fence_marker)) + r"+\s*$")
            index += 1
            while index < len(lines):
                if close_pattern.match(lines[index].content):
                    index += 1
                    break
                index += 1
            # Include any trailing blank lines as part of the fence block.
            index = _consume_blank_lines(lines, index)
            blocks.append(
                StructuralBlock(
                    start=lines[start_index].start,
                    end=lines[index - 1].end,
                    boundary_kind=BoundaryKind.CODE_FENCE,
                )
            )
            continue
        if _ATX_HEADING.match(content):
            kind = BoundaryKind.HEADING
            index += 1
        elif _is_setext_heading(lines, index):
            kind = BoundaryKind.HEADING
            index += 2
        elif _LIST_ITEM.match(content):
            kind = BoundaryKind.LIST
            index += 1
            while index < len(lines) and lines[index].content:
                if _ATX_HEADING.match(lines[index].content) or _is_setext_heading(
                    lines, index
                ):
                    break
                if _FENCE_OPEN.match(lines[index].content):
                    break
                index += 1
        else:
            kind = BoundaryKind.PARAGRAPH
            index += 1
            while index < len(lines) and lines[index].content:
                if (
                    _ATX_HEADING.match(lines[index].content)
                    or _LIST_ITEM.match(lines[index].content)
                    or _is_setext_heading(lines, index)
                    or _FENCE_OPEN.match(lines[index].content)
                ):
                    break
                index += 1
        index = _consume_blank_lines(lines, index)
        blocks.append(
            StructuralBlock(
                start=lines[start_index].start,
                end=lines[index - 1].end,
                boundary_kind=kind,
            )
        )
    return blocks


_SENTENCE_ABBREVS = {
    "co",
    "dept",
    "dr",
    "e.g",
    "etc",
    "fig",
    "i.e",
    "inc",
    "jr",
    "ltd",
    "mr",
    "mrs",
    "ms",
    "no",
    "prof",
    "sr",
    "st",
    "u.k",
    "u.s",
    "vs",
}
_SENTENCE_TOKENIZER = PunktSentenceTokenizer()
_SENTENCE_TOKENIZER._params.abbrev_types.update(_SENTENCE_ABBREVS)


def _count_tokens(counter: TokenCounter, text: str) -> int:
    count = counter.count(text)
    if count < 0:
        raise TokenAccountingError(
            f"counter {counter.identity!r} returned a negative token count"
        )
    return count


def _largest_fitting_prefix(
    text: str,
    start: int,
    end: int,
    counter: TokenCounter,
    max_tokens: int,
) -> int:
    if isinstance(counter, PrefixTokenCounter):
        candidate = counter.fitting_prefix(text, start, end, max_tokens)
        if candidate <= start or candidate > end:
            raise SegmentationError("token counter could not produce a valid prefix")
        if _count_tokens(counter, text[start:candidate]) > max_tokens:
            raise SegmentationError("token counter produced an oversized prefix")
        return candidate
    if not counter.monotonic:
        raise SegmentationError(
            "non-monotonic token counters must implement fitting_prefix"
        )
    if _count_tokens(counter, text[start : start + 1]) > max_tokens:
        raise SegmentationError("token budget cannot fit one source character")

    best = start + 1
    distance = 2
    while start + distance <= end:
        candidate = start + distance
        if _count_tokens(counter, text[start:candidate]) > max_tokens:
            break
        best = candidate
        distance *= 2
    if best == end:
        return best

    low = best + 1
    high = min(end, start + distance)
    while low <= high:
        candidate = (low + high) // 2
        if _count_tokens(counter, text[start:candidate]) <= max_tokens:
            best = candidate
            low = candidate + 1
        else:
            high = candidate - 1
    return best


def _prefer_whitespace_boundary(
    text: str,
    start: int,
    candidate: int,
    counter: TokenCounter,
    max_tokens: int,
) -> int:
    boundary = next(
        (
            index + 1
            for index in range(candidate - 1, start - 1, -1)
            if text[index].isspace()
        ),
        start,
    )
    if boundary > start and _count_tokens(counter, text[start:boundary]) <= max_tokens:
        return boundary
    return candidate


def _hard_split(
    text: str,
    start: int,
    end: int,
    counter: TokenCounter,
    max_tokens: int,
) -> list[_CoreUnit]:
    units: list[_CoreUnit] = []
    while start < end:
        next_end = _largest_fitting_prefix(
            text,
            start,
            end,
            counter,
            max_tokens,
        )
        if next_end < end:
            next_end = _prefer_whitespace_boundary(
                text,
                start,
                next_end,
                counter,
                max_tokens,
            )
        units.append(_CoreUnit(start, next_end, BoundaryKind.HARD))
        start = next_end
    return units


def _sentence_units(
    text: str,
    block: StructuralBlock,
    counter: TokenCounter,
    max_tokens: int,
) -> list[_CoreUnit]:
    block_text = text[block.start : block.end]
    spans = list(_SENTENCE_TOKENIZER.span_tokenize(block_text))
    if len(spans) <= 1:
        return _hard_split(text, block.start, block.end, counter, max_tokens)

    units: list[_CoreUnit] = []
    start = block.start
    for index, _ in enumerate(spans):
        if index + 1 < len(spans):
            end = block.start + spans[index + 1][0]
        else:
            end = block.end
        if _count_tokens(counter, text[start:end]) <= max_tokens:
            units.append(_CoreUnit(start, end, BoundaryKind.SENTENCE))
        else:
            units.extend(_hard_split(text, start, end, counter, max_tokens))
        start = end
    return units


def _budgeted_units(
    document: SourceDocument,
    counter: TokenCounter,
    max_tokens: int,
) -> list[_CoreUnit]:
    units: list[_CoreUnit] = []
    for block in detect_structural_blocks(document.text):
        if _count_tokens(counter, document.text[block.start : block.end]) <= max_tokens:
            units.append(_CoreUnit(block.start, block.end, block.boundary_kind))
        else:
            units.extend(
                _sentence_units(document.text, block, counter, max_tokens)
            )
    return units


def _pack_units(
    text: str,
    units: list[_CoreUnit],
    counter: TokenCounter,
    max_tokens: int,
) -> list[_CoreUnit]:
    if not units:
        return []
    packed: list[_CoreUnit] = []
    index = 0
    unit_count = len(units)
    while index < unit_count:
        run_end_index = index + 1
        while (
            run_end_index < unit_count
            and units[run_end_index].boundary_kind is not BoundaryKind.HEADING
        ):
            run_end_index += 1

        start = units[index].start
        covered_index = index
        next_index = covered_index + 1
        if next_index < run_end_index and (
            _count_tokens(counter, text[start : units[next_index].end])
            <= max_tokens
        ):
            # At least one more unit fits; find the full extent of the run
            # with a bounded search instead of re-summing every growing
            # prefix one unit at a time.
            limit_end = units[run_end_index - 1].end
            fitting_end = max(
                _largest_fitting_prefix(text, start, limit_end, counter, max_tokens),
                units[next_index].end,
            )
            covered_index = next_index
            while (
                covered_index + 1 < run_end_index
                and units[covered_index + 1].end <= fitting_end
            ):
                covered_index += 1
            # `fitting_end` is only verified at its own offset. A shorter slice
            # of a fitting slice can still exceed the budget under a
            # non-monotonic counter, so the unit boundary actually chosen has
            # to be recounted, falling back toward the verified `next_index`.
            while covered_index > next_index and (
                _count_tokens(counter, text[start : units[covered_index].end])
                > max_tokens
            ):
                covered_index -= 1

        packed.append(
            _CoreUnit(
                start,
                units[covered_index].end,
                units[covered_index].boundary_kind,
            )
        )
        index = covered_index + 1
    return packed


def _sentence_boundaries(text: str, start: int, end: int) -> list[int]:
    block_text = text[start:end]
    return [
        start + span_start
        for span_start, _ in _SENTENCE_TOKENIZER.span_tokenize(block_text)
    ]


def _largest_fitting_suffix(
    text: str,
    floor: int,
    end: int,
    counter: TokenCounter,
    max_tokens: int,
) -> int:
    """Return a start in [floor, end] whose suffix fits, mirroring the prefix search."""
    if max_tokens <= 0 or end <= floor:
        return end
    if isinstance(counter, SuffixTokenCounter):
        candidate = counter.fitting_suffix(text, floor, end, max_tokens)
        if candidate < floor or candidate > end:
            raise SegmentationError("token counter could not produce a valid suffix")
        if (
            candidate < end
            and _count_tokens(counter, text[candidate:end]) > max_tokens
        ):
            raise SegmentationError("token counter produced an oversized suffix")
        return candidate
    if _count_tokens(counter, text[floor:end]) <= max_tokens:
        return floor
    if not counter.monotonic:
        raise SegmentationError(
            "non-monotonic token counters must implement fitting_suffix"
        )
    low, high = floor, end
    while low < high:
        mid = (low + high) // 2
        if _count_tokens(counter, text[mid:end]) <= max_tokens:
            high = mid
        else:
            low = mid + 1
    return low


def _leading_overlap_start(
    text: str,
    floor: int,
    core_start: int,
    counter: TokenCounter,
    max_tokens: int,
) -> int:
    if max_tokens <= 0 or core_start <= floor:
        return core_start
    fitting_start = _largest_fitting_suffix(
        text, floor, core_start, counter, max_tokens
    )
    if fitting_start >= core_start:
        return core_start
    # A boundary later than `fitting_start` shortens the slice, which is not by
    # itself proof that it fits: recount before preferring it.
    for boundary in _sentence_boundaries(text, floor, core_start):
        if fitting_start <= boundary < core_start and (
            _count_tokens(counter, text[boundary:core_start]) <= max_tokens
        ):
            return boundary
    return fitting_start


def _trailing_overlap_end(
    text: str,
    core_end: int,
    ceiling: int,
    counter: TokenCounter,
    max_tokens: int,
) -> int:
    if max_tokens <= 0 or ceiling <= core_end:
        return core_end
    fitting_end = _largest_fitting_prefix(text, core_end, ceiling, counter, max_tokens)
    if fitting_end <= core_end:
        return core_end
    # Prefer the latest sentence boundary within the fitting window, but only
    # once recounted: a shorter slice is not guaranteed to fit.
    for boundary in sorted(
        _sentence_boundaries(text, core_end, ceiling), reverse=True
    ):
        if core_end < boundary <= fitting_end and (
            _count_tokens(counter, text[core_end:boundary]) <= max_tokens
        ):
            return boundary
    return fitting_end


def _validate_segments(
    document: SourceDocument,
    segments: list[SourceSegment],
    counter: TokenCounter,
    max_tokens: int,
    *,
    strict_cached: bool = False,
) -> None:
    expected_start = 0
    for order, segment in enumerate(segments):
        if segment.segment_id != f"S{order + 1:06d}" or segment.order != order:
            raise SegmentationError("segment identifiers or order are unstable")
        if segment.source_id != document.source_id:
            raise SegmentationError("segment source identity does not match document")
        if segment.core_start != expected_start:
            raise SegmentationError("segment core ranges are not contiguous")
        if document.text[segment.context_start : segment.context_end] != segment.text:
            raise SegmentationError("segment text does not match its source range")
        if strict_cached and segment.core_token_count != _count_tokens(
            counter, document.text[segment.core_start : segment.core_end]
        ):
            raise SegmentationError("segment core token count does not match its source")
        if _count_tokens(counter, segment.text) != segment.token_count:
            raise SegmentationError("segment token count does not match its text")
        if strict_cached and (
            segment.leading_overlap_tokens
            != _count_tokens(counter, document.text[segment.context_start : segment.core_start])
            or segment.trailing_overlap_tokens
            != _count_tokens(counter, document.text[segment.core_end : segment.context_end])
        ):
            raise SegmentationError("segment overlap token counts do not match context")
        previous_start = segments[order - 1].core_start if order else segment.core_start
        next_end = (
            segments[order + 1].core_end
            if order + 1 < len(segments)
            else segment.core_end
        )
        if segment.context_start < previous_start or segment.context_end > next_end:
            raise SegmentationError("segment context reaches beyond neighbouring cores")
        if segment.token_count > max_tokens:
            raise SegmentationError("segment exceeds the configured token budget")
        expected_start = segment.core_end
    if expected_start != len(document.text):
        raise SegmentationError("segment core ranges do not reconstruct the source")


def segment_document(
    document: SourceDocument,
    counter: TokenCounter,
    config: SegmentationConfig,
) -> list[SourceSegment]:
    """Split a canonical document into stable, budget-compliant segments."""
    units = _budgeted_units(document, counter, config.max_tokens)
    cores = _pack_units(document.text, units, counter, config.max_tokens)
    segments = []
    for order, core in enumerate(cores):
        core_text = document.text[core.start : core.end]
        core_token_count = _count_tokens(counter, core_text)
        remaining = config.max_tokens - core_token_count

        # Leading context is resolved first and trailing context receives only
        # what it leaves behind, so look-back continuity wins the shared room
        # when the core leaves too little for both.
        leading_floor = cores[order - 1].start if order > 0 else core.start
        context_start = _leading_overlap_start(
            document.text,
            leading_floor,
            core.start,
            counter,
            min(config.overlap_tokens, remaining),
        )
        leading_overlap_tokens = (
            0
            if context_start == core.start
            else _count_tokens(counter, document.text[context_start : core.start])
        )
        remaining -= leading_overlap_tokens

        trailing_ceiling = cores[order + 1].end if order + 1 < len(cores) else core.end
        context_end = _trailing_overlap_end(
            document.text,
            core.end,
            trailing_ceiling,
            counter,
            min(config.overlap_tokens, remaining),
        )
        trailing_overlap_tokens = (
            0
            if context_end == core.end
            else _count_tokens(counter, document.text[core.end : context_end])
        )

        text = document.text[context_start:context_end]
        if context_start == core.start and context_end == core.end:
            token_count = core_token_count
        else:
            token_count = _count_tokens(counter, text)
        segments.append(
            SourceSegment(
                segment_id=f"S{order + 1:06d}",
                source_id=document.source_id,
                order=order,
                text=text,
                core_start=core.start,
                core_end=core.end,
                context_start=context_start,
                context_end=context_end,
                core_token_count=core_token_count,
                token_count=token_count,
                leading_overlap_tokens=leading_overlap_tokens,
                trailing_overlap_tokens=trailing_overlap_tokens,
                boundary_kind=core.boundary_kind,
            )
        )
    _validate_segments(document, segments, counter, config.max_tokens)
    return segments


def cached_segment_document(
    document: SourceDocument,
    counter: TokenCounter,
    config: SegmentationConfig,
    *,
    coordinator: CacheCoordinator | None = None,
) -> list[SourceSegment]:
    """Segment normally, or reuse a fully revalidated segment sequence."""
    if coordinator is None:
        return segment_document(document, counter, config)

    def decode(payload: object) -> list[SourceSegment]:
        if not isinstance(payload, list):
            raise SegmentationError("cached segments must be a list")
        try:
            segments = [
                SourceSegment(
                    **{
                        **item,
                        "boundary_kind": BoundaryKind(item["boundary_kind"]),
                    }
                )
                for item in payload
                if isinstance(item, dict)
            ]
        except (KeyError, TypeError, ValueError) as error:
            raise SegmentationError("cached segments are malformed") from error
        if len(segments) != len(payload) or any(
            segment.source_id != document.source_id for segment in segments
        ):
            raise SegmentationError("cached segments identify another source")
        _validate_segments(
            document, segments, counter, config.max_tokens, strict_cached=True
        )
        return segments

    def encode(segments: list[SourceSegment]) -> object:
        return [
            {
                "segment_id": segment.segment_id,
                "source_id": segment.source_id,
                "order": segment.order,
                "text": segment.text,
                "core_start": segment.core_start,
                "core_end": segment.core_end,
                "context_start": segment.context_start,
                "context_end": segment.context_end,
                "core_token_count": segment.core_token_count,
                "token_count": segment.token_count,
                "leading_overlap_tokens": segment.leading_overlap_tokens,
                "trailing_overlap_tokens": segment.trailing_overlap_tokens,
                "boundary_kind": segment.boundary_kind.value,
            }
            for segment in segments
        ]

    return coordinator.resolve(
        stage="segmentation",
        work_id="segmentation",
        prompt_version=SEGMENTATION_CACHE_VERSION,
        schema_version=SEGMENTATION_CACHE_VERSION,
        input_value={"source_id": document.source_id, "config": {"max_tokens": config.max_tokens, "overlap_tokens": config.overlap_tokens}},
        behavior={"segmentation": {"max_tokens": config.max_tokens, "overlap_tokens": config.overlap_tokens}},
        decode=decode,
        encode=encode,
        compute=lambda: segment_document(document, counter, config),
    )
