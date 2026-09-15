from dataclasses import dataclass

import pytest

from summarizer.ingestion import ingest_text
from summarizer.segmentation import (
    BoundaryKind,
    SegmentationConfig,
    SegmentationError,
    segment_document,
)

# Packing decides boundaries that only a non-monotonic counter can expose as
# unsafe, and reaching that through whole documents depends on an accidental
# alignment between unit boundaries and a tokenizer dip. The private packer is
# imported so the invariant can be pinned directly.
from summarizer.segmentation import _CoreUnit, _pack_units
from summarizer.tokenization import TokenAccountingError


@dataclass(frozen=True)
class CharacterCounter:
    identity: str = "test:characters"
    exact: bool = True
    monotonic: bool = True

    def count(self, text: str) -> int:
        return len(text)


def test_prefers_heading_and_paragraph_boundaries() -> None:
    document = ingest_text("# H\n\nBody")

    segments = segment_document(
        document,
        CharacterCounter(),
        SegmentationConfig(max_tokens=5),
    )

    assert [segment.text for segment in segments] == ["# H\n\n", "Body"]
    assert [segment.boundary_kind for segment in segments] == [
        BoundaryKind.HEADING,
        BoundaryKind.PARAGRAPH,
    ]


def test_greedily_packs_complete_blocks_at_exact_budget() -> None:
    document = ingest_text("aa\n\nbb\n\ncc")

    segments = segment_document(
        document,
        CharacterCounter(),
        SegmentationConfig(max_tokens=8),
    )

    assert [segment.text for segment in segments] == ["aa\n\nbb\n\n", "cc"]
    assert [segment.core_token_count for segment in segments] == [8, 2]


def test_oversized_paragraph_falls_back_to_sentence_boundaries() -> None:
    document = ingest_text("One. Two.")

    segments = segment_document(
        document,
        CharacterCounter(),
        SegmentationConfig(max_tokens=5),
    )

    assert [segment.text for segment in segments] == ["One. ", "Two."]
    assert all(
        segment.boundary_kind is BoundaryKind.SENTENCE for segment in segments
    )


def test_oversized_sentence_uses_token_safe_hard_fallback() -> None:
    document = ingest_text("abcdefgh")

    segments = segment_document(
        document,
        CharacterCounter(),
        SegmentationConfig(max_tokens=3),
    )

    assert [segment.text for segment in segments] == ["abc", "def", "gh"]
    assert all(segment.boundary_kind is BoundaryKind.HARD for segment in segments)


def test_hard_fallback_prefers_whitespace_without_exceeding_budget() -> None:
    document = ingest_text("abc def ghi")

    segments = segment_document(
        document,
        CharacterCounter(),
        SegmentationConfig(max_tokens=6),
    )

    assert [segment.text for segment in segments] == ["abc ", "def ", "ghi"]


def test_hard_fallback_recognizes_unicode_whitespace() -> None:
    document = ingest_text("abc\u00a0def ghi")

    segments = segment_document(
        document,
        CharacterCounter(),
        SegmentationConfig(max_tokens=5),
    )

    assert [segment.text for segment in segments] == ["abc\u00a0", "def ", "ghi"]


def test_heading_starts_a_new_section_and_packs_forward() -> None:
    document = ingest_text("A\n\n# H\n\nBody")

    segments = segment_document(
        document,
        CharacterCounter(),
        SegmentationConfig(max_tokens=9),
    )

    assert [segment.text for segment in segments] == ["A\n\n", "# H\n\nBody"]


def test_tiny_budget_makes_progress_through_unicode() -> None:
    document = ingest_text("é🙂東")

    segments = segment_document(
        document,
        CharacterCounter(),
        SegmentationConfig(max_tokens=1),
    )

    assert [segment.text for segment in segments] == ["é", "🙂", "東"]


def test_segments_are_stable_ordered_and_reconstruct_the_source() -> None:
    document = ingest_text("# H\n\nRepeated.\n\nRepeated.\n\nabcdefgh")
    config = SegmentationConfig(max_tokens=9)

    first = segment_document(document, CharacterCounter(), config)
    second = segment_document(document, CharacterCounter(), config)

    assert first == second
    assert [segment.segment_id for segment in first] == [
        f"S{index:06d}" for index in range(1, len(first) + 1)
    ]
    assert [segment.order for segment in first] == list(range(len(first)))
    assert "".join(
        document.text[segment.core_start : segment.core_end]
        for segment in first
    ) == document.text
    assert all(
        segment.text
        == document.text[segment.context_start : segment.context_end]
        for segment in first
    )
    assert all(segment.token_count <= config.max_tokens for segment in first)


class NegativeCounter:
    identity = "test:negative"
    exact = False
    monotonic = True

    def count(self, text: str) -> int:
        return -1


def test_rejects_negative_token_counts() -> None:
    with pytest.raises(TokenAccountingError, match="negative"):
        segment_document(
            ingest_text("content"),
            NegativeCounter(),
            SegmentationConfig(max_tokens=10),
        )


class NonMonotonicPrefixCounter:
    identity = "test:non-monotonic"
    exact = True
    monotonic = False

    def count(self, text: str) -> int:
        return {"删": 2, "删除": 1}.get(text, len(text))

    def fitting_prefix(
        self,
        text: str,
        start: int,
        end: int,
        max_tokens: int,
    ) -> int:
        if text.startswith("删除", start, end) and max_tokens == 1:
            return start + 2
        return min(end, start + max_tokens)


def test_non_monotonic_counter_controls_hard_prefix_boundaries() -> None:
    segments = segment_document(
        ingest_text("删除x"),
        NonMonotonicPrefixCounter(),
        SegmentationConfig(max_tokens=1),
    )

    assert [segment.text for segment in segments] == ["删除", "x"]


class TrackingMonotonicCounter:
    identity = "test:tracking"
    exact = True
    monotonic = True

    def __init__(self) -> None:
        self.calls = 0
        self.largest_input = 0
        self.large_calls = 0
        self.total_characters = 0

    def count(self, text: str) -> int:
        self.calls += 1
        self.largest_input = max(self.largest_input, len(text))
        self.large_calls += len(text) > 2
        self.total_characters += len(text)
        return len(text)


def test_hard_splitting_does_not_recount_the_remaining_document() -> None:
    counter = TrackingMonotonicCounter()

    segments = segment_document(
        ingest_text("x" * 5_000),
        counter,
        SegmentationConfig(max_tokens=1),
    )

    assert len(segments) == 5_000
    assert counter.largest_input == 5_000
    assert counter.large_calls == 1
    assert counter.total_characters < 50_000
    assert counter.calls < 30_000


class OffsetPrefixCounter:
    identity = "test:offset-prefix"
    exact = True
    monotonic = False

    def __init__(self) -> None:
        self.source_ids: set[int] = set()

    def count(self, text: str) -> int:
        return len(text)

    def fitting_prefix(
        self,
        text: str,
        start: int,
        end: int,
        max_tokens: int,
    ) -> int:
        self.source_ids.add(id(text))
        return min(end, start + max_tokens)


def test_prefix_counter_receives_original_text_with_offsets() -> None:
    document = ingest_text("x" * 100)
    counter = OffsetPrefixCounter()

    segments = segment_document(
        document,
        counter,
        SegmentationConfig(max_tokens=1),
    )

    assert len(segments) == 100
    assert counter.source_ids == {id(document.text)}


def test_tiktoken_treats_special_token_spelling_as_inert_source() -> None:
    from types import SimpleNamespace

    from summarizer.tokenization import TiktokenCounter

    def reject_special(text: str) -> list[str]:
        if "<|endoftext|>" in text:
            raise ValueError("special token")
        return list(text)

    encoding = SimpleNamespace(
        name="test",
        encode=reject_special,
        encode_ordinary=lambda text: list(text),
    )

    segments = segment_document(
        ingest_text("before <|endoftext|> after"),
        TiktokenCounter(encoding=encoding),  # type: ignore[arg-type]
        SegmentationConfig(max_tokens=8),
    )

    assert "".join(segment.text for segment in segments) == (
        "before <|endoftext|> after"
    )


def test_structural_packing_does_not_recount_growing_document_prefixes() -> None:
    counter = TrackingMonotonicCounter()
    document = ingest_text("\n\n".join("x" for _ in range(1_000)))

    segments = segment_document(
        document,
        counter,
        SegmentationConfig(max_tokens=len(document.text)),
    )

    assert len(segments) == 1
    assert counter.total_characters < 50_000


_OVERLAP_DOCUMENT = "# A\n\nFirst. Second sentence here.\n\n# B\n\nBody B."


def test_overlap_defaults_to_zero_and_matches_core_ranges() -> None:
    document = ingest_text(_OVERLAP_DOCUMENT)

    segments = segment_document(
        document, CharacterCounter(), SegmentationConfig(max_tokens=40)
    )

    assert [s.context_start for s in segments] == [s.core_start for s in segments]
    assert [s.context_end for s in segments] == [s.core_end for s in segments]
    assert all(s.leading_overlap_tokens == 0 for s in segments)
    assert all(s.trailing_overlap_tokens == 0 for s in segments)


def test_overlap_expands_backward_when_no_boundary_fits_the_budget() -> None:
    document = ingest_text(_OVERLAP_DOCUMENT)

    segments = segment_document(
        document,
        CharacterCounter(),
        SegmentationConfig(max_tokens=40, overlap_tokens=22),
    )

    second = segments[1]
    assert (second.core_start, second.core_end) == (35, 47)
    assert second.context_start == 13
    assert second.text.startswith("econd sentence here.\n\n")
    assert second.leading_overlap_tokens == 22


def test_overlap_prefers_a_sentence_boundary_over_the_raw_budget_limit() -> None:
    document = ingest_text(_OVERLAP_DOCUMENT)

    segments = segment_document(
        document,
        CharacterCounter(),
        SegmentationConfig(max_tokens=40, overlap_tokens=24),
    )

    second = segments[1]
    assert second.context_start == 12
    assert second.text.startswith("Second sentence here.\n\n")
    # A clean sentence start is preferred even though it uses one token less
    # than the full 24-token overlap budget would otherwise allow.
    assert second.leading_overlap_tokens == 23


def test_overlap_is_reduced_when_the_core_nearly_fills_the_budget() -> None:
    document = ingest_text("AAAA\n\nBBBB\n\nCCCC\n\nDDDD")

    segments = segment_document(
        document,
        CharacterCounter(),
        SegmentationConfig(max_tokens=6, overlap_tokens=100),
    )

    assert [s.leading_overlap_tokens for s in segments] == [0, 0, 0, 2]
    assert [s.trailing_overlap_tokens for s in segments] == [0, 0, 0, 0]
    assert all(s.token_count <= 6 for s in segments)


def test_overlap_does_not_change_core_ranges_ids_or_order() -> None:
    document = ingest_text(_OVERLAP_DOCUMENT)
    counter = CharacterCounter()

    baseline = segment_document(document, counter, SegmentationConfig(max_tokens=40))
    overlapped = segment_document(
        document, counter, SegmentationConfig(max_tokens=40, overlap_tokens=24)
    )

    assert [(s.segment_id, s.order, s.core_start, s.core_end) for s in baseline] == [
        (s.segment_id, s.order, s.core_start, s.core_end) for s in overlapped
    ]


def test_overlap_is_unambiguous_with_repeated_text() -> None:
    document = ingest_text("# H\n\nRepeated.\n\nRepeated.\n\nabcdefgh")
    config = SegmentationConfig(max_tokens=9, overlap_tokens=3)

    segments = segment_document(document, CharacterCounter(), config)

    assert all(
        segment.text == document.text[segment.context_start : segment.context_end]
        for segment in segments
    )
    assert "".join(
        document.text[segment.core_start : segment.core_end] for segment in segments
    ) == document.text
    assert all(segment.token_count <= config.max_tokens for segment in segments)


class DippingCounter:
    """Counts non-monotonically: 15 characters exceed a budget that 16 meet.

    Real BPE behaves this way. With cl100k_base and a 2-token budget, 15
    characters of `"a" * 3000` cost 3 tokens while 16 cost 2, so a slice that
    fits is no proof that a shorter slice of it also fits.
    """

    identity = "test:dipping"
    exact = True
    monotonic = False

    _COUNTS = {5: 1, 10: 2, 15: 3, 16: 2, 20: 5}

    def count(self, text: str) -> int:
        return self._COUNTS.get(len(text), len(text))

    def fitting_prefix(
        self,
        text: str,
        start: int,
        end: int,
        max_tokens: int,
    ) -> int:
        return min(end, start + 16)

    def fitting_suffix(
        self,
        text: str,
        floor: int,
        end: int,
        max_tokens: int,
    ) -> int:
        return max(floor, end - 16)


def test_packing_recounts_the_boundary_it_snaps_to() -> None:
    counter = DippingCounter()
    text = "a" * 20
    units = [
        _CoreUnit(offset, offset + 5, BoundaryKind.PARAGRAPH)
        for offset in range(0, 20, 5)
    ]

    packed = _pack_units(text, units, counter, 2)

    # Snapping to the largest unit boundary within the fitting offset lands on
    # 15, which costs 3 tokens; packing must fall back to the verified 10.
    assert packed[0].end == 10
    assert all(counter.count(text[unit.start : unit.end]) <= 2 for unit in packed)


class NonMonotonicOverlapCounter:
    """Exact but non-monotonic, and able to search in both directions."""

    identity = "test:non-monotonic-overlap"
    exact = True
    monotonic = False

    def count(self, text: str) -> int:
        return len(text)

    def fitting_prefix(
        self,
        text: str,
        start: int,
        end: int,
        max_tokens: int,
    ) -> int:
        return min(end, start + max_tokens)

    def fitting_suffix(
        self,
        text: str,
        floor: int,
        end: int,
        max_tokens: int,
    ) -> int:
        return max(floor, end - max_tokens)


class ForwardOnlyCounter(NonMonotonicOverlapCounter):
    """Non-monotonic with no backward search, so overlap cannot be resolved."""

    identity = "test:forward-only"
    fitting_suffix = None  # type: ignore[assignment]


def test_overlap_works_for_a_non_monotonic_counter_that_searches_backward() -> None:
    document = ingest_text(_OVERLAP_DOCUMENT)

    segments = segment_document(
        document,
        NonMonotonicOverlapCounter(),
        SegmentationConfig(max_tokens=40, overlap_tokens=5),
    )

    assert segments[1].context_start < segments[1].core_start
    assert segments[1].leading_overlap_tokens > 0
    assert all(segment.token_count <= 40 for segment in segments)


def test_overlap_rejects_a_non_monotonic_counter_without_a_backward_search() -> None:
    document = ingest_text(_OVERLAP_DOCUMENT)

    with pytest.raises(SegmentationError, match="fitting_suffix"):
        segment_document(
            document,
            ForwardOnlyCounter(),
            SegmentationConfig(max_tokens=40, overlap_tokens=5),
        )


# ---------------------------------------------------------------------------
# Bug 2: abbreviation-aware sentence tokenizer
# ---------------------------------------------------------------------------

def test_abbreviation_tokenizer_does_not_split_on_dr_and_us() -> None:
    from summarizer.segmentation import _SENTENCE_TOKENIZER

    text = "Dr. Smith reviewed the U.S. market. It rose."
    spans = list(_SENTENCE_TOKENIZER.span_tokenize(text))
    assert len(spans) == 2, (
        f"Expected 2 sentences, got {len(spans)}: {[text[s:e] for s, e in spans]}"
    )
