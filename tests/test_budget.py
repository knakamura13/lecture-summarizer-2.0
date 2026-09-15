from dataclasses import dataclass

import pytest

from summarizer.budget import (
    BudgetError,
    ContextWindow,
    measure_overhead,
    safety_margin,
    usable_input_capacity,
)
from summarizer.config import StrategyConfig


@dataclass(frozen=True)
class CharacterCounter:
    identity: str = "test:characters"
    exact: bool = True
    monotonic: bool = True

    def count(self, text: str) -> int:
        return len(text)


def test_overhead_counts_instructions_schema_and_fencing() -> None:
    overhead = measure_overhead(CharacterCounter(), with_overlap=False)

    assert overhead.instructions > 0
    assert overhead.schema > 0
    assert overhead.fencing > 0
    assert overhead.total == (
        overhead.instructions + overhead.schema + overhead.fencing
    )


def test_overlap_increases_overhead() -> None:
    """Overhead is not one constant: overlap adds a second instruction block."""
    plain = measure_overhead(CharacterCounter(), with_overlap=False)
    overlapping = measure_overhead(CharacterCounter(), with_overlap=True)

    assert overlapping.total > plain.total
    assert overlapping.instructions > plain.instructions


def test_overhead_is_measured_not_assumed() -> None:
    """A hard-coded constant would silently rot when the prompt changes.

    Counting with a character counter makes the measurement equal the actual
    character length of what is sent, so this fails if the schema or the
    instruction template stops being measured at all.
    """
    from summarizer.summaries import leaf_summary_schema
    import json

    overhead = measure_overhead(CharacterCounter(), with_overlap=False)
    schema_chars = len(json.dumps(leaf_summary_schema(), separators=(",", ":")))

    assert overhead.schema == schema_chars


def test_overhead_with_a_real_encoding_matches_the_measured_scale() -> None:
    """One case against a real tokenizer, since every other test uses a fake.

    The skip covers vocabulary availability alone, so a broken counter still
    fails loudly rather than skipping.
    """
    import tiktoken

    from summarizer.tokenization import TiktokenCounter

    try:
        tiktoken.encoding_for_model("gpt-4o-mini")
    except Exception as error:  # pragma: no cover - depends on the local cache
        pytest.skip(f"tiktoken vocabulary is unavailable offline: {error}")

    overhead = measure_overhead(
        TiktokenCounter.for_model("gpt-4o-mini"), with_overlap=False
    )

    # Pinned exactly rather than banded: a band this wide would not notice the
    # prompt or the record changing, which is the thing worth noticing.
    assert (overhead.instructions, overhead.schema, overhead.fencing) == (
        270,
        610,
        25,
    )
    assert overhead.total == 905


def test_safety_margin_takes_the_larger_term() -> None:
    config = StrategyConfig(safety_margin_tokens=256, safety_margin_fraction=0.02)

    # Fixed term dominates a small window.
    assert safety_margin(4_096, config) == 256
    # Fractional term dominates a large one.
    assert safety_margin(128_000, config) == 2_560


def test_capacity_subtracts_every_term() -> None:
    config = StrategyConfig(
        max_output_tokens=1_000,
        safety_margin_tokens=100,
        safety_margin_fraction=0,
    )
    overhead = measure_overhead(CharacterCounter(), with_overlap=False)
    window = ContextWindow(tokens=overhead.total + 5_000, assumed=False)

    capacity = usable_input_capacity(
        window=window, overhead=overhead, config=config
    )

    assert capacity == 5_000 - 1_000 - 100


def test_non_positive_capacity_reports_every_term() -> None:
    """Reachable on default local configuration, so it must be a named error.

    The byte estimator charges over four times a real tokenizer on the
    project's own ASCII, which drives a small window negative.
    """
    config = StrategyConfig(max_output_tokens=1_024, safety_margin_fraction=0.05)
    overhead = measure_overhead(CharacterCounter(), with_overlap=False)

    with pytest.raises(BudgetError) as error:
        usable_input_capacity(
            window=ContextWindow(tokens=1_024, assumed=True),
            overhead=overhead,
            config=config,
        )

    message = str(error.value)
    # Each term's own value, not just its label: "1024" alone would match the
    # window as well as the reserve.
    assert f"overhead of {overhead.total}" in message
    assert f"instructions {overhead.instructions}" in message
    assert f"schema {overhead.schema}" in message
    assert "reserved output of 1024" in message
    # The fixed floor dominates at this window: max(256, 0.05 * 1024) == 256.
    assert "safety margin of 256" in message


def test_capacity_of_exactly_zero_is_refused() -> None:
    """The boundary itself, not merely a deeply negative case."""
    config = StrategyConfig(
        max_output_tokens=1, safety_margin_tokens=0, safety_margin_fraction=0
    )
    overhead = measure_overhead(CharacterCounter(), with_overlap=False)

    with pytest.raises(BudgetError, match="leaves 0"):
        usable_input_capacity(
            window=ContextWindow(tokens=overhead.total + 1, assumed=False),
            overhead=overhead,
            config=config,
        )


def test_fencing_excludes_the_probe_text_it_measured_with() -> None:
    """The fencing term is defined by that subtraction, so pin it.

    Without this, dropping the subtraction leaves every assertion satisfied
    while the fencing figure silently absorbs the probe's own text.
    """
    counter = CharacterCounter()
    overhead = measure_overhead(counter, with_overlap=False)

    from summarizer.budget import _overhead_probe_segment
    from summarizer.leaf import build_leaf_request

    probe = _overhead_probe_segment(with_overlap=False)
    request = build_leaf_request(probe, model="probe", timeout_seconds=1)

    assert overhead.fencing == len(request.input_text) - len(probe.text)
    assert "probe" not in str(overhead.fencing)


def test_capacity_is_deterministic() -> None:
    config = StrategyConfig()
    overhead = measure_overhead(CharacterCounter(), with_overlap=False)
    window = ContextWindow(tokens=overhead.total + 10_000, assumed=False)

    first = usable_input_capacity(window=window, overhead=overhead, config=config)
    second = usable_input_capacity(window=window, overhead=overhead, config=config)

    assert first == second
