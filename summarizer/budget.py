from __future__ import annotations

import json
from dataclasses import dataclass

from summarizer.config import StrategyConfig, StrategyName
from summarizer.ingestion import SourceDocument
from summarizer.leaf import build_leaf_request
from summarizer.segmentation import BoundaryKind, SourceSegment
from summarizer.tokenization import TokenCounter

# Context windows have no offline source of truth, and for OpenAI no online one
# either: the installed openai client exposes only {id, created, object,
# owned_by, shutdown_date} per model, and tiktoken maps names to encodings
# rather than to sizes. Ollama does report an architectural length through
# show(), but the key is architecture-prefixed rather than tag-named (the tag
# `qwen3.8` reports `qwen35.context_length`) and reaching it is a network call
# that is unavailable before a pull. So this table is maintained by hand, and
# anything missing from it is reported as assumed rather than as knowledge.
_MODEL_CONTEXT_WINDOWS: dict[str, int] = {
    "gpt-4": 8_192,
    "gpt-4-32k": 32_768,
    "gpt-3.5-turbo": 16_385,
}

# Consulted by longest prefix after the exact table. Only model families listed
# here accept suffixed names; repeating an exact entry here explicitly opts its
# dated snapshots into family resolution.
_MODEL_PREFIX_CONTEXT_WINDOWS: dict[str, int] = {
    "gpt-4": 8_192,
    "gpt-4-32k": 32_768,
    "gpt-4o": 128_000,
    "gpt-4o-mini": 128_000,
    "gpt-4-turbo": 128_000,
    "gpt-4.1": 1_047_576,
    "gpt-5": 400_000,
    "o1": 200_000,
    # Nested under "o1" with a different window, which is what makes the
    # longest-prefix rule observable rather than incidental.
    "o1-mini": 128_000,
    "o3": 200_000,
    "o4": 200_000,
}

# Deliberately small: an assumed window should not let an unknown model gamble
# a large request. Selection routes an assumed window to hierarchical rather
# than trusting it.
ASSUMED_CONTEXT_WINDOW = 8_192


class BudgetError(ValueError):
    """A configuration leaves no room to send a request safely."""


@dataclass(frozen=True)
class ContextWindow:
    """A model's total context size, and whether that size is actually known."""

    tokens: int
    assumed: bool

    def __post_init__(self) -> None:
        if self.tokens <= 0:
            raise ValueError("context window tokens must be positive")


def resolve_context_window(
    *,
    provider: str,
    model: str,
    explicit: int | None = None,
) -> ContextWindow:
    """Resolve a model's context window without contacting a provider.

    An explicit value is authoritative. For OpenAI, the exact-name table is
    consulted first and then the longest matching known family prefix. Other
    providers skip those OpenAI tables. An unresolved model yields an assumed
    window flagged as such, so a caller can decline to rely on it.
    """
    if explicit is not None:
        if explicit <= 0:
            raise ValueError("context window must be positive")
        return ContextWindow(tokens=explicit, assumed=False)

    if provider.strip().lower() == "openai":
        name = model.strip()
        exact = _MODEL_CONTEXT_WINDOWS.get(name)
        if exact is not None:
            return ContextWindow(tokens=exact, assumed=False)

        matches = [
            prefix
            for prefix in _MODEL_PREFIX_CONTEXT_WINDOWS
            if name.startswith(prefix)
        ]
        if matches:
            longest = max(matches, key=len)
            return ContextWindow(
                tokens=_MODEL_PREFIX_CONTEXT_WINDOWS[longest], assumed=False
            )

    return ContextWindow(tokens=ASSUMED_CONTEXT_WINDOW, assumed=True)


@dataclass(frozen=True)
class OverheadMeasurement:
    """What a request costs before any source text is added."""

    instructions: int
    schema: int
    fencing: int

    @property
    def total(self) -> int:
        return self.instructions + self.schema + self.fencing


def measure_overhead(
    counter: TokenCounter,
    *,
    with_overlap: bool,
) -> OverheadMeasurement:
    """Measure per-request overhead rather than assuming a constant.

    A hard-coded figure would be wrong the moment the prompt or the record
    changes, and the schema dominates: it is roughly two thirds of the total,
    because the record's own docstrings are rendered into it and shipped on
    every call.

    The schema is counted in its compact serialization. That approximates what
    a provider's tokenizer sees; an indented form would overstate it by about
    two thirds. Approximating downward is why a safety margin exists.
    """
    # The probe uses a region framing, which is the longer of the two by three
    # tokens, so the measurement covers a whole-document request as well.
    probe = _overhead_probe_segment(with_overlap=with_overlap)
    request = build_leaf_request(probe, model="probe", timeout_seconds=1)

    schema = counter.count(
        json.dumps(request.response_schema, separators=(",", ":"), sort_keys=True)
    )
    fencing = counter.count(request.input_text) - counter.count(probe.text)
    return OverheadMeasurement(
        instructions=counter.count(request.instructions),
        schema=schema,
        fencing=max(fencing, 0),
    )


def _overhead_probe_segment(*, with_overlap: bool) -> SourceSegment:
    """Build the smallest segment that exercises the real request builder.

    Measuring the assembled request rather than its parts means the fences and
    the overlap instruction block are counted exactly as they are sent.
    """
    text = "probe"
    if not with_overlap:
        return SourceSegment(
            segment_id="S000001",
            source_id="0" * 64,
            order=0,
            text=text,
            core_start=0,
            core_end=len(text),
            context_start=0,
            context_end=len(text),
            core_token_count=1,
            token_count=1,
            leading_overlap_tokens=0,
            trailing_overlap_tokens=0,
            boundary_kind=BoundaryKind.PARAGRAPH,
        )

    padded = f"a{text}b"
    return SourceSegment(
        segment_id="S000001",
        source_id="0" * 64,
        order=0,
        text=padded,
        core_start=1,
        core_end=1 + len(text),
        context_start=0,
        context_end=len(padded),
        core_token_count=1,
        token_count=1,
        leading_overlap_tokens=1,
        trailing_overlap_tokens=1,
        boundary_kind=BoundaryKind.PARAGRAPH,
    )


def safety_margin(window_tokens: int, config: StrategyConfig) -> int:
    """Reserve the larger of a fixed floor and a proportional share.

    A fixed margin alone is negligible against a very large window; a
    proportional one alone discards thousands of tokens there while
    under-protecting a small window.
    """
    proportional = int(window_tokens * config.safety_margin_fraction)
    return max(config.safety_margin_tokens, proportional)


def usable_input_capacity(
    *,
    window: ContextWindow,
    overhead: OverheadMeasurement,
    config: StrategyConfig,
) -> int:
    """Return how many tokens of source text a request may carry.

    Raises `BudgetError` rather than returning a non-positive number, because
    that outcome is reachable on default local configuration: the conservative
    byte estimator charges over four times a real tokenizer on this project's
    own ASCII.
    """
    margin = safety_margin(window.tokens, config)
    capacity = (
        window.tokens - overhead.total - config.max_output_tokens - margin
    )
    if capacity <= 0:
        raise BudgetError(
            f"no usable input capacity: a context window of {window.tokens} tokens "
            f"leaves {capacity} after overhead of {overhead.total} "
            f"(instructions {overhead.instructions}, schema {overhead.schema}, "
            f"fencing {overhead.fencing}), reserved output of "
            f"{config.max_output_tokens}, and a safety margin of {margin}"
        )
    return capacity


@dataclass(frozen=True)
class BudgetReport:
    """Everything a strategy decision saw, and why it decided.

    Returned to callers as run metadata, and safe to hash for later cache keys
    because identical inputs produce an identical report.
    """

    strategy: StrategyName
    reason: str
    context_window_tokens: int
    context_window_assumed: bool
    counter_identity: str
    counter_exact: bool
    overhead: OverheadMeasurement
    reserved_output_tokens: int
    safety_margin_tokens: int
    usable_input_capacity: int
    document_tokens: int
    fits: bool


def select_strategy(
    document: SourceDocument,
    counter: TokenCounter,
    *,
    provider: str,
    model: str,
    config: StrategyConfig,
) -> BudgetReport:
    """Choose an execution path from a measured budget.

    The document is measured with `counter.count(document.text)` - exactly the
    text a direct request sends - rather than by summing segment counts, which
    differs under a byte-pair encoder and diverges further under overlap.
    """
    window = resolve_context_window(
        provider=provider, model=model, explicit=config.context_window
    )
    # A direct request carries no overlap. A stage that sends overlap-carrying
    # requests must measure its own overhead: the overlap variant is about 120
    # tokens larger, and sizing against this figure would under-reserve.
    overhead = measure_overhead(counter, with_overlap=False)
    margin = safety_margin(window.tokens, config)
    capacity = usable_input_capacity(
        window=window, overhead=overhead, config=config
    )
    document_tokens = counter.count(document.text)
    fits = document_tokens <= capacity

    capped = (
        config.max_direct_tokens is not None
        and document_tokens > config.max_direct_tokens
    )

    if config.strategy == "direct" and window.assumed:
        raise BudgetError(
            f"direct summarization was requested but the context window for "
            f"model {model!r} is not known, so a fit cannot be established; "
            f"pass an explicit context window to proceed. This matters most "
            f"on a provider that truncates an oversized prompt silently "
            f"rather than rejecting it."
        )

    if config.strategy == "direct" and not fits:
        raise BudgetError(
            f"direct summarization was requested but the document does not fit: "
            f"{document_tokens} tokens against a usable input capacity of "
            f"{capacity} (context window {window.tokens}, overhead "
            f"{overhead.total}, reserved output {config.max_output_tokens}, "
            f"safety margin {margin})"
        )

    strategy, reason = _decide(
        config=config,
        fits=fits,
        capped=capped,
        assumed=window.assumed,
        document_tokens=document_tokens,
        capacity=capacity,
    )

    return BudgetReport(
        strategy=strategy,
        reason=reason,
        context_window_tokens=window.tokens,
        context_window_assumed=window.assumed,
        counter_identity=counter.identity,
        counter_exact=counter.exact,
        overhead=overhead,
        reserved_output_tokens=config.max_output_tokens,
        safety_margin_tokens=margin,
        usable_input_capacity=capacity,
        document_tokens=document_tokens,
        fits=fits,
    )


def _decide(
    *,
    config: StrategyConfig,
    fits: bool,
    capped: bool,
    assumed: bool,
    document_tokens: int,
    capacity: int,
) -> tuple[StrategyName, str]:
    if config.strategy == "direct":
        return "direct", (
            f"direct was requested and {document_tokens} tokens fit a capacity "
            f"of {capacity}"
        )
    if config.strategy == "hierarchical":
        return "hierarchical", "hierarchical was requested explicitly"

    if not fits:
        return "hierarchical", (
            f"{document_tokens} tokens exceed a usable input capacity of "
            f"{capacity}"
        )
    if assumed:
        return "hierarchical", (
            "the context window is assumed rather than known, so a direct "
            "request cannot be shown to fit"
        )
    if capped:
        return "hierarchical", (
            f"{document_tokens} tokens exceed the configured direct cap of "
            f"{config.max_direct_tokens}"
        )
    return "direct", (
        f"{document_tokens} tokens fit a usable input capacity of {capacity}"
    )
