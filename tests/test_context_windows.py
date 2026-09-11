import pytest

from summarizer.budget import (
    ASSUMED_CONTEXT_WINDOW,
    ContextWindow,
    resolve_context_window,
)


def test_resolves_a_known_model_exactly() -> None:
    window = resolve_context_window(provider="openai", model="gpt-4o-mini")

    assert window.tokens > 0
    assert window.assumed is False


def test_resolves_a_known_model_family_by_prefix() -> None:
    """`gpt-4o-mini` has no exact entry in tiktoken either; prefixes carry it."""
    exact = resolve_context_window(provider="openai", model="gpt-4o")
    variant = resolve_context_window(provider="openai", model="gpt-4o-2024-11-20")

    assert variant.tokens == exact.tokens
    assert variant.assumed is False


@pytest.mark.parametrize(
    ("model", "expected_tokens"),
    [
        ("gpt-4-0613", 8_192),
        ("gpt-4-32k-0613", 32_768),
    ],
)
def test_resolves_dated_gpt_4_snapshots_by_family(
    model: str,
    expected_tokens: int,
) -> None:
    window = resolve_context_window(provider="openai", model=model)

    assert window == ContextWindow(tokens=expected_tokens, assumed=False)


def test_does_not_prefix_match_gpt_3_5_turbo_snapshots() -> None:
    window = resolve_context_window(
        provider="openai",
        model="gpt-3.5-turbo-0613",
    )

    assert window == ContextWindow(tokens=ASSUMED_CONTEXT_WINDOW, assumed=True)


def test_prefers_the_longest_matching_prefix() -> None:
    """A more specific family must win over a shorter one that also matches.

    `o1-mini` matches both `o1` and `o1-mini`, and the two carry different
    windows - which is what makes the longest-prefix rule observable rather
    than incidental.
    """
    broad = resolve_context_window(provider="openai", model="o1-preview")
    specific = resolve_context_window(provider="openai", model="o1-mini-2024")

    assert broad.tokens == 200_000
    assert specific.tokens == 128_000


def test_an_explicit_window_overrides_the_table() -> None:
    window = resolve_context_window(
        provider="openai", model="gpt-4o-mini", explicit=4_096
    )

    assert window == ContextWindow(tokens=4_096, assumed=False)


def test_an_unknown_openai_model_is_assumed() -> None:
    window = resolve_context_window(provider="openai", model="gpt-nonexistent-9")

    assert window.tokens == ASSUMED_CONTEXT_WINDOW
    assert window.assumed is True


def test_an_arbitrary_ollama_tag_is_assumed() -> None:
    """Neither the installed clients nor tiktoken expose a window offline.

    An arbitrary local tag therefore cannot be known, and the epic requires
    such tags keep working, so it resolves to an assumed window rather than
    failing.
    """
    window = resolve_context_window(provider="ollama", model="qwen3.8")

    assert window.tokens == ASSUMED_CONTEXT_WINDOW
    assert window.assumed is True


def test_an_explicit_window_rescues_an_unknown_model() -> None:
    window = resolve_context_window(
        provider="ollama", model="qwen3.8", explicit=262_144
    )

    assert window == ContextWindow(tokens=262_144, assumed=False)


def test_rejects_a_non_positive_explicit_window() -> None:
    for value in (0, -1):
        # Matched on the resolver's own wording, so it cannot be satisfied by
        # ContextWindow's near-identical message from a layer further in.
        with pytest.raises(ValueError, match="^context window must be positive$"):
            resolve_context_window(
                provider="openai", model="gpt-4o-mini", explicit=value
            )


def test_context_window_rejects_a_non_positive_size() -> None:
    with pytest.raises(ValueError, match="positive"):
        ContextWindow(tokens=0, assumed=True)
