from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from summarizer.tokenization import (
    ConservativeUtf8TokenCounter,
    TiktokenCounter,
    TokenAccountingError,
    TokenCounter,
    resolve_token_counter,
)


@dataclass(frozen=True)
class CharacterCounter:
    identity: str = "test:characters"
    exact: bool = True
    monotonic: bool = True

    def count(self, text: str) -> int:
        return len(text)


def test_protocol_accepts_an_injected_deterministic_counter() -> None:
    counter: TokenCounter = CharacterCounter()

    assert counter.count("café") == 4
    assert counter.identity == "test:characters"
    assert counter.exact is True
    assert counter.monotonic is True


def test_tiktoken_counter_uses_known_model_encoding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import tiktoken

    encoding = SimpleNamespace(
        name="o200k_base",
        encode_ordinary=lambda text: list(text),
    )

    def encoding_for_model(model: str) -> SimpleNamespace:
        assert model == "gpt-4o-mini"
        return encoding

    monkeypatch.setattr(tiktoken, "encoding_for_model", encoding_for_model)

    counter = TiktokenCounter.for_model("gpt-4o-mini")

    assert counter.count("summary") == 7
    assert counter.identity == "tiktoken:o200k_base"
    assert counter.exact is True
    assert counter.monotonic is False


def test_openai_unknown_model_requires_explicit_encoding_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import tiktoken

    def reject_unknown_model(model: str) -> None:
        assert model == "future-model"
        raise KeyError(model)

    monkeypatch.setattr(tiktoken, "encoding_for_model", reject_unknown_model)

    with pytest.raises(TokenAccountingError, match="encoding"):
        resolve_token_counter(provider="openai", model="future-model")


def test_openai_explicit_encoding_fallback_is_exact_for_that_encoding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import tiktoken

    encoding = SimpleNamespace(
        name="cl100k_base",
        encode_ordinary=lambda text: [text],
    )

    def get_encoding(name: str) -> SimpleNamespace:
        assert name == "cl100k_base"
        return encoding

    monkeypatch.setattr(tiktoken, "get_encoding", get_encoding)

    counter = resolve_token_counter(
        provider="openai",
        model="future-model",
        encoding_name="cl100k_base",
    )

    assert isinstance(counter, TiktokenCounter)
    assert counter.identity == "tiktoken:cl100k_base"
    assert counter.exact is True
    assert counter.monotonic is False


def test_non_openai_explicit_encoding_is_exact_for_that_encoding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import tiktoken

    encoding = SimpleNamespace(
        name="cl100k_base",
        encode_ordinary=lambda text: [text],
    )

    def get_encoding(name: str) -> SimpleNamespace:
        assert name == "cl100k_base"
        return encoding

    monkeypatch.setattr(tiktoken, "get_encoding", get_encoding)

    counter = resolve_token_counter(
        provider="ollama",
        model="qwen3.8",
        encoding_name="cl100k_base",
    )

    assert isinstance(counter, TiktokenCounter)
    assert counter.identity == "tiktoken:cl100k_base"
    assert counter.exact is True
    assert counter.monotonic is False


def test_conservative_counter_uses_utf8_byte_length() -> None:
    counter = ConservativeUtf8TokenCounter()

    assert counter.count("abc") == 3
    assert counter.count("é") == 2
    assert counter.count("🙂") == 4
    assert counter.identity == "estimate:utf8-bytes"
    assert counter.exact is False
    assert counter.monotonic is True


@pytest.mark.parametrize("provider", ["ollama", "custom"])
def test_non_openai_provider_uses_offline_conservative_counter(
    provider: str,
) -> None:
    counter = resolve_token_counter(provider=provider, model="org/model:tag")

    assert isinstance(counter, ConservativeUtf8TokenCounter)
    assert counter.identity == "estimate:utf8-bytes"
    assert counter.count("é") == 2


def test_default_model_counter_constructs_against_a_real_encoding() -> None:
    """The one test here that touches a real encoding rather than a fake.

    A fully mocked suite cannot catch a counter that only fails against real
    tiktoken data. The skip covers vocabulary availability alone — the offline
    guard blocks the download on a cold cache — so construction itself stays
    unguarded and a broken counter fails loudly instead of skipping.
    """
    import tiktoken

    try:
        tiktoken.encoding_for_model("gpt-4o-mini")
    except Exception as error:  # pragma: no cover - depends on the local cache
        pytest.skip(f"tiktoken vocabulary is unavailable offline: {error}")

    counter = TiktokenCounter.for_model("gpt-4o-mini")

    assert counter.identity == "tiktoken:o200k_base"
    assert counter.exact is True
    assert counter.count("structured leaf summarization") >= 1

    text = "alpha bravo charlie delta echo foxtrot golf hotel india"
    boundary = counter.fitting_prefix(text, 0, len(text), max_tokens=4)
    assert 0 < boundary <= len(text)
    assert counter.count(text[:boundary]) <= 4

    floor = counter.fitting_suffix(text, 0, len(text), max_tokens=4)
    assert 0 <= floor < len(text)
    assert counter.count(text[floor:]) <= 4


def test_tiktoken_counter_does_not_decode_the_vocabulary() -> None:
    """Real encodings reserve ids that have no byte mapping at all.

    `o200k_base`, the encoding behind the default `gpt-4o-mini` model, declares
    200019 ids of which 19 raise `KeyError` from `decode_single_token_bytes`.
    Deriving anything by walking `range(n_vocab)` therefore crashes on the
    project's own default configuration.
    """

    def reject_reserved_id(token: int) -> bytes:
        raise KeyError(token)

    encoding = SimpleNamespace(
        name="gap-ids",
        encode_ordinary=lambda text: list(text),
        n_vocab=200_019,
        decode_single_token_bytes=reject_reserved_id,
    )

    counter = TiktokenCounter(encoding=encoding)  # type: ignore[arg-type]

    assert counter.count("summary") == 7
    assert counter.fitting_prefix("summary", 0, 7, max_tokens=3) == 3


def test_tiktoken_suffix_mirrors_the_prefix_search() -> None:
    encoding = SimpleNamespace(
        name="test",
        encode_ordinary=lambda text: list(text),
    )
    counter = TiktokenCounter(encoding=encoding)  # type: ignore[arg-type]

    assert counter.fitting_suffix("abcdefgh", 0, 8, max_tokens=3) == 5
    assert counter.fitting_suffix("abcdefgh", 6, 8, max_tokens=3) == 6
    assert counter.fitting_suffix("abcdefgh", 0, 8, max_tokens=0) == 8


def test_conservative_suffix_counts_utf8_bytes_backward() -> None:
    counter = ConservativeUtf8TokenCounter()

    assert counter.fitting_suffix("abcdef", 0, 6, max_tokens=2) == 4
    assert counter.fitting_suffix("aé", 0, 2, max_tokens=2) == 1
    assert counter.fitting_suffix("abc", 0, 3, max_tokens=99) == 0


def test_tiktoken_prefix_uses_complete_token_bytes() -> None:
    encoding = SimpleNamespace(
        name="test",
        encode_ordinary=lambda text: {
            "删除x": [1, 2],
            "删除": [1],
            "删": [3, 4],
        }.get(text, list(text)),
    )
    counter = TiktokenCounter(encoding=encoding)  # type: ignore[arg-type]

    assert counter.fitting_prefix("删除x", 0, 3, max_tokens=1) == 2


def test_tiktoken_prefix_extends_past_unstable_full_text_boundary() -> None:
    encoding = SimpleNamespace(
        name="test",
        encode_ordinary=lambda text: {
            "X": [1],
            "XB": [2],
            "XBg": [2, 3],
        }.get(text, list(text)),
    )
    counter = TiktokenCounter(encoding=encoding)  # type: ignore[arg-type]

    assert counter.fitting_prefix("XBg", 0, 3, max_tokens=1) == 2


def test_tiktoken_prefix_search_is_bounded_near_the_budget() -> None:
    class TrackingEncoding:
        name = "tracking"

        def __init__(self) -> None:
            self.calls = 0
            self.total_characters = 0

        def encode_ordinary(self, text: str) -> list[str]:
            self.calls += 1
            self.total_characters += len(text)
            return list(text)

    encoding = TrackingEncoding()
    counter = TiktokenCounter(encoding=encoding)  # type: ignore[arg-type]

    assert counter.fitting_prefix("x" * 5_000, 0, 5_000, 2_049) == 2_049
    assert encoding.calls < 50
    assert encoding.total_characters < 100_000
