"""Provider-independent token accounting for segmentation budgets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import tiktoken
from tiktoken.core import Encoding


class TokenAccountingError(ValueError):
    """Raised when a token counter cannot provide a valid budget value."""


class TokenCounter(Protocol):
    """Minimal injectable token-counting boundary."""

    @property
    def identity(self) -> str: ...

    @property
    def exact(self) -> bool: ...

    @property
    def monotonic(self) -> bool: ...

    def count(self, text: str) -> int: ...


@runtime_checkable
class PrefixTokenCounter(Protocol):
    def fitting_prefix(
        self,
        text: str,
        start: int,
        end: int,
        max_tokens: int,
    ) -> int: ...


@runtime_checkable
class SuffixTokenCounter(Protocol):
    def fitting_suffix(
        self,
        text: str,
        floor: int,
        end: int,
        max_tokens: int,
    ) -> int: ...


# Width of the window searched exactly at the end of a boundary search. BPE
# token counts are not monotonic in text length, and the instability is *not*
# bounded by any single token's byte length: with cl100k_base, "a" * 3000 at a
# 12-token budget fits through 96 characters even though 93 characters already
# exceeds it. No window makes a coarse search exact, so this is a packing
# density knob only. Every returned boundary is verified by an exact recount,
# and callers must not assume a returned boundary is the largest one possible.
_UNSTABLE_TOKEN_WINDOW_CHARS = 16


@dataclass(frozen=True)
class TiktokenCounter:
    """Count exactly for a selected tiktoken encoding."""

    encoding: Encoding

    @classmethod
    def for_model(cls, model: str) -> TiktokenCounter:
        try:
            encoding = tiktoken.encoding_for_model(model)
        except KeyError as error:
            raise TokenAccountingError(
                f"no tiktoken encoding is registered for model {model!r}; "
                "provide encoding_name explicitly"
            ) from error
        return cls(encoding)

    @classmethod
    def for_encoding(cls, encoding_name: str) -> TiktokenCounter:
        try:
            encoding = tiktoken.get_encoding(encoding_name)
        except ValueError as error:
            raise TokenAccountingError(
                f"unknown tiktoken encoding {encoding_name!r}"
            ) from error
        return cls(encoding)

    @property
    def identity(self) -> str:
        return f"tiktoken:{self.encoding.name}"

    @property
    def exact(self) -> bool:
        return True

    @property
    def monotonic(self) -> bool:
        return False

    def count(self, text: str) -> int:
        return len(self.encoding.encode_ordinary(text))

    def fitting_prefix(
        self,
        text: str,
        start: int,
        end: int,
        max_tokens: int,
    ) -> int:
        """Return an end offset whose prefix is verified to fit `max_tokens`.

        The result is exact for its own slice but is not guaranteed to be the
        largest fitting offset: a coarse search cannot be exact against a
        non-monotonic encoding. Under-packing is safe; callers must never treat
        the result as maximal, nor assume a smaller offset also fits.
        """
        if self.count(text[start:end]) <= max_tokens:
            return end

        low, high = start, end
        while high - low > _UNSTABLE_TOKEN_WINDOW_CHARS:
            mid = (low + high) // 2
            if self.count(text[start:mid]) <= max_tokens:
                low = mid
            else:
                high = mid

        best = low
        for candidate in range(low + 1, high + 1):
            if self.count(text[start:candidate]) <= max_tokens:
                best = candidate
        return best

    def fitting_suffix(
        self,
        text: str,
        floor: int,
        end: int,
        max_tokens: int,
    ) -> int:
        """Return a start offset whose suffix is verified to fit `max_tokens`.

        The backward mirror of `fitting_prefix`, carrying the same contract: the
        returned slice is verified, but is not guaranteed to be the longest
        fitting suffix.
        """
        if self.count(text[floor:end]) <= max_tokens:
            return floor

        low, high = floor, end
        while high - low > _UNSTABLE_TOKEN_WINDOW_CHARS:
            mid = (low + high) // 2
            if self.count(text[mid:end]) <= max_tokens:
                high = mid
            else:
                low = mid

        best = high
        for candidate in range(high - 1, low - 1, -1):
            if self.count(text[candidate:end]) <= max_tokens:
                best = candidate
        return best


@dataclass(frozen=True)
class ConservativeUtf8TokenCounter:
    """Use UTF-8 bytes as a conservative offline token estimate."""

    identity: str = "estimate:utf8-bytes"
    exact: bool = False
    monotonic: bool = True

    def count(self, text: str) -> int:
        return len(text.encode("utf-8"))

    def fitting_prefix(
        self,
        text: str,
        start: int,
        end: int,
        max_tokens: int,
    ) -> int:
        byte_count = 0
        for index in range(start, end):
            byte_count += len(text[index].encode("utf-8"))
            if byte_count > max_tokens:
                return index
        return end

    def fitting_suffix(
        self,
        text: str,
        floor: int,
        end: int,
        max_tokens: int,
    ) -> int:
        byte_count = 0
        for index in range(end - 1, floor - 1, -1):
            byte_count += len(text[index].encode("utf-8"))
            if byte_count > max_tokens:
                return index + 1
        return floor


def resolve_token_counter(
    *,
    provider: str,
    model: str,
    encoding_name: str | None = None,
) -> TokenCounter:
    """Resolve a counter without constructing or calling a model provider.

    Constructing a tiktoken counter may download an uncached vocabulary.
    Once constructed, counter calls perform local encoding only.
    """
    if encoding_name is not None:
        return TiktokenCounter.for_encoding(encoding_name)
    if provider.strip().lower() != "openai":
        return ConservativeUtf8TokenCounter()
    return TiktokenCounter.for_model(model)
