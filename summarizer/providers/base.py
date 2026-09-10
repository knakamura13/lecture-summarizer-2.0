from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Protocol, runtime_checkable


@dataclass(frozen=True)
class GenerationRequest:
    model: str
    instructions: str
    input_text: str
    timeout_seconds: float
    operation_id: str | None = None
    # A JSON Schema the response should conform to, or None for prose. This is
    # the one representation both supported clients accept natively, so it
    # keeps structured output from coupling orchestration to a single SDK.
    # Adapters that cannot constrain decoding may ignore it; callers must
    # parse defensively either way.
    response_schema: Mapping[str, object] | None = None
    schema_name: str | None = None
    # Stable internal work identity for ordered diagnostics. Provider adapters
    # do not transmit it or include it in model input.
    audit_work_id: str | None = None

    def __post_init__(self) -> None:
        for field_name in ("model", "instructions", "input_text"):
            if not getattr(self, field_name).strip():
                raise ValueError(f"{field_name} must not be empty")
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if self.response_schema is not None and not (self.schema_name or "").strip():
            raise ValueError("schema_name is required when response_schema is set")


@dataclass(frozen=True)
class GenerationResult:
    text: str
    provider: str
    model: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    finish_status: str | None = None
    request_id: str | None = None
    retry_attempts: tuple[RetryAttempt, ...] = ()

    def __post_init__(self) -> None:
        for field_name in ("text", "provider", "model"):
            if not getattr(self, field_name).strip():
                raise ValueError(f"{field_name} must not be empty")
        for field_name in ("input_tokens", "output_tokens"):
            value = getattr(self, field_name)
            if value is not None and value < 0:
                raise ValueError(f"{field_name} must not be negative")
        retry_attempts = tuple(self.retry_attempts)
        if not all(isinstance(attempt, RetryAttempt) for attempt in retry_attempts):
            raise ValueError("retry_attempts must contain RetryAttempt values")
        object.__setattr__(self, "retry_attempts", retry_attempts)


def normalize_output_text(text: str, request: GenerationRequest) -> str:
    """Collapse whitespace in a prose response, but never in a structured one.

    The collapse tidies prose summaries. Applied to a structured response it
    silently corrupts verbatim quotations: a quote copied out of a segment
    containing a newline or a run of spaces comes back single-spaced and can no
    longer be located in the source it came from.
    """
    if request.response_schema is not None:
        return text
    return re.sub(r"\s+", " ", text.strip()).strip()


@runtime_checkable
class ModelProvider(Protocol):
    """Generate model output.

    `BoundedScheduler` may call `generate` concurrently when `max_in_flight`
    exceeds one. First-party providers support that usage; custom providers
    are responsible for making their own implementation thread-safe.
    """

    def generate(self, request: GenerationRequest) -> GenerationResult: ...


class ProviderError(RuntimeError):
    """Base class for provider-independent generation failures."""


class TransientProviderError(ProviderError):
    """A provider failure that may succeed when retried."""


class ProviderTimeoutError(TransientProviderError):
    pass


class ProviderRateLimitError(TransientProviderError):
    pass


class ProviderConnectionError(TransientProviderError):
    pass


class ProviderServerError(TransientProviderError):
    pass


class ProviderAuthenticationError(ProviderError):
    pass


class ProviderRequestError(ProviderError):
    pass


class ProviderResponseError(ProviderError):
    pass


class RetryErrorCategory(str, Enum):
    """Safe, closed categories for retry diagnostics."""

    TIMEOUT = "timeout"
    RATE_LIMIT = "rate_limit"
    CONNECTION = "connection"
    SERVER = "server"
    TRANSIENT = "transient"


@dataclass(frozen=True, slots=True)
class RetryAttempt:
    """Sanitized metadata for one transient provider failure."""

    attempt: int
    error_category: RetryErrorCategory
    planned_delay_seconds: float | None
    exhausted: bool
    recorded_at_seconds: float


class ProviderRetriesExhaustedError(ProviderError):
    def __init__(
        self,
        attempts: int,
        detail: str | None = None,
        retry_attempts: tuple[RetryAttempt, ...] = (),
    ) -> None:
        self.attempts = attempts
        self.retry_attempts = tuple(retry_attempts)
        message = f"Provider request failed after {attempts} attempts"
        if detail:
            message = f"{message}: {detail}"
        super().__init__(message)
