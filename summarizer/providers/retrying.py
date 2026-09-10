from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from random import random
from time import monotonic, sleep

from summarizer.config import RetryPolicy
from summarizer.providers.base import (
    GenerationRequest,
    GenerationResult,
    ModelProvider,
    ProviderConnectionError,
    ProviderRateLimitError,
    ProviderRetriesExhaustedError,
    ProviderServerError,
    ProviderTimeoutError,
    RetryAttempt,
    RetryErrorCategory,
    TransientProviderError,
)


class RetryingProvider:
    def __init__(
        self,
        provider: ModelProvider,
        policy: RetryPolicy,
        sleeper: Callable[[float], None] = sleep,
        clock: Callable[[], float] = monotonic,
        random_source: Callable[[], float] = random,
    ) -> None:
        self._provider = provider
        self._policy = policy
        self._sleeper = sleeper
        self._clock = clock
        self._random_source = random_source

    def generate(self, request: GenerationRequest) -> GenerationResult:
        retry_attempts: list[RetryAttempt] = []
        delay = min(
            self._policy.initial_delay_seconds,
            self._policy.max_delay_seconds,
        )
        for attempt in range(1, self._policy.max_attempts + 1):
            try:
                result = self._provider.generate(request)
                if not retry_attempts:
                    return result
                return replace(result, retry_attempts=tuple(retry_attempts))
            except TransientProviderError as error:
                exhausted = attempt == self._policy.max_attempts
                planned_delay = None if exhausted else self._jittered_delay(delay)
                retry_attempts.append(
                    RetryAttempt(
                        attempt=attempt,
                        error_category=self._error_category(error),
                        planned_delay_seconds=planned_delay,
                        exhausted=exhausted,
                        recorded_at_seconds=self._clock(),
                    )
                )
                if exhausted:
                    raise ProviderRetriesExhaustedError(
                        attempt,
                        str(error),
                        tuple(retry_attempts),
                    ) from error
                assert planned_delay is not None
                self._sleeper(planned_delay)
                delay = min(
                    self._policy.max_delay_seconds,
                    delay * self._policy.backoff_multiplier,
                )
        raise AssertionError("retry loop completed without a result")

    def _jittered_delay(self, delay: float) -> float:
        if self._policy.jitter_fraction == 0:
            return delay
        random_value = min(1.0, max(0.0, self._random_source()))
        multiplier = 1 + self._policy.jitter_fraction * (2 * random_value - 1)
        return min(self._policy.max_delay_seconds, delay * multiplier)

    @staticmethod
    def _error_category(error: TransientProviderError) -> RetryErrorCategory:
        if isinstance(error, ProviderTimeoutError):
            return RetryErrorCategory.TIMEOUT
        if isinstance(error, ProviderRateLimitError):
            return RetryErrorCategory.RATE_LIMIT
        if isinstance(error, ProviderConnectionError):
            return RetryErrorCategory.CONNECTION
        if isinstance(error, ProviderServerError):
            return RetryErrorCategory.SERVER
        return RetryErrorCategory.TRANSIENT
