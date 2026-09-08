from collections import deque
from dataclasses import FrozenInstanceError, dataclass, field
from threading import Event, Thread

import pytest

from summarizer.config import RetryPolicy
from summarizer.providers.base import (
    GenerationRequest,
    GenerationResult,
    ProviderAuthenticationError,
    ProviderConnectionError,
    ProviderRateLimitError,
    ProviderRequestError,
    ProviderResponseError,
    ProviderRetriesExhaustedError,
    ProviderServerError,
    ProviderTimeoutError,
    TransientProviderError,
)
from summarizer.providers.retrying import RetryingProvider

REQUEST = GenerationRequest("model", "instructions", "input", 30)
RESULT = GenerationResult("summary", "fake", "model")


@dataclass
class ScriptedProvider:
    outcomes: deque[object]
    calls: list[GenerationRequest] = field(default_factory=list)

    def generate(self, request: GenerationRequest) -> GenerationResult:
        self.calls.append(request)
        outcome = self.outcomes.popleft()
        if isinstance(outcome, BaseException):
            raise outcome
        assert isinstance(outcome, GenerationResult)
        return outcome


def test_transient_failures_retry_with_configured_backoff() -> None:
    provider = ScriptedProvider(
        deque(
            [
                ProviderConnectionError("first"),
                ProviderConnectionError("second"),
                RESULT,
            ]
        )
    )
    delays: list[float] = []
    retrying = RetryingProvider(
        provider,
        RetryPolicy(max_attempts=3),
        sleeper=delays.append,
    )

    result = retrying.generate(REQUEST)

    assert result.text == RESULT.text
    assert provider.calls == [REQUEST, REQUEST, REQUEST]
    assert delays == [1, 2]
    assert tuple(record.planned_delay_seconds for record in result.retry_attempts) == (
        1,
        2,
    )


def test_success_without_retries_preserves_provider_result() -> None:
    retrying = RetryingProvider(ScriptedProvider(deque([RESULT])), RetryPolicy())

    assert retrying.generate(REQUEST) is RESULT


def test_transient_failures_use_bounded_jitter_and_record_safe_attempts() -> None:
    provider = ScriptedProvider(
        deque(
            [
                ProviderConnectionError("https://api.example.test leaked"),
                ProviderConnectionError("second"),
                RESULT,
            ]
        )
    )
    delays: list[float] = []
    clock_values = iter([100.0, 101.0])
    random_values = iter([0.0, 1.0])
    retrying = RetryingProvider(
        provider,
        RetryPolicy(
            max_attempts=3,
            initial_delay_seconds=2,
            backoff_multiplier=3,
            max_delay_seconds=7,
            jitter_fraction=0.5,
        ),
        sleeper=delays.append,
        clock=lambda: next(clock_values),
        random_source=lambda: next(random_values),
    )

    result = retrying.generate(REQUEST)
    assert delays == [1, 7]
    assert len(result.retry_attempts) == 2
    record = result.retry_attempts[0]
    assert record.attempt == 1
    assert record.error_category.value == "connection"
    assert record.planned_delay_seconds == 1
    assert record.exhausted is False
    assert record.recorded_at_seconds == 100
    assert "api.example" not in repr(record)
    with pytest.raises(FrozenInstanceError):
        record.attempt = 2  # type: ignore[misc]

    assert result.retry_attempts[1].attempt == 2
    assert result.retry_attempts[1].planned_delay_seconds == 7
    assert result.retry_attempts[1].recorded_at_seconds == 101


def test_concurrent_generations_keep_retry_attempts_on_their_own_results() -> None:
    class InterleavingProvider:
        def __init__(self) -> None:
            self.a_calls = 0

        def generate(self, request: GenerationRequest) -> GenerationResult:
            if request.operation_id == "a":
                self.a_calls += 1
                if self.a_calls == 1:
                    raise ProviderConnectionError("a transient failure")
                return GenerationResult("a result", "fake", "model")
            return GenerationResult("b result", "fake", "model")

    a_sleeping = Event()
    release_a = Event()

    def blocking_sleeper(_: float) -> None:
        a_sleeping.set()
        assert release_a.wait(timeout=1)

    retrying = RetryingProvider(
        InterleavingProvider(),
        RetryPolicy(max_attempts=2),
        sleeper=blocking_sleeper,
        clock=lambda: 1,
    )
    results: dict[str, GenerationResult] = {}
    a_request = GenerationRequest("model", "instructions", "a input", 30, "a")
    b_request = GenerationRequest("model", "instructions", "b input", 30, "b")
    a_thread = Thread(target=lambda: results.setdefault("a", retrying.generate(a_request)))

    a_thread.start()
    assert a_sleeping.wait(timeout=1)
    results["b"] = retrying.generate(b_request)
    release_a.set()
    a_thread.join(timeout=1)
    assert not a_thread.is_alive()

    assert [record.error_category.value for record in results["a"].retry_attempts] == [
        "connection"
    ]
    assert results["b"].retry_attempts == ()


@pytest.mark.parametrize(
    ("failure", "expected_category"),
    [
        (ProviderTimeoutError("timeout"), "timeout"),
        (ProviderRateLimitError("rate limit"), "rate_limit"),
        (ProviderConnectionError("connection"), "connection"),
        (ProviderServerError("server"), "server"),
        (TransientProviderError("other"), "transient"),
    ],
)
def test_transient_attempts_use_closed_error_categories(
    failure: TransientProviderError,
    expected_category: str,
) -> None:
    retrying = RetryingProvider(
        ScriptedProvider(deque([failure])),
        RetryPolicy(max_attempts=1),
        clock=lambda: 1,
    )

    with pytest.raises(ProviderRetriesExhaustedError) as caught:
        retrying.generate(REQUEST)

    assert caught.value.retry_attempts[0].error_category.value == expected_category


@pytest.mark.parametrize(
    "failure",
    [
        ProviderAuthenticationError("unauthorized"),
        ProviderRequestError("invalid request"),
        ProviderResponseError("invalid response"),
        ValueError("invalid configuration"),
    ],
)
def test_terminal_failures_are_not_retried(failure: BaseException) -> None:
    provider = ScriptedProvider(deque([failure]))
    delays: list[float] = []
    retrying = RetryingProvider(provider, RetryPolicy(), sleeper=delays.append)

    try:
        retrying.generate(REQUEST)
    except type(failure) as caught:
        assert caught is failure
    else:
        raise AssertionError("fatal provider failure was not raised")

    assert provider.calls == [REQUEST]
    assert delays == []


def test_exhaustion_preserves_last_failure_as_cause() -> None:
    first = ProviderConnectionError("first")
    last = ProviderConnectionError("last")
    provider = ScriptedProvider(deque([first, last]))
    delays: list[float] = []
    retrying = RetryingProvider(
        provider,
        RetryPolicy(max_attempts=2),
        sleeper=delays.append,
    )

    exhaustion: ProviderRetriesExhaustedError | None = None
    try:
        retrying.generate(REQUEST)
    except ProviderRetriesExhaustedError as caught:
        assert caught.attempts == 2
        assert caught.__cause__ is last
        assert "last" in str(caught)
        exhaustion = caught
    else:
        raise AssertionError("retry exhaustion was not raised")

    assert delays == [1]
    assert exhaustion is not None
    assert len(exhaustion.retry_attempts) == 2
    assert exhaustion.retry_attempts[-1].error_category.value == "connection"
    assert exhaustion.retry_attempts[-1].planned_delay_seconds is None
    assert exhaustion.retry_attempts[-1].exhausted is True


def test_single_attempt_never_sleeps() -> None:
    failure = ProviderConnectionError("offline")
    provider = ScriptedProvider(deque([failure]))
    delays: list[float] = []
    retrying = RetryingProvider(
        provider,
        RetryPolicy(max_attempts=1),
        sleeper=delays.append,
    )

    try:
        retrying.generate(REQUEST)
    except ProviderRetriesExhaustedError:
        pass
    else:
        raise AssertionError("retry exhaustion was not raised")

    assert provider.calls == [REQUEST]
    assert delays == []
