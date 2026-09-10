"""Thread-safe, secret-free observations for audit/3 reliability metadata."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from threading import Lock

from summarizer.providers.base import GenerationRequest, GenerationResult, RetryAttempt

_INVALIDATION_CODES = frozenset(
    {
        "source_changed",
        "prompt_changed",
        "schema_changed",
        "model_changed",
        "behavior_changed",
    }
)


@dataclass(frozen=True)
class ReliabilitySnapshot:
    cache: Mapping[str, tuple[str, ...]]
    resumed: bool
    reused_count: int
    recomputed_count: int
    attempts: tuple[Mapping[str, object], ...]


class ReliabilityTracker:
    """Aggregate closed outcomes in the manifest's deterministic work order."""

    def __init__(
        self, work_order: Callable[[], tuple[str, ...]], *, resumed: bool
    ) -> None:
        self._work_order = work_order
        self._resumed = resumed
        self._cache: dict[str, tuple[bool, str]] = {}
        self._invalidation_reasons: set[str] = set()
        self._attempt_counts: dict[str, int] = {}
        self._failures: dict[str, list[str]] = {}
        self._lock = Lock()

    def record_cache_hit(self, work_id: str) -> None:
        self._record_cache(work_id, hit=True, code="hit")

    def record_cache_miss(self, work_id: str, reason: str) -> None:
        self._record_cache(work_id, hit=False, code=reason)

    def _record_cache(self, work_id: str, *, hit: bool, code: str) -> None:
        if work_id not in self._work_order():
            return
        with self._lock:
            self._cache[work_id] = (hit, code)

    def record_invalidation_reasons(self, reasons: tuple[str, ...]) -> None:
        with self._lock:
            self._invalidation_reasons.update(
                reason for reason in reasons if reason in _INVALIDATION_CODES
            )

    def record_generation(
        self, request: GenerationRequest, result: GenerationResult
    ) -> None:
        self._record_attempts(
            request,
            attempt_count=1 + len(result.retry_attempts),
            retry_attempts=result.retry_attempts,
        )

    def manifest_work_order(self) -> tuple[str, ...]:
        """Return the stable work order used by reliability projections."""
        return self._work_order()

    def record_retry_exhaustion(
        self,
        request: GenerationRequest,
        *,
        attempt_count: int,
        retry_attempts: tuple[RetryAttempt, ...],
    ) -> None:
        self._record_attempts(
            request,
            attempt_count=attempt_count,
            retry_attempts=retry_attempts,
        )

    def _record_attempts(
        self,
        request: GenerationRequest,
        *,
        attempt_count: int,
        retry_attempts: tuple[RetryAttempt, ...],
    ) -> None:
        work_id = request.audit_work_id or request.operation_id
        if work_id not in self._work_order():
            return
        with self._lock:
            self._attempt_counts[work_id] = (
                self._attempt_counts.get(work_id, 0) + attempt_count
            )
            self._failures.setdefault(work_id, []).extend(
                attempt.error_category.value for attempt in retry_attempts
            )

    def snapshot(self) -> ReliabilitySnapshot:
        order = self._work_order()
        with self._lock:
            cache = dict(self._cache)
            invalidation_reasons = tuple(sorted(self._invalidation_reasons))
            attempt_counts = dict(self._attempt_counts)
            failures = {
                work_id: tuple(sorted(set(codes)))
                for work_id, codes in self._failures.items()
            }
        ordered_cache = tuple(cache[work_id] for work_id in order if work_id in cache)
        hits = tuple(code for hit, code in ordered_cache if hit)
        misses = tuple(code for hit, code in ordered_cache if not hit)
        attempts = tuple(
            {
                "work_id": work_id,
                "attempt_count": attempt_counts[work_id],
                "failure_reasons": failures.get(work_id, ()),
            }
            for work_id in order
            if work_id in attempt_counts
        )
        return ReliabilitySnapshot(
            cache={
                "cache_hits": hits,
                "cache_misses": misses,
                "invalidation_reasons": invalidation_reasons,
            },
            resumed=self._resumed,
            reused_count=len(hits),
            recomputed_count=len(misses),
            attempts=attempts,
        )
