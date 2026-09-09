"""Thread-safe, secret-free observations for audit/3 reliability metadata."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from threading import Lock

from summarizer.providers.base import GenerationRequest, GenerationResult


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
        self._logical_calls: dict[str, int] = {}
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

    def record_generation(
        self, request: GenerationRequest, result: GenerationResult
    ) -> None:
        work_id = request.audit_work_id or request.operation_id
        if work_id not in self._work_order():
            return
        with self._lock:
            self._logical_calls[work_id] = self._logical_calls.get(work_id, 0) + 1
            self._failures.setdefault(work_id, []).extend(
                attempt.error_category.value for attempt in result.retry_attempts
            )

    def snapshot(self) -> ReliabilitySnapshot:
        order = self._work_order()
        with self._lock:
            cache = dict(self._cache)
            logical_calls = dict(self._logical_calls)
            failures = {
                work_id: tuple(sorted(codes))
                for work_id, codes in self._failures.items()
            }
        ordered_cache = tuple(cache[work_id] for work_id in order if work_id in cache)
        hits = tuple(code for hit, code in ordered_cache if hit)
        misses = tuple(code for hit, code in ordered_cache if not hit)
        attempts = tuple(
            {
                "work_id": work_id,
                "attempt_count": logical_calls[work_id] + len(failures.get(work_id, ())),
                "failure_reasons": failures.get(work_id, ()),
            }
            for work_id in order
            if work_id in logical_calls
        )
        return ReliabilitySnapshot(
            cache={
                "cache_hits": hits,
                "cache_misses": misses,
                "invalidation_reasons": (),
            },
            resumed=self._resumed,
            reused_count=len(hits),
            recomputed_count=len(misses),
            attempts=attempts,
        )
