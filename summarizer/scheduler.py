"""Bounded, ordered local work with durable failure recovery state."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from concurrent.futures import CancelledError, Executor, Future, ThreadPoolExecutor
from dataclasses import dataclass
from queue import Empty, SimpleQueue

from summarizer.cache import CacheDescriptor, CacheStore
from summarizer.checkpoint import (
    CheckpointSession,
    CompletedRef,
    NonReusableReason,
    NonReusableRef,
)


@dataclass(frozen=True)
class ScheduledWork:
    """One independently executable, cacheable unit with a stable work ID."""

    descriptor: CacheDescriptor
    operation: Callable[[], object]
    validate: Callable[[object], object]

    @property
    def work_id(self) -> str:
        return self.descriptor.work_id


@dataclass(frozen=True)
class ScheduledResult:
    work_id: str
    payload: object


class BoundedScheduler:
    """Run independent work locally without changing result or level order."""

    def __init__(
        self,
        *,
        max_in_flight: int,
        cache: CacheStore,
        executor_factory: Callable[[int], Executor] = ThreadPoolExecutor,
    ) -> None:
        if not isinstance(max_in_flight, int) or isinstance(max_in_flight, bool):
            raise TypeError("max_in_flight must be an integer")
        if max_in_flight <= 0:
            raise ValueError("max_in_flight must be positive")
        self._max_in_flight = max_in_flight
        self._cache = cache
        self._executor_factory = executor_factory

    def run(
        self,
        work: tuple[ScheduledWork, ...],
        session: CheckpointSession,
    ) -> tuple[ScheduledResult, ...]:
        self._validate_work(work, session)
        if not work:
            return ()

        positions = {
            work_id: index for index, work_id in enumerate(session.manifest.work_ids)
        }
        descriptors = {item.work_id: item.descriptor for item in work}
        completed: dict[str, CompletedRef] = {}
        results: dict[str, ScheduledResult] = {}
        non_reusable: dict[str, NonReusableReason] = {}
        pending: dict[Future[object], ScheduledWork] = {}
        completions: SimpleQueue[Future[object]] = SimpleQueue()
        next_index = 0
        terminal_error: Exception | None = None
        terminal_work_id: str | None = None
        cleanup_error: Exception | None = None
        executor: Executor | None = None

        def record_non_reusable(work_id: str, reason: NonReusableReason) -> None:
            non_reusable.setdefault(work_id, reason)

        def collect(future: Future[object], item: ScheduledWork) -> Exception | None:
            try:
                payload = future.result()
            except CancelledError as error:
                record_non_reusable(item.work_id, NonReusableReason.CANCELLED)
                return error
            except Exception as error:  # noqa: BLE001 - checkpoint all worker failures
                record_non_reusable(item.work_id, NonReusableReason.FAILED)
                return error
            try:
                validated = self._cache.store_winner(
                    item.descriptor, payload, item.validate
                )
                self._cache.record_descriptor_projection(item.descriptor)
            except Exception as error:  # noqa: BLE001 - preserve validation failures
                record_non_reusable(item.work_id, NonReusableReason.FAILED)
                return error
            completed[item.work_id] = CompletedRef(
                work_id=item.work_id,
                cache_key=item.descriptor.key,
            )
            results[item.work_id] = ScheduledResult(item.work_id, validated)
            return None

        def completed_futures() -> set[Future[object]]:
            done: set[Future[object]] = set()
            while not done:
                future = completions.get()
                if future in pending:
                    done.add(future)
            while True:
                try:
                    future = completions.get_nowait()
                except Empty:
                    return done
                if future in pending:
                    done.add(future)

        try:
            try:
                executor = self._executor_factory(self._max_in_flight)
            except Exception as error:  # noqa: BLE001 - no executor was observable
                terminal_error = error

            while executor is not None and (
                pending or (next_index < len(work) and terminal_error is None)
            ):
                while (
                    terminal_error is None
                    and next_index < len(work)
                    and len(pending) < self._max_in_flight
                ):
                    item = work[next_index]
                    next_index += 1
                    try:
                        future = executor.submit(item.operation)
                    except Exception as error:  # noqa: BLE001 - future is unavailable
                        record_non_reusable(
                            item.work_id, NonReusableReason.UNOBSERVABLE
                        )
                        terminal_error = error
                        terminal_work_id = item.work_id
                        break
                    pending[future] = item
                    future.add_done_callback(completions.put)

                if not pending:
                    continue

                done = completed_futures()
                for future in sorted(
                    done, key=lambda item: positions[pending[item].work_id]
                ):
                    item = pending.pop(future)
                    error = collect(future, item)
                    if terminal_error is None and error is not None:
                        terminal_error = error
                        terminal_work_id = item.work_id

                if terminal_error is not None:
                    for future, item in tuple(pending.items()):
                        if future.cancel():
                            pending.pop(future)
                            collect(future, item)
                    break

            while executor is not None and pending:
                done = completed_futures()
                for future in sorted(
                    done, key=lambda item: positions[pending[item].work_id]
                ):
                    item = pending.pop(future)
                    error = collect(future, item)
                    if terminal_error is None and error is not None:
                        terminal_error = error
                        terminal_work_id = item.work_id
        finally:
            try:
                if executor is not None:
                    executor.shutdown(wait=True)
            except Exception as error:  # noqa: BLE001 - retain cleanup failure as cause
                cleanup_error = error

        if terminal_error is None and cleanup_error is not None:
            terminal_error = cleanup_error

        if terminal_error is not None:
            for item in work[next_index:]:
                record_non_reusable(item.work_id, NonReusableReason.UNKNOWN)
            checkpoint_error = self._checkpoint(
                session,
                completed,
                descriptors,
                non_reusable,
                terminal_work_id,
                positions,
                terminal_failure=True,
            )
            if checkpoint_error is not None:
                raise terminal_error from checkpoint_error
            if cleanup_error is not None and cleanup_error is not terminal_error:
                raise terminal_error from cleanup_error
            raise terminal_error

        checkpoint_error = self._checkpoint(
            session,
            completed,
            descriptors,
            non_reusable,
            None,
            positions,
            terminal_failure=False,
        )
        if checkpoint_error is not None:
            raise checkpoint_error
        return tuple(
            results[work_id] for work_id in sorted(results, key=positions.__getitem__)
        )

    def run_levels(
        self,
        levels: Iterable[tuple[ScheduledWork, ...]],
        session: CheckpointSession,
    ) -> tuple[ScheduledResult, ...]:
        results: list[ScheduledResult] = []
        for level in levels:
            results.extend(self.run(level, session))
        positions = {
            work_id: index for index, work_id in enumerate(session.manifest.work_ids)
        }
        return tuple(sorted(results, key=lambda item: positions[item.work_id]))

    @staticmethod
    def _validate_work(
        work: tuple[ScheduledWork, ...], session: CheckpointSession
    ) -> None:
        work_ids = tuple(item.work_id for item in work)
        planned = set(session.manifest.work_ids)
        unavailable = {reference.work_id for reference in session.manifest.completed}
        if (
            len(set(work_ids)) != len(work_ids)
            or not set(work_ids).issubset(planned)
            or set(work_ids) & unavailable
        ):
            raise ValueError("work must be unique planned work without prior state")

    @staticmethod
    def _checkpoint(
        session: CheckpointSession,
        completed: Mapping[str, CompletedRef],
        descriptors: Mapping[str, CacheDescriptor],
        non_reusable: Mapping[str, NonReusableReason],
        terminal_work_id: str | None,
        positions: Mapping[str, int],
        *,
        terminal_failure: bool,
    ) -> Exception | None:
        try:
            session.checkpoint_scheduler_state(
                completed=tuple(
                    completed[work_id]
                    for work_id in sorted(completed, key=positions.__getitem__)
                ),
                descriptors=descriptors,
                non_reusable=tuple(
                    NonReusableRef(work_id=work_id, reason=non_reusable[work_id])
                    for work_id in sorted(non_reusable, key=positions.__getitem__)
                ),
                terminal_failure=terminal_failure,
                terminal_failure_work_id=terminal_work_id,
                clear_terminal_failure=terminal_work_id is None,
            )
        except Exception as error:  # noqa: BLE001 - caller receives the terminal work error
            return error
        return None
