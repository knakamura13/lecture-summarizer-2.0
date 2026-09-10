from __future__ import annotations

import hashlib
from collections.abc import Callable
from concurrent.futures import CancelledError, Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from threading import Event, Lock, Thread

import pytest

from summarizer.cache import CacheDescriptor, CacheStore
from summarizer.checkpoint import (
    CheckpointStore,
    NonReusableReason,
    RunPlan,
)
from summarizer.scheduler import BoundedScheduler, ScheduledWork


def _digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _plan(*work_ids: str) -> RunPlan:
    return RunPlan(
        run_id="run-20260908",
        descriptor_sha256=_digest(b"run descriptor"),
        source_sha256=_digest(b"canonical source"),
        work_ids=work_ids,
    )


def _descriptor(work_id: str) -> CacheDescriptor:
    return CacheDescriptor(
        source_id=_digest(b"canonical source"),
        input_hash=_digest(work_id.encode("utf-8")),
        stage="leaf",
        work_id=work_id,
        prompt_version="leaf/1",
        schema_version="summary/1",
        provider="openai",
        model="gpt-4o-mini",
        counter_identity="tiktoken:o200k_base",
        counter_exact=True,
        context_window_tokens=128_000,
        behavior={"max_output_tokens": 1024},
    )


def _validate(payload: object) -> dict[str, str]:
    if not isinstance(payload, dict) or set(payload) != {"summary"}:
        raise ValueError("expected a summary payload")
    summary = payload["summary"]
    if not isinstance(summary, str) or not summary:
        raise ValueError("summary must be nonblank")
    return {"summary": summary}


def _work(work_id: str, operation: Callable[[], object]) -> ScheduledWork:
    return ScheduledWork(_descriptor(work_id), operation, _validate)


def test_scheduler_caps_in_flight_work_and_returns_work_id_order(
    tmp_path: Path,
) -> None:
    active = 0
    maximum_active = 0
    started = Event()
    release = Event()
    lock = Lock()

    def operation(work_id: str) -> Callable[[], object]:
        def run() -> object:
            nonlocal active, maximum_active
            with lock:
                active += 1
                maximum_active = max(maximum_active, active)
                if active == 2:
                    started.set()
            release.wait(timeout=2)
            with lock:
                active -= 1
            return {"summary": work_id}

        return run

    work_ids = ("S000001", "S000002", "S000003")
    store = CheckpointStore(tmp_path / "cache")
    scheduler = BoundedScheduler(max_in_flight=2, cache=CacheStore(tmp_path / "cache"))
    result: list[object] = []

    with store.open(_plan(*work_ids), resume=False) as session:
        thread = Thread(
            target=lambda: result.extend(
                scheduler.run(
                    tuple(_work(work_id, operation(work_id)) for work_id in work_ids),
                    session,
                )
            )
        )
        thread.start()
        assert started.wait(timeout=2)
        assert maximum_active == 2
        release.set()
        thread.join(timeout=2)
        assert not thread.is_alive()

    assert [item.work_id for item in result] == list(work_ids)
    assert maximum_active == 2


def test_scheduler_waits_for_an_entire_level_before_submitting_the_next(
    tmp_path: Path,
) -> None:
    second_started = Event()
    release_second = Event()
    next_level_started = Event()
    store = CheckpointStore(tmp_path / "cache")
    scheduler = BoundedScheduler(max_in_flight=2, cache=CacheStore(tmp_path / "cache"))

    def wait_for_release() -> object:
        second_started.set()
        assert release_second.wait(timeout=2)
        return {"summary": "second"}

    def next_level() -> object:
        next_level_started.set()
        return {"summary": "next"}

    with store.open(_plan("S000001", "S000002", "M000001"), resume=False) as session:
        thread = Thread(
            target=lambda: scheduler.run_levels(
                (
                    (
                        _work("S000001", lambda: {"summary": "first"}),
                        _work("S000002", wait_for_release),
                    ),
                    (_work("M000001", next_level),),
                ),
                session,
            )
        )
        thread.start()
        assert second_started.wait(timeout=2)
        assert not next_level_started.is_set()
        release_second.set()
        thread.join(timeout=2)
        assert not thread.is_alive()

    assert next_level_started.is_set()


def test_scheduler_returns_the_locked_first_writer_payload_for_concurrent_runs(
    tmp_path: Path,
) -> None:
    cache_root = tmp_path / "cache"
    checkpoint_store = CheckpointStore(cache_root)
    descriptor = _descriptor("D000001")
    started = Event()
    release = Event()

    def delayed_loser() -> object:
        started.set()
        assert release.wait(timeout=2)
        return {"summary": "loser"}

    def plan(run_id: str) -> RunPlan:
        return RunPlan(
            run_id=run_id,
            descriptor_sha256=_digest(run_id.encode("utf-8")),
            source_sha256=_digest(b"canonical source"),
            work_ids=("D000001",),
        )

    with checkpoint_store.open(plan("scheduler-race-first"), resume=False) as first:
        with checkpoint_store.open(plan("scheduler-race-second"), resume=False) as second:
            with ThreadPoolExecutor(max_workers=1) as executor:
                first_result = executor.submit(
                    BoundedScheduler(
                        max_in_flight=1, cache=CacheStore(cache_root)
                    ).run,
                    (_work("D000001", delayed_loser),),
                    first,
                )
                assert started.wait(timeout=2)
                second_result = BoundedScheduler(
                    max_in_flight=1, cache=CacheStore(cache_root)
                ).run(
                    (_work("D000001", lambda: {"summary": "winner"}),),
                    second,
                )
                release.set()
                first_result_value = first_result.result(timeout=2)

            winner = CacheStore(cache_root).load(descriptor, _validate).payload
            assert winner == {"summary": "winner"}
            assert first_result_value[0].payload == winner
            assert second_result[0].payload == winner
            for session in (first, second):
                assert session.reusable_for(
                    work_ids=("D000001",),
                    descriptors={"D000001": descriptor},
                    validators={"D000001": _validate},
                )[0].payload == winner


class _DrainingFuture(Future[object]):
    def __init__(self, value: object) -> None:
        super().__init__()
        self._value = value
        self.cancel_attempted = False

    def cancel(self) -> bool:
        self.cancel_attempted = True
        self.set_result(self._value)
        return False


@dataclass
class _Executor:
    futures: list[Future[object]]
    submitted: list[Callable[[], object]]

    def shutdown(self, *, wait: bool) -> None:
        assert wait

    def submit(self, operation: Callable[[], object]) -> Future[object]:
        self.submitted.append(operation)
        return self.futures.pop(0)


def test_failure_drains_successful_sibling_and_resume_reuses_it(tmp_path: Path) -> None:
    failing: Future[object] = Future()
    failing.set_exception(RuntimeError("first failure"))
    sibling = _DrainingFuture({"summary": "durable sibling"})
    executor = _Executor([failing, sibling], [])
    work_ids = ("S000001", "S000002", "S000003")
    calls = {work_id: 0 for work_id in work_ids}

    def operation(work_id: str) -> Callable[[], object]:
        def run() -> object:
            calls[work_id] += 1
            return {"summary": work_id}

        return run

    cache_root = tmp_path / "cache"
    scheduler = BoundedScheduler(
        max_in_flight=2,
        cache=CacheStore(cache_root),
        executor_factory=lambda _: executor,
    )
    store = CheckpointStore(cache_root)

    with store.open(_plan(*work_ids), resume=False) as session:
        with pytest.raises(RuntimeError, match="first failure"):
            scheduler.run(
                tuple(_work(work_id, operation(work_id)) for work_id in work_ids),
                session,
            )
        assert tuple(reference.work_id for reference in session.manifest.completed) == (
            "S000002",
        )
        assert tuple(
            reference.work_id for reference in session.manifest.non_reusable
        ) == (
            "S000001",
            "S000003",
        )
        assert session.manifest.non_reusable[0].reason is NonReusableReason.FAILED
        assert session.manifest.non_reusable[1].reason is NonReusableReason.UNKNOWN
        assert session.manifest.terminal_failure_work_id == "S000001"

    assert sibling.cancel_attempted
    assert len(executor.submitted) == 2
    assert calls == {work_id: 0 for work_id in work_ids}

    with store.open(_plan(*work_ids), resume=True) as session:
        BoundedScheduler(max_in_flight=1, cache=CacheStore(cache_root)).run(
            (
                _work("S000001", operation("S000001")),
                _work("S000003", operation("S000003")),
            ),
            session,
        )
        reusable = session.reusable(
            descriptors={"S000002": _descriptor("S000002")},
            validators={"S000002": _validate},
        )
        assert (
            tuple(reference.work_id for reference in session.manifest.completed)
            == work_ids
        )
        assert not session.manifest.non_reusable
        assert session.manifest.terminal_failure_work_id is None

    assert reusable[1].payload == {"summary": "durable sibling"}
    assert calls == {"S000001": 1, "S000002": 0, "S000003": 1}


def test_failure_marks_a_cancelled_sibling_non_reusable(tmp_path: Path) -> None:
    failing: Future[object] = Future()
    failing.set_exception(RuntimeError("first failure"))
    cancelled: Future[object] = Future()
    executor = _Executor([failing, cancelled], [])
    cache_root = tmp_path / "cache"
    scheduler = BoundedScheduler(
        max_in_flight=2,
        cache=CacheStore(cache_root),
        executor_factory=lambda _: executor,
    )
    store = CheckpointStore(cache_root)

    with store.open(_plan("S000001", "S000002"), resume=False) as session:
        with pytest.raises(RuntimeError, match="first failure"):
            scheduler.run(
                (
                    _work("S000001", lambda: {"summary": "failure"}),
                    _work("S000002", lambda: {"summary": "cancelled"}),
                ),
                session,
            )

        assert tuple(
            (entry.work_id, entry.reason) for entry in session.manifest.non_reusable
        ) == (
            ("S000001", NonReusableReason.FAILED),
            ("S000002", NonReusableReason.CANCELLED),
        )


def test_submission_failure_marks_unobservable_and_unsubmitted_work_unknown(
    tmp_path: Path,
) -> None:
    class _BrokenExecutor(_Executor):
        def submit(self, operation: Callable[[], object]) -> Future[object]:
            self.submitted.append(operation)
            raise OSError("executor unavailable")

    executor = _BrokenExecutor([], [])
    work_ids = ("S000001", "S000002")
    cache_root = tmp_path / "cache"
    scheduler = BoundedScheduler(
        max_in_flight=1,
        cache=CacheStore(cache_root),
        executor_factory=lambda _: executor,
    )
    store = CheckpointStore(cache_root)

    with store.open(_plan(*work_ids), resume=False) as session:
        with pytest.raises(OSError, match="executor unavailable"):
            scheduler.run(
                tuple(
                    _work(work_id, lambda work_id=work_id: {"summary": work_id})
                    for work_id in work_ids
                ),
                session,
            )
        assert tuple(
            (entry.work_id, entry.reason) for entry in session.manifest.non_reusable
        ) == (
            ("S000001", NonReusableReason.UNOBSERVABLE),
            ("S000002", NonReusableReason.UNKNOWN),
        )


def test_factory_failure_checkpoints_all_unsubmitted_work_before_reraising(
    tmp_path: Path,
) -> None:
    def factory(_: int) -> _Executor:
        raise OSError("factory failed")

    work_ids = ("S000001", "S000002")
    cache_root = tmp_path / "cache"
    scheduler = BoundedScheduler(
        max_in_flight=1,
        cache=CacheStore(cache_root),
        executor_factory=factory,
    )
    store = CheckpointStore(cache_root)

    with store.open(_plan(*work_ids), resume=False) as session:
        with pytest.raises(OSError, match="factory failed"):
            scheduler.run(
                tuple(
                    _work(work_id, lambda work_id=work_id: {"summary": work_id})
                    for work_id in work_ids
                ),
                session,
            )

        assert not session.manifest.completed
        assert tuple(
            (entry.work_id, entry.reason) for entry in session.manifest.non_reusable
        ) == tuple((work_id, NonReusableReason.UNKNOWN) for work_id in work_ids)
        assert session.manifest.terminal_failure
        assert session.manifest.terminal_failure_work_id is None


def test_shutdown_failure_checkpoints_drained_success_before_reraising(
    tmp_path: Path,
) -> None:
    class _ShutdownFailingExecutor(_Executor):
        shutdown_calls = 0

        def shutdown(self, *, wait: bool) -> None:
            assert wait
            self.shutdown_calls += 1
            raise OSError("shutdown failed")

    completed: Future[object] = Future()
    completed.set_result({"summary": "durable"})
    executor = _ShutdownFailingExecutor([completed], [])
    cache_root = tmp_path / "cache"
    scheduler = BoundedScheduler(
        max_in_flight=1,
        cache=CacheStore(cache_root),
        executor_factory=lambda _: executor,
    )
    store = CheckpointStore(cache_root)

    with store.open(_plan("S000001"), resume=False) as session:
        with pytest.raises(OSError, match="shutdown failed"):
            scheduler.run(
                (_work("S000001", lambda: {"summary": "not used"}),),
                session,
            )

        assert tuple(reference.work_id for reference in session.manifest.completed) == (
            "S000001",
        )
        assert not session.manifest.non_reusable
        assert session.manifest.terminal_failure
        assert session.manifest.terminal_failure_work_id is None

    assert executor.shutdown_calls == 1


def test_shutdown_failure_is_chained_without_replacing_work_failure(
    tmp_path: Path,
) -> None:
    class _ShutdownFailingExecutor(_Executor):
        def shutdown(self, *, wait: bool) -> None:
            assert wait
            raise OSError("shutdown failed")

    failing: Future[object] = Future()
    failing.set_exception(RuntimeError("work failed"))
    executor = _ShutdownFailingExecutor([failing], [])
    cache_root = tmp_path / "cache"
    scheduler = BoundedScheduler(
        max_in_flight=1,
        cache=CacheStore(cache_root),
        executor_factory=lambda _: executor,
    )
    store = CheckpointStore(cache_root)

    with store.open(_plan("S000001"), resume=False) as session:
        with pytest.raises(RuntimeError, match="work failed") as raised:
            scheduler.run(
                (_work("S000001", lambda: {"summary": "not used"}),),
                session,
            )

        assert isinstance(raised.value.__cause__, OSError)
        assert tuple(
            (entry.work_id, entry.reason) for entry in session.manifest.non_reusable
        ) == (("S000001", NonReusableReason.FAILED),)
        assert session.manifest.terminal_failure_work_id == "S000001"


def test_pre_cancelled_future_does_not_block_and_checkpoints_terminal_state(
    tmp_path: Path,
) -> None:
    cancelled: Future[object] = Future()
    assert cancelled.cancel()
    executor = _Executor([cancelled], [])
    cache_root = tmp_path / "cache"
    scheduler = BoundedScheduler(
        max_in_flight=1,
        cache=CacheStore(cache_root),
        executor_factory=lambda _: executor,
    )
    store = CheckpointStore(cache_root)
    errors: list[BaseException] = []

    with store.open(_plan("S000001"), resume=False) as session:

        def run() -> None:
            try:
                scheduler.run(
                    (_work("S000001", lambda: {"summary": "not used"}),),
                    session,
                )
            except Exception as error:  # noqa: BLE001 - surface any scheduler failure
                errors.append(error)

        thread = Thread(target=run, daemon=True)
        thread.start()
        thread.join(timeout=1)

        assert not thread.is_alive()
        assert len(errors) == 1
        assert isinstance(errors[0], CancelledError)
        assert tuple(
            (entry.work_id, entry.reason) for entry in session.manifest.non_reusable
        ) == (("S000001", NonReusableReason.CANCELLED),)
        assert session.manifest.terminal_failure
        assert session.manifest.terminal_failure_work_id == "S000001"
