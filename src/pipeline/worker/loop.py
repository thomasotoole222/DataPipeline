"""The worker loop: claim tasks, run them under a lease with bounded concurrency, and drain on shutdown.

Public API:
    TaskRunner: Protocol for the callable that runs one claimed task.
    HealthStatus: Point-in-time health snapshot of one worker.
    Worker: Claims tasks and supervises them until stopped.
    IDLE_WAIT, DRAIN_TIMEOUT, CLAIM_BACKOFF_MAX, BREAKER_LOSSES, BREAKER_WINDOW, BREAKER_PAUSE: ``Worker`` defaults.

"""

import asyncio
import contextlib
import logging
from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Final, Protocol

import psycopg
from opentelemetry import propagate, trace
from opentelemetry.context import Context

from pipeline import telemetry
from pipeline.config import Settings
from pipeline.db import control
from pipeline.db.control import ClaimedTask, LeaseLoss, TaskReport
from pipeline.db.pool import Pool
from pipeline.errors import (
    Interrupted,
    JobNotRunnable,
    LeaseConflict,
    QualityFailed,
    UnknownJob,
    classify,
)
from pipeline.telemetry import Instruments
from pipeline.timeutil import ensure_utc, utc_now
from pipeline.worker.lease import HEARTBEAT_INTERVAL, SILENCE_LIMIT, Lease


class TaskRunner(Protocol):
    """Define the callable that executes one claimed task under its lease.

    ``Worker`` looks a runner up by job name. ``Worker._supervise`` runs it in its own ``asyncio`` task and cancels
    that task if the lease is lost or the drain cut-off arrives first.

    """

    async def __call__(self, task: ClaimedTask, lease: Lease, /) -> None:  # Load calls lease.fence first, complete last
        """Run one claimed task.

        Preconditions:
            ``lease`` is the lease this worker holds on ``task``.

        Args:
            task: The claimed task to run.
            lease: The lease that grants this worker exclusive ownership of ``task``.

        Returns:
            None.

        Raises:
            Exception: Any. ``Worker._supervise`` records it as a lease loss when ``classify`` returns ``None``, and
                otherwise reports it as the task's outcome.
            asyncio.CancelledError: When the worker cancels the runner.

        """
        ...


#: Default longest wait for a ``NOTIFY`` while the queue is empty.
IDLE_WAIT: Final = timedelta(seconds=30)
#: Default grace period for running tasks during shutdown, before they are cut off.
DRAIN_TIMEOUT: Final = timedelta(seconds=15)
#: Default cap on the retry delay after a claim or listener error.
CLAIM_BACKOFF_MAX: Final = timedelta(seconds=30)
#: Default first retry delay after a claim or listener error; doubled per further consecutive failure.
_CLAIM_BACKOFF_FIRST: Final = timedelta(seconds=1)  # doubles on each failure in a row, up to CLAIM_BACKOFF_MAX
#: Default number of lease losses within ``BREAKER_WINDOW`` that pauses claiming.
BREAKER_LOSSES: Final = 3
#: Default window over which lease losses are counted.
BREAKER_WINDOW: Final = timedelta(minutes=10)
#: Default time the worker stops claiming once the breaker trips.
BREAKER_PAUSE: Final = timedelta(minutes=5)


@dataclass(frozen=True, slots=True, kw_only=True)
class HealthStatus:
    """Hold a point-in-time health snapshot of one worker.

    Instances are immutable and keyword-only. ``busy`` and ``slots`` are checked first, then ``paused_until`` and
    ``last_db_contact`` are normalised to UTC with ``ensure_utc``, which rejects a naive datetime with ``ValueError``.

    Preconditions:
        ``paused_until`` and ``last_db_contact`` are aware ``datetime`` values or ``None``.

    Args:
        worker_id: Identifier of the reporting worker.
        busy: Number of slots currently running a task, >= 0.
        slots: Total task slots the worker has, >= 1.
        draining: ``True`` once the worker has stopped claiming new tasks and is finishing current ones.
        paused_until: End of the current circuit-breaker pause, or ``None`` if not paused. Stored in UTC.
        last_db_contact: Time of the last successful database round trip, or ``None`` if there has been none. Stored
            in UTC.

    Returns:
        None. The generated ``__init__`` initialises a frozen instance in place.

    Raises:
        TypeError: If an argument is missing, passed positionally or unknown (raised by the generated ``__init__``),
            or if ``paused_until`` or ``last_db_contact`` is neither ``None`` nor a ``datetime`` (raised by
            ``ensure_utc``).
        ValueError: If ``busy`` is negative or ``slots`` is less than 1.
        ValueError: If ``paused_until`` or ``last_db_contact`` is naive (raised by ``ensure_utc``).
        OverflowError: If converting a timestamp to UTC falls outside the ``datetime`` range (raised by
            ``ensure_utc``).

    """

    worker_id: str
    busy: int  # slots running a task, including one still unwinding
    slots: int  # concurrency
    draining: bool
    paused_until: datetime | None  # circuit breaker; UTC
    last_db_contact: datetime | None  # latest successful claim, wake-up wait or heartbeat; UTC

    def __post_init__(self) -> None:
        """Check ``busy`` and ``slots``, then normalise ``paused_until`` and ``last_db_contact`` to UTC.

        Preconditions:
            Called by the generated ``__init__`` after every field is set.

        Args:
            None.

        Returns:
            None.

        Raises:
            ValueError: If ``busy`` is negative or ``slots`` is less than 1, or a timestamp is naive (raised by
                ``ensure_utc``).
            TypeError: If a timestamp is neither ``None`` nor a ``datetime`` (raised by ``ensure_utc``).
            OverflowError: If converting a timestamp to UTC falls outside the ``datetime`` range (raised by
                ``ensure_utc``).

        """
        if self.busy < 0:
            raise ValueError(f"busy must be >= 0, got {self.busy}")
        if self.slots < 1:
            raise ValueError(f"slots must be >= 1, got {self.slots}")
        # busy <= slots is not checked: the claim loop never starts a task while every slot is busy.
        for name in ("paused_until", "last_db_contact"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, ensure_utc(value))


#: Logger for this module.
_log: Final = logging.getLogger(__name__)
#: Tracer for the per-task spans.
_tracer: Final = trace.get_tracer(__name__)


def _backoff(first: timedelta, cap: timedelta, failures: int) -> timedelta:
    """Return the retry delay after ``failures`` consecutive failures.

    The delay is ``first`` doubled once per failure after the first, with the doubling capped at ``2**32``, and then
    capped at ``cap``.

    Preconditions:
        ``failures`` >= 1.

    Args:
        first: Delay after the first failure.
        cap: Largest delay returned.
        failures: Number of consecutive failures so far.

    Returns:
        ``min(first * 2 ** min(failures - 1, 32), cap)``.

    Raises:
        ValueError: If ``failures`` < 1, because the shift count is negative.
        OverflowError: If ``first * 2 ** min(failures - 1, 32)`` falls outside the ``timedelta`` range.

    """
    return min(first * (1 << min(failures - 1, 32)), cap)


def _count(err: BaseException, name: str) -> int:
    """Read a non-negative row count from an attribute of ``err``.

    Preconditions:
        None.

    Args:
        err: The exception to read from.
        name: Name of the attribute.

    Returns:
        The attribute's value if it is an ``int`` (``bool`` included) and >= 0; otherwise 0, including when the
        attribute is missing.

    Raises:
        None. An exception other than ``AttributeError`` raised by the attribute's getter propagates.

    """
    value = getattr(err, name, 0)
    return value if isinstance(value, int) and value >= 0 else 0


def _unwind(lease: Lease, ended: float) -> timedelta | None:
    """Return how long the runner took to end after the lease was first marked lost.

    Preconditions:
        ``ended`` was read from the same event loop's ``time()`` as ``lease.lost_at``, once the runner's task was done.

    Args:
        lease: The lost lease.
        ended: ``loop.time()`` when the runner's task ended.

    Returns:
        ``ended - lease.lost_at`` as a ``timedelta``, or ``None`` when ``lease.lost_at`` is ``None`` because no loss
        moment is known.

    Raises:
        Nothing.

    """
    lost_at = lease.lost_at
    return None if lost_at is None else timedelta(seconds=ended - lost_at)


class Worker:
    """Claim tasks from the queue and supervise them with bounded concurrency.

    ``run`` claims one task at a time while fewer than ``settings.concurrency`` tasks are running, and waits on the
    ``LISTEN`` connection for up to ``idle_wait`` when the queue is empty. Each task for a known job with a registered
    runner gets a ``Lease`` and is supervised under ``Lease.hold``; other tasks are reported as failed. Claim and
    listener errors are retried with exponential backoff from ``claim_backoff_first`` up to ``claim_backoff_max``. If
    ``breaker_losses`` leases are lost within ``breaker_window``, claiming stops for ``breaker_pause``. On return,
    ``run`` drains: running tasks get ``drain_timeout`` to finish and are then cut off. ``stop`` ends the claim loop.

    Attributes:
        None public. All state is private.

    """

    def __init__(
        self,
        *,
        settings: Settings,
        pool: Pool,
        listener: psycopg.AsyncConnection[Any],
        runners: Mapping[str, TaskRunner],
        known_jobs: frozenset[str],
        instruments: Instruments,
        idle_wait: timedelta = IDLE_WAIT,
        drain_timeout: timedelta = DRAIN_TIMEOUT,
        claim_backoff_first: timedelta = _CLAIM_BACKOFF_FIRST,
        claim_backoff_max: timedelta = CLAIM_BACKOFF_MAX,
        breaker_losses: int = BREAKER_LOSSES,
        breaker_window: timedelta = BREAKER_WINDOW,
        breaker_pause: timedelta = BREAKER_PAUSE,
        heartbeat_interval: timedelta = HEARTBEAT_INTERVAL,
        silence_limit: timedelta = SILENCE_LIMIT,
    ) -> None:
        """Store the dependencies and settings, and create the worker's events and bookkeeping.

        ``listener`` is owned by the caller and is not closed by the
        worker; a replacement that the worker opens is owned and closed by the worker.

        Preconditions:
            Called with a running event loop or before one is used for ``run``. ``listener`` is open and has run
            ``LISTEN`` on ``control.TASK_CHANNEL``. ``settings.concurrency`` >= 1. The durations are positive and
            ``breaker_losses`` >= 1; none of these is checked.

        Args:
            settings: Supplies ``worker_id`` and ``concurrency``.
            pool: Connection pool used for claims, heartbeats and outcome reports.
            listener: Open ``LISTEN`` connection, owned by the caller.
            runners: Map from job name to the runner for that job.
            known_jobs: Job names the registry recognises. A task naming any other job fails as ``UnknownJob``.
            instruments: Metric instruments to record into.
            idle_wait: Longest wait for a notification while the queue is empty.
            drain_timeout: Grace period for running tasks during shutdown.
            claim_backoff_first: First retry delay after a claim or listener error.
            claim_backoff_max: Upper bound on that retry delay.
            breaker_losses: Number of lease losses within ``breaker_window`` that pauses claiming.
            breaker_window: Window over which lease losses are counted.
            breaker_pause: How long the worker stops claiming once the breaker trips.
            heartbeat_interval: Lease heartbeat period.
            silence_limit: How long a lease may go without a successful heartbeat before it counts as lost.

        Returns:
            None.

        Raises:
            Nothing.

        """
        self._settings = settings
        self._pool = pool
        self._listener: psycopg.AsyncConnection[Any] | None = listener
        self._owned: psycopg.AsyncConnection[Any] | None = None  # a listener this worker reopened; it closes it
        self._runners = runners
        self._known_jobs = known_jobs
        self._instruments = instruments
        self._idle_wait = idle_wait
        self._drain_timeout = drain_timeout
        self._backoff_first = claim_backoff_first
        self._backoff_max = claim_backoff_max
        self._breaker_losses = breaker_losses
        self._breaker_window = breaker_window
        self._breaker_pause = breaker_pause
        self._heartbeat_interval = heartbeat_interval
        self._silence_limit = silence_limit
        self._stopping = asyncio.Event()  # set by stop(): no more claims
        self._wake = asyncio.Event()  # a slot freed, or stop()
        self._cutoff = asyncio.Event()  # the drain ran out: cancel what still runs
        self._running: dict[asyncio.Task[None], Lease | None] = {}
        self._losses: deque[datetime] = deque()
        self._paused_until: datetime | None = None
        self._last_contact: datetime | None = None

    async def run(self) -> None:
        """Run the claim loop until it stops or raises, then drain and close any listener the worker opened.

        ``_drain`` leaves the worker stopping, so a second call drains again and returns without claiming.

        Preconditions:
            Awaited from the event loop the worker's events belong to.

        Args:
            None.

        Returns:
            None, after ``stop`` has been called and running tasks have been drained.

        Raises:
            asyncio.CancelledError: If the call is cancelled. Draining still runs.
            Exception: Any exception from ``_drain`` or from closing the worker's own listener.

        """
        try:
            await self._claim_loop()
        finally:
            await self._drain()
            if self._owned is not None:
                await self._owned.close()

    def stop(self) -> None:
        """Ask the claim loop to stop.

        Sets the stopping event, which also ends an idle wait on the listener, and the wake event, which ends a wait
        for a free slot. Calling it again has no further effect.

        Preconditions:
            Called from the event loop thread; ``asyncio.Event`` is not thread-safe.

        Args:
            None.

        Returns:
            None.

        Raises:
            None.

        """
        self._stopping.set()
        self._wake.set()

    def health(self) -> HealthStatus:
        """Return a health snapshot of the worker.

        Preconditions:
            None.

        Args:
            None.

        Returns:
            A ``HealthStatus`` with ``busy`` as the number of running tasks, ``slots`` as ``settings.concurrency``,
            ``draining`` set once stopping has begun, ``paused_until`` as the end of the breaker pause if it is still
            in the future and otherwise ``None``, and ``last_db_contact`` as the latest of the last successful claim or
            listener wait and each running lease's ``last_heartbeat``, or ``None`` if there is none.

        Raises:
            ValueError: If ``settings.concurrency`` is less than 1 (raised by ``HealthStatus``).

        """
        now = utc_now()
        paused = self._paused_until if self._paused_until is not None and self._paused_until > now else None
        contacts = [self._last_contact, *(lease.last_heartbeat for lease in self._running.values() if lease)]
        return HealthStatus(
            worker_id=self._settings.worker_id,
            busy=len(self._running),
            slots=self._settings.concurrency,
            draining=self._stopping.is_set(),
            paused_until=paused,
            last_db_contact=max((c for c in contacts if c is not None), default=None),
        )

    async def _claim_loop(self) -> None:
        """Claim and start tasks until the worker is stopping.

        Each round:
        - waits on the wake event while every slot is busy;
        - sleeps out a breaker pause that is still running;
        - otherwise claims a task on a pool connection.

        A failed claim is logged and retried after ``_backoff``. A successful claim resets the failure count and
        records the contact time. It then starts the task, or waits for a notification if there was none. ``_finished``
        and ``stop`` set the wake event, so a full worker resumes claiming when a slot frees.

        Preconditions:
            Called once, from ``run``.

        Args:
            None.

        Returns:
            None, once the worker is stopping.

        Raises:
            asyncio.CancelledError: If the call is cancelled. An ``Exception`` from taking a connection or claiming
                is logged at ``WARNING`` and retried, not raised.

        """
        failures = 0
        while not self._stopping.is_set():
            if len(self._running) >= self._settings.concurrency:
                self._wake.clear()
                if len(self._running) >= self._settings.concurrency and not self._stopping.is_set():
                    await self._wake.wait()
                continue
            if self._paused_until is not None and (pause := self._paused_until - utc_now()) > timedelta(0):
                await self._sleep(pause)
                continue
            try:
                async with self._pool.connection() as conn:
                    task = await control.claim(conn, self._settings.worker_id)
            except Exception:
                failures += 1
                wait = _backoff(self._backoff_first, self._backoff_max, failures)
                _log.warning("claim failed; retrying in %.1f s", wait.total_seconds(), exc_info=True)
                await self._sleep(wait)
                continue
            failures = 0
            self._last_contact = utc_now()
            if task is None:
                await self._wait_for_task()
            else:
                self._start(task)

    async def _sleep(self, wait: timedelta) -> None:
        """Wait for ``wait`` or until ``stop``, whichever comes first.

        Preconditions:
            None.

        Args:
            wait: The longest wait.

        Returns:
            None.

        Raises:
            Nothing.

        """
        with contextlib.suppress(TimeoutError):
            async with asyncio.timeout(wait.total_seconds()):
                await self._stopping.wait()

    async def _wait_for_task(self) -> None:
        """Wait for a task notification, the idle timeout or ``stop``, whichever comes first.

        Races ``control.wait_for_task`` on the listener against the stopping event, then cancels whichever is still
        pending and waits for both. If the wait finished without an error, with or without a notification, the
        contact time is updated. If it raised, the error is logged at ``WARNING`` and ``self._drop_listener`` is
        called.

        Preconditions:
            No other coroutine uses the listener during the call.

        Args:
            None.

        Returns:
            None. It returns at once if the worker stops before a listener is available.

        Raises:
            asyncio.CancelledError: If the call is cancelled, after both inner tasks have been cancelled and awaited.
                An ``Exception`` from the listener wait is logged and handled, not raised.

        """
        listener = await self._usable_listener()
        if listener is None:
            return
        waiter = asyncio.create_task(control.wait_for_task(listener, self._idle_wait))
        stopper = asyncio.create_task(self._stopping.wait())
        try:
            await asyncio.wait([waiter, stopper], return_when=asyncio.FIRST_COMPLETED)
        finally:
            stopper.cancel()
            if not waiter.done():
                waiter.cancel()
            await asyncio.wait([waiter, stopper])
        if waiter.cancelled():
            return
        if (err := waiter.exception()) is None:
            self._last_contact = utc_now()
            return
        _log.warning("task listener failed; reopening it", exc_info=err)
        await self._drop_listener()

    async def _usable_listener(self) -> psycopg.AsyncConnection[Any] | None:
        """Return the listener, opening a replacement while there is none and the worker is not stopping.

        A replacement comes from ``control.open_listener(self._settings)`` and is stored as both the listener and the
        worker-owned connection. A failed open is logged at ``WARNING`` and retried after ``_backoff``, or sooner if
        ``stop`` is called.

        Preconditions:
            None.

        Args:
            None.

        Returns:
            The current listener, or ``None`` if the worker is stopping and there is none.

        Raises:
            asyncio.CancelledError: If the call is cancelled. An ``Exception`` from opening the listener is logged at
                ``WARNING`` and retried, not raised.

        """
        failures = 0
        while self._listener is None and not self._stopping.is_set():
            try:
                self._listener = self._owned = await control.open_listener(self._settings)
            except Exception:
                failures += 1
                wait = _backoff(self._backoff_first, self._backoff_max, failures)
                _log.warning(
                    "could not reopen the task listener; retrying in %.1f s", wait.total_seconds(), exc_info=True
                )
                await self._sleep(wait)
        return self._listener

    async def _drop_listener(self) -> None:
        """Forget a failed listener, closing it if this worker opened it.

        Preconditions:
            None.

        Returns:
            None.

        Raises:
            Nothing.

        """
        if self._owned is not None and self._owned is self._listener:
            with contextlib.suppress(Exception):
                await self._owned.close()
            self._owned = None
        self._listener = None

    def _start(self, task: ClaimedTask) -> None:
        """Start ``task`` in its own ``asyncio`` task and track it as running.

        Adds 1 to ``slots_busy``, records the queue wait from the later of ``created_at`` and ``not_before`` to
        ``claimed_at``, creates the task running ``_run_task``, adds it to the running map with no lease yet, and
        registers ``_finished`` as its done callback.

        Preconditions:
            Called from the claim loop with a free slot.

        Args:
            task: The task just claimed.

        Returns:
            None.

        Raises:
            None.

        """
        self._instruments.slots_busy.add(1)
        waited = task.claimed_at - max(task.created_at, task.not_before)
        self._instruments.queue_wait.record(waited.total_seconds())
        running = asyncio.create_task(self._run_task(task), name=f"task-{task.id}")
        self._running[running] = None
        running.add_done_callback(self._finished)

    def _finished(self, running: asyncio.Task[None]) -> None:
        """Remove a finished task from the running map and free its slot.

        Removes ``running``, subtracts 1 from ``slots_busy``, sets the wake event so the claim loop can fill the slot,
        and logs at ``ERROR`` if the task ended with an exception.

        Preconditions:
            Called as the done callback ``_start`` registers. ``running`` is in the running map and is done.

        Args:
            running: The finished task.

        Returns:
            None.

        Raises:
            KeyError: If ``running`` is not in the running map.

        """
        del self._running[running]
        self._instruments.slots_busy.add(-1)
        self._wake.set()
        if not running.cancelled() and (err := running.exception()) is not None:
            _log.error("task slot %s ended with an error", running.get_name(), exc_info=err)

    async def _run_task(self, task: ClaimedTask) -> None:
        """Run one attempt at ``task`` inside a ``task`` span and with the task's fields bound for logging.

        The span's parent is the context extracted from ``task.trace_context`` as a ``traceparent`` header, or an
        empty context when there is none. ``telemetry.bind_task`` binds the task's id, run, attempt, connector, job,
        source and partition as ``pipeline.*`` fields.

        Preconditions:
            None.

        Args:
            task: The claimed task.

        Returns:
            None.

        Raises:
            Exception: Whatever ``_attempt`` raises, unchanged. ``telemetry.bind_task`` restores the bound fields
                and the span ends even when it does.

        """
        parent = propagate.extract({"traceparent": task.trace_context}) if task.trace_context else Context()
        with (
            _tracer.start_as_current_span("task", context=parent),
            telemetry.bind_task(
                task_id=str(task.id),
                run_id=str(task.run_id),
                attempt=task.attempt,
                connector=task.connector,
                job=task.job,
                source_id=str(task.source_id),
                partition=task.partition_key,
            ),
        ):
            await self._attempt(task)

    async def _attempt(self, task: ClaimedTask) -> None:
        """Create the task's lease, then report the task as failed or supervise it under the lease.

        A job not in ``known_jobs`` is logged at ``ERROR``, counted in ``unknown_jobs`` and reported as
        ``UnknownJob``. A known job without a runner is reported as ``JobNotRunnable``. Otherwise the lease is stored
        against the current task in the running map and ``_supervise`` runs inside ``lease.hold()``.

        Preconditions:
            Runs inside the ``asyncio`` task that ``_start`` created for ``task``.

        Args:
            task: The claimed task.

        Returns:
            None.

        Raises:
            Exception: Whatever ``_supervise`` or ``_report_failure`` raises.

        """
        lease = Lease(
            self._pool,
            task,
            worker_id=self._settings.worker_id,
            instruments=self._instruments,
            interval=self._heartbeat_interval,
            silence_limit=self._silence_limit,
        )
        if task.job not in self._known_jobs:
            _log.error(
                "task %s names job %r, which is not in the job registry (created %s)",
                task.id,
                task.job,
                task.created_at,
            )
            self._instruments.unknown_jobs.add(1)
            await self._report_failure(lease, UnknownJob(f"job {task.job!r} is not in the job registry"))
            return
        runner = self._runners.get(task.job)
        if runner is None:
            await self._report_failure(lease, JobNotRunnable(f"no runner for job {task.job!r}"))
            return
        current = asyncio.current_task()
        if current is not None and current in self._running:
            self._running[current] = lease
        async with lease.hold():
            await self._supervise(lease, runner)

    async def _supervise(self, lease: Lease, runner: TaskRunner) -> None:
        """Run the runner until it ends, the lease is lost or the drain runs out, and act on which came first.

        Preconditions:
            ``lease`` is held.

        Args:
            lease: The task's lease.
            runner: The job's runner.

        Returns:
            None.

        Every unwind passed to ``_lose`` is measured by ``_unwind`` from ``lease.lost_at`` to when the runner's task
        ended, or is ``None`` when no loss moment is known. Once ``lease.complete`` has succeeded, any ending other
        than a normal return (the runner raising an ``Exception``, its task being cancelled, or this method cancelling
        it on a lost lease or at the drain cutoff) first goes to ``_landed``, which asks the database whether the
        outcome landed. If it did, or the check failed, ``_landed`` has logged the ending at error level and nothing
        else happens: no second report, no loss and no breaker count. If it did not (Load's transaction rolled back
        after ``complete``), ``_landed`` logs a warning and the ending is handled as if ``complete`` had never run.

        Raises:
            asyncio.CancelledError: If this slot is cancelled from outside; the runner is cancelled and awaited first.
            BaseException: A runner's exception that is not an ``Exception``, unchanged.

        """
        loop = asyncio.get_running_loop()
        work = asyncio.create_task(runner(lease.task, lease), name=f"runner-{lease.task.id}")
        lost = asyncio.create_task(lease.lost.wait())
        cutoff = asyncio.create_task(self._cutoff.wait())
        try:
            await asyncio.wait([work, lost, cutoff], return_when=asyncio.FIRST_COMPLETED)
        except asyncio.CancelledError:
            await self._cancel(work)
            raise
        finally:
            lost.cancel()
            cutoff.cancel()
        if not work.done():
            await self._cancel(work)
            if lease.completed and await self._landed(lease, "was cancelled", None):
                return
            if lease.lost.is_set():
                await self._lose(lease, _unwind(lease, loop.time()))
            else:
                drained = self._drain_timeout.total_seconds()
                await self._report_failure(lease, Interrupted(f"cancelled by shutdown after the {drained:g} s drain"))
            return
        ended = loop.time()
        if work.cancelled():
            if lease.completed and await self._landed(lease, "was cancelled", None):
                return
            await self._report_failure(lease, RuntimeError("runner was cancelled"))
            return
        err = work.exception()
        if err is None:
            if not lease.completed:
                await self._report_failure(lease, RuntimeError("runner returned without completing the lease"))
            return
        if not isinstance(err, Exception):
            raise err
        if lease.completed and await self._landed(lease, "raised", err):
            return
        if classify(err) is None:  # LeaseConflict: Load's transaction already rolled back
            await self._lose(lease, _unwind(lease, ended))
            return
        await self._report_failure(lease, err)

    async def _landed(self, lease: Lease, ending: str, runner_err: Exception | None) -> bool:
        """After ``lease.complete`` succeeded but the runner ended abnormally, check whether the outcome landed.

        One ``control.heartbeat`` on a pool connection decides it. ``control.heartbeat`` returns ``False`` once the
        task is no longer running under this token, which a committed ``control.report_outcome`` brings about, and
        ``True`` while it still is: then Load's transaction rolled back after ``complete`` and the outcome did not
        land, which is logged as a warning. As in ``Lease._heartbeats``, the connection is borrowed first and only the
        call runs under ``asyncio.timeout(heartbeat_interval)``; a timeout is a failed check.

        Preconditions:
            ``lease.completed`` is ``True`` and the runner's task is done.

        Args:
            lease: The task's lease.
            ending: How the runner ended, as the log line reads: ``"raised"`` or ``"was cancelled"``.
            runner_err: The runner's exception when it raised, else ``None``.

        Returns:
            ``True`` when this method has dealt with the ending: the outcome landed, or the check failed or timed
            out. Either is logged at error level, and the caller reports nothing and records no loss. ``False`` when
            the task is still running under this token: a warning is logged, and the caller handles the ending as if
            ``complete`` had never run.

        Raises:
            Nothing; a failed check is logged and the lease left to expire.

        """
        task = lease.task
        deadline: asyncio.Timeout | None = None  # control.heartbeat has no timeout of its own
        try:
            async with self._pool.connection() as conn:
                async with (deadline := asyncio.timeout(self._heartbeat_interval.total_seconds())):
                    running = await control.heartbeat(conn, task)
        except Exception as err:
            cause = "timed out" if deadline is not None and deadline.expired() else "failed"
            _log.error(
                "runner for task %s %s after completing the lease%s, and checking whether the outcome landed %s: "
                "%r; its lease will expire",
                task.id,
                ending,
                "" if runner_err is None else f" ({runner_err!r})",
                cause,
                err,
                exc_info=err,
            )
            return True
        if running:  # Load's transaction rolled back after complete: the outcome never landed
            _log.warning(
                "outcome for task %s rolled back after complete; the runner %s%s, handled as if complete never ran",
                task.id,
                ending,
                "" if runner_err is None else f" ({runner_err!r})",
            )
            return False
        _log.error("runner for task %s %s after completing the lease", task.id, ending, exc_info=runner_err)
        return True

    async def _cancel(self, work: asyncio.Task[None]) -> None:
        """Cancel the runner and wait for it to unwind.

        Preconditions:
            None.

        Args:
            work: The runner's asyncio task.

        Returns:
            None, once the runner's task is done.

        Raises:
            Nothing; whatever the runner raises while unwinding is logged at debug level and dropped.

        """
        work.cancel()
        await asyncio.wait([work])
        if not work.cancelled() and (err := work.exception()) is not None:
            _log.debug("runner raised while unwinding", exc_info=err)

    async def _report_failure(self, lease: Lease, err: Exception) -> None:
        """Report ``err`` as the outcome of the lease's task.

        When ``classify`` returns ``None``, the lease is recorded as lost through ``self._lose``. Otherwise the failure
        is logged at ``WARNING``, and a ``TaskReport`` is built from it:
        - ``error_message`` is ``str(err)``, or the exception's type name when that is empty;
        - ``retry_after`` comes from ``err.retry_after`` when that is a ``timedelta``;
        - ``rows_fetched`` and ``rows_written`` come from ``_count``;
        - ``rows_rejected`` is ``err.rows_quarantined`` for a ``QualityFailed`` and 0 otherwise;
        - ``duration`` is the time since the claim, at least zero.

        The report is sent with ``lease.complete`` on a pool connection. A ``LeaseConflict``, meaning the lease is no
        longer held, is recorded as a lease loss through ``self._lose``. Any other ``Exception`` from reporting is
        logged at ``ERROR`` and swallowed, and the lease is left to expire.

        Preconditions:
            None.

        Args:
            lease: The lease on the failed task.
            err: The exception to report.

        Returns:
            None.

        Raises:
            Exception: Whatever ``_lose`` raises, when ``classify`` returns ``None``.
            ValueError: If ``err.retry_after`` is a negative ``timedelta`` (raised by ``TaskReport``).
            asyncio.CancelledError: If the call is cancelled. Any other ``Exception`` from taking a connection or
                reporting is logged at ``ERROR`` and swallowed.

        """
        task = lease.task
        verdict = classify(err)
        if verdict is None:
            await self._lose(lease, None)
            return
        outcome, error_class = verdict
        _log.warning("task %s failed: %s: %s", task.id, error_class, err, exc_info=err)
        retry_after = getattr(err, "retry_after", None)
        report = TaskReport(
            outcome=outcome,
            error_class=error_class,
            error_message=str(err) or type(err).__name__,
            retry_after=retry_after if isinstance(retry_after, timedelta) else None,
            rows_fetched=_count(err, "rows_fetched"),
            rows_written=_count(err, "rows_written"),
            rows_rejected=err.rows_quarantined if isinstance(err, QualityFailed) else 0,
            duration=max(utc_now() - task.claimed_at, timedelta(0)),
        )
        try:
            async with self._pool.connection() as conn:
                await lease.complete(conn, report)
        except LeaseConflict:
            await self._lose(lease, None)
        except Exception:
            _log.error("could not report the outcome of task %s; its lease will expire", task.id, exc_info=True)

    async def _lose(self, lease: Lease, unwind: timedelta | None) -> None:
        """Record a lease loss and count it towards the circuit breaker.

        Preconditions:
            The task has unwound.

        A lease not yet marked lost (a ``LeaseConflict`` that did not come through ``fence`` or ``complete``) is
        marked with ``lease.lose(LeaseLoss.COMMIT)``, which also records its ``lost_at``.

        Args:
            lease: The lost lease.
            unwind: From ``lease.lost_at`` to when the runner's task ended, or ``None`` if unknown.

        Returns:
            None.

        Raises:
            Nothing.

        """
        if lease.loss is None:  # a LeaseConflict that did not come through fence or complete
            lease.lose(LeaseLoss.COMMIT)
        await lease.record_loss(unwind)
        now = utc_now()
        self._losses.append(now)
        while self._losses and now - self._losses[0] > self._breaker_window:
            self._losses.popleft()
        if len(self._losses) >= self._breaker_losses:
            self._losses.clear()
            self._paused_until = now + self._breaker_pause
            self._instruments.worker_paused.add(1)
            _log.warning(
                "%d lease losses within %s; not claiming until %s",
                self._breaker_losses,
                self._breaker_window,
                self._paused_until.isoformat(),
            )

    async def _drain(self) -> None:
        """Stop claiming and wait for running tasks: up to ``drain_timeout``, then without a bound after the cut-off.

        Sets the stopping event. If tasks are running, waits for all of them for up to ``drain_timeout``. If any are
        still tracked afterwards, it sets the cut-off event, which makes ``_supervise`` cancel its work, and waits for
        all of them again with no time limit. Exceptions of the tasks themselves are not re-raised.

        Preconditions:
            None.

        Args:
            None.

        Returns:
            None.

        Raises:
            asyncio.CancelledError: If the call is cancelled. The drain timeout's ``TimeoutError`` is suppressed.

        """
        self._stopping.set()
        if self._running:
            with contextlib.suppress(TimeoutError):
                async with asyncio.timeout(self._drain_timeout.total_seconds()):
                    await asyncio.wait(list(self._running))
        if self._running:
            self._cutoff.set()
            await asyncio.wait(list(self._running))
