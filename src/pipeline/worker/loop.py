"""The worker loop: claim tasks, run them under a lease with bounded concurrency, and drain on shutdown.

Public API:
    TaskRunner: Protocol for the callable that runs one claimed task.
    HealthStatus: Point-in-time health snapshot of one worker.
    Worker: Claims tasks and supervises them until stopped.
    IDLE_WAIT: Default longest wait for a ``NOTIFY`` while the queue is empty.
    DRAIN_TIMEOUT: Default grace period for running tasks during shutdown.
    CLAIM_BACKOFF_MAX: Default cap on the retry delay after a claim error.
    BREAKER_LOSSES: Default number of lease losses or ``Lease`` refusals that pauses claiming.
    BREAKER_WINDOW: Default window over which those are counted.
    BREAKER_PAUSE: Default time claiming stops once the breaker trips.

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
    AlreadyReported,
    Interrupted,
    JobNotRunnable,
    LeaseConflict,
    Outcome,
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
    that task if the lease is lost or the drain cut-off arrives first. A runner must end once cancelled: the worker
    awaits a cancelled runner without a time limit (``Worker._cancel``, and ``Worker._drain``'s second wait).

    """

    async def __call__(self, task: ClaimedTask, lease: Lease, /) -> None:  # Load completes inside lease.fenced
        """Run one claimed task.

        Preconditions:
            ``lease`` is the lease this worker holds on ``task``.

        Args:
            task: The claimed task to run.
            lease: The lease that grants this worker exclusive ownership of ``task``.

        Returns:
            None.

        Raises:
            Exception: Any. ``Worker._supervise`` only logs it once the outcome has landed (``lease.completed``),
                records it as a lease loss when ``classify`` returns ``None`` or the lease is already lost, and
                otherwise reports it as the task's outcome.
            BaseException: ``KeyboardInterrupt``, ``SystemExit`` or ``GeneratorExit`` stops the worker without a
                report (see ``_FATAL``); any other ``BaseException`` is treated as an ``Exception`` above.
            asyncio.CancelledError: When the worker cancels the runner.

        """
        ...


#: Default longest wait for a ``NOTIFY`` while the queue is empty.
IDLE_WAIT: Final = timedelta(seconds=30)
#: Default grace period for running tasks during shutdown, before they are cut off. It leaves 5 s of ``__main__``'s
#: 15 s ``SHUTDOWN_TIMEOUT`` for the cut-off tasks' reports and the closes.
DRAIN_TIMEOUT: Final = timedelta(seconds=10)
#: Default cap on the retry delay after a claim error.
CLAIM_BACKOFF_MAX: Final = timedelta(seconds=30)
#: Default first retry delay after a claim error; doubled per further consecutive failure.
_CLAIM_BACKOFF_FIRST: Final = timedelta(seconds=1)  # doubles on each failure in a row, up to CLAIM_BACKOFF_MAX
#: A runner's ``BaseException`` that is not an ``Exception`` and that ends the worker instead of being reported as
#: the task's outcome. ``Worker._supervise`` records the first one, calls ``Worker.stop`` and re-raises it, which ends
#: the task slot; the task is not reported and stays running until its lease expires. ``Worker.run`` then drains the
#: other running tasks as on a normal stop and re-raises the recorded exception. ``asyncio`` also re-raises
#: ``KeyboardInterrupt`` and ``SystemExit`` from a task into the event loop; ``GeneratorExit`` reaches the caller only
#: through ``run``. ``asyncio.CancelledError`` is not listed: a runner's task that raises it counts as cancelled.
_FATAL: Final = (KeyboardInterrupt, SystemExit, GeneratorExit)
#: Default number of lease losses or ``Lease`` refusals within ``BREAKER_WINDOW`` that pauses claiming.
BREAKER_LOSSES: Final = 3
#: Default window over which lease losses and ``Lease`` refusals are counted.
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
            in UTC. ``Worker.health`` sets it to the snapshot's ``utc_now()`` minus the contact's age on the monotonic
            clock, so ``now - last_db_contact`` is that age when ``now`` is read right after the snapshot.

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
        Exception: Any exception other than ``AttributeError`` raised by the attribute's getter, unchanged.

    """
    value = getattr(err, name, 0)
    return value if isinstance(value, int) and value >= 0 else 0


def _failure_report(task: ClaimedTask, err: BaseException) -> TaskReport:
    """Log ``err`` as the failure of ``task`` and build the ``TaskReport`` that reports it.

    An ``Exception`` is classified by ``classify``; any other ``BaseException`` is ``Outcome.UNEXPECTED`` with its
    class name as the error class. The failure is logged at ``WARNING``, and the report is built from it:
    - ``error_type`` is the exception's type name, so an ``"Unexpected"`` error class still names its exception;
    - ``error_message`` is ``str(err)``, or the exception's type name when that is empty;
    - ``retry_after`` comes from ``err.retry_after`` when that is a ``timedelta``;
    - ``rows_fetched`` and ``rows_written`` come from ``_count``;
    - ``rows_rejected`` is ``err.rows_quarantined`` for a ``QualityFailed`` and 0 otherwise;
    - ``duration`` is the time since the claim, at least zero.

    Preconditions:
        ``err`` is not ``asyncio.CancelledError``, ``KeyboardInterrupt``, ``SystemExit`` or ``GeneratorExit``, and
        ``classify`` does not map it to ``None``. Not checked; such an ``err`` is reported as ``Outcome.UNEXPECTED``
        with its class name.

    Args:
        task: The failed task.
        err: The exception to report.

    Returns:
        The report.

    Raises:
        ValueError: From ``TaskReport``, for a value an exception's own constructor did not check (for example a
            ``retry_after`` above ``MAX_RETRY_AFTER`` or a row count above ``2**63 - 1``).
        Exception: Any exception from ``str(err)``, from reading ``err.retry_after`` or from ``_count``, unchanged.

    """
    verdict = classify(err) if isinstance(err, Exception) else None  # never None for an Exception here
    outcome, error_class = verdict or (Outcome.UNEXPECTED, type(err).__name__)
    _log.warning("task %s failed: %s: %s", task.id, error_class, err, exc_info=err)
    retry_after = getattr(err, "retry_after", None)
    return TaskReport(
        outcome=outcome,
        error_class=error_class,
        error_type=type(err).__name__,
        error_message=str(err) or type(err).__name__,
        retry_after=retry_after if isinstance(retry_after, timedelta) else None,
        rows_fetched=_count(err, "rows_fetched"),
        rows_written=_count(err, "rows_written"),
        rows_rejected=err.rows_quarantined if isinstance(err, QualityFailed) else 0,
        duration=max(utc_now() - task.claimed_at, timedelta(0)),
    )


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


def _ending(work: asyncio.Task[None]) -> tuple[str, BaseException | None]:
    """Name how a runner's task ended, as ``Worker._landed``'s log line reads it.

    Preconditions:
        ``work`` is done.

    Args:
        work: The runner's asyncio task.

    Returns:
        ``("was cancelled", None)``, ``("raised", err)`` with the task's exception, or ``("returned", None)``.

    Raises:
        asyncio.InvalidStateError: If ``work`` is not done (raised by ``Task.exception``).

    """
    if work.cancelled():
        return "was cancelled", None
    if (err := work.exception()) is not None:
        return "raised", err
    return "returned", None


class Worker:
    """Claim tasks from the queue and supervise them with bounded concurrency.

    ``run`` claims one task at a time while fewer than ``settings.concurrency`` tasks are running, and waits on the
    ``LISTEN`` connection for up to ``idle_wait`` when the queue is empty. Each claimed task first gets a ``Lease``; a
    task whose ``Lease`` cannot be built is reported as failed without one, a task for an unknown job or a job with no
    runner is reported as failed under its lease, and any other task is supervised under ``Lease.hold``. Claim errors
    are retried with exponential backoff from ``claim_backoff_first`` up to ``claim_backoff_max``; a failed wait on
    the listener drops it, and the next idle round reopens it, claiming every ``idle_wait`` while it cannot. If
    ``breaker_losses`` lease losses or ``Lease`` refusals happen within ``breaker_window``, claiming stops for
    ``breaker_pause``. On return, ``run`` drains: running tasks get ``drain_timeout`` to finish and are then cut
    off. ``stop`` ends the claim loop. A runner raising a ``_FATAL`` exception also stops the worker; ``run``
    re-raises it after the drain.

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

        ``listener`` is owned by the caller and is not closed by the worker; a replacement that the worker opens is
        owned and closed by the worker.

        Preconditions:
            Called with a running event loop or before one is used for ``run``. ``listener`` is open and has run
            ``LISTEN`` on ``control.TASK_CHANNEL``. ``settings.concurrency`` >= 1. The durations are positive and
            ``breaker_losses`` >= 1; none of these is checked. ``heartbeat_interval`` and ``silence_limit`` are
            checked by ``Lease`` only when a task is attempted; a task they fail is reported (see ``_attempt``).

        Args:
            settings: Supplies ``worker_id`` and ``concurrency``.
            pool: Connection pool used for claims, heartbeats and outcome reports.
            listener: Open ``LISTEN`` connection, owned by the caller.
            runners: Map from job name to the runner for that job.
            known_jobs: Job names the registry recognises. A task naming any other job fails as ``UnknownJob``.
            instruments: Metric instruments to record into.
            idle_wait: Longest wait for a notification while the queue is empty.
            drain_timeout: Grace period for running tasks during shutdown.
            claim_backoff_first: First retry delay after a claim error.
            claim_backoff_max: Upper bound on that retry delay.
            breaker_losses: Number of lease losses and ``Lease`` refusals within ``breaker_window`` that pauses
                claiming.
            breaker_window: Window over which they are counted.
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
        self._losses: deque[float] = deque()  # loop.time() of each loss in the breaker window
        self._paused_until: float | None = None  # loop.time() the breaker pause ends
        self._last_contact: float | None = None  # loop.time() of the last successful claim or listener wait
        self._fatal: BaseException | None = None  # the first _FATAL a runner raised; run re-raises it

    async def run(self) -> None:
        """Run the claim loop until it stops or raises, then drain and close any listener the worker opened.

        ``_drain`` leaves the worker stopping, so a second call drains again and returns without claiming. If
        ``_drain`` raises (for example when the call is cancelled during the drain), the worker's own listener is not
        closed. If a runner raised a ``_FATAL`` exception, ``_supervise`` has called ``stop``; the other running tasks
        are drained as on a normal stop and the first such exception is then re-raised.

        Preconditions:
            Awaited from the event loop the worker's events belong to.

        Args:
            None.

        Returns:
            None, after ``stop`` has been called and running tasks have been drained.

        Raises:
            asyncio.CancelledError: If the call is cancelled. Draining still runs when the claim loop is cancelled; a
                cancelled drain cancels the running tasks (see ``_drain``).
            BaseException: The first ``_FATAL`` exception a runner raised (``KeyboardInterrupt``, ``SystemExit`` or
                ``GeneratorExit``), after the drain and the listener close, unless ``asyncio`` has already raised
                the first two into the event loop.
            Exception: Whatever ``_claim_loop`` raises (from ``_start``), after the drain, or any exception from
                closing the worker's own listener. Either takes the place of a recorded ``_FATAL`` exception.

        """
        try:
            await self._claim_loop()
        finally:
            await self._drain()
            if self._owned is not None:
                await self._owned.close()
        if self._fatal is not None:
            raise self._fatal

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

        The pause and the contact times are kept on the event loop's monotonic ``time()``; each is turned into a
        wall-clock time here by its distance from ``utc_now()``, so a step of the system clock moves the reported
        times with it and never stretches the remaining pause or the age of the last contact.

        Preconditions:
            Called from the thread running the worker's event loop, while it runs.

        Args:
            None.

        Returns:
            A ``HealthStatus`` with ``busy`` as the number of running tasks, ``slots`` as ``settings.concurrency``,
            ``draining`` set once stopping has begun, ``paused_until`` as ``utc_now()`` plus what remains of the
            breaker pause if any remains and otherwise ``None``, and ``last_db_contact`` as ``utc_now()`` minus the
            time since the latest of the last successful claim or listener wait and each running lease's
            ``last_heartbeat``, or ``None`` if there is none.

        Raises:
            RuntimeError: If no event loop is running in this thread (raised by ``asyncio.get_running_loop``).
            ValueError: If ``settings.concurrency`` is less than 1 (raised by ``HealthStatus``).

        """
        now, clock = utc_now(), asyncio.get_running_loop().time()
        left = None if self._paused_until is None else self._paused_until - clock
        contacts = [self._last_contact, *(lease.last_heartbeat for lease in self._running.values() if lease)]
        latest = max((c for c in contacts if c is not None), default=None)
        return HealthStatus(
            worker_id=self._settings.worker_id,
            busy=len(self._running),
            slots=self._settings.concurrency,
            draining=self._stopping.is_set(),
            paused_until=now + timedelta(seconds=left) if left is not None and left > 0 else None,
            last_db_contact=None if latest is None else now - timedelta(seconds=clock - latest),
        )

    async def _claim_loop(self) -> None:
        """Claim and start tasks until the worker is stopping.

        Each round:
        - waits on the wake event while every slot is busy;
        - sleeps out a breaker pause that is still running;
        - otherwise claims a task on a pool connection.

        A failed claim is logged and retried after ``_backoff``. A successful claim resets the failure count and
        records the contact time. It then starts the task with the time noted just before the claim call, or waits
        for a notification if there was none. ``_finished``
        and ``stop`` set the wake event, so a full worker resumes claiming when a slot frees.

        Preconditions:
            Called from ``run``. Once ``run`` has drained, the worker stays stopping, so a later call returns at once.

        Args:
            None.

        Returns:
            None, once the worker is stopping.

        Raises:
            asyncio.CancelledError: If the call is cancelled. An ``Exception`` from taking a connection or claiming
                is logged at ``WARNING`` and retried, not raised.
            Exception: Whatever ``_start`` raises.

        """
        loop = asyncio.get_running_loop()
        failures = 0
        while not self._stopping.is_set():
            if len(self._running) >= self._settings.concurrency:
                self._wake.clear()
                if len(self._running) >= self._settings.concurrency and not self._stopping.is_set():
                    await self._wake.wait()
                continue
            if self._paused_until is not None and (pause := self._paused_until - loop.time()) > 0:
                await self._sleep(timedelta(seconds=pause))
                continue
            try:
                async with self._pool.connection() as conn:
                    claim_started = (
                        loop.time()
                    )  # no later than control.claim's renewal: the lease's first silence clock
                    task = await control.claim(conn, self._settings.worker_id)
            except Exception:
                failures += 1
                wait = _backoff(self._backoff_first, self._backoff_max, failures)
                _log.warning("claim failed; retrying in %.1f s", wait.total_seconds(), exc_info=True)
                await self._sleep(wait)
                continue
            failures = 0
            self._last_contact = loop.time()
            if task is None:
                await self._wait_for_task()
            else:
                self._start(task, claim_started)

    async def _sleep(self, wait: timedelta) -> None:
        """Wait for ``wait`` or until ``stop``, whichever comes first.

        Preconditions:
            None.

        Args:
            wait: The longest wait.

        Returns:
            None.

        Raises:
            asyncio.CancelledError: If the call is cancelled. The wait's own ``TimeoutError`` is suppressed.

        """
        with contextlib.suppress(TimeoutError):
            async with asyncio.timeout(wait.total_seconds()):
                await self._stopping.wait()

    async def _wait_for_task(self) -> None:
        """Wait for a task notification, the idle timeout or ``stop``, whichever comes first.

        With no listener (a failed one was dropped), tries once to open a replacement (``_reopen_listener``) instead
        of waiting. When that succeeds it returns at once, so the claim loop claims once before waiting on the new
        listener: a task enqueued while there was none sent its ``NOTIFY`` to no listener. When it fails it waits for
        ``idle_wait`` or ``stop``, so the claim loop keeps claiming on that timer, as claiming needs only the pool.

        With a listener, races ``control.wait_for_task`` on it against the stopping event, then cancels whichever is
        still pending and waits for both. If the wait was cancelled because ``stop`` came first, nothing else happens.
        If it finished without an error, with or without a notification, the contact time is updated. If it raised,
        the error is logged at ``WARNING`` and ``self._drop_listener`` is called.

        Preconditions:
            No other coroutine uses the listener during the call.

        Args:
            None.

        Returns:
            None.

        Raises:
            asyncio.CancelledError: If the call is cancelled, after both inner tasks have been cancelled and awaited.
                An ``Exception`` from the listener wait is logged and handled, not raised.

        """
        if self._listener is None:
            if not await self._reopen_listener():
                await self._sleep(self._idle_wait)  # poll: claiming needs only the pool
            return  # a replacement opened: claim once before waiting on it
        listener = self._listener
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
            self._last_contact = asyncio.get_running_loop().time()
            return
        _log.warning("task listener failed; reopening it", exc_info=err)
        await self._drop_listener()

    async def _reopen_listener(self) -> bool:
        """Try once to open a replacement listener, unless the worker is stopping.

        A replacement comes from ``control.open_listener(self._settings)`` and is stored as both the listener and the
        worker-owned connection. A failed open is logged at ``WARNING``; the next idle round tries again.

        Preconditions:
            There is no listener (``self._listener`` is ``None``).

        Args:
            None.

        Returns:
            ``True`` if a replacement opened; ``False`` if the open failed or the worker is stopping.

        Raises:
            asyncio.CancelledError: If the call is cancelled. An ``Exception`` from opening the listener is logged at
                ``WARNING``, not raised.

        """
        if self._stopping.is_set():
            return False
        try:
            self._listener = self._owned = await control.open_listener(self._settings)
        except Exception:
            _log.warning(
                "could not reopen the task listener; claiming every %.1f s until it opens",
                self._idle_wait.total_seconds(),
                exc_info=True,
            )
            return False
        return True

    async def _drop_listener(self) -> None:
        """Forget a failed listener, closing it if this worker opened it.

        The caller's listener is only forgotten, never closed. An ``Exception`` from closing the worker's own listener
        is suppressed.

        Preconditions:
            None.

        Args:
            None.

        Returns:
            None.

        Raises:
            asyncio.CancelledError: If the call is cancelled while closing the worker's own listener; the listener is
                then still stored. An ``Exception`` from the close is suppressed, not raised.

        """
        if self._owned is not None and self._owned is self._listener:
            with contextlib.suppress(Exception):
                await self._owned.close()
            self._owned = None
        self._listener = None

    def _start(self, task: ClaimedTask, claim_started: float) -> None:
        """Start ``task`` in its own ``asyncio`` task and track it as running.

        Adds 1 to ``slots_busy``, records the queue wait from the later of ``created_at`` and ``not_before`` to
        ``claimed_at``, creates the task running ``_run_task``, adds it to the running map with no lease yet, and
        registers ``_finished`` as its done callback.

        Preconditions:
            Called from the claim loop with a free slot.

        Args:
            task: The task just claimed.
            claim_started: The event loop's ``time()`` just before the claim, passed on to the lease.

        Returns:
            None.

        Raises:
            Exception: Any exception from the instruments, unchanged; ``_claim_loop`` does not catch it.

        """
        self._instruments.slots_busy.add(1)
        waited = task.claimed_at - max(task.created_at, task.not_before)
        self._instruments.queue_wait.record(waited.total_seconds())
        running = asyncio.create_task(self._run_task(task, claim_started), name=f"task-{task.id}")
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
            Exception: Any exception from the ``slots_busy`` instrument, unchanged, before the wake event is set.

        """
        del self._running[running]
        self._instruments.slots_busy.add(-1)
        self._wake.set()
        if not running.cancelled() and (err := running.exception()) is not None:
            _log.error("task slot %s ended with an error", running.get_name(), exc_info=err)

    async def _run_task(self, task: ClaimedTask, claim_started: float) -> None:
        """Run one attempt at ``task`` inside a ``task`` span and with the task's fields bound for logging.

        The span's parent is the context extracted from ``task.trace_context`` as a ``traceparent`` header, or an
        empty context when there is none. ``telemetry.bind_task`` binds the task's id, run, attempt, connector, job,
        source and partition as ``pipeline.*`` fields.

        Preconditions:
            None.

        Args:
            task: The claimed task.
            claim_started: The event loop's ``time()`` just before the claim, passed on to ``_attempt``.

        Returns:
            None.

        Raises:
            BaseException: Whatever ``_attempt`` raises, unchanged: an ``Exception``, ``asyncio.CancelledError`` or a
                runner's ``_FATAL`` exception. ``telemetry.bind_task`` restores the bound fields and the span ends
                even when it does.

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
            await self._attempt(task, claim_started)

    async def _attempt(self, task: ClaimedTask, claim_started: float) -> None:
        """Create the task's lease, then report the task as failed or supervise it under the lease.

        A ``ValueError`` from ``Lease`` (the claimed lease is too short for ``silence_limit``) is reported as the
        task's failure through ``_report_unleased`` and counted towards the circuit breaker
        (``_count_towards_breaker``), so a misconfigured worker pauses instead of draining the queue. A job not in
        ``known_jobs`` is logged at ``ERROR``, counted in ``unknown_jobs`` and reported as ``UnknownJob``. A known job
        without a runner is reported as ``JobNotRunnable``. Otherwise the lease is stored against the current task in
        the running map and ``_supervise`` runs inside ``lease.hold()``.

        Preconditions:
            Runs inside the ``asyncio`` task that ``_start`` created for ``task``.

        Args:
            task: The claimed task.
            claim_started: The event loop's ``time()`` just before the claim, where the lease's silence clock starts.

        Returns:
            None.

        Raises:
            Exception: Whatever ``_supervise`` or ``_report_failure`` raises, or the ``unknown_jobs`` or
                ``worker_paused`` instrument raises. ``_report_unleased`` swallows every ``Exception``.
            asyncio.CancelledError: If the slot is cancelled.
            BaseException: A runner's ``_FATAL`` exception, re-raised by ``_supervise`` after it has called ``stop``.

        """
        try:
            lease = Lease(
                self._pool,
                task,
                worker_id=self._settings.worker_id,
                instruments=self._instruments,
                claim_started=claim_started,
                interval=self._heartbeat_interval,
                silence_limit=self._silence_limit,
            )
        except ValueError as err:
            await self._report_unleased(task, err)
            self._count_towards_breaker()
            return
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

        Waits for the first of: the runner's task ending, ``lease.lost`` being set, or the drain cut-off.
        - Lease lost or cut-off first: the runner is cancelled and awaited (``_cancel``), and the ending goes to
          ``_settle`` with ``Interrupted`` as the failure and the time after the cancel as the end.
        - Runner raised a ``_FATAL`` exception (``KeyboardInterrupt``, ``SystemExit``, ``GeneratorExit``): it is
          recorded for ``run`` if none was before, ``stop`` is called and it is re-raised without a report.
        - Runner returned after ``lease.complete`` (``lease.completed`` set): nothing more happens.
        - Any other ending goes to ``_settle``, with the runner's own exception (an ``Exception`` or any other
          ``BaseException``, such as pyo3's ``PanicException`` from Polars) as the failure, or a ``RuntimeError``
          for a runner that was cancelled or returned without completing the lease.

        ``_settle`` then logs the ending when the outcome has landed (``_landed``), records a lease loss when the lease
        is lost or the failure classifies to ``None`` (``_lose``, with the unwind ``_unwind`` measures from
        ``lease.lost_at``), and otherwise reports the failure (``_report_failure``).

        Preconditions:
            ``lease`` is held. The runner ends once cancelled, as ``TaskRunner`` states; ``_drain``'s unbounded second
            wait relies on it.

        Args:
            lease: The task's lease.
            runner: The job's runner.

        Returns:
            None.

        Raises:
            asyncio.CancelledError: If this slot is cancelled from outside. While the first wait runs, the runner is
                cancelled and awaited first; a cancellation that arrives later (during ``_cancel`` or ``_settle``) is
                raised from that await, and the runner may then still be unwinding.
            KeyboardInterrupt: The runner raised it; it is recorded, ``stop`` is called and it is re-raised
                unchanged. Likewise ``SystemExit`` and ``GeneratorExit``. ``asyncio`` has usually already re-raised the
                first two into the event loop.
            Exception: Whatever ``_report_failure``, ``_lose`` or the instruments raise (see ``_report_failure``).

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
            ending, runner_err = _ending(work)
            drained = self._drain_timeout.total_seconds()
            failure: BaseException = Interrupted(f"cancelled by shutdown after the {drained:g} s drain")
            await self._settle(lease, ending, runner_err, failure, loop.time())
            return
        ended = loop.time()
        ending, runner_err = _ending(work)
        if isinstance(runner_err, _FATAL):
            if self._fatal is None:
                self._fatal = runner_err
            self.stop()
            raise runner_err
        if runner_err is not None:
            failure = runner_err
        elif ending == "was cancelled":
            failure = RuntimeError("runner was cancelled")
        elif lease.completed:  # a normal return after complete: nothing to check or report
            return
        else:
            failure = RuntimeError("runner returned without completing the lease")
        await self._settle(lease, ending, runner_err, failure, ended)

    async def _settle(
        self, lease: Lease, ending: str, runner_err: BaseException | None, failure: BaseException, ended: float
    ) -> None:
        """Act on a runner ending that is not a normal return after ``lease.complete``: check, lose or report it.

        In order: once ``lease.completed`` is set (the outcome has landed), ``_landed`` logs the ending and
        nothing else happens. Otherwise, a lost lease (including one lost in the step the
        runner ended) or a ``failure`` that ``classify`` maps to ``None`` (``LeaseConflict``: Load's transaction
        rolled back) goes to ``_lose`` with the unwind ``_unwind`` measures up to ``ended``. Anything else is
        reported as ``failure`` through ``_report_failure``.

        Preconditions:
            The runner's task is done. ``failure`` is not a ``_FATAL`` exception or ``asyncio.CancelledError``.

        Args:
            lease: The task's lease.
            ending: How the runner ended, as ``_ending`` names it.
            runner_err: The runner's exception when it raised, else ``None``.
            failure: The exception to report when the ending is neither landed nor a loss: the runner's own
                exception, or the one that stands for a cut-off, a cancellation or a return without completing.
            ended: ``loop.time()`` when the runner's task ended.

        Returns:
            None.

        Raises:
            Exception: Whatever ``_report_failure``, ``_lose`` or the instruments raise (see ``_report_failure``).
            asyncio.CancelledError: If the call is cancelled.

        """
        if lease.completed:
            self._landed(lease, ending, runner_err)
            return
        if lease.lost.is_set() or (isinstance(failure, Exception) and classify(failure) is None):
            await self._lose(lease, _unwind(lease, ended))
            return
        await self._report_failure(lease, failure)

    def _landed(self, lease: Lease, ending: str, runner_err: BaseException | None) -> None:
        """After ``lease.completed`` was set but the runner ended abnormally, log the ending.

        ``Lease`` sets ``completed`` only once the outcome has committed, or once ``control`` has refused a report or
        fence with ``AlreadyReported`` because an outcome already landed under the lease, so the outcome has landed:
        nothing is checked, reported or counted as a loss. The ending is logged at error level, with the runner's
        exception when it raised.

        Preconditions:
            ``lease.completed`` is ``True`` and the runner's task is done.

        Args:
            lease: The task's lease.
            ending: How the runner ended, as the log line reads: ``"raised"``, ``"was cancelled"`` or ``"returned"``
                (a runner that returned once cancelled on the lost-lease or cut-off path).
            runner_err: The runner's exception when it raised, else ``None``.

        Returns:
            None. The caller reports nothing and records no loss.

        Raises:
            None.

        """
        _log.error("runner for task %s %s after completing the lease", lease.task.id, ending, exc_info=runner_err)

    async def _cancel(self, work: asyncio.Task[None]) -> None:
        """Cancel the runner and wait for it to unwind.

        Preconditions:
            None.

        Args:
            work: The runner's asyncio task.

        Returns:
            None, once the runner's task is done.

        Raises:
            asyncio.CancelledError: If the call is cancelled while waiting; the runner may then still be unwinding.
                Whatever the runner raises while unwinding, a ``_FATAL`` exception included, is logged at debug
                level and dropped.

        """
        work.cancel()
        await asyncio.wait([work])
        if not work.cancelled() and (err := work.exception()) is not None:
            _log.debug("runner raised while unwinding", exc_info=err)

    async def _report_failure(self, lease: Lease, err: BaseException) -> None:
        """Report ``err`` as the outcome of the lease's task.

        The failure is logged and its report built by ``_failure_report``, and the report is sent with
        ``lease.complete`` on a pool connection, inside one ``try``. A ``LeaseConflict``, meaning the lease is no
        longer held, is recorded as a lease loss through ``self._lose`` with no unwind time. An ``AlreadyReported``,
        meaning an outcome already landed under the lease (as when the runner's own report committed at the drain
        cut-off but its reply was cancelled), is logged at ``WARNING`` as landed: no loss is recorded and nothing is
        counted towards the breaker. Any other ``Exception``
        from building the report (such as ``TaskReport``'s ``ValueError`` for a value an exception's own constructor
        did not check), taking a connection or reporting is logged at ``ERROR`` and swallowed, and the lease is left
        to expire. ``err.retry_after`` is read inside that ``try``.

        Preconditions:
            ``err`` is not ``asyncio.CancelledError``, ``KeyboardInterrupt``, ``SystemExit`` or ``GeneratorExit``.
            ``classify`` does not map ``err`` to ``None``: ``_settle`` sends those to ``_lose`` instead, and
            ``_attempt`` passes only ``UnknownJob`` and ``JobNotRunnable``. Not checked; such an ``err`` would be
            reported as ``Outcome.UNEXPECTED`` with its class name.

        Args:
            lease: The lease on the failed task.
            err: The exception to report.

        Returns:
            None.

        Raises:
            Exception: Whatever ``_lose`` raises, when reporting raises ``LeaseConflict``.
            asyncio.CancelledError: If the call is cancelled. Any other ``Exception`` from reading ``err.retry_after``,
                building the report, taking a connection or reporting is logged at ``ERROR`` and swallowed.

        """
        task = lease.task
        try:
            report = _failure_report(task, err)
            async with self._pool.connection() as conn:
                await lease.complete(conn, report)
        except LeaseConflict:
            await self._lose(lease, None)
        except AlreadyReported:
            _log.warning("the outcome of task %s had already landed under its lease; not reported again", task.id)
        except Exception:
            _log.error("could not report the outcome of task %s; its lease will expire", task.id, exc_info=True)

    async def _report_unleased(self, task: ClaimedTask, err: ValueError) -> None:
        """Report ``err``, ``Lease``'s refusal to hold ``task``, as the task's outcome without a lease.

        As ``_report_failure``, but the report built by ``_failure_report`` is sent with ``control.report_outcome`` on
        a pool connection, and its duration is recorded in ``task_duration`` (``pipeline.task.duration``) with ``job``
        and ``outcome`` attributes once it has committed, as ``Lease.complete`` records it. Any ``Exception`` from
        building, sending or recording the report, a ``LeaseConflict`` included, is logged at ``ERROR`` and
        swallowed, and the lease is left to expire.

        Preconditions:
            ``Lease`` refused ``task`` with ``err``. ``task_duration`` is a ``Histogram``, as ``Instruments`` declares.

        Args:
            task: The claimed task no ``Lease`` could be built for.
            err: The ``ValueError`` ``Lease`` raised.

        Returns:
            None.

        Raises:
            asyncio.CancelledError: If the call is cancelled. Any ``Exception`` is logged and swallowed.

        """
        try:
            report = _failure_report(task, err)
            async with self._pool.connection() as conn:
                await control.report_outcome(conn, task, report)
            self._instruments.task_duration.record(
                report.duration.total_seconds(), {"job": task.job, "outcome": report.outcome.value}
            )
        except Exception:
            _log.error("could not report the outcome of task %s; its lease will expire", task.id, exc_info=True)

    async def _lose(self, lease: Lease, unwind: timedelta | None) -> None:
        """Record a lease loss and count it towards the circuit breaker.

        A lease not yet marked lost (a ``LeaseConflict`` that did not come through ``fence`` or ``complete``) is
        marked with ``lease.lose(LeaseLoss.COMMIT)``, which also records its ``lost_at``. The loss is then recorded
        through ``lease.record_loss`` and counted by ``_count_towards_breaker``.

        The loss times and ``_paused_until`` are the event loop's monotonic ``time()``, so a step of the system clock
        moves neither the window nor the pause. Only the logged end of the pause is a wall-clock ``utc_now`` time.

        Preconditions:
            The task has unwound. Called on the worker's running event loop.

        Args:
            lease: The lost lease.
            unwind: From ``lease.lost_at`` to when the runner's task ended, or ``None`` if unknown.

        Returns:
            None.

        Raises:
            Exception: Any exception from the instruments, raised by ``lease.record_loss`` or ``worker_paused``,
                unchanged. ``lease.record_loss`` logs and swallows a failure to store the loss.
            asyncio.CancelledError: If the call is cancelled while ``lease.record_loss`` stores the loss; the loss is
                then not counted towards the breaker.

        """
        if lease.loss is None:  # a LeaseConflict that did not come through fence or complete
            lease.lose(LeaseLoss.COMMIT)
        await lease.record_loss(unwind)
        self._count_towards_breaker()

    def _count_towards_breaker(self) -> None:
        """Count one lease loss or ``Lease`` refusal towards the circuit breaker, pausing claiming when it trips.

        The time is appended to the breaker window, entries older than ``breaker_window`` are dropped, and once
        ``breaker_losses`` remain the window is cleared, ``worker_paused`` is counted and claiming pauses for
        ``breaker_pause``. The times and ``_paused_until`` are the event loop's monotonic ``time()``.

        Preconditions:
            Called on the worker's running event loop.

        Args:
            None.

        Returns:
            None.

        Raises:
            Exception: Any exception from the ``worker_paused`` instrument, unchanged.

        """
        now = asyncio.get_running_loop().time()
        self._losses.append(now)
        while self._losses and now - self._losses[0] > self._breaker_window.total_seconds():
            self._losses.popleft()
        if len(self._losses) >= self._breaker_losses:
            self._losses.clear()
            self._paused_until = now + self._breaker_pause.total_seconds()
            self._instruments.worker_paused.add(1)
            _log.warning(
                "%d lease losses or refusals within %s; not claiming until %s",
                self._breaker_losses,
                self._breaker_window,
                (utc_now() + self._breaker_pause).isoformat(),
            )

    async def _drain(self) -> None:
        """Stop claiming and wait for running tasks: up to ``drain_timeout``, then without a bound after the cut-off.

        Sets the stopping event. If tasks are running, waits for all of them for up to ``drain_timeout``. If any are
        still tracked afterwards, it sets the cut-off event, which makes ``_supervise`` cancel its work, and waits for
        all of them again with no time limit. Exceptions of the tasks themselves are not re-raised. If the drain is
        cancelled while tasks are running, they are cancelled too, without being awaited, and their names are logged at
        ``ERROR``.

        Preconditions:
            None.

        Args:
            None.

        Returns:
            None.

        Raises:
            asyncio.CancelledError: If the call is cancelled; the running tasks are cancelled first. The drain
                timeout's ``TimeoutError`` is suppressed.

        """
        self._stopping.set()
        try:
            if self._running:
                with contextlib.suppress(TimeoutError):
                    async with asyncio.timeout(self._drain_timeout.total_seconds()):
                        await asyncio.wait(list(self._running))
            if self._running:
                self._cutoff.set()
                await asyncio.wait(list(self._running))
        except asyncio.CancelledError:
            if self._running:
                _log.error("drain cancelled; cancelling %s", ", ".join(sorted(t.get_name() for t in self._running)))
                for running in self._running:
                    running.cancel()
            raise
