"""The lease a worker holds on a claimed task: background heartbeats, fencing, completion and loss recording.

Public API:
    HEARTBEAT_INTERVAL: Default time between heartbeats.
    SILENCE_LIMIT: Default time without a successful heartbeat after which the lease is given up.
    Lease: The lease on one claimed task.

"""

import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from typing import Final, Self

from psycopg.pq import TransactionStatus

from pipeline.db import control
from pipeline.db.control import ClaimedTask, LeaseLoss, TaskReport
from pipeline.db.pool import Connection, Pool
from pipeline.errors import LeaseConflict
from pipeline.telemetry import Instruments

#: Default time between heartbeats; also the timeout on each heartbeat call.
HEARTBEAT_INTERVAL: Final = timedelta(seconds=30)
#: Default time without a successful heartbeat after which the lease is given up.
SILENCE_LIMIT: Final = timedelta(minutes=4)  # below the 5-minute lease

#: Logger for this module.
_log: Final = logging.getLogger(__name__)
#: Transaction states in which a connection counts as having a load open, which pauses the heartbeats.
_IN_TRANSACTION: Final = frozenset({TransactionStatus.ACTIVE, TransactionStatus.INTRANS, TransactionStatus.INERROR})


class Lease:
    """Track the lease a worker holds on one claimed task.

    Attributes:
        task: The claimed task the lease is on.
        lost: Event set when the lease is lost: by ``lose``, when a heartbeat, ``fence`` or ``complete`` finds the
            lease gone or the heartbeats fall silent, or by the worker.
        loss: Kind of the first lease loss, or ``None`` while none has been recorded.
        completed: ``True`` once ``complete`` has reported the outcome.

    """

    def __init__(
        self,
        pool: Pool,
        task: ClaimedTask,
        *,
        worker_id: str,
        instruments: Instruments,
        interval: timedelta = HEARTBEAT_INTERVAL,
        silence_limit: timedelta = SILENCE_LIMIT,
    ) -> None:
        """Store the task, pool, worker ID, instruments and heartbeat settings.

        ``lost`` starts unset, ``loss`` and ``last_heartbeat`` as ``None``, ``completed`` as ``False``, and no
        connection is fenced. No heartbeat starts until ``hold`` is entered.

        Preconditions:
            ``interval`` and ``silence_limit`` are positive. They are not checked.

        Args:
            pool: Pool the heartbeats and the loss record take connections from.
            task: The claimed task the lease is on.
            worker_id: ID of the worker holding the lease, used in the loss log and record.
            instruments: Instruments that time heartbeats and reported task outcomes, and count and time lease losses.
            interval: Time between heartbeats.
            silence_limit: Time without a successful heartbeat after which the lease is given up.

        Returns:
            None.

        Raises:
            None.

        """
        self.task = task
        self.lost = asyncio.Event()
        self.loss: LeaseLoss | None = None
        self.completed = False
        self._fenced: Connection | None = None  # Load's connection, kept only to see whether its transaction is open
        self._fences = 0  # successful fences so far
        self._last_heartbeat: datetime | None = None
        self._lost_at: float | None = None  # loop.time() of the first loss
        self._pool = pool
        self._worker_id = worker_id
        self._instruments = instruments
        self._interval = interval
        self._silence_limit = silence_limit

    @property
    def last_heartbeat(self) -> datetime | None:
        """Return the time of the last successful heartbeat.

        Preconditions:
            None.

        Args:
            None.

        Returns:
            The aware UTC time of the last heartbeat that confirmed the lease, or ``None`` before the first one.

        Raises:
            None.

        """
        return self._last_heartbeat

    @property
    def lost_at(self) -> float | None:
        """Return the time the lease was lost.

        Nothing sets ``_lost_at`` after ``__init__``, so this is always ``None``.

        Preconditions:
            None.

        Args:
            None.

        Returns:
            ``None``.

        Raises:
            None.

        """
        return self._lost_at

    @asynccontextmanager
    async def hold(self) -> AsyncIterator[Self]:
        """Run a background heartbeat task for the duration of the ``async with`` block.

        On entry, creates a task named ``heartbeat-<task id>`` running ``_heartbeats``. On exit, cancels it and waits
        for it to finish. If the heartbeats stop early because the lease is lost, the block keeps running; the caller
        watches ``lost``. An exception the heartbeat task raised is not re-raised or logged, because ``asyncio.wait``
        does not re-raise it.

        Preconditions:
            Entered with a running event loop.

        Args:
            None.

        Yields:
            ``self``, once.

        Raises:
            Exception: Any exception raised in the block, unchanged, after the heartbeat task has finished.

        """
        beats = asyncio.create_task(self._heartbeats(), name=f"heartbeat-{self.task.id}")
        try:
            yield self
        finally:
            beats.cancel()
            await asyncio.wait([beats])

    async def fence(self, conn: Connection) -> None:
        """Check that the lease is still held by running ``control.fence``.

        On success, counts the fence and remembers ``conn`` as the fenced connection. While ``conn`` has a
        transaction open the load holds the task's row lock, so the heartbeats pause rather than queue behind it. On
        ``LeaseConflict``, records a ``COMMIT`` loss through ``lose`` and re-raises; the fence is then not counted.

        Preconditions:
            ``conn`` is open and not used by another coroutine.

        Args:
            conn: Connection the fence runs on, normally the one the load then writes on.

        Returns:
            None.

        Raises:
            LeaseConflict: If the lease is no longer held, after a ``COMMIT`` loss has been recorded and ``lost`` set.
            DatabaseConflict: If the statement raises ``SerializationFailure`` or ``DeadlockDetected``.
            DatabaseUnavailable: If the connection is lost or the warehouse is unavailable.
            psycopg.Error: Any other database error, unchanged. Nothing is swallowed.

        """
        try:
            await control.fence(conn, self.task)
        except LeaseConflict:
            self.lose(LeaseLoss.COMMIT)
            raise
        self._fences += 1
        self._fenced = conn  # while its transaction is open Load holds the row lock: heartbeats would queue behind it

    async def complete(self, conn: Connection, report: TaskReport) -> str:
        """Report the task's outcome through ``control.report_outcome`` and mark the lease completed.

        On success, sets ``completed`` to ``True``, records ``report.duration`` in seconds in ``task_duration``
        (``pipeline.task.duration``) with ``job`` and ``outcome`` attributes, and returns the result. On
        ``LeaseConflict``, records a ``COMMIT`` loss through ``lose`` and re-raises; nothing is recorded in
        ``task_duration``. This one place covers every reported outcome: failures from ``Worker._report_failure``
        and successes from the job runner (stage 9).

        Preconditions:
            ``conn`` is open and not used by another coroutine. If it is not in autocommit mode, the caller commits
            or rolls back afterwards.
            ``task_duration`` is a ``Histogram``, as ``Instruments`` declares.

        Args:
            conn: Connection the outcome is reported on.
            report: The outcome to report.

        Returns:
            The text ``control.report_outcome`` returned.

        Raises:
            LeaseConflict: If the lease is no longer held, after a ``COMMIT`` loss has been recorded and ``lost`` set.
                ``completed`` stays ``False``.
            RuntimeError: If ``control.report_outcome`` returns no row.
            TypeError: If ``report.duration`` is not a ``timedelta``.
            DatabaseConflict: If the statement raises ``SerializationFailure`` or ``DeadlockDetected``.
            DatabaseUnavailable: If the connection is lost or the warehouse is unavailable.
            psycopg.Error: Any other database error, unchanged. Nothing is swallowed.

        """
        try:
            state = await control.report_outcome(conn, self.task, report)
        except LeaseConflict:
            self.lose(LeaseLoss.COMMIT)
            raise
        self.completed = True
        self._instruments.task_duration.record(  # every reported outcome, success or failure
            report.duration.total_seconds(), {"job": self.task.job, "outcome": report.outcome.value}
        )
        return state

    async def record_loss(self, unwind: timedelta | None) -> None:
        """Count, log and store the loss of the lease.

        Adds 1 to ``lease_losses`` and, when ``unwind`` is given, records its seconds in ``lease_unwind``, both with
        ``kind`` and ``job`` attributes. Then logs the loss at ``ERROR`` with the task, lease token, attempt, worker,
        kind and unwind time. Last, stores the loss with ``control.record_lease_loss`` on a pool connection.

        Preconditions:
            The lease has been lost: ``loss`` is not ``None``.
            ``lease_losses`` is a ``Counter`` and ``lease_unwind`` a ``Histogram``, as ``Instruments`` declares.

        Args:
            unwind: Time the task took to unwind after the loss, or ``None`` when not known.

        Returns:
            None.

        Raises:
            RuntimeError: If ``loss`` is ``None``.
            Exception: Any exception from the instruments, unchanged. An ``Exception`` from taking a connection or
                storing the loss is logged at ``WARNING`` with its traceback and swallowed.

        """
        kind = self.loss
        if kind is None:
            raise RuntimeError(f"task {self.task.id}: record_loss called before the lease was lost")
        attributes = {"kind": kind.value, "job": self.task.job}
        self._instruments.lease_losses.add(1, attributes)
        if unwind is not None:
            self._instruments.lease_unwind.record(unwind.total_seconds(), attributes)
        _log.error(
            "lost the lease on task %s (token %s, attempt %s, worker %s): %s; unwound in %s",
            self.task.id,
            self.task.lease_token,
            self.task.attempt,
            self._worker_id,
            kind.value,
            "unknown" if unwind is None else f"{unwind.total_seconds():.3f} s",
        )
        try:
            async with self._pool.connection() as conn:
                await control.record_lease_loss(conn, self.task, kind, self._worker_id, unwind)
        except Exception:
            _log.warning("could not record the lease loss on task %s", self.task.id, exc_info=True)

    def lose(self, kind: LeaseLoss) -> None:
        """Record ``kind`` as the loss if none is recorded yet, and set ``lost``.

        A later call keeps the first ``kind`` but sets ``lost`` again.

        Preconditions:
            None.

        Args:
            kind: Kind of lease loss.

        Returns:
            None.

        Raises:
            None.

        """
        if self.loss is None:
            self.loss = kind
            self._lost_at = asyncio.get_running_loop().time()
        self.lost.set()

    def _load_open(self) -> bool:
        """Report whether the fenced connection has a transaction open.

        The fenced connection is kept after its transaction ends, so this turns ``False`` again once the load commits
        or rolls back.

        Preconditions:
            None.

        Args:
            None.

        Returns:
            ``True`` if ``fence`` has succeeded and that connection's transaction status is in ``_IN_TRANSACTION``.

        Raises:
            None.

        """
        return self._fenced is not None and self._fenced.info.transaction_status in _IN_TRANSACTION

    async def _heartbeats(self) -> None:
        """Renew the lease every heartbeat interval until it is lost, the task completes, or this is cancelled.

        Each round sleeps for one interval. While a load is open (``_load_open``), the round is skipped and the
        heartbeats count as paused; the first round after a pause resets the last success to now. Otherwise the round
        takes a pool connection and calls ``control.heartbeat``, with a timeout of one interval on the call itself
        but not on taking the connection.
        - A heartbeat that raises is logged at ``WARNING`` as ``timed out`` or ``failed``. If ``fence`` ran during
          the attempt or a load is now open, the heartbeats pause. Otherwise, once the silence limit has passed since
          the last success (initially ``task.claimed_at``), a ``SILENCE`` loss is recorded and the loop returns.
        - A ``False`` result means the lease is gone: a ``HEARTBEAT`` loss is recorded unless ``completed`` is set,
          and the loop returns.
        - A ``True`` result records the time since the last success in ``heartbeat_lag`` and sets the last success
          and ``last_heartbeat`` to now.

        Preconditions:
            Runs as the task ``hold`` creates.

        Args:
            None.

        Returns:
            None, after a loss is recorded or the heartbeat reports the lease gone.

        Raises:
            asyncio.CancelledError: When the task running it is cancelled, as ``hold`` does on exit.
            Exception: Any exception from ``heartbeat_lag``, unchanged. An ``Exception`` from taking a connection or
                from a heartbeat is logged and handled as above, not raised.

        """
        last_success = self.task.claimed_at
        paused = False
        while True:
            await asyncio.sleep(self._interval.total_seconds())
            if self._load_open():
                paused = True
                continue
            if paused:
                paused = False
                last_success = datetime.now(UTC)  # the pause is not silence
            fences = self._fences
            deadline: asyncio.Timeout | None = None  # control.heartbeat has no timeout of its own
            try:
                async with self._pool.connection() as conn:
                    async with (deadline := asyncio.timeout(self._interval.total_seconds())):
                        held = await control.heartbeat(conn, self.task)
            except Exception:
                cause = "timed out" if deadline is not None and deadline.expired() else "failed"
                _log.warning("heartbeat on task %s %s", self.task.id, cause, exc_info=True)
                if self._fences != fences or self._load_open():
                    paused = True  # it waited on Load's row lock; the lease cannot be lost meanwhile
                    continue
                if datetime.now(UTC) - last_success >= self._silence_limit:
                    if not self.completed:
                        self.lose(LeaseLoss.SILENCE)
                    return
                continue
            if not held:
                if not self.completed:
                    self.lose(LeaseLoss.HEARTBEAT)
                return
            now = datetime.now(UTC)
            self._instruments.heartbeat_lag.record((now - last_success).total_seconds())
            last_success = self._last_heartbeat = now
