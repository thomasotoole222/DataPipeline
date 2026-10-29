"""The lease a worker holds on a claimed task: background heartbeats, fencing, completion and loss recording.

Public API:
    HEARTBEAT_INTERVAL: Default time between heartbeats.
    SILENCE_LIMIT: Default time without a successful heartbeat after which the lease is given up.
    Lease: The lease on one claimed task.

"""

import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import AsyncExitStack, asynccontextmanager
from datetime import timedelta
from typing import Final, Self

from psycopg.pq import TransactionStatus

from pipeline.db import control
from pipeline.db.control import ClaimedTask, LeaseLoss, TaskReport
from pipeline.db.pool import Connection, Pool
from pipeline.errors import AlreadyReported, LeaseConflict
from pipeline.telemetry import Instruments

#: Default time between heartbeats; also the timeout on each heartbeat call.
HEARTBEAT_INTERVAL: Final = timedelta(seconds=30)
#: Default time without a successful heartbeat after which the lease is given up.
#: With the default interval and the worker pool's 10 s checkout timeout, ``Lease`` counts one heartbeat round as
#: 80 s (``_CANCEL_MARGIN`` included), so the limit plus that round (260 s) stays below the 5-minute lease
#: ``control.claim`` grants; ``Lease`` checks this.
SILENCE_LIMIT: Final = timedelta(minutes=3)
#: How long a heartbeat call cancelled by its timeout can run on before the cancellation propagates. psycopg 3.3.6's
#: ``AsyncConnection.wait`` has no constant for it: on cancellation it sends the server a cancel request with a 5 s
#: timeout, then waits up to 5 s more for the query to end, both literals in its code.
_CANCEL_MARGIN: Final = timedelta(seconds=10)

#: Logger for this module.
_log: Final = logging.getLogger(__name__)


class Lease:
    """Track the lease a worker holds on one claimed task.

    Attributes:
        task: The claimed task the lease is on.
        lost: Event set when the lease is lost: by ``lose``, when a heartbeat, ``fenced`` or ``complete`` finds the
            lease gone or the heartbeats fall silent, or by the worker.
        loss: Kind of the first lease loss, or ``None`` while none has been recorded.
        completed: ``True`` once an outcome under the lease is known to have committed: when ``complete`` returns
            outside ``fenced``, when the ``fenced`` transaction it reported in commits, when ``complete`` outside
            ``fenced`` or a fence meets ``AlreadyReported``, or when a ``fenced`` transaction in which ``complete``
            met ``AlreadyReported`` (and the block went on) commits.

    """

    def __init__(
        self,
        pool: Pool,
        task: ClaimedTask,
        *,
        worker_id: str,
        instruments: Instruments,
        claim_started: float,
        interval: timedelta = HEARTBEAT_INTERVAL,
        silence_limit: timedelta = SILENCE_LIMIT,
    ) -> None:
        """Check the silence limit, then store the task, pool, worker ID, instruments and heartbeat settings.

        ``lost`` starts unset, ``loss`` and ``last_heartbeat`` as ``None``, ``completed`` as ``False``, and no
        connection is fenced. No heartbeat starts until ``hold`` is entered.

        The check counts one heartbeat round as one ``interval`` of sleep, then the longer of its two ways to end: a
        checkout cancelled at its ``pool.timeout`` timeout (``pool.timeout`` plus ``_CANCEL_MARGIN``), or a checkout
        of up to ``pool.timeout`` then a call cancelled at its one-``interval`` timeout (``pool.timeout`` plus
        ``interval`` plus ``_CANCEL_MARGIN``), so ``2 * interval + pool.timeout + _CANCEL_MARGIN``. It requires
        ``silence_limit`` plus that round to stay below the lease length the claim granted
        (``task.lease_expires_at - task.claimed_at``, which every heartbeat grants again). ``_heartbeats`` checks the
        silence limit every round no open ``fenced`` block pauses, and as soon as a block that paused it closes,
        measuring from its last success: ``claim_started`` (noted before the claim), the time noted just before a
        successful heartbeat's call, or a committed ``fenced`` block's fence time (noted before the fence). Each is no
        later than the renewal the database counts the lease expiry from. A block that rolls back leaves the clock at
        the previous success. The pause adds nothing to the round: no check is due while a block is open, and the
        first one comes when it closes, so the bound holds for every check outside a block. A check that finds
        ``silence_limit`` passed still tries one heartbeat first when the lease's known expiry (the last success plus
        the lease length, which a heartbeat and a fence grant again as the claim did) is more than one such round away,
        so the attempt ends before the expiry; a committed block open for ``silence_limit`` or longer therefore ends in
        a ``SILENCE`` loss as it closes only when that heartbeat cannot be tried in time or fails, unless ``completed``
        is set (see ``_heartbeats``).

        Preconditions:
            ``interval`` and ``silence_limit`` are positive. They are not checked. ``claim_started`` is the running
            event loop's ``time()`` noted before the ``control.claim`` call that returned ``task`` (not checked).

        Args:
            pool: Pool the heartbeats and the loss record take connections from.
            task: The claimed task the lease is on.
            worker_id: ID of the worker holding the lease, used in the loss log and record.
            instruments: Instruments that time heartbeats and reported task outcomes, and count and time lease losses.
            claim_started: The event loop's ``time()`` just before the claim; the first silence clock starts there.
            interval: Time between heartbeats.
            silence_limit: Time without a successful heartbeat after which the lease is given up.

        Returns:
            None.

        Raises:
            ValueError: If ``silence_limit`` plus one heartbeat round, counted as above, reaches the task's lease
                length.

        """
        round_length = 2 * interval + timedelta(seconds=pool.timeout) + _CANCEL_MARGIN  # sleep, checkout, call, cancel
        lease_length = task.lease_expires_at - task.claimed_at
        if silence_limit + round_length >= lease_length:
            raise ValueError(
                f"silence limit {silence_limit} plus a heartbeat round of up to {round_length} reaches the "
                f"{lease_length} lease"
            )
        self.task = task
        self.lost = asyncio.Event()
        self.loss: LeaseLoss | None = None
        self.completed = False
        self._fenced: Connection | None = None  # the connection inside fenced(), while its transaction is open
        self._unfenced = asyncio.Event()  # set when a fenced() block closes; the paused heartbeats wait on it
        self._fences = 0  # successful fences so far
        self._completing: TaskReport | None = None  # complete()'s report on the fenced connection, awaiting commit
        self._already_reported = False  # complete() met AlreadyReported on the fenced connection, awaiting commit
        self._last_heartbeat: float | None = None  # loop.time() just before the last successful heartbeat's call
        self._claim_started = claim_started  # loop.time() just before the claim: the first silence clock
        self._lost_at: float | None = None  # loop.time() of the first loss
        self._fence_renewed: float | None = None  # loop.time() of the latest committed fence's renewal
        self._pool = pool
        self._worker_id = worker_id
        self._instruments = instruments
        self._interval = interval
        self._silence_limit = silence_limit
        self._round = round_length  # one worst-case heartbeat round, as checked above
        self._lease_length = lease_length  # granted again by every heartbeat and committed fence
        self._unreported = asyncio.Event()  # cleared while complete() reports outside fenced
        self._unreported.set()

    @property
    def last_heartbeat(self) -> float | None:
        """Return the time of the last successful heartbeat.

        Preconditions:
            None.

        Args:
            None.

        Returns:
            The event loop's ``time()`` just before the call of the last heartbeat that confirmed the lease, or
            ``None`` before the first one.

        Raises:
            None.

        """
        return self._last_heartbeat

    @property
    def lost_at(self) -> float | None:
        """Return the time the lease was first lost.

        ``lose`` sets ``_lost_at`` on its first call only, so later losses do not move it.

        Preconditions:
            None.

        Args:
            None.

        Returns:
            The event loop's ``time()`` when ``lose`` was first called, or ``None`` before any loss.

        Raises:
            None.

        """
        return self._lost_at

    @property
    def pool(self) -> Pool:
        """Return the pool the lease was created with, which its heartbeats and the task's stages use.

        Preconditions:
            None.

        Args:
            None.

        Returns:
            The ``pool`` given to ``__init__``.

        Raises:
            None.

        """
        return self._pool

    @asynccontextmanager
    async def hold(self) -> AsyncIterator[Self]:
        """Run a background heartbeat task for the duration of the ``async with`` block.

        On entry, creates a task named ``heartbeat-<task id>`` running ``_heartbeats``. On exit, cancels it and waits
        for it to finish. If the heartbeats stop early because the lease is lost, the block keeps running; the caller
        watches ``lost``. An exception the heartbeat task raised is not re-raised here, because ``asyncio.wait``
        does not retrieve it; ``asyncio`` logs it as never retrieved only when the task is garbage-collected. A
        heartbeat task that died that way leaves the lease neither renewed nor marked lost.

        Preconditions:
            Entered with a running event loop.

        Args:
            None.

        Yields:
            ``self``, once.

        Raises:
            Exception: Any exception raised in the block, unchanged, after the heartbeat task has finished.
            asyncio.CancelledError: If the task running the block is cancelled while it waits for the heartbeat task
                to finish; it replaces any exception from the block.

        """
        beats = asyncio.create_task(self._heartbeats(), name=f"heartbeat-{self.task.id}")
        try:
            yield self
        finally:
            beats.cancel()
            await asyncio.wait([beats])

    @asynccontextmanager
    async def fenced(self, conn: Connection) -> AsyncIterator[None]:
        """Open a transaction on ``conn``, check the lease in it with ``control.fence``, and run the block inside it.

        On entry, notes the time (before the transaction opens, so no later than the fence's renewal), opens
        ``conn.transaction()`` and runs ``control.fence``. On success, counts the fence and records
        ``conn`` as the fenced connection: the transaction holds the task's row lock, so the heartbeats pause rather
        than queue behind it. On ``LeaseConflict`` from the fence, the block does not run, the transaction rolls back,
        and then a ``COMMIT`` loss is recorded through ``lose`` and the error re-raised. On ``AlreadyReported`` from
        the fence (an outcome already landed under this lease), the block does not run, ``completed`` is set, no loss
        is recorded and the error is re-raised. On exit the transaction
        commits, or rolls back when the block raises. ``control.fence`` also renews the lease in the database, so the
        renewal commits or rolls back with the block; once it commits, the noted time becomes the last committed
        renewal ``_heartbeats`` measures silence from. If ``complete`` reported on ``conn`` inside the block,
        ``completed`` is set and the report's duration recorded (``_record_duration``) only once the commit has
        succeeded; likewise, if ``complete`` met ``AlreadyReported`` on ``conn`` inside the block, ``completed`` is set
        only once the commit has succeeded (no duration is recorded for it). A block that returns with the transaction
        in a failed state (it caught a database error) is not committed: ``RuntimeError`` is raised and the
        transaction rolls back, so ``completed`` stays ``False`` and the silence clock does not move. Whether it
        commits or rolls back, the fenced connection is then cleared, so a connection back in the pool is never taken
        for this lease's fenced connection, and the paused heartbeats are woken so they check the silence limit at once.

        Preconditions:
            ``conn`` is open, in autocommit mode with no transaction open (a pool connection), and not used by another
            coroutine. Only one ``fenced`` block is open on the lease at a time.

        Args:
            conn: Connection the fence runs on and the block writes on.

        Yields:
            None, once, inside the fenced transaction.

        Raises:
            LeaseConflict: If the lease is no longer held, after a ``COMMIT`` loss has been recorded and ``lost`` set.
            AlreadyReported: If an outcome was already recorded under the lease, after ``completed`` has been set.
            DatabaseConflict: If the fence raises ``SerializationFailure`` or ``DeadlockDetected``.
            DatabaseUnavailable: If the connection is lost or the warehouse is unavailable during the fence.
            psycopg.Error: Any other database error from the fence, unchanged, or from the commit. Nothing is
                swallowed; ``completed`` stays ``False`` and no duration is recorded when the commit raises.
            RuntimeError: If the block returns with the transaction in a failed state (it swallowed a database
                error), after the transaction has rolled back; ``completed`` stays ``False`` and the fence time is not
                recorded as a renewal.
            Exception: Any exception raised in the block, unchanged, after the transaction has rolled back.
            Exception: Any exception from ``task_duration``, unchanged, raised after the commit and after
                ``completed`` is set.
            asyncio.CancelledError: If the task running the block is cancelled, after the transaction has rolled
                back; no loss is recorded.

        """
        checked = False
        fenced_at = asyncio.get_running_loop().time()  # no later than control.fence's now()
        try:
            async with conn.transaction():
                await control.fence(conn, self.task)
                checked = True
                self._fences += 1
                self._unfenced.clear()
                self._fenced = conn  # the transaction holds the row lock: heartbeats would queue behind it
                yield
                if conn.info.transaction_status == TransactionStatus.INERROR:  # COMMIT would silently roll back
                    raise RuntimeError("the fenced block swallowed a database error; its writes were rolled back")
            if self._fence_renewed is None or fenced_at > self._fence_renewed:  # the fence's renewal has committed
                self._fence_renewed = fenced_at
            if self._already_reported:  # the transaction complete() met AlreadyReported in has committed
                self._already_reported = False
                self.completed = True
            if self._completing is not None:  # the reported outcome has committed
                report, self._completing = self._completing, None
                self.completed = True
                self._record_duration(report)
        except LeaseConflict:
            if not checked:  # after the rollback, so the caller gets the error in the step the loss is marked
                self.lose(LeaseLoss.COMMIT)
            raise
        except AlreadyReported:
            if not checked:  # the outcome landed earlier: not a loss
                self.completed = True
            raise
        finally:
            self._fenced = None
            self._completing = None
            self._already_reported = False
            self._unfenced.set()  # the paused heartbeats check the silence limit at once

    async def complete(self, conn: Connection, report: TaskReport) -> str:
        """Report the task's outcome through ``control.report_outcome`` and mark the lease completed once it commits.

        On success, marks the lease completed, records the report's duration (``_record_duration``) and returns the
        result, both only for an outcome that has committed. On ``conn`` inside ``fenced``, ``fenced`` does both when
        its transaction commits, so a rollback leaves ``completed`` ``False`` and records nothing, and an attempt whose
        fenced commit fails is recorded once, by the failure reported after it; on any other connection the report has
        committed already (autocommit) and both happen here. On ``LeaseConflict``, records a ``COMMIT`` loss through
        ``lose`` and re-raises; nothing is recorded in ``task_duration``. This one place covers every outcome
        reported under a lease: failures from ``Worker._report_failure`` and successes from the job runner (stage 9);
        ``Worker._report_unleased`` reports without a lease and does not come here. On
        ``AlreadyReported`` (an outcome already landed under this lease), records no loss and nothing in
        ``task_duration``, and re-raises; on any connection but the fenced one it sets ``completed``, and on the fenced
        one ``fenced`` sets it only when its transaction commits, since the earlier outcome may be this same
        uncommitted transaction's, which a rollback undoes. While a report on any connection but the fenced one is in
        progress, a heartbeat that returns ``False`` waits for it to end before it judges the lease (see
        ``_heartbeats``).

        Nothing checks that a ``fenced`` block is open on another connection. If one is, ``control.report_outcome``
        waits on the task's row lock that block's transaction holds, with no timeout; called from inside that block,
        the call never returns, and the paused heartbeats never check the silence limit.

        TODO(review) 1: reject a call on any connection but the fenced one while a ``fenced`` block is open (raise
        ``RuntimeError`` when ``self._fenced is not None and conn is not self._fenced``); the precondition below admits
        ``async with lease.fenced(a): await lease.complete(b, report)``, which deadlocks on the row lock ``a`` holds.

        Preconditions:
            ``conn`` is open and not used by another coroutine. It is either the connection of the open ``fenced``
            block or in autocommit mode with no transaction open.
            ``task_duration`` is a ``Histogram``, as ``Instruments`` declares.

        Args:
            conn: Connection the outcome is reported on.
            report: The outcome to report.

        Returns:
            The text ``control.report_outcome`` returned.

        Raises:
            LeaseConflict: If the lease is no longer held, after a ``COMMIT`` loss has been recorded and ``lost`` set.
                ``completed`` stays ``False``.
            AlreadyReported: If an outcome was already recorded under the lease, after ``completed`` has been set
                (outside ``fenced``) or marked to be set when the ``fenced`` transaction commits.
            RuntimeError: If ``control.report_outcome`` returns no row.
            TypeError: If ``report.duration`` is not a ``timedelta``.
            DatabaseConflict: If the statement raises ``SerializationFailure`` or ``DeadlockDetected``.
            DatabaseUnavailable: If the connection is lost or the warehouse is unavailable.
            psycopg.Error: Any other database error, unchanged. Nothing is swallowed.
            Exception: Any exception from ``task_duration``, unchanged, raised after the lease is marked completed
                (outside ``fenced`` only).

        """
        outside = conn is not self._fenced
        if outside:  # a heartbeat that returns False meanwhile waits for completed to be settled
            self._unreported.clear()
        try:
            try:
                state = await control.report_outcome(conn, self.task, report)
            except LeaseConflict:
                self.lose(LeaseLoss.COMMIT)
                raise
            except AlreadyReported:
                if conn is self._fenced:
                    self._already_reported = True  # fenced sets completed once its transaction commits
                else:
                    self.completed = True  # the outcome landed earlier, under this lease
                raise
            if conn is self._fenced:
                self._completing = report  # fenced completes and records it once its transaction commits
            else:
                self.completed = True  # autocommit: the report has committed
                self._record_duration(report)
            return state
        finally:
            if outside:
                self._unreported.set()

    def _record_duration(self, report: TaskReport) -> None:
        """Record a committed report's duration in ``task_duration``.

        Records ``report.duration`` in seconds in ``task_duration`` (``pipeline.task.duration``) with ``job`` and
        ``outcome`` attributes: every reported outcome, success or failure, once it has committed.

        Preconditions:
            ``report`` has committed through ``control.report_outcome``.
            ``task_duration`` is a ``Histogram``, as ``Instruments`` declares.

        Args:
            report: The committed outcome.

        Returns:
            None.

        Raises:
            Exception: Any exception from ``task_duration``, unchanged.

        """
        self._instruments.task_duration.record(
            report.duration.total_seconds(), {"job": self.task.job, "outcome": report.outcome.value}
        )

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
            asyncio.CancelledError: If cancelled while taking the connection or storing the loss; it is not an
                ``Exception``, so it is not swallowed, and the loss may not be stored.

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

        A later call keeps the first ``kind`` and ``lost_at`` but sets ``lost`` again.

        Preconditions:
            Called with a running event loop, as every caller on this page and in the worker is a coroutine.

        Args:
            kind: Kind of lease loss.

        Returns:
            None.

        Raises:
            RuntimeError: If no event loop is running and no loss is recorded yet (from ``asyncio.get_running_loop``).

        """
        if self.loss is None:
            self.loss = kind
            self._lost_at = asyncio.get_running_loop().time()
        self.lost.set()

    def _fell_silent(self, last_success: float, *, may_try: bool = False) -> bool:
        """Report whether the silence limit has passed since the last success, recording a ``SILENCE`` loss if so.

        The loss is recorded through ``lose`` unless ``completed`` is set. With ``may_try``, a passed limit is not
        reported while the lease's known expiry, ``last_success`` plus the lease length, is more than one worst-case
        heartbeat round (the one ``__init__`` checks) away: one heartbeat can still be tried and end before it.

        Preconditions:
            Called with a running event loop. ``last_success`` is a time from that loop's ``time()``.

        Args:
            last_success: The event loop's ``time()`` of the last success: ``claim_started``, the time noted just before
                a successful heartbeat's call, or a committed ``fenced`` block's fence time, whichever is latest.
            may_try: Whether one heartbeat may still be tried past the limit when the known expiry allows it.

        Returns:
            ``True`` if at least ``silence_limit`` has passed since ``last_success`` and, with ``may_try``, the known
            expiry is no more than one round away; the heartbeats then stop.

        Raises:
            None.

        """
        now = asyncio.get_running_loop().time()
        if now - last_success < self._silence_limit.total_seconds():
            return False
        if may_try and now + self._round.total_seconds() < last_success + self._lease_length.total_seconds():
            return False  # one heartbeat ends before the expiry the last renewal gave; the database decides
        if not self.completed:
            self.lose(LeaseLoss.SILENCE)
        return True

    async def _heartbeats(self) -> None:
        """Renew the lease every heartbeat interval until it is lost, the task completes, or this is cancelled.

        Each round sleeps for one interval, except right after a failed heartbeat that paused (below). While a
        ``fenced`` block is open, the heartbeats count as paused and wait for it to close (``fenced`` wakes them), not
        for another sleep; the first round after a pause moves the last success to the fence time of the latest
        committed ``fenced`` block when that is later, and leaves it otherwise (a rolled-back block renewed nothing).
        The round then checks the silence limit (``_fell_silent``, which past the limit still lets one heartbeat be
        tried per last success while the lease's known expiry is more than one round away), then takes a pool
        connection and calls ``control.heartbeat``, with a timeout of one interval on the call itself. Taking the
        connection, the wait for a free connection and the pool's connection check (``check_connection``, an empty
        query, which the pool does not time) together, has a timeout of ``pool.timeout``; a checkout cancelled at that
        timeout ends within ``_CANCEL_MARGIN``, and the pool puts the connection back. A checkout that times out is a
        failed heartbeat.
        - A heartbeat that raises is logged at ``WARNING`` as ``timed out`` or ``failed``. If a fence succeeded during
          the attempt or a ``fenced`` block is now open, the heartbeats pause, and the next round starts without a
          sleep, so the silence limit is checked as soon as no block is open. Otherwise the silence limit is checked
          again, with no further heartbeat allowed past it, so a silent lease is given up at the end of the failed
          round rather than one sleep later.
        - A ``False`` result while a fence succeeded during the attempt, a ``fenced`` block is open, or a ``complete``
          outside ``fenced`` is in progress may come from a call that waited on that work's row lock while its
          outcome committed: the round waits for such a ``complete`` to end, then returns if ``completed`` is set, and
          otherwise the heartbeats pause as for a raised heartbeat and the lease is checked again.
        - Any other ``False`` result means the lease is gone: a ``HEARTBEAT`` loss is recorded unless ``completed`` is
          set, and the loop returns.
        - A ``True`` result records the time from the last success to the reply in ``heartbeat_lag`` and sets the
          last success and ``last_heartbeat`` to the time noted just before the call (after the connection was
          taken), which is no later than the renewal the database made. The last success is not always a heartbeat:
          the first sample is measured from ``claim_started`` and a sample after a committed ``fenced`` block from
          its fence time, and the end is the reply rather than the call.

        The last success starts as ``claim_started``, noted before the claim, so no later than the claim's renewal.
        Every time here is the event loop's monotonic ``time()``, so a step of the system clock neither fakes nor
        hides a silence. The limit is checked before each attempt and after each failed one, so a ``SILENCE`` loss
        comes at most one round after the limit, counted from the last success: the round ``__init__`` keeps, with the
        limit, below the lease length. Past the limit, one heartbeat per last success is still tried when the lease's
        known expiry (the last success plus the lease length) is more than one round away, so it ends before the
        expiry and the database decides; if it fails, the check after it records the ``SILENCE`` loss. After a pause
        the last success is the latest committed fence's time when that is later, which is noted before the fence and
        so is never later than its renewal. While a ``fenced`` block stays open no round checks the limit; the check
        comes as soon as the block closes, measured from the last committed renewal: a committed block open for
        ``silence_limit`` or longer ends in a ``SILENCE`` loss as it closes only when the expiry its fence gave is no
        more than one round away, and otherwise a heartbeat is tried first.

        Preconditions:
            Runs as the task ``hold`` creates.

        Args:
            None.

        Returns:
            None, after ``_fell_silent`` finds the silence limit passed or the heartbeat reports the lease gone; a
            loss is recorded then unless ``completed`` is set.

        Raises:
            asyncio.CancelledError: When the task running it is cancelled, as ``hold`` does on exit.
            Exception: Any exception from ``heartbeat_lag``, unchanged. An ``Exception`` from taking a connection or
                from a heartbeat, ``TimeoutError`` from either timeout included, is logged and handled as above, not
                raised.

        """
        loop = asyncio.get_running_loop()
        last_success = self._claim_started  # no later than the claim's renewal
        tried_from: float | None = None  # the last success a heartbeat past the silence limit was tried from
        paused = False
        while True:
            if not paused:  # a paused round checks as soon as no block is open, not one sleep later
                await asyncio.sleep(self._interval.total_seconds())
            while self._fenced is not None:  # the block holds the row lock; fenced() sets _unfenced as it closes
                paused = True
                await self._unfenced.wait()
            if paused:
                paused = False
                renewed = self._fence_renewed
                if renewed is not None and renewed > last_success:  # a committed fence renewed the lease
                    last_success = renewed
            if self._fell_silent(last_success, may_try=tried_from != last_success):
                return
            if loop.time() - last_success >= self._silence_limit.total_seconds():
                tried_from = last_success  # the one heartbeat the known expiry leaves time for
            fences = self._fences
            checkout: asyncio.Timeout | None = None  # the pool's connection check has no timeout of its own
            deadline: asyncio.Timeout | None = None  # control.heartbeat has no timeout of its own
            try:
                async with AsyncExitStack() as stack:
                    async with (checkout := asyncio.timeout(self._pool.timeout)):
                        conn = await stack.enter_async_context(self._pool.connection())
                    called = loop.time()  # no later than control.heartbeat's renewal
                    async with (deadline := asyncio.timeout(self._interval.total_seconds())):
                        held = await control.heartbeat(conn, self.task)
            except Exception:
                expired = any(timeout is not None and timeout.expired() for timeout in (checkout, deadline))
                cause = "timed out" if expired else "failed"
                _log.warning("heartbeat on task %s %s", self.task.id, cause, exc_info=True)
                if self._fences != fences or self._fenced is not None:
                    paused = True  # it waited on the fenced transaction's row lock; the lease cannot be lost meanwhile
                    continue
                if self._fell_silent(last_success):
                    return
                continue
            if not held:
                if self._fences != fences or self._fenced is not None or not self._unreported.is_set():
                    await self._unreported.wait()  # a complete() outside fenced sets completed before it ends
                    if not self.completed:
                        paused = True  # it may have waited on a committed outcome's row lock: check again
                        continue
                if not self.completed:
                    self.lose(LeaseLoss.HEARTBEAT)
                return
            self._instruments.heartbeat_lag.record(loop.time() - last_success)
            last_success = self._last_heartbeat = called
