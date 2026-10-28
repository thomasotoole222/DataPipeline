"""Calls to the warehouse's ``control`` schema: claim, heartbeat, fence and report tasks, and listen for new ones.

Public API:
    TASK_CHANNEL: The ``NOTIFY`` channel signalled when a task becomes runnable.
    ClaimedTask: One claimed task, as returned by ``control.claim``.
    TaskReport: The outcome of one task run.
    LeaseLoss: The kinds of lease loss.
    claim: Claim a task through ``control.claim``.
    heartbeat: Renew a task's lease through ``control.heartbeat``.
    fence: Check a task's lease through ``control.fence``.
    Fence: The way a stage fences a transaction on a claimed task's lease.
    report_outcome: Report a task's outcome through ``control.report_outcome``.
    record_lease_loss: Record a lost lease through ``control.record_lease_loss``.
    open_listener: Open a connection that listens on ``TASK_CHANNEL``.
    wait_for_task: Wait for a notification on a listening connection.

"""

from collections.abc import Iterator
from contextlib import AbstractAsyncContextManager, contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import StrEnum
from typing import Any, Final, Protocol
from uuid import UUID

import psycopg
from psycopg import sql
from psycopg.conninfo import make_conninfo
from psycopg.rows import class_row

from pipeline.config import Settings
from pipeline.db.pool import DEAD_CONNECTION_PARAMS, Connection, storable_text
from pipeline.errors import MAX_RETRY_AFTER, AlreadyReported, LeaseConflict, Outcome, db_errors
from pipeline.timeutil import ensure_utc

#: Name of the ``NOTIFY`` channel the orchestrator signals on when a new task becomes runnable.
TASK_CHANNEL: Final = "control_task"  # NOTIFY channel the Orchestrator signals on a new runnable task


@dataclass(frozen=True, slots=True, kw_only=True)
class ClaimedTask:
    """Hold one claimed task, with its timestamp fields normalised by ``ensure_utc``.

    Instances are immutable, hashable and keyword-only. After construction, ``created_at``, ``not_before``,
    ``claimed_at`` and ``lease_expires_at`` hold the values ``ensure_utc`` returned for them. Every field,
    ``lease_token`` included, appears in ``repr``. Built only from ``control.claim`` rows. The column types and
    constraints of ``control.task`` are its guarantee, so no field is checked beyond ``ensure_utc``, which rejects a
    naive datetime with ``ValueError``, and the invariant that ``lease_expires_at`` is after ``claimed_at``, which the
    worker's lease timing relies on. ``dataclasses.replace`` runs ``__post_init__`` again on the copy, which relies on
    ``ensure_utc`` returning a datetime already in ``UTC`` as is.

    Preconditions:
        ``created_at``, ``not_before``, ``claimed_at`` and ``lease_expires_at`` are aware ``datetime`` values.

    Args:
        id: Task identifier.
        run_id: Identifier of the run the task belongs to.
        connector: The connector the task fetches from, for example ``"tiktok_shop"``.
        job: Job name.
        source_id: Identifier of the source, the account the task is for.
        partition_key: Partition key for the task.
        attempt: Attempt number.
        lease_token: Token for the lease held on the task.
        created_at: Creation time.
        not_before: Earliest time the task may run.
        claimed_at: Time the task was claimed.
        lease_expires_at: Time the lease the claim granted expires; ``lease_expires_at - claimed_at`` is the lease
            length, which every heartbeat and every fence grants again.
        trace_context: Serialised trace context, or ``None``.

    Returns:
        None.

    Raises:
        TypeError: If an argument is passed positionally, a required argument is missing or an unknown keyword is
            given, or if a timestamp is not a ``datetime`` (raised by ``ensure_utc``).
        ValueError: If a timestamp is naive (raised by ``ensure_utc``), or ``lease_expires_at`` is not after
            ``claimed_at``.
        OverflowError: If converting a timestamp to ``UTC`` falls outside the ``datetime`` range (raised by
            ``ensure_utc``).

    """

    id: int
    run_id: int
    connector: str
    job: str
    source_id: UUID
    partition_key: str
    attempt: int  # 1 on the first claim
    lease_token: UUID  # fencing token; new on every claim
    created_at: datetime  # UTC
    not_before: datetime  # UTC
    claimed_at: datetime  # UTC; the task's started_at
    lease_expires_at: datetime  # UTC; after claimed_at
    trace_context: str | None  # W3C traceparent the Orchestrator stored

    def __post_init__(self) -> None:
        """Apply ``ensure_utc`` to each timestamp field, then check that the lease expires after the claim.

        Preconditions:
            Called by the generated ``__init__`` after every field is set.

        Args:
            None.

        Returns:
            None.

        Raises:
            TypeError: If one of the four timestamp fields is not a ``datetime`` (raised by ``ensure_utc``).
            ValueError: If one of the four timestamp fields is naive (raised by ``ensure_utc``), or
                ``lease_expires_at`` is not after ``claimed_at``.
            OverflowError: If converting a field to ``UTC`` falls outside the ``datetime`` range (raised by
                ``ensure_utc``).

        """
        for name in ("created_at", "not_before", "claimed_at", "lease_expires_at"):
            object.__setattr__(self, name, ensure_utc(getattr(self, name)))
        if self.lease_expires_at <= self.claimed_at:
            raise ValueError(
                f"task {self.id}: lease_expires_at {self.lease_expires_at} is not after claimed_at {self.claimed_at}"
            )


#: Maximum number of characters of ``error_message`` that ``TaskReport`` keeps.
_MESSAGE_LIMIT: Final = 2000  # error_message is cut to this many characters
#: Largest row count ``TaskReport`` accepts: ``control.report_outcome`` takes the counts as ``bigint``.
_MAX_ROWS: Final = 2**63 - 1


@dataclass(frozen=True, slots=True, kw_only=True)
class TaskReport:
    """Record the outcome, row counts and duration of one task run.

    Fields are validated in ``__post_init__``, and the first failed check raises. An ``error_message`` longer than
    ``_MESSAGE_LIMIT`` characters is stored as its first ``_MESSAGE_LIMIT`` characters, with no truncation marker. Row
    counts are not checked against each other, because one fetched record can produce several rows. Every character
    of ``error_class`` and ``error_message`` that Postgres text cannot store (NUL and lone surrogates; see
    ``storable_text``) is stored as U+FFFD, because psycopg cannot send it and ``report_outcome`` would then fail to
    record the outcome. ``control.report_outcome`` takes the row counts as ``bigint``, so each must be at most
    ``_MAX_ROWS`` (``2**63 - 1``). ``retry_after`` must be at most ``MAX_RETRY_AFTER`` (six hours), the longest
    retry delay the pipeline stores; ``retry_after_from`` clamps a ``Retry-After`` header to it.

    Preconditions:
        ``outcome`` is an ``Outcome`` member, and the other fields have the types they are annotated with.

    Args:
        outcome: The result of the run.
        error_class: Error class of an unsuccessful run. ``None`` exactly when ``outcome`` is ``Outcome.SUCCEEDED``,
            and never ``""``. Characters ``storable_text`` replaces become U+FFFD.
        error_message: Error text, or ``None``. Must be ``None`` when ``outcome`` is ``Outcome.SUCCEEDED``.
            Characters ``storable_text`` replaces become U+FFFD, then the text is truncated to ``_MESSAGE_LIMIT``
            characters.
        retry_after: Delay before a retry, or ``None``. Must be ``None`` when ``outcome`` is ``Outcome.SUCCEEDED``,
            and must be from ``timedelta(0)`` to ``MAX_RETRY_AFTER``.
        rows_fetched: Number of rows fetched, >= 0 and at most ``_MAX_ROWS``.
        rows_written: Number of rows written, >= 0 and at most ``_MAX_ROWS``.
        rows_rejected: Number of rows rejected, >= 0 and at most ``_MAX_ROWS``.
        duration: Duration of the run, >= ``timedelta(0)``.

    Returns:
        None. The generated ``__init__`` initialises a frozen instance in place.

    Raises:
        TypeError: If an argument is missing, passed positionally or unknown (raised by the generated ``__init__``).
        ValueError: If ``error_class`` is ``None`` while ``outcome`` is not ``Outcome.SUCCEEDED``, or is not ``None``
            while it is.
        ValueError: If ``error_class`` is ``""``.
        ValueError: If ``outcome`` is ``Outcome.SUCCEEDED`` and ``error_message`` or ``retry_after`` is not ``None``.
        ValueError: If ``rows_fetched``, ``rows_written`` or ``rows_rejected`` is negative or above ``_MAX_ROWS``.
        ValueError: If ``duration`` is negative.
        ValueError: If ``retry_after`` is negative or above ``MAX_RETRY_AFTER``.

    """

    outcome: Outcome
    error_class: str | None  # None exactly when outcome is SUCCEEDED; never ""
    error_message: str | None  # None when SUCCEEDED; cut to _MESSAGE_LIMIT characters
    retry_after: timedelta | None  # the delay a connector asked for, if any; 0 to MAX_RETRY_AFTER; None when SUCCEEDED
    rows_fetched: int  # each count >= 0; not checked against each other
    rows_written: int
    rows_rejected: int
    duration: timedelta  # >= 0

    def __post_init__(self) -> None:
        """Validate the fields, make ``error_class`` and ``error_message`` storable, and truncate ``error_message``.

        Both texts go through ``storable_text`` (NUL and lone surrogates become U+FFFD), and ``error_message`` is
        then cut to ``_MESSAGE_LIMIT`` characters.

        Preconditions:
            Called once, by the generated ``__init__``.

        Args:
            None.

        Returns:
            None.

        Raises:
            ValueError: Under the conditions listed in the ``TaskReport`` docstring.

        """
        succeeded = self.outcome == Outcome.SUCCEEDED
        if succeeded != (self.error_class is None):
            raise ValueError(f"error_class must be None exactly when the outcome is succeeded, got {self.outcome}")
        if self.error_class == "":
            raise ValueError("error_class must not be empty")
        if succeeded and self.error_message is not None:
            raise ValueError("error_message must be None when the outcome is succeeded")
        if succeeded and self.retry_after is not None:
            raise ValueError("retry_after must be None when the outcome is succeeded")
        if min(self.rows_fetched, self.rows_written, self.rows_rejected) < 0:
            raise ValueError("row counts must be 0 or more")
        if max(self.rows_fetched, self.rows_written, self.rows_rejected) > _MAX_ROWS:
            raise ValueError(f"row counts must be at most {_MAX_ROWS}")
        if self.duration < timedelta(0):
            raise ValueError(f"duration must be 0 or more, got {self.duration}")
        if self.retry_after is not None and self.retry_after < timedelta(0):
            raise ValueError(f"retry_after must be 0 or more, got {self.retry_after}")
        if self.retry_after is not None and self.retry_after > MAX_RETRY_AFTER:
            raise ValueError(f"retry_after must be at most {MAX_RETRY_AFTER}, got {self.retry_after}")
        if self.error_class is not None:
            object.__setattr__(self, "error_class", storable_text(self.error_class))
        if self.error_message is not None:
            object.__setattr__(self, "error_message", storable_text(self.error_message)[:_MESSAGE_LIMIT])


class LeaseLoss(StrEnum):
    """Enumerate the kinds of lease loss as string members.

    Each member is a ``str`` equal to its value, so it compares equal to, hashes as and formats as that plain string.
    Members compare with ``<`` by string value, so ``sorted()`` on members gives alphabetical order, while iterating
    the class gives definition order. The enum has no ``__init__`` of its own, so the contract of the lookup
    ``LeaseLoss(value)`` is given here.

    Attributes:
        HEARTBEAT: ``"heartbeat"``.
        SILENCE: ``"silence"``.
        COMMIT: ``"commit"``.

    Preconditions:
        None.

    Args:
        value: The string to look up when calling ``LeaseLoss(value)``. Matching is exact and case-sensitive.

    Returns:
        The member whose value equals ``value``.

    Raises:
        ValueError: If ``value`` is not one of ``"heartbeat"``, ``"silence"`` or ``"commit"``.

    """

    HEARTBEAT = "heartbeat"
    SILENCE = "silence"
    COMMIT = "commit"


#: SQL that calls ``control.claim`` with the worker ID as its only parameter and selects every result column.
#: If the columns stop matching ``ClaimedTask``, building it raises ``TypeError``.
_CLAIM: Final = "SELECT * FROM control.claim(%s)"


async def claim(conn: Connection, worker_id: str) -> ClaimedTask | None:
    """Claim a task for a worker by calling ``control.claim``.

    Runs ``_CLAIM`` on a new cursor of ``conn`` with ``worker_id`` as the only parameter and maps the first row of the
    result to a ``ClaimedTask``. The cursor is closed before the function returns. Exceptions raised inside the
    cursor block pass through ``db_errors``. Only the first row is read; ``None`` is returned when the result has no
    row. If the call is cancelled after ``execute``, the claimed task is not returned: its lease is left to expire and
    be re-queued by the sweeper. The same happens when building ``ClaimedTask`` from the row raises, because the claim
    has already taken effect on an autocommit connection.

    Preconditions:
        ``conn`` is open, not in a failed transaction, and not used by another coroutine during the call.
        If ``conn`` is not in autocommit mode (pool connections are), the caller commits or rolls back the
        transaction this call runs in, and the claim takes effect only when it commits.
        ``worker_id`` is a value ``control.claim`` accepts as a worker ID. The function does not check it.

    Args:
        conn: The connection to run the query on.
        worker_id: The ID of the worker claiming the task.

    Returns:
        The claimed task as a ``ClaimedTask``, or ``None`` if ``control.claim`` returned no row.

    Raises:
        DatabaseConflict: If the query raises ``SerializationFailure`` or ``DeadlockDetected`` (mapped by
            ``db_errors``).
        DatabaseUnavailable: If the connection is lost or the warehouse is unavailable (mapped by ``db_errors``).
        TypeError: If the row's columns do not match the fields of ``ClaimedTask``, or a timestamp column is not a
            ``datetime``.
        ValueError: If a timestamp column is naive (raised by ``ensure_utc``), or ``lease_expires_at`` is not after
            ``claimed_at``.
        psycopg.Error: Any other database error, unchanged. Nothing is swallowed.

    """
    with db_errors():
        async with conn.cursor(row_factory=class_row(ClaimedTask)) as cursor:
            await cursor.execute(_CLAIM, (worker_id,))
            return await cursor.fetchone()


#: SQL that calls ``control.heartbeat`` with a task id and a lease token, bound as positional parameters.
_HEARTBEAT: Final = "SELECT control.heartbeat(%s, %s)"


async def heartbeat(conn: Connection, task: ClaimedTask) -> bool:
    """Renew the lease on a claimed task.

    Executes ``control.heartbeat`` with the task's id and lease token inside ``db_errors`` and reads the first row of
    the result. Pool connections run in autocommit, so the renewal commits when the statement ends.
    ``control.heartbeat`` returns ``boolean``. The call has no timeout of its own: on a pool connection the server's
    ``statement_timeout`` bounds the query, and a caller that needs a shorter bound cancels the call. A cancelled call
    does not end at once: psycopg 3.3's ``AsyncConnection.wait`` first sends the server a cancel request (up to 5 s)
    and then waits for the query to end (up to 5 s more) before the cancellation propagates, so a cancelling caller
    must allow up to 10 s after it cancels. The cursor is client-side, so leaving it unclosed holds no server
    resource.

    Preconditions:
        ``conn`` is open and is not used by another coroutine during the call.
        ``conn`` is in autocommit mode, as pool connections are, or the caller commits the renewal.

    Args:
        conn: Connection on which the heartbeat query runs.
        task: Claimed task whose lease is renewed.

    Returns:
        ``True`` if ``control.heartbeat`` returned true. ``False`` if it returned false or ``NULL``, or no row.

    Raises:
        DatabaseConflict: If the query raises ``SerializationFailure`` or ``DeadlockDetected`` (mapped by
            ``db_errors``).
        DatabaseUnavailable: If the connection is lost or the warehouse is unavailable (mapped by ``db_errors``).
        psycopg.Error: Any other database error, unchanged.
        asyncio.CancelledError: If the call is cancelled. ``db_errors`` is a synchronous context manager that
            swallows nothing and lets it through.

    """
    with db_errors():
        cursor = await conn.execute(_HEARTBEAT, (task.id, task.lease_token))
        row = await cursor.fetchone()
    return row is not None and row[0] is True


#: SQLSTATE that ``_fenced`` translates into ``LeaseConflict``.
_LEASE_CONFLICT_SQLSTATE: Final = "CH001"  # raised by control.fence and control.report_outcome on a stale token
#: SQLSTATE that ``_fenced`` translates into ``AlreadyReported``.
_ALREADY_REPORTED_SQLSTATE: Final = "CH002"  # raised by them when an outcome was already recorded under the token


@contextmanager
def _fenced(task: ClaimedTask) -> Iterator[None]:
    """Translate a lease-conflict database error raised in the block into ``LeaseConflict`` or ``AlreadyReported``.

    Runs the ``with`` block inside ``db_errors()``. A ``psycopg.Error`` that propagates out of ``db_errors()``
    with ``sqlstate`` equal to ``_LEASE_CONFLICT_SQLSTATE`` is replaced by ``LeaseConflict``, and one with
    ``_ALREADY_REPORTED_SQLSTATE`` by ``AlreadyReported``, chained to the original error. Every other exception
    propagates unchanged. ``db_errors`` does not map SQLSTATEs ``CH001`` and ``CH002``, so those errors reach the
    check unchanged.

    Preconditions:
        Used as a context manager: ``with _fenced(task):``.

    Args:
        task: The claimed task whose ``id`` and ``lease_token`` appear in the ``LeaseConflict`` message.

    Yields:
        None.

    Raises:
        LeaseConflict: If the block raises a ``psycopg.Error`` with ``sqlstate`` equal to ``"CH001"``. The original
            error is set as ``__cause__``.
        AlreadyReported: If the block raises a ``psycopg.Error`` with ``sqlstate`` equal to ``"CH002"``. The original
            error is set as ``__cause__``.
        DatabaseConflict: If the block raises ``SerializationFailure`` or ``DeadlockDetected`` (mapped by
            ``db_errors``). It is not a ``psycopg.Error``, so it passes the lease check unchanged.
        DatabaseUnavailable: If the connection is lost or the warehouse is unavailable (mapped by ``db_errors``). It
            is not a ``psycopg.Error``, so it passes the lease check unchanged.
        psycopg.Error: Any other ``psycopg.Error`` from the block, re-raised unchanged.
        Exception: Any other exception from the block, unchanged. Nothing is swallowed.

    """
    try:
        with db_errors():
            yield
    except psycopg.Error as err:
        # db_errors lets unknown SQLSTATEs through unchanged, so the fencing error arrives here as it was raised.
        if err.sqlstate == _ALREADY_REPORTED_SQLSTATE:
            raise AlreadyReported(
                f"task {task.id}: an outcome was already reported under lease {task.lease_token}"
            ) from err
        if err.sqlstate != _LEASE_CONFLICT_SQLSTATE:
            raise
        raise LeaseConflict(f"task {task.id}: lease {task.lease_token} is no longer held") from err


#: SQL that calls ``control.fence`` with a task id and a lease token as positional ``%s`` parameters.
_FENCE: Final = "SELECT control.fence(%s, %s)"


async def fence(conn: Connection, task: ClaimedTask) -> None:
    """Run ``control.fence`` for ``task`` on ``conn`` inside ``_fenced(task)``.

    Executes ``_FENCE`` with ``task.id`` and ``task.lease_token`` as parameters. Any result the statement produces is
    discarded. ``control.fence`` returns ``void`` and raises SQLSTATE ``CH001`` when the lease is no longer held,
    which ``_fenced`` turns into ``LeaseConflict``, or ``CH002`` when it is not held because an outcome was already
    recorded under it, which ``_fenced`` turns into ``AlreadyReported``. When the lease is held, ``control.fence``
    locks the task's row and grants the lease length again, as a heartbeat does, in one statement. The renewal commits
    or rolls back with the caller's transaction; ``Lease`` measures silence from the last committed renewal, so a
    committed fence counts from its time and a rolled-back one not at all. ``_fenced`` is a synchronous context
    manager that does no blocking work.

    The check covers only the transaction it runs in: on an autocommit connection the fence commits on its own, so
    the lease can be lost before the next statement. Its one caller in the pipeline, the worker's ``Lease.fenced``,
    runs it inside ``conn.transaction()`` ahead of the writes it guards; stages fence through ``Fence``.

    Preconditions:
        ``conn`` is an open connection that is free to execute a statement. ``task.id`` and ``task.lease_token``
        are the values ``control.fence`` expects for this task. To guard later writes, ``conn`` has a transaction
        open and those writes run in the same transaction.

    Args:
        conn: Connection on which the fence statement is executed.
        task: Claimed task whose id and lease token are passed to ``control.fence``.

    Returns:
        None.

    Raises:
        LeaseConflict: If the lease is no longer held (SQLSTATE ``CH001``).
        AlreadyReported: If the lease is no longer held because an outcome was already recorded under it (SQLSTATE
            ``CH002``).
        DatabaseConflict: If the statement raises ``SerializationFailure`` or ``DeadlockDetected`` (mapped by
            ``db_errors``).
        DatabaseUnavailable: If the connection is lost or the warehouse is unavailable (mapped by ``db_errors``).
        psycopg.Error: Any other database error from ``conn.execute``, unchanged. Nothing is swallowed.

    """
    with _fenced(task):
        await conn.execute(_FENCE, (task.id, task.lease_token))


class Fence(Protocol):
    """Fence a transaction on a claimed task's lease: the one way a stage guards its writes.

    A stage wraps each transaction that writes for the task in ``async with fence.fenced(conn):`` and does not call
    ``fence`` itself. The worker's ``Lease`` implements this protocol structurally, so ``pipeline.db`` and
    ``pipeline.stages`` need no import of ``pipeline.worker``.

    """

    def fenced(self, conn: Connection) -> AbstractAsyncContextManager[None]:
        """Return a context manager that runs its block in one transaction on ``conn``, fenced on the lease.

        On entry the context manager opens a transaction on ``conn`` and runs ``fence`` in it; the block then runs
        inside that transaction, which commits on a normal exit and rolls back when the block raises.

        Preconditions:
            ``conn`` is open, in autocommit mode with no transaction open, and not used by another coroutine.

        Args:
            conn: Connection the fence runs on and the block writes on.

        Returns:
            An async context manager yielding ``None``.

        Raises:
            LeaseConflict: On entering the context manager, if the lease is no longer held; the block does not run.
            AlreadyReported: On entering, if an outcome was already recorded under the lease; the block does not
                run.
            DatabaseConflict: On entering, if the fence raises ``SerializationFailure`` or ``DeadlockDetected``.
            DatabaseUnavailable: On entering, if the connection is lost or the warehouse is unavailable.
            psycopg.Error: Any other database error from the fence or the commit, unchanged.
            Exception: Any exception from the block, unchanged, after the transaction has rolled back.

        """
        ...


#: SQL that calls ``control.report_outcome`` with ten positional ``%s`` parameters.
_REPORT: Final = "SELECT control.report_outcome(%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"


async def report_outcome(conn: Connection, task: ClaimedTask, report: TaskReport) -> str:
    """Report a claimed task's outcome through ``control.report_outcome`` and return the result as text.

    Builds ten positional parameters from ``task`` and ``report``, executes ``_REPORT`` on ``conn`` inside
    ``_fenced(task)``, fetches one row and returns its first column converted with ``str``. A ``NULL`` first column is
    not rejected. ``report.duration`` is sent as whole milliseconds, rounded towards negative infinity. The function
    does not commit. ``TaskReport`` keeps ``retry_after`` within ``MAX_RETRY_AFTER``, and ``control.report_outcome``
    waits at most 6 hours before a retry, so the retry time it stores stays within ``timestamptz``'s range.

    Preconditions:
        ``conn`` is open, usable from the running event loop, and not used by another coroutine during the call. If
        ``conn`` is not in autocommit mode, the caller commits or rolls back afterwards. The row counts in ``report``
        fit in a PostgreSQL ``bigint``, which ``TaskReport`` guarantees by rejecting counts above ``_MAX_ROWS``.

    Args:
        conn: The connection the query is executed on.
        task: The claimed task. ``task.id`` and ``task.lease_token`` are the first two parameters.
        report: The outcome to record. ``outcome.value``, ``error_class``, ``error_message``, ``retry_after``,
            ``rows_fetched``, ``rows_written``, ``rows_rejected`` and ``duration`` (in whole milliseconds) are the
            remaining parameters, in that order. ``retry_after`` is sent as a ``timedelta``, which psycopg adapts to
            the ``interval`` parameter of ``control.report_outcome``.

    Returns:
        ``str(row[0])``, where ``row`` is the first row returned by ``control.report_outcome``. The string ``"None"``
        if that column is ``NULL``. Never ``None``.

    Raises:
        RuntimeError: If ``control.report_outcome`` returns no row.
        TypeError: If ``report.duration`` is not a ``timedelta``.
        LeaseConflict: If the lease is no longer held (SQLSTATE ``CH001``, translated by ``_fenced``).
        AlreadyReported: If an outcome was already recorded under the lease (SQLSTATE ``CH002``, translated by
            ``_fenced``); nothing is recorded again.
        DatabaseConflict: If the statement raises ``SerializationFailure`` or ``DeadlockDetected`` (mapped by
            ``db_errors``).
        DatabaseUnavailable: If the connection is lost or the warehouse is unavailable (mapped by ``db_errors``).
        psycopg.Error: Any other database error from ``conn.execute`` or ``cursor.fetchone``, unchanged. Nothing is
            swallowed.

    """
    params = (
        task.id,
        task.lease_token,
        report.outcome.value,
        report.error_class,
        report.error_message,
        report.retry_after,
        report.rows_fetched,
        report.rows_written,
        report.rows_rejected,
        report.duration // timedelta(milliseconds=1),
    )
    with _fenced(task):
        cursor = await conn.execute(_REPORT, params)
        row = await cursor.fetchone()
    if row is None:
        raise RuntimeError("control.report_outcome returned no row")
    return str(row[0])


#: SQL that records a lease loss through ``control.record_lease_loss``, taking five positional parameters.
_LEASE_LOSS: Final = "SELECT control.record_lease_loss(%s, %s, %s, %s, %s)"


async def record_lease_loss(
    conn: Connection, task: ClaimedTask, kind: LeaseLoss, worker_id: str, unwind: timedelta | None
) -> None:
    """Record that ``worker_id`` lost its lease on ``task``.

    Calls ``control.record_lease_loss`` on ``conn`` with ``task.id``, ``task.lease_token``, ``kind.value``,
    ``worker_id`` and ``unwind`` as whole milliseconds (floored), or SQL ``NULL`` when ``unwind`` is ``None``.
    Any result of the SQL function is discarded.

    Preconditions:
        ``conn`` is open and not in use by another concurrent operation. ``unwind``, when given, is non-negative; it
        is not checked, and a negative value is sent as a negative number of milliseconds.

    Args:
        conn: Connection the statement is executed on.
        task: Claimed task whose ``id`` and ``lease_token`` are recorded.
        kind: Kind of lease loss; its ``value`` is recorded.
        worker_id: Identifier of the worker that lost the lease.
        unwind: Duration recorded as the unwind time, or ``None`` to record ``NULL``.

    Returns:
        None.

    Raises:
        TypeError: If ``unwind`` is neither ``None`` nor a ``timedelta``.
        DatabaseConflict: If the statement raises ``SerializationFailure`` or ``DeadlockDetected`` (mapped by
            ``db_errors``).
        DatabaseUnavailable: If the connection is lost or the warehouse is unavailable (mapped by ``db_errors``).
        psycopg.Error: Any other database error from ``conn.execute``, unchanged. ``db_errors`` swallows nothing.

    """
    with db_errors():
        unwind_ms = None if unwind is None else unwind // timedelta(milliseconds=1)
        await conn.execute(_LEASE_LOSS, (task.id, task.lease_token, kind.value, worker_id, unwind_ms))


async def open_listener(settings: Settings) -> psycopg.AsyncConnection[Any]:
    """Open an autocommit connection to the warehouse and subscribe it to ``TASK_CHANNEL``.

    Connects to ``settings.warehouse_url`` with ``DEAD_CONNECTION_PARAMS`` merged in by ``make_conninfo``, and with
    ``application_name`` set to ``"<worker_id>-listen"``, then issues ``LISTEN`` on ``TASK_CHANNEL``, quoted as an
    identifier. Database errors are passed through ``db_errors``. If ``LISTEN`` raises anything, including
    ``asyncio.CancelledError``, the connection is closed and the error re-raised; if ``close`` itself raises, that
    exception propagates instead, mapped by ``db_errors`` like any other, with the ``LISTEN`` error as its
    ``__context__``. ``db_errors`` never suppresses an exception, so a failed ``connect`` or ``LISTEN`` always
    propagates. ``TASK_CHANNEL`` is lowercase, so the quoted name matches the ``pg_notify('control_task', ...)``
    sender. PostgreSQL truncates ``application_name`` to 63 bytes; ``Settings`` caps ``worker_id`` at 56 bytes in
    UTF-8, so ``"<worker_id>-listen"`` always fits.

    Preconditions:
        Must be awaited inside a running event loop. ``settings.warehouse_url`` must hold a valid PostgreSQL
        connection string. The caller takes ownership of the returned connection and must close it.

    Args:
        settings: Settings providing ``warehouse_url`` and ``worker_id``.

    Returns:
        An open ``psycopg.AsyncConnection`` in autocommit mode on which ``LISTEN`` for ``TASK_CHANNEL`` has
        succeeded.

    Raises:
        DatabaseUnavailable: If the warehouse cannot be reached or the connection is lost (mapped by ``db_errors``).
        DatabaseConflict: If ``LISTEN`` raises ``SerializationFailure`` or ``DeadlockDetected`` (mapped by
            ``db_errors``).
        psycopg.Error: Any other database error from ``connect``, ``LISTEN`` or ``close``, unchanged, or
            ``psycopg.ProgrammingError`` from ``make_conninfo`` if ``warehouse_url`` is not a valid connection string.
        asyncio.CancelledError: If the call is cancelled. During ``LISTEN``, the connection is closed first.
        Exception: Any exception from ``close`` after a failed ``LISTEN``, in place of the original error. Nothing
            is swallowed.

    """
    with db_errors():
        listener = await psycopg.AsyncConnection.connect(
            make_conninfo(settings.warehouse_url.get_secret_value(), **DEAD_CONNECTION_PARAMS),
            autocommit=True,
            application_name=f"{settings.worker_id}-listen",
        )
        try:
            await listener.execute(sql.SQL("LISTEN {}").format(sql.Identifier(TASK_CHANNEL)))
        except BaseException:
            await listener.close()
            raise
    return listener


async def wait_for_task(listener: psycopg.AsyncConnection[Any], wait: timedelta) -> bool:
    """Wait up to ``wait`` for a notification on ``listener``.

    Iterates ``listener.notifies()`` to completion with ``stop_after=1``, so it returns after the first batch of
    notifications or when ``wait`` elapses, whichever comes first. A notification only wakes the worker: extra
    notifications in the same batch are discarded, and the loop keeps claiming until ``claim`` returns ``None``. A
    negative ``wait`` is not rejected: ``notifies`` clamps the remaining time to zero, so the call polls once for
    what has already arrived and returns without waiting.

    Preconditions:
        ``listener`` has already executed ``LISTEN`` for every channel that should wake the caller.
        ``listener`` is in autocommit mode, or otherwise has no open transaction, so notifications are delivered.
        No other coroutine uses ``listener`` while this call is running.

    Args:
        listener: Connection to receive notifications on.
        wait: Maximum time to wait. A negative value behaves as zero.

    Returns:
        ``True`` if at least one notification on any listened channel arrived within ``wait``. ``False`` if
        ``wait`` elapsed with none.

    Raises:
        DatabaseUnavailable: If the connection is lost while waiting (mapped by ``db_errors``).
        psycopg.Error: Any other database error from ``listener.notifies``, unchanged. ``db_errors`` swallows
            nothing.
        asyncio.CancelledError: If the awaiting task is cancelled.

    """
    received = False
    with db_errors():
        async for _ in listener.notifies(timeout=wait.total_seconds(), stop_after=1):
            received = True
    return received
