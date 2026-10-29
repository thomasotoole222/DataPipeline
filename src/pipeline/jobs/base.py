"""The base class of every batch job: one claimed task extracted, validated, transformed and loaded, in that order.

Public API:
    Partition: How a job's tasks slice its endpoint.
    TaskLease: Protocol for the lease a job runs a task under.
    BatchJob: The base class an application package's jobs subclass.
"""

from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from datetime import date, timedelta
from typing import Any, Literal, Protocol, final
from zoneinfo import ZoneInfo

from pipeline.checks.checks import Validation
from pipeline.checks.record import Record
from pipeline.checks.targets import ReplacePartition, target_of
from pipeline.connectors.client import ConnectorClient
from pipeline.connectors.spec import EndpointSpec
from pipeline.db.control import ClaimedTask, Fence, TaskReport
from pipeline.db.pool import Connection, Pool
from pipeline.errors import Outcome
from pipeline.stages.extract import extract
from pipeline.stages.load import load
from pipeline.stages.transform import Transformed, transform
from pipeline.stages.validate import validate
from pipeline.telemetry import Instruments
from pipeline.timeutil import TimeFrame, utc_now

#: How a job's tasks slice its endpoint: ``"day"`` is one local day per task, keyed ``YYYY-MM-DD``; ``"none"`` is the
#: whole endpoint in one task, with no time frame.
type Partition = Literal["day", "none"]


class TaskLease(Fence, Protocol):
    """Define the lease a job runs a task under: it fences writes, gives the pool and reports the outcome.

    The worker's ``Lease`` satisfies it structurally. ``BatchJob.__call__`` uses ``fenced`` (through ``extract`` and
    ``_Completing``), ``pool`` and ``complete``.

    """

    @property
    def pool(self) -> Pool:
        """Return the pool the task's stages use.

        Preconditions:
            None.

        Returns:
            The pool every stage takes its connections from.

        Raises:
            Not specified by the protocol.

        """
        ...

    async def complete(self, conn: Connection, report: TaskReport) -> str:
        """Report the task's outcome on ``conn``, as ``Lease.complete`` does.

        Preconditions:
            ``conn`` is open and not used by another coroutine; it is the connection of an open ``fenced`` block or
            in autocommit mode with no transaction open.

        Args:
            conn: Connection the outcome is reported on.
            report: The outcome to report.

        Returns:
            The text ``control.report_outcome`` returned.

        Raises:
            LeaseConflict: If the lease is no longer held.
            AlreadyReported: If an outcome was already recorded under the lease.
            Exception: Whatever else the implementation raises; ``Lease.complete`` documents its own.

        """
        ...


class _Completing:
    """Fence Load's transaction on the lease and report the task's success in that transaction, as its last write.

    Satisfies ``Fence`` structurally, so ``load`` takes it in place of the lease.

    Attributes:
        None public. All state is private.

    """

    def __init__(self, lease: TaskLease, report: Callable[[], TaskReport]) -> None:
        """Store the lease and the function that builds the success report.

        Preconditions:
            None.

        Args:
            lease: The lease the task runs under.
            report: Builds the report when Load's writes are done, so its duration ends then.

        Returns:
            None.

        Raises:
            Nothing.

        """
        self._lease = lease
        self._report = report

    @asynccontextmanager
    async def fenced(self, conn: Connection) -> AsyncIterator[None]:
        """Run the block inside ``lease.fenced(conn)``, then report success on ``conn`` before the transaction ends.

        The report is built by calling the stored function after the block returns, and sent with
        ``lease.complete``; the text it returns is discarded. The report is skipped when the block raises, so the
        transaction rolls back with nothing reported. An error from building or sending the report also rolls the
        transaction back, Load's writes included.

        Preconditions:
            As ``Fence.fenced``: ``conn`` is open, in autocommit mode with no transaction open, and not used by
            another coroutine.

        Args:
            conn: The connection Load writes on.

        Yields:
            None, once, inside the fenced transaction.

        Raises:
            LeaseConflict: From ``lease.fenced`` or ``lease.complete``, when the lease is no longer held.
            AlreadyReported: From ``lease.fenced`` or ``lease.complete``, when an outcome already landed under it.
            DatabaseConflict: From ``lease.fenced`` or ``lease.complete``, on a serialization failure or deadlock.
            DatabaseUnavailable: From ``lease.fenced`` or ``lease.complete``, on a lost or unavailable connection.
            psycopg.Error: Any other database error from the fence, the report or the commit, unchanged.
            ValueError: From the report function, when ``TaskReport`` refuses its fields.
            Exception: Anything the block raises, or anything else ``lease.complete`` raises, unchanged, after the
                transaction has rolled back.

        """
        async with self._lease.fenced(conn):
            yield
            await self._lease.complete(conn, self._report())


def _rows_written(transformed: Transformed, *, held: bool) -> int:
    """Count the rows Load writes: every frame's rows, less a ``ReplacePartition`` frame's when records are held.

    ``load`` skips a ``ReplacePartition`` target when records are held and writes every other frame whole (its
    contract), so the count is the sum of the heights of the frames ``load`` reports as ``sent``. It is computed here
    because ``load`` returns its own counts only after the transaction the report must be part of has committed.

    Preconditions:
        ``held`` is true exactly when ``load`` receives a non-empty ``validated.held``.

    Args:
        transformed: The frames handed to Load.
        held: Whether Validate held any record.

    Returns:
        The number of rows in the frames Load writes.

    Raises:
        Nothing.

    """
    return sum(
        data.height for target, data in transformed.frames if not (held and isinstance(target.writer, ReplacePartition))
    )


class BatchJob[R: Record]:
    """Run one claimed task of a job: Extract, Validate, Transform and Load, then report success inside Load.

    A subclass sets ``name``, ``spec``, ``window``, ``partition`` and ``validation`` as class attributes and nothing
    else; the lifecycle, ``__call__``, is final. An instance is the job's ``TaskRunner``. ``__init__`` checks that the
    attributes are set and agree with each other, with the client and with the record model's root target (see
    ``__init__``).

    Attributes:
        name (str): The job's name, as tasks name it in ``control.task.job``.
        spec (EndpointSpec): The one endpoint the job fetches.
        window (str | None): The name in ``spec.windows`` the time frame is applied with; ``None`` exactly when
            ``partition`` is ``"none"``.
        partition (Partition): How a task's ``partition_key`` becomes its time frame.
        validation (Validation[R]): The record model, its checks and the batch policy.

    """

    name: str
    spec: EndpointSpec
    window: str | None
    partition: Partition
    validation: Validation[R]

    def __init__(self, *, client: ConnectorClient[Any], zone: ZoneInfo, instruments: Instruments) -> None:
        """Check the job's declaration against itself and the client, then store the client, zone and instruments.

        The record model's root target is built with ``target_of``, so a model that would fail every run after
        Extract is refused here instead: one that declares no table or loads as child rows, and one whose root
        target writes with ``ReplacePartition`` under a ``"none"`` partition, which has no frame to replace. A child
        target always writes with ``ReplaceChildren`` (the ``Rows`` mark in ``pipeline.checks.targets`` refuses any
        other; not among this page's contracts), so only the root is checked.

        Preconditions:
            None.

        Args:
            client: The client of the connector ``spec`` belongs to.
            zone: The source's home zone; a ``"day"`` partition is one local day in it.
            instruments: Where the stages record their metrics.

        Returns:
            None.

        Raises:
            AttributeError: A class attribute is not set.
            TypeError: From ``target_of``: ``validation.model`` declares no table, or loads as child rows.
            ValueError: ``client.connector`` is not ``spec.connector``; ``partition`` is neither ``"day"`` nor
                ``"none"``; ``window`` is ``None`` for a ``"day"`` partition, set for a ``"none"`` one, or not a name
                in ``spec.windows``; or the model's root target writes with ``ReplacePartition`` under a ``"none"``
                partition.

        """
        for attribute in ("name", "spec", "window", "partition", "validation"):
            getattr(self, attribute)
        if client.connector != self.spec.connector:
            raise ValueError(f"job {self.name}: client is for {client.connector}, spec is for {self.spec.connector}")
        if self.partition not in ("day", "none"):
            raise ValueError(f"job {self.name}: unknown partition {self.partition!r}")
        if (self.window is None) != (self.partition == "none"):
            raise ValueError(f"job {self.name}: a {self.partition} partition needs window None exactly when 'none'")
        if self.window is not None and self.window not in self.spec.windows:
            raise ValueError(f"job {self.name}: {self.spec.name} has no window {self.window!r}")
        root = target_of(self.validation.model)
        if self.partition == "none" and isinstance(root.writer, ReplacePartition):
            raise ValueError(f"job {self.name}: {root.table} replaces a partition, but a none partition has no frame")
        self._client = client
        self._zone = zone
        self._instruments = instruments

    def frame(self, partition_key: str) -> TimeFrame | None:
        """Return the time frame a task's partition key covers.

        Preconditions:
            None.

        Args:
            partition_key: The task's ``partition_key``.

        Returns:
            For a ``"day"`` partition, the local day ``partition_key`` names in the job's zone, from its local midnight
            to the next (``TimeFrame.days``); for ``"none"``, ``None``, whatever the key.

        Raises:
            ValueError: A ``"day"`` key that is not a date written ``YYYY-MM-DD`` (``date.fromisoformat`` also reads
                forms such as ``"20240101"``, which the round-trip check refuses); or a date the zone skips entirely
                (``TimeFrame.days``), e.g. ``"2011-12-30"`` in ``Pacific/Apia``.
            OverflowError: A ``"day"`` key of ``"9999-12-31"`` (the day after is out of range) or ``"9999-12-30"``
                (``TimeFrame.days`` needs ``end`` before ``date.max``); or a key whose local midnight leaves the
                ``datetime`` range in UTC, e.g. ``"0001-01-01"`` in a zone east of UTC such as ``Asia/Tokyo``.

        A date the zone skips (``"2011-12-30"`` in ``Pacific/Apia``) holds no instants, so there is nothing to fetch,
        yet ``frame`` raises ``ValueError`` and the task fails on every attempt (todo.txt 13).

        """
        if self.partition == "none":
            return None
        day = date.fromisoformat(partition_key)
        if day.isoformat() != partition_key:
            raise ValueError(f"job {self.name}: partition key {partition_key!r} is not a YYYY-MM-DD date")
        return TimeFrame.days(day, day + timedelta(days=1), self._zone)

    @final
    async def __call__(self, task: ClaimedTask, lease: TaskLease, /) -> None:
        """Run the task: Extract, Validate, Transform, then Load with the success report as its last write.

        Steps:
            1. ``frame`` turns the task's partition key into its time frame.
            2. ``extract`` lands every page of ``spec`` in ``raw.page``, each write fenced on the lease.
            3. ``validate`` splits the landed records into accepted and held.
            4. ``transform`` turns the accepted records into one frame per target table.
            5. ``load`` writes the held rows and the frames in one transaction fenced on the lease; its last write
               is ``lease.complete`` with a ``SUCCEEDED`` report, so the rows and the outcome commit together.

        The report counts the records fetched, the rows Load writes (``_rows_written``) and the rows held, and its
        duration runs from ``task.claimed_at`` until Load's writes are done, floored at zero when the wall clock is
        behind the claim time. The text ``lease.complete`` returns is discarded.

        Preconditions:
            ``lease`` is the lease this worker holds on ``task``.

        Args:
            task: The claimed task.
            lease: The lease on ``task``.

        Returns:
            None, once the outcome has been reported.

        Raises:
            ValueError: From ``frame``, for a partition key the job's partition cannot read or a skipped local date,
                before anything is written.
            OverflowError: From ``frame``, for a ``"day"`` key of ``"9999-12-30"`` or ``"9999-12-31"``, or one whose
                local midnight leaves the ``datetime`` range in UTC, before anything is written.
            Exception: Anything a stage, ``lease.complete`` or ``TaskReport`` raises, unchanged; the worker reports it
                as the task's outcome. Pages landed by ``extract`` stay committed when a later stage raises.

        """
        frame = self.frame(task.partition_key)
        pool = lease.pool
        fetched = await extract(pool, task, lease, self._client, self.spec, self.window, frame, self._instruments)
        validated = await validate(pool, task, self.spec, frame, self.validation, self._instruments)
        transformed = transform(self.validation.model, validated.accepted, task.source_id)
        written = _rows_written(transformed, held=bool(validated.held))

        def report() -> TaskReport:
            return TaskReport(
                outcome=Outcome.SUCCEEDED,
                error_class=None,
                error_message=None,
                retry_after=None,
                rows_fetched=fetched,
                rows_written=written,
                rows_rejected=len(validated.held),
                duration=max(utc_now() - task.claimed_at, timedelta(0)),
            )

        await load(pool, task, _Completing(lease, report), validated, transformed, frame, self._instruments)
