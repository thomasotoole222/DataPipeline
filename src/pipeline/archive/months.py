"""Write raw pages to the archive: one Parquet object per job per month, written once that month has settled.

A page's month is ``raw.data_month`` of its task: the month of a day partition's key, otherwise the UTC month the task
was created in. A job's month is settled once it has ended and the job's longest re-fetch, plus ``SETTLE_MARGIN``, has
passed, so the object holds the month's final pages and is never added to. ``raw.archive_manifest`` records each
object; its ``UNIQUE (job, month)`` refuses a second object for the same job and month, and pages fetched for an
archived month later are not archived (``control.purge_raw`` deletes them once they are old).

Public API:
    SETTLE_MARGIN: Days added to a job's longest re-fetch before its month is written.
    ROW_GROUP_BYTES: Response bytes buffered before a row group is written.
    COLUMNS: The archive's Parquet schema.
    ArchiveCheckFailed: An object did not read back as written, or the manifest and the store disagree.
    MonthKey: One job and month.
    Plan: What one archive run finds: the months ready to write, the months waiting, and pages of unknown jobs.
    Archived: What writing one month produced.
    object_key: The key a job and month's object is stored under.
    settled_on: The first day a job's month may be written.
    plan_months: Find the months ready to write.
    archive_month: Write one job and month, read it back, then record it and mark its pages archived.
    check_manifest: Check every recorded object exists with the size and SHA-256 the manifest holds.

"""

import asyncio
import hashlib
import logging
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Final

import pyarrow as pa
import pyarrow.parquet as pq

from pipeline.archive.store import ObjectStore
from pipeline.db.control import Fence
from pipeline.db.pool import Pool
from pipeline.errors import PermanentError

_log = logging.getLogger(__name__)

#: Days added to a job's longest re-fetch before its month is written, so a re-fetch that runs a day late still lands
#: before the month is archived.
SETTLE_MARGIN: Final = timedelta(days=2)

#: Response bytes buffered before a row group is written; a row group also ends where the source changes, so a reader
#: can skip to one shop.
ROW_GROUP_BYTES: Final = 64 * 1024 * 1024

#: Pages fetched from Postgres per round trip.
_FETCH_ROWS: Final = 200

#: Raise the pool's 60 s ``statement_timeout`` for the archive's own transactions only: ``_PAGES`` sorts a whole
#: (job, month) before its first row (``video_daily`` for 2026-06 took over 60 s), and ``_MARK`` updates every page of
#: it. ``SET LOCAL`` ends with the transaction, so the connection goes back to the pool with the 60 s limit.
_LONG_STATEMENTS: Final = "SET LOCAL statement_timeout = '30min'"

#: zstd level for every column; the archive is written once and read rarely.
_ZSTD_LEVEL: Final = 9

#: The archive's columns, in ``_PAGES``'s order.
_FIELDS: Final[list[pa.Field[Any]]] = [
    pa.field("connector", pa.string(), nullable=False),
    pa.field("endpoint", pa.string(), nullable=False),
    pa.field("job", pa.string(), nullable=False),
    pa.field("source_id", pa.string(), nullable=False),
    pa.field("partition_key", pa.string(), nullable=False),  # the data day 'YYYY-MM-DD', or 'none'
    pa.field("task_id", pa.int64(), nullable=False),
    pa.field("page_no", pa.int32(), nullable=False),
    pa.field("cursor_in", pa.string()),
    pa.field("cursor_next", pa.string()),
    pa.field("fetched_at", pa.timestamp("us", tz="UTC"), nullable=False),
    pa.field("request_id", pa.string()),  # the envelope's request_id, when the source sends one
    pa.field("ids", pa.list_(pa.string())),  # the ids a fanned-out call was for
    pa.field("error_class", pa.string()),  # set: a failed fanned-out call's marker page
    pa.field("response", pa.large_string(), nullable=False),  # the whole response body as JSON text
]

#: The archive's Parquet schema: one row per raw page, the response rebuilt as fetched (envelope and payload).
COLUMNS: Final = pa.schema(_FIELDS)

#: Every column but ``response`` is dictionary-encoded; responses rarely repeat.
_DICTIONARY: Final = [name for name in COLUMNS.names if name != "response"]

#: The months that have unarchived pages of closed tasks, with their page counts.
_PENDING: Final = (
    "SELECT t.job, raw.data_month(t.partition_key, t.created_at) AS month, count(*) "
    "FROM raw.page AS p JOIN control.task AS t ON t.id = p.task_id "
    "WHERE p.archived_month IS NULL AND t.state NOT IN ('pending', 'running', 'retry_wait', 'blocked') "
    "GROUP BY 1, 2"
)

#: The months that have a task that may still fetch pages.
_WAITING: Final = (
    "SELECT DISTINCT job, raw.data_month(partition_key, created_at) FROM control.task "
    "WHERE state IN ('pending', 'running', 'retry_wait')"
)

#: The months already in the archive.
_ARCHIVED: Final = "SELECT job, month FROM raw.archive_manifest"

#: Unarchived pages whose task_id names no task: never archived, never purged; counted for the log.
_ORPHANS: Final = (
    "SELECT count(*) FROM raw.page AS p "
    "WHERE p.archived_month IS NULL AND NOT EXISTS (SELECT FROM control.task AS t WHERE t.id = p.task_id)"
)

#: One job and month's unarchived pages of closed tasks, in the archive's order. The parameters are the job and the
#: month's first day.
_PAGES: Final = (
    "SELECT p.connector, p.endpoint, t.job, t.source_id::text, t.partition_key, p.task_id, p.page_no, "
    "p.cursor_in, p.cursor_next, p.fetched_at, p.envelope ->> 'request_id', p.ids, p.error_class, "
    "(p.envelope || y.body)::text "
    "FROM control.task AS t "
    "JOIN raw.page AS p ON p.task_id = t.id "
    "JOIN raw.payload AS y ON y.hash = p.payload_hash "
    "WHERE t.job = %s AND raw.data_month(t.partition_key, t.created_at) = %s "
    "AND t.state NOT IN ('pending', 'running', 'retry_wait', 'blocked') AND p.archived_month IS NULL "
    "ORDER BY t.source_id, t.partition_key, p.fetched_at, p.task_id, p.page_no"
)

#: Record one object; ``UNIQUE (job, month)`` refuses a second.
_RECORD: Final = (
    "INSERT INTO raw.archive_manifest (job, month, object_key, rows, bytes, sha256) VALUES (%s, %s, %s, %s, %s, %s)"
)

#: Mark the pages written to an object as archived. The parameters are the month and the pages' task ids and page
#: numbers, as two arrays of the same length.
_MARK: Final = (
    "UPDATE raw.page AS p SET archived_month = %s "
    "FROM unnest(%s::bigint[], %s::int[]) AS k (task_id, page_no) "
    "WHERE p.task_id = k.task_id AND p.page_no = k.page_no AND p.archived_month IS NULL"
)

#: Every recorded object.
_MANIFEST: Final = "SELECT job, month, object_key, bytes, sha256 FROM raw.archive_manifest ORDER BY month, job"


class ArchiveCheckFailed(PermanentError):
    """Signal that an object did not read back as written, or that the manifest and the store disagree.

    Adds no constructor or attributes of its own; the raiser supplies the message. Nothing is recorded for an object
    whose check failed, so its pages stay unarchived and the next run writes it again.

    Attributes:
        outcome (Outcome): ``Outcome.FAILED``, inherited from ``PermanentError``.

    """

    pass


@dataclass(frozen=True, slots=True, order=True)
class MonthKey:
    """Name one job's month; sorts by month, then job.

    Attributes:
        month: The month's first day.
        job: The job's name.

    """

    month: date
    job: str


@dataclass(frozen=True, slots=True)
class Plan:
    """Hold what one archive run finds.

    Attributes:
        ready: The settled months with pages and no open task, not yet archived, oldest first.
        waiting: The settled months not yet archived that still have a pending, running or retry_wait task.
        unknown: Pages of closed tasks per job with no re-fetch horizon, which are never archived.
        orphans: Unarchived pages whose task_id names no task.

    """

    ready: list[MonthKey] = field(default_factory=list)
    waiting: list[MonthKey] = field(default_factory=list)
    unknown: dict[str, int] = field(default_factory=dict)
    orphans: int = 0


@dataclass(frozen=True, slots=True)
class Archived:
    """Describe the object written for one job and month.

    Attributes:
        key: The object's key.
        rows: The pages written, one row each; 0 when the month had none left and nothing was written.
        size: The object's length in bytes; 0 when nothing was written.
        sha256: The object's hex SHA-256; ``""`` when nothing was written.

    """

    key: str
    rows: int
    size: int
    sha256: str


def object_key(job: str, month: date) -> str:
    """Return the key a job and month's object is stored under, e.g. ``raw/month=2026-05/job=orders.parquet``.

    Preconditions:
        ``job`` holds no ``/``.

    Args:
        job: The job's name.
        month: Any day of the month.

    Returns:
        The key.

    Raises:
        Nothing.

    """
    return f"raw/month={month:%Y-%m}/job={job}.parquet"


def settled_on(month: date, refetch_days: int) -> date:
    """Return the first day ``month`` may be written for a job whose longest re-fetch is ``refetch_days``.

    That is the first day of the next month, plus ``refetch_days``, plus ``SETTLE_MARGIN``: the 5th of the next month
    for a job re-fetched up to 2 days back, the 3rd for a snapshot job.

    Preconditions:
        ``refetch_days`` is 0 or more.

    Args:
        month: Any day of the month.
        refetch_days: The job's longest re-fetch, in days of age.

    Returns:
        The day.

    Raises:
        OverflowError: For a month in 9999.

    """
    following = (month.replace(day=28) + timedelta(days=4)).replace(day=1)
    return following + timedelta(days=refetch_days) + SETTLE_MARGIN


async def plan_months(pool: Pool, refetch_days: Mapping[str, int], today: date) -> Plan:
    """Find the months ready to write: settled on or before ``today``, not archived, with no open task.

    A month of a job with no entry in ``refetch_days`` is counted in ``unknown``, never archived. A settled month that
    still has a pending, running or retry_wait task waits, so its pages are archived together once the task ends. A
    blocked task does not hold its month back: its pages are not read, and it is not waited for.

    Preconditions:
        None.

    Args:
        pool: The warehouse pool.
        refetch_days: Each archived job's longest re-fetch, in days of age, by job name.
        today: The UTC day the run counts as today.

    Returns:
        The ``Plan``.

    Raises:
        psycopg.Error: A query failed.

    """
    async with pool.connection() as conn:
        pending = await (await conn.execute(_PENDING)).fetchall()
        open_months = {MonthKey(month, job) for job, month in await (await conn.execute(_WAITING)).fetchall()}
        archived = {MonthKey(month, job) for job, month in await (await conn.execute(_ARCHIVED)).fetchall()}
        orphan_row = await (await conn.execute(_ORPHANS)).fetchone()
    plan = Plan(orphans=int(orphan_row[0]) if orphan_row else 0)
    for job, month, pages in pending:
        key = MonthKey(month, job)
        if key in archived:
            continue  # later re-fetches of an archived month: control.purge_raw deletes them once old
        if job not in refetch_days:
            plan.unknown[job] = plan.unknown.get(job, 0) + int(pages)
        elif settled_on(month, refetch_days[job]) > today:
            continue
        elif key in open_months:
            plan.waiting.append(key)
        else:
            plan.ready.append(key)
    plan.ready.sort()
    plan.waiting.sort()
    return plan


def _digest(path: Path) -> str:
    """Return the hex SHA-256 of the file at ``path``, read in 1 MiB chunks."""
    sha = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(1 << 20):
            sha.update(chunk)
    return sha.hexdigest()


def _read_back(store: ObjectStore, key: str, path: Path) -> tuple[int | None, int, str]:
    """Download ``key`` to ``path``; return its rows (Parquet footer; ``None`` if unreadable), size and SHA-256."""
    store.get(key, path)
    try:
        rows: int | None = pq.read_metadata(path).num_rows
    except pa.ArrowException:  # not a Parquet file any more: the check below fails on it
        rows = None
    return rows, path.stat().st_size, _digest(path)


def _table(rows: list[tuple[Any, ...]], schema: pa.Schema) -> pa.Table:
    """Build one row group's table from page rows in ``_PAGES``'s column order."""
    columns = list(zip(*rows, strict=True))
    return pa.Table.from_arrays(
        [pa.array(values, type=column.type) for values, column in zip(columns, schema, strict=True)], schema=schema
    )


async def _write(pool: Pool, month_key: MonthKey, path: Path) -> tuple[list[int], list[int]]:
    """Stream the month's pages into a Parquet file at ``path``; return the task ids and page numbers written."""
    task_ids: list[int] = []
    page_nos: list[int] = []
    schema = COLUMNS.with_metadata({"job": month_key.job, "month": f"{month_key.month:%Y-%m}"})
    writer = pq.ParquetWriter(
        path,
        schema,
        compression="zstd",
        compression_level=_ZSTD_LEVEL,
        use_dictionary=_DICTIONARY,  # type: ignore[arg-type]  # the stub says bool; pyarrow takes column names too
    )
    try:
        buffered: list[tuple[Any, ...]] = []
        buffered_bytes = 0
        async with pool.connection() as conn, conn.transaction(), conn.cursor(name="archive_pages") as cur:
            await conn.execute(_LONG_STATEMENTS)
            await cur.execute(_PAGES, (month_key.job, month_key.month))
            while batch := await cur.fetchmany(_FETCH_ROWS):
                for row in batch:
                    if buffered and (row[3] != buffered[-1][3] or buffered_bytes >= ROW_GROUP_BYTES):
                        await asyncio.to_thread(writer.write_table, _table(buffered, schema))
                        buffered, buffered_bytes = [], 0
                    buffered.append(tuple(row))
                    buffered_bytes += len(row[13])
                    task_ids.append(row[5])
                    page_nos.append(row[6])
        if buffered:
            await asyncio.to_thread(writer.write_table, _table(buffered, schema))
    finally:
        await asyncio.to_thread(writer.close)
    return task_ids, page_nos


async def archive_month(
    pool: Pool, fence: Fence, store: ObjectStore, month_key: MonthKey, scratch: Path | None = None
) -> Archived:
    """Write one job and month's pages to one object, read it back, then record it and mark its pages archived.

    Steps:
        1. Stream the month's unarchived pages of closed tasks out of Postgres, in ``_PAGES``'s order, into a local
           Parquet file (zstd), a row group per source or per ``ROW_GROUP_BYTES`` of responses.
        2. Upload it as one object, its SHA-256 stored with it.
        3. Download it again and check its row count (from the Parquet footer), size and SHA-256 against the file
           written, and the SHA-256 the store holds for it.
        4. In one transaction fenced on the task's lease: insert the manifest row and mark exactly the pages written
           as archived.

    A month with no such pages left writes nothing and returns ``rows == 0``. Each local file is removed before this
    returns, whether or not it raises. A failed check records nothing: the pages stay unarchived. An object uploaded
    before a failure stays in the store under its key, and the next run's ``put`` replaces it.

    Preconditions:
        No other archive run writes the same job and month; ``control.task``'s live-task index keeps one archive task
        live at a time.

    Args:
        pool: The warehouse pool.
        fence: The task's lease; the final transaction is fenced on it.
        store: Where the object is written.
        month_key: The job and month.
        scratch: The directory local files are written in; the system's temporary directory when ``None``.

    Returns:
        The ``Archived`` object.

    Raises:
        ArchiveCheckFailed: The object read back differently, or the pages marked were not exactly the pages written.
        LeaseConflict: From the fence, when the lease is no longer held; nothing is recorded.
        psycopg.errors.UniqueViolation: The manifest already holds the job and month; nothing is recorded.
        Exception: Whatever the store or Postgres raises.

    """
    key = object_key(month_key.job, month_key.month)
    with tempfile.TemporaryDirectory(prefix="archive-", dir=scratch) as tmp:
        written = Path(tmp) / "written.parquet"
        task_ids, page_nos = await _write(pool, month_key, written)
        rows = len(task_ids)
        if not rows:
            return Archived(key=key, rows=0, size=0, sha256="")
        size = written.stat().st_size
        sha256 = await asyncio.to_thread(_digest, written)
        await asyncio.to_thread(store.put, key, written, sha256)
        written.unlink()  # one file at a time on the worker's disk
        back_rows, back_size, back_sha = await asyncio.to_thread(_read_back, store, key, Path(tmp) / "read.parquet")
        stored = await asyncio.to_thread(store.head, key)
    if (back_rows, back_size, back_sha) != (rows, size, sha256) or stored is None or stored.sha256 != sha256:
        raise ArchiveCheckFailed(
            f"{key}: wrote {rows} rows, {size} bytes, sha256 {sha256}; read back {back_rows} rows, {back_size} bytes, "
            f"sha256 {back_sha}; stored sha256 {stored.sha256 if stored else None}"
        )
    async with pool.connection() as conn, fence.fenced(conn):
        await conn.execute(_LONG_STATEMENTS)
        await conn.execute(_RECORD, (month_key.job, month_key.month, key, rows, size, bytes.fromhex(sha256)))
        marked = (await conn.execute(_MARK, (month_key.month, task_ids, page_nos))).rowcount
        if marked != rows:  # raising rolls the manifest row back
            raise ArchiveCheckFailed(f"{key}: wrote {rows} pages but {marked} were still unarchived to mark")
    _log.info("archived %s: %d pages, %d bytes", key, rows, size)
    return Archived(key=key, rows=rows, size=size, sha256=sha256)


async def check_manifest(pool: Pool, store: ObjectStore) -> list[str]:
    """Check every recorded object exists in the store with the size and SHA-256 the manifest holds.

    Only the store's object metadata is read (``ObjectStore.head``), not the objects themselves.

    Preconditions:
        None.

    Args:
        pool: The warehouse pool.
        store: The archive's store.

    Returns:
        One message per object that is missing or differs, in manifest order; empty when all match.

    Raises:
        psycopg.Error: The manifest could not be read.
        Exception: Whatever the store raises other than for a missing object.

    """
    async with pool.connection() as conn:
        manifest = await (await conn.execute(_MANIFEST)).fetchall()
    problems = []
    for _job, _month, key, size, sha256 in manifest:
        stored = await asyncio.to_thread(store.head, key)
        if stored is None:
            problems.append(f"{key}: in the manifest but not in the store")
        elif stored.size != size or stored.sha256 != bytes(sha256).hex():
            problems.append(
                f"{key}: manifest {size} bytes, sha256 {bytes(sha256).hex()}; store {stored.size} bytes, "
                f"sha256 {stored.sha256}"
            )
    return problems
