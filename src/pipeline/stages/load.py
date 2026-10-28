"""Write one task's transformed frames and held rows to Postgres in a single transaction.

Public API:
    Loaded: Per-table rows sent and changed, and the number of rows held, from one ``load`` call.
    load: Fence the task, hold its rejected rows, write every target's frame and record row metrics.

"""

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any, Final
from uuid import UUID

import polars as pl

from pipeline.checks.targets import (
    ReplaceChildren,
    ReplacePartition,
    Snapshot,
    Target,
    Update,
    Upsert,
    mismatches,
)
from pipeline.db import quarantine
from pipeline.db.catalog import table_info
from pipeline.db.control import ClaimedTask, Fence
from pipeline.db.pool import Connection, Pool
from pipeline.errors import NothingLoaded, TableMismatch
from pipeline.stages.transform import Transformed
from pipeline.stages.validate import Validated
from pipeline.telemetry import Instruments
from pipeline.timeutil import TimeFrame
from pipeline.writers import sql
from pipeline.writers.staging import copy_in

#: Module logger; ``load`` warns through it once per table left unpruned or not written.
_log: Final = logging.getLogger(__name__)
#: Targets already compared with their tables in this worker process; ``_columns_checked`` adds a target only when
#: it matches, and nothing removes one, so a table altered later is not compared again until the process restarts.
_checked: Final[set[Target]] = set()


@dataclass(frozen=True, slots=True, kw_only=True)
class Loaded:
    """Frozen, slotted, keyword-only result of ``load``.

    No field is checked; there is no ``__post_init__``. ``load`` builds ``sent`` and ``changed`` with the same keys:
    every target it wrote, which is every target but a ``ReplacePartition`` skipped because records were held.

    Attributes:
        sent: Rows in each written target's frame, keyed by table name (``"schema.table"``).
        changed: The count each written table's ``sql`` writer returned, keyed by table name.
        held: Number of rows ``quarantine.hold`` copied into the hold table (``len(validated.held)``).

    """

    sent: Mapping[str, int]  # rows sent per written table, in load order
    changed: Mapping[str, int]  # rows the writer inserted, updated, deleted or marked, per written table
    held: int  # held rows written to the hold table


async def load(
    pool: Pool,
    task: ClaimedTask,
    fence: Fence,
    validated: Validated[Any],
    transformed: Transformed,
    frame: TimeFrame | None,  # the task's time window; needed by a ReplacePartition target
    instruments: Instruments,
) -> Loaded:
    """Fence the task, hold its rejected rows, write every target's frame, then record row metrics.

    Everything up to the writes runs on one pooled connection inside one transaction that ``fence.fenced`` opens and
    fences on the task's lease: ``quarantine.hold`` of ``validated.held``, then ``_columns_checked`` for each target,
    then a check that no ``Snapshot`` or ``Update(missing="zero")`` frame is empty when pruning, then ``_write`` for
    each pair in ``transformed.frames`` order. After a written target, its children's ``parents`` entry is the target's
    staged table. ``prune`` is true only when nothing was held; ``_write`` passes it to ``Snapshot`` and ``Update``
    writers. When records were held, a ``ReplacePartition`` target is skipped, not written at all (no ``sent`` or
    ``changed`` entry), since replacing the partition would delete the held records' earlier rows. A held parent's
    children are safe without ``prune``, because ``sql.replace_children`` deletes only under parents in the staged
    parent table. Any error rolls the whole transaction back. The metrics run only after commit: for each written table,
    ``sent`` is added to ``instruments.rows`` with ``stage="written"`` and ``changed`` with ``stage="changed"``; for
    each skipped table, its frame height is added with ``stage="skipped"`` and a warning says it was not written because
    records are held. When records were held, every ``Snapshot`` and ``Update(missing="zero")`` target is left unpruned:
    for each, ``held`` is added with ``stage="prune_skipped"`` and a warning is logged.

    Preconditions:
        ``frame`` is not ``None`` when a target uses ``ReplacePartition``. A ``ReplacePartition`` target has no
        children: it has no key, so ``build_target`` refuses ``Rows`` on it. Every frame came from ``frame_of``, so a
        non-nullable column has no nulls; ``Transformed`` checks each frame's columns and types but not its nulls.
        ``Transformed`` guarantees every child target's parent is in ``transformed.frames`` before it.

    Args:
        pool: Pool the connection is taken from.
        task: The claimed task; its ``source_id``, ``id`` and ``job`` are used.
        fence: Fences the load transaction on the task's lease (the worker's ``Lease``).
        validated: The validate result; only ``held`` is read.
        transformed: One (target, frame) pair per target, in load order.
        frame: The task's time frame, or ``None``; only ``ReplacePartition`` targets read it.
        instruments: Instruments whose ``rows`` counter gets the metrics.

    Returns:
        ``Loaded`` with each written table's frame height and changed count, and the number of rows held.

    Raises:
        LeaseConflict: From ``fence.fenced``, when the task's lease is no longer held; nothing is written.
        TableMismatch: From ``_columns_checked``, when a target not yet in ``_checked`` does not match its table; a
            target cached earlier in this process is not compared again, so a table altered since fails as a
            ``psycopg.Error`` from the writes instead.
        NothingLoaded: When pruning and a ``Snapshot`` or ``Update(missing="zero")`` target's frame has no rows.
        ValueError: From ``_write``, for a ``ReplacePartition`` target with no usable frame or with a row outside the
            frame's bounds.
        TypeError: From ``_write`` (``TimeFrame.dates``), for a date partition when ``frame.zone`` is not a ``tzinfo``.
        OverflowError: From ``_write`` (``TimeFrame.dates``), for a date partition whose bounds leave the ``datetime``
            range in ``frame.zone``.
        DatabaseConflict: From ``db_errors`` in the callees, on a serialization failure or deadlock.
        DatabaseUnavailable: From ``db_errors`` in the callees, on a lost or refused connection.
        psycopg.Error: Any other database error from the callees, unchanged.

    """
    prune = not validated.held  # a held record is missing from the frames, so it must not be pruned
    skipped = {  # would delete held records' earlier rows
        target.table: data.height
        for target, data in transformed.frames
        if not prune and isinstance(target.writer, ReplacePartition)
    }
    sent = {target.table: data.height for target, data in transformed.frames if target.table not in skipped}
    unpruned = [
        target.table
        for target, _ in transformed.frames
        if not prune
        and (
            isinstance(target.writer, Snapshot)
            or (isinstance(target.writer, Update) and target.writer.missing == "zero")
        )
    ]
    changed: dict[str, int] = {}
    async with pool.connection() as conn, fence.fenced(conn):
        held = await quarantine.hold(conn, validated.held)
        for target, _ in transformed.frames:
            await _columns_checked(conn, target)
        for target, data in transformed.frames:
            if isinstance(target.writer, Snapshot) and data.height == 0 and prune:  # held records skip the prune
                raise NothingLoaded(f"{target.table}: a snapshot with no rows would remove every row of the source")
            zeroing = isinstance(target.writer, Update) and target.writer.missing == "zero"
            if zeroing and data.height == 0 and prune:
                raise NothingLoaded(f"{target.table}: an update with no rows would zero every row of the source")
        parents: dict[Target, str] = {}  # a child target's parent's staged table
        for target, data in transformed.frames:
            if target.table in skipped:
                continue
            staged, changed[target.table] = await _write(conn, task.source_id, target, data, parents, frame, prune)
            parents.update((child, staged) for _, child in target.children)
    for table in sent:
        instruments.rows.add(sent[table], {"job": task.job, "stage": "written", "table": table})
        instruments.rows.add(changed[table], {"job": task.job, "stage": "changed", "table": table})
    for table in skipped:
        instruments.rows.add(skipped[table], {"job": task.job, "stage": "skipped", "table": table})
        _log.warning("task %s %s: %s not written because %d records are held", task.id, task.job, table, held)
    for table in unpruned:
        instruments.rows.add(held, {"job": task.job, "stage": "prune_skipped", "table": table})
        _log.warning("task %s %s: %s not pruned, %d records held", task.id, task.job, table, held)
    return Loaded(sent=sent, changed=changed, held=held)


async def _write(
    conn: Connection,
    source_id: UUID,
    target: Target,
    data: pl.DataFrame,
    parents: Mapping[Target, str],
    frame: TimeFrame | None,
    prune: bool,
) -> tuple[str, int]:
    """Stage one frame in a temporary table and write it to the target's table with the target's writer.

    The frame is copied in with each target column's Postgres type. The task's source is passed only when the target
    has a source column. Each of the five writers calls its ``sql`` function: ``Upsert``, ``ReplaceChildren``,
    ``ReplacePartition``, ``Snapshot`` (with ``prune``) and ``Update`` (with its ``missing`` and ``prune``). For
    ``ReplacePartition``, a ``date`` column is bounded by ``frame.dates`` and a timestamptz column by ``frame.start``
    and ``frame.end``, and no row's partition value may lie outside ``[start, end)``; both are checked before the
    frame is staged. A null partition value is not counted as outside. The column lookup always finds the column,
    because ``build_target`` (``_check_writer``) refuses a ``ReplacePartition`` column that is not a target column.

    Preconditions:
        ``conn`` is inside a transaction (``copy_in`` checks this). For ``ReplaceChildren``, ``parents`` has
        ``target`` (``load`` ensures this when the parent is in the frames before the child). For
        ``ReplacePartition``, the partition column has no nulls (``_check_writer`` makes it non-nullable and
        ``frame_of`` refuses ``None`` in a non-nullable column).

    Args:
        conn: Connection inside the load transaction.
        source_id: The task's source.
        target: The target whose table is written.
        data: The target's frame, columns in ``target.columns`` order.
        parents: Staged table of each child target's parent, keyed by the child target.
        frame: The task's time frame, or ``None``; read only for ``ReplacePartition``.
        prune: Passed to ``sql.snapshot`` and ``sql.update``.

    Returns:
        The staged table's name and the count the ``sql`` writer returned.

    Raises:
        ValueError: For ``ReplacePartition`` when ``frame`` is ``None``, when the column is a ``date`` and ``frame`` is
            not whole local days in ``frame.zone``, or when a row's partition value lies outside ``[start, end)``
            (naming the table, the column, the number of such rows and the bounds); from the ``sql`` writers for a
            malformed table name (unreachable: ``build_target`` checks ``target.table`` against ``TABLE_PATTERN``).
        TypeError: From ``TimeFrame.dates``, for a date column when ``frame.zone`` is not a ``tzinfo``.
        OverflowError: From ``TimeFrame.dates``, for a date column when a bound in ``frame.zone`` leaves the
            ``datetime`` range.
        KeyError: For ``ReplaceChildren`` when ``parents`` lacks ``target``.
        RuntimeError: From ``copy_in``, when ``conn`` is not inside a transaction.
        DatabaseConflict: From ``db_errors`` in the callees, on a serialization failure or deadlock.
        DatabaseUnavailable: From ``db_errors`` in the callees, on a lost or refused connection.
        psycopg.Error: Any other database error from the callees, unchanged.

    """
    table, columns = target.table, [column.name for column in target.columns]
    bounds: tuple[datetime, datetime] | tuple[date, date] | None = None
    if isinstance(target.writer, ReplacePartition):  # checked before staging, so a bad frame stages nothing
        partition = target.writer.column
        if frame is None:
            raise ValueError(f"{table}: ReplacePartition needs the task's time frame")
        by_date = next(c.postgres for c in target.columns if c.name == partition) == "date"
        bounds = frame.dates if by_date else (frame.start, frame.end)  # local dates for a date column
        if bounds is None:
            raise ValueError(f"{table}: a date partition needs a frame of whole days in {frame.zone}")
        start, end = bounds
        outside = data.filter((pl.col(partition) < start) | (pl.col(partition) >= end)).height
        if outside:
            raise ValueError(f"{table}: {outside} rows have {partition} outside [{start}, {end})")
    staged = await copy_in(conn, data, [(column.name, column.postgres) for column in target.columns])
    source = source_id if target.source_column is not None else None
    match target.writer:
        case Upsert(key=key):
            changed = await sql.upsert(conn, table, staged, columns, key)
        case ReplaceChildren():
            changed = await sql.replace_children(conn, table, staged, columns, target.inherited, parents[target])
        case ReplacePartition(column=column) if bounds is not None:  # bounds is always set for ReplacePartition
            changed = await sql.replace_partition(
                conn, table, staged, columns, target.source_column, source, column, *bounds
            )
        case Snapshot(key=key):
            changed = await sql.snapshot(conn, table, staged, columns, key, target.source_column, source, prune)
        case Update(key=key, missing=missing):
            changed = await sql.update(conn, table, staged, columns, key, target.source_column, source, missing, prune)
    return staged, changed


async def _columns_checked(conn: Connection, target: Target) -> None:
    """Compare a target with its table once per worker process and refuse a target that does not match.

    A target already in ``_checked`` returns at once. Otherwise its table's catalog entry is fetched with
    ``table_info`` and compared with ``mismatches``; a target that matches is added to ``_checked``, so later calls
    in this process skip the lookup. A mismatched target is not cached and is checked again on its next call.

    Preconditions:
        ``conn`` is open.

    Args:
        conn: Connection the catalog is read on; inside the load transaction when called from ``load``.
        target: The target to compare with its table.

    Returns:
        None.

    Raises:
        TableMismatch: When the table is missing or differs from the target; the message lists every mismatch,
            separated by ``"; "``.
        DatabaseConflict: From ``table_info``, on a serialization failure or deadlock.
        DatabaseUnavailable: From ``table_info``, on a lost or refused connection.
        psycopg.Error: Any other database error from ``table_info``, unchanged.

    """
    if target in _checked:
        return
    found = mismatches(target, await table_info(conn, target.table))
    if found:
        raise TableMismatch("; ".join(found))
    _checked.add(target)
