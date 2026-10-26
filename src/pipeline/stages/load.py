"""Write one task's transformed frames and held rows to Postgres in a single transaction.

Public API:
    Loaded: Per-table rows sent and changed, and the number of rows held, from one ``load`` call.
    load: Fence the task, hold its rejected rows, write every target's frame and record row metrics.

"""

import logging
from collections.abc import Mapping
from dataclasses import dataclass
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
from pipeline.db import control, quarantine
from pipeline.db.catalog import table_info
from pipeline.db.control import ClaimedTask
from pipeline.db.pool import Connection, Pool
from pipeline.errors import NothingLoaded, TableMismatch
from pipeline.stages.transform import Transformed
from pipeline.stages.validate import Validated
from pipeline.telemetry import Instruments
from pipeline.timeutil import TimeFrame
from pipeline.writers import sql
from pipeline.writers.staging import copy_in

#: Module logger; ``load`` warns through it once per table left unpruned.
_log: Final = logging.getLogger(__name__)
#: Targets already compared with their tables in this worker process.
#: Targets already compared with their tables in this worker process.
_checked: Final[set[Target]] = set()


@dataclass(frozen=True, slots=True, kw_only=True)
class Loaded:
    """Frozen, slotted, keyword-only result of ``load``.

    No field is checked; there is no ``__post_init__``. ``load`` builds ``sent`` and ``changed`` with the same keys.

    Attributes:
        sent: Rows in each target's frame, keyed by table name (``"schema.table"``).
        changed: The count each table's ``sql`` writer returned, keyed by table name.
        held: Number of rows ``quarantine.hold`` copied into the hold table (``len(validated.held)``).

    """

    sent: Mapping[str, int]  # rows sent per table, in load order
    changed: Mapping[str, int]  # rows the writer inserted, updated, deleted or marked, per table
    held: int  # held rows written to the hold table


async def load(
    pool: Pool,
    task: ClaimedTask,
    validated: Validated[Any],
    transformed: Transformed,
    frame: TimeFrame | None,  # the task's time window; needed by a ReplacePartition target
    instruments: Instruments,
) -> Loaded:
    """Fence the task, hold its rejected rows, write every target's frame, then record row metrics.

    Everything up to the writes runs on one pooled connection inside one transaction: ``control.fence``, then
    ``quarantine.hold`` of ``validated.held``, then ``_columns_checked`` for each target, then a check that no
    ``Snapshot`` frame is empty when pruning, then ``_write`` for each pair in ``transformed.frames`` order. After a
    written target, its children's ``parents`` entry is the target's staged table. ``prune`` is true only when nothing
    was held. Any error rolls the whole transaction back. The metrics run only after commit: for each table, ``sent``
    is added to ``instruments.rows`` with ``stage="written"`` and ``changed`` with ``stage="changed"``. When records
    were held, every ``Snapshot`` target and every ``Update(missing="zero")`` target is left unpruned: for each,
    ``held`` is added with ``stage="prune_skipped"`` and a warning is logged.

    Preconditions:
        ``transformed.frames`` lists each parent before its children (``Transformed`` states this), so a
        ``ReplaceChildren`` target's parent is in ``parents`` when it is written. ``frame`` is not ``None`` when a
        target uses ``ReplacePartition``.

    Args:
        pool: Pool the connection is taken from.
        task: The claimed task; its lease is fenced, and its ``source_id``, ``id`` and ``job`` are used.
        validated: The validate result; only ``held`` is read.
        transformed: One (target, frame) pair per target, in load order.
        frame: The task's time frame, or ``None``; only ``ReplacePartition`` targets read it.
        instruments: Instruments whose ``rows`` counter gets the metrics.

    Returns:
        ``Loaded`` with each table's frame height, each writer's changed count and the number of rows held.

    Raises:
        LeaseConflict: From ``control.fence``, when the task's lease is no longer held.
        TableMismatch: From ``_columns_checked``, when a target does not match its table.
        NothingLoaded: When pruning and a ``Snapshot`` target's frame has no rows.
        ValueError: From ``_write``, for a ``ReplacePartition`` target with no usable frame.
        DatabaseConflict: From ``db_errors`` in the callees, on a serialization failure or deadlock.
        DatabaseUnavailable: From ``db_errors`` in the callees, on a lost or refused connection.
        psycopg.Error: Any other database error from the callees, unchanged.

    """
    sent = {target.table: data.height for target, data in transformed.frames}
    prune = not validated.held  # a held record is missing from the frames, so it must not be pruned
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
    async with pool.connection() as conn, conn.transaction():
        await control.fence(conn, task)
        held = await quarantine.hold(conn, validated.held)
        for target, _ in transformed.frames:
            await _columns_checked(conn, target)
        for target, data in transformed.frames:
            if isinstance(target.writer, Snapshot) and data.height == 0 and prune:  # held records skip the prune
                raise NothingLoaded(f"{target.table}: a snapshot with no rows would remove every row of the source")
        parents: dict[Target, str] = {}  # a child target's parent's staged table
        for target, data in transformed.frames:
            staged, changed[target.table] = await _write(conn, task.source_id, target, data, parents, frame, prune)
            parents.update((child, staged) for _, child in target.children)
    for table in sent:
        instruments.rows.add(sent[table], {"job": task.job, "stage": "written", "table": table})
        instruments.rows.add(changed[table], {"job": task.job, "stage": "changed", "table": table})
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
    and ``frame.end``. The column lookup always finds the column, because ``build_target`` (``_check_writer``)
    refuses a ``ReplacePartition`` column that is not a target column.

    Preconditions:
        ``conn`` is inside a transaction (``copy_in`` checks this). For ``ReplaceChildren``, ``parents`` has
        ``target`` (``load`` ensures this when frames are in load order).

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
        ValueError: For ``ReplacePartition`` when ``frame`` is ``None``, or when the column is a ``date`` and
            ``frame`` is not whole local days in ``frame.zone``; from the ``sql`` writers for a malformed table name.
        KeyError: For ``ReplaceChildren`` when ``parents`` lacks ``target``.
        RuntimeError: From ``copy_in``, when ``conn`` is not inside a transaction.
        DatabaseConflict: From ``db_errors`` in the callees, on a serialization failure or deadlock.
        DatabaseUnavailable: From ``db_errors`` in the callees, on a lost or refused connection.
        psycopg.Error: Any other database error from the callees, unchanged.

    """
    staged = await copy_in(conn, data, [(column.name, column.postgres) for column in target.columns])
    table, columns = target.table, [column.name for column in target.columns]
    source = source_id if target.source_column is not None else None
    match target.writer:
        case Upsert(key=key):
            changed = await sql.upsert(conn, table, staged, columns, key)
        case ReplaceChildren():
            changed = await sql.replace_children(conn, table, staged, columns, target.inherited, parents[target])
        case ReplacePartition(column=column):
            if frame is None:
                raise ValueError(f"{table}: ReplacePartition needs the task's time frame")
            by_date = next(c.postgres for c in target.columns if c.name == column) == "date"
            bounds = frame.dates if by_date else (frame.start, frame.end)  # local dates for a date column
            if bounds is None:
                raise ValueError(f"{table}: a date partition needs a frame of whole days in {frame.zone}")
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
