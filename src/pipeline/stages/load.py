"""Write one task's transformed frames and held rows to Postgres in a single transaction.

Public API:
    Loaded: Per-table rows sent and changed, and the number of rows held, from one ``load`` call.
    load: Hold the records that share a key or misfit, then fence the task, hold every rejected row and write the rest.

"""

import logging
from collections import Counter
from collections.abc import Callable, Collection, Mapping
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Any, Final
from uuid import UUID
from zoneinfo import ZoneInfo

import polars as pl

from pipeline.checks.checks import BatchPolicy
from pipeline.checks.targets import (
    ReplaceChildren,
    ReplacePartition,
    Snapshot,
    Target,
    Update,
    Upsert,
    Versioned,
    mismatches,
    row_key,
)
from pipeline.db import quarantine
from pipeline.db.catalog import table_info
from pipeline.db.control import ClaimedTask, Fence
from pipeline.db.pool import Connection, Pool
from pipeline.db.quarantine import Failure
from pipeline.errors import KeyViolation, NothingLoaded, TableMismatch
from pipeline.stages.transform import Transformed
from pipeline.stages.validate import Validated, held_accepted, judge
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
    every target it wrote, which is every target but a ``ReplacePartition`` skipped because records were held and a
    container, which has no table.

    Attributes:
        sent: Rows in each written target's frame, keyed by table name (``"schema.table"``).
        changed: The count each written table's ``sql`` writer returned, keyed by table name: the rows it inserted,
            updated, deleted or marked, so a table left unchanged counts 0.
        held: Number of rows ``quarantine.hold`` copied into the hold table: Validate's held rows, then the records
            held for a key shared in their batch or a value that misfits.

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
    *,
    policy: BatchPolicy,  # the job's batch policy, applied again with the records held here
    zone: ZoneInfo | None = None,  # the job's home zone; dates a Versioned target's observation day
    on_loaded: Callable[[Loaded], None] | None = None,  # called inside the transaction once every write is done
) -> Loaded:
    """Hold the records that share a key or misfit, then fence the task, hold every rejected row and write the rest.

    Before anything is written, ``_key_violations`` finds the records whose rows share a row key in a frame, and each
    becomes a held row whose failure's rule is ``KeyViolation``'s ``error_class`` and whose version is its model's
    ``record_version``; Transform's misfits become held rows with their ``Overflow`` failures. Both are built with
    ``held_accepted`` and follow ``validated.held`` in position order, and every row they own is taken out of the
    frames (``_without``), so the rest of the task loads. When it holds any, ``judge`` applies ``policy`` again, to
    ``validated.judged`` followed by these held rows, out of ``validated.judged_total``: they count towards the batch
    policy as Validate's held rows do, and a batch that now breaks it raises ``QualityFailed`` before anything is
    written.

    Everything up to the writes runs on one pooled connection inside one transaction that ``fence.fenced`` opens and
    fences on the task's lease: ``quarantine.hold`` of the held rows, then ``_columns_checked`` for each target, then
    a check that no top-level ``Snapshot`` or ``Update(missing="zero")`` frame is empty when pruning, then ``_write``
    for each pair in ``transformed.frames`` order (parents before their children, at every depth). After a written
    target, its children's ``parents`` entry is the target's staged table. A container (a target with no table) is
    staged with ``copy_in`` but written nowhere, has no ``sent``, ``changed`` or ``stale`` entry and is not compared
    with the catalog; its staged keys are its children's ``parents`` entry, so they are scoped to the container
    records loaded, and its children are written and pruned as any child is. ``prune`` is true only when nothing was
    held; ``_write`` passes it to top-level ``Snapshot``, ``Versioned`` and ``Update`` writers. When records were held,
    a ``ReplacePartition`` target is skipped, not written at all (no ``sent`` or ``changed`` entry), since replacing
    the partition would delete the held records' earlier rows. Children are pruned whether or not records are held,
    and an empty child frame is valid: a held record's rows are taken out of every frame, its parent row included, and
    ``sql.replace_children`` and a ``Versioned`` child's ``sql.versioned`` delete or close only under parents in the
    staged parent table. An ``Upsert`` child never prunes. Once every write is done, the ``Loaded`` result is built
    and passed to ``on_loaded`` while the transaction is still open, so a success report written in the same
    transaction can count it. Any error rolls the whole transaction back. A ``Versioned`` target's observation day
    (its children's too) is ``task.claimed_at``'s date in ``zone``, computed before the transaction opens. The
    metrics run only after commit: for each written table, ``sent`` is added to ``instruments.rows`` with
    ``stage="written"`` and ``changed`` with ``stage="changed"``; for each written table whose ``Versioned`` writer
    left stale keys alone (keys with a version a later day's task wrote), their count is added with ``stage="stale"``
    and a warning gives it; for each skipped table, its frame height is added with ``stage="skipped"`` and a warning
    says it was not written because records are held.
    When records were held, every top-level ``Snapshot`` and ``Update(missing="zero")`` target is left unpruned (a
    ``Versioned`` one included, as a ``Snapshot``; children are never counted here, as they are pruned): for each,
    ``held`` is added with ``stage="prune_skipped"`` and a warning is logged. The records held here are added with
    ``stage="quarantined"``, and a warning per rule gives their count, never their keys.

    Preconditions:
        ``frame`` is not ``None`` when a target uses ``ReplacePartition``. A ``ReplacePartition`` target has no
        children: it has no key, so ``build_target`` refuses ``Rows`` on it. Every frame came from ``frame_of``, so a
        non-nullable column has no nulls; ``Transformed`` checks each frame's columns and types but not its nulls.
        ``Transformed`` guarantees every child target's parent is in ``transformed.frames`` before it, and one owner
        per frame row. ``transformed`` is ``transform``'s result for ``validated.accepted``, so every owner and misfit
        is a position in it.

    Args:
        pool: Pool the connection is taken from.
        task: The claimed task; its ``source_id``, ``id`` and ``job`` are used, and it is the task of the held rows.
        fence: Fences the load transaction on the task's lease (the worker's ``Lease``).
        validated: The validate result; ``held``, ``accepted`` and ``landed`` are read.
        transformed: One (target, frame) pair per target, in load order, with each row's owner and the misfits.
        frame: The task's time frame, or ``None``; only ``ReplacePartition`` targets read it.
        instruments: Instruments whose ``rows`` counter gets the metrics.
        policy: Keyword-only. The job's batch policy, which Validate applied first.
        zone: Keyword-only. The job's home zone; a ``Versioned`` target's observation day is the task's ``claimed_at``
            date in it. ``None``, the default, for a job with no ``Versioned`` target.
        on_loaded: Keyword-only. Called once with the result inside the transaction, after the writes; ``None``, the
            default, calls nothing.

    Returns:
        ``Loaded`` with each written table's frame height and changed count, and the number of rows held.

    Raises:
        QualityFailed: From ``judge``, when Validate's counted held rows and the records held here break
            ``policy``; nothing is written.
        LeaseConflict: From ``fence.fenced``, when the task's lease is no longer held; nothing is written.
        TableMismatch: From ``_columns_checked``, when a target not yet in ``_checked`` does not match its table; a
            target cached earlier in this process is not compared again, so a table altered since fails as a
            ``psycopg.Error`` from the writes instead.
        NothingLoaded: When pruning and a top-level ``Snapshot`` or ``Update(missing="zero")`` target's frame has no
            rows; never for a child.
        ValueError: From ``_write``, for a ``ReplacePartition`` target with no usable frame or with a row outside the
            frame's bounds, or for a ``Versioned`` target when ``zone`` is ``None``; raised inside the transaction, so
            the hold and any earlier writes roll back.
        TypeError: From ``_write`` (``TimeFrame.dates``), for a date partition when ``frame.zone`` is not a ``tzinfo``.
        OverflowError: From ``_write`` (``TimeFrame.dates``), for a date partition whose bounds leave the ``datetime``
            range in ``frame.zone``.
        DatabaseConflict: From ``db_errors`` in the callees, on a serialization failure or deadlock.
        DatabaseUnavailable: From ``db_errors`` in the callees, on a lost or refused connection.
        KeyViolation: From ``db_errors`` in the callees, when a written row violates a unique key of its table.
        psycopg.Error: Any other database error from the callees, unchanged.
        Exception: Whatever ``on_loaded`` raises, after the transaction has rolled back.

    """
    rejected = dict(transformed.misfits)
    for position, table in _key_violations(transformed).items():
        version = type(validated.accepted[position].record).record_version
        rejected[position] = Failure(KeyViolation.error_class, version, f"{table}: another row has the same key")
    later = tuple(held_accepted(task, validated, position, failure) for position, failure in sorted(rejected.items()))
    if later:  # they count towards the batch policy with Validate's, before anything is written
        judge(policy, validated.judged_total, (*validated.judged, *later))
    holding = (*validated.held, *later)
    frames = _without(transformed, rejected.keys())
    prune = not holding  # a held record is missing from the frames, so it must not be pruned
    skipped = {  # would delete held records' earlier rows
        target.table: data.height
        for target, data in frames
        if not prune and target.table is not None and isinstance(target.writer, ReplacePartition)
    }
    sent = {  # a container writes no table, so it sends nothing
        target.table: data.height for target, data in frames if target.table is not None and target.table not in skipped
    }
    unpruned = [  # top-level targets only: a child is pruned under its staged parents
        target.table
        for target, _ in frames
        if not prune
        and target.table is not None
        and not target.inherited
        and (
            isinstance(target.writer, Snapshot)
            or (isinstance(target.writer, Update) and target.writer.missing == "zero")
        )
    ]
    changed: dict[str, int] = {}
    stale: dict[str, int] = {}
    observed = task.claimed_at.astimezone(zone).date() if zone is not None else None  # a Versioned target's day
    async with pool.connection() as conn, fence.fenced(conn):
        held = await quarantine.hold(conn, holding)
        for target, _ in frames:
            await _columns_checked(conn, target)
        for target, data in frames:
            if target.inherited:  # a child: an empty frame is valid, and its prune is scoped to its staged parents
                continue
            if isinstance(target.writer, Snapshot) and data.height == 0 and prune:  # held records skip the prune
                raise NothingLoaded(f"{target.table}: a snapshot with no rows would remove every row of the source")
            zeroing = isinstance(target.writer, Update) and target.writer.missing == "zero"
            if zeroing and data.height == 0 and prune:
                raise NothingLoaded(f"{target.table}: an update with no rows would zero every row of the source")
        parents: dict[Target, str] = {}  # a child target's parent's staged table
        for target, data in frames:
            if target.table in skipped:
                continue
            if target.table is None:  # a container: its keys are staged for its children, and nothing is written
                staged = await copy_in(conn, data, [(column.name, column.postgres) for column in target.columns])
            else:
                staged, changed[target.table], stale[target.table] = await _write(
                    conn, task.source_id, target, data, parents, frame, prune, observed
                )
            parents.update((child, staged) for _, child in target.children)
        loaded = Loaded(sent=sent, changed=changed, held=held)
        if on_loaded is not None:
            on_loaded(loaded)
    for table in sent:
        instruments.rows.add(sent[table], {"job": task.job, "stage": "written", "table": table})
        instruments.rows.add(changed[table], {"job": task.job, "stage": "changed", "table": table})
    for table, count in stale.items():
        if count:
            instruments.rows.add(count, {"job": task.job, "stage": "stale", "table": table})
            _log.warning(
                "task %s %s: %s left %d versions alone, observed before they started", task.id, task.job, table, count
            )
    for table in skipped:
        instruments.rows.add(skipped[table], {"job": task.job, "stage": "skipped", "table": table})
        _log.warning("task %s %s: %s not written because %d records are held", task.id, task.job, table, held)
    for table in unpruned:
        instruments.rows.add(held, {"job": task.job, "stage": "prune_skipped", "table": table})
        _log.warning("task %s %s: %s not pruned, %d records held", task.id, task.job, table, held)
    if rejected:
        instruments.rows.add(len(rejected), {"job": task.job, "stage": "quarantined"})
    for rule, count in sorted(Counter(failure.rule for failure in rejected.values()).items()):
        _log.warning("task %s %s: %d quarantined by %s at load", task.id, task.job, count, rule)
    return loaded


def _key_violations(transformed: Transformed) -> dict[int, str]:
    """Find the records whose rows share a row key with another row of the same frame.

    For each frame whose target has a table and a row key (``row_key``), every row whose key columns equal another
    row's is found, and the record that owns it is named; a record owning such rows in several frames is named for the
    first. A container's frame is skipped: its key is the record key, on which Validate already keeps one record.

    Preconditions:
        ``transformed`` gives one owner per frame row, as ``Transformed`` checks.

    Args:
        transformed: The frames to check, with each row's owner.

    Returns:
        Each such record's position in the accepted records, mapped to the table of the first frame its rows share a
        key in; empty when no key is shared.

    Raises:
        Nothing.

    """
    found: dict[int, str] = {}
    for (target, data), owners in zip(transformed.frames, transformed.owners, strict=True):
        key = row_key(target)
        if key is None or target.table is None or data.height < 2:  # a container's key is the record key
            continue
        shared = data.select(key).is_duplicated()
        for owner, duplicate in zip(owners, shared, strict=True):
            if duplicate:
                found.setdefault(owner, target.table)
    return found


def _without(transformed: Transformed, positions: Collection[int]) -> list[tuple[Target, pl.DataFrame]]:
    """Return the frames without the rows the given records own.

    Preconditions:
        ``transformed`` gives one owner per frame row, as ``Transformed`` checks.

    Args:
        transformed: The frames and each row's owner.
        positions: The positions, in the accepted records, of the records whose rows are left out.

    Returns:
        One ``(target, frame)`` pair per frame of ``transformed``, in order; a frame is returned as it is when none of
        its rows is left out.

    Raises:
        Nothing.

    """
    if not positions:
        return list(transformed.frames)
    return [
        (target, data.filter(pl.Series([owner not in positions for owner in owners], dtype=pl.Boolean)))
        for (target, data), owners in zip(transformed.frames, transformed.owners, strict=True)
    ]


async def _write(
    conn: Connection,
    source_id: UUID,
    target: Target,
    data: pl.DataFrame,
    parents: Mapping[Target, str],
    frame: TimeFrame | None,
    prune: bool,
    observed: date | None = None,
) -> tuple[str, int, int]:
    """Stage one frame in a temporary table and write it to the target's table with the target's writer.

    The frame is copied in with each target column's Postgres type. The task's source is passed only when the target
    has a source column. Each writer calls its ``sql`` function: ``Upsert``, ``ReplaceChildren``,
    ``ReplacePartition``, ``Versioned``, ``Snapshot`` (with ``prune``) and ``Update`` (with its ``missing`` and
    ``prune``). ``Upsert``, ``ReplaceChildren`` and ``Versioned`` are keyed by the target's ``row_key``, which for a
    child may differ from its writer's ``key``. ``Versioned`` is matched before ``Snapshot``, its base. Its
    ``valid_from`` column is not a target column, so its type is read live from the catalog with ``table_info`` on
    every call (one extra query); ``observed`` and the day before it are passed to ``sql.versioned`` in that type
    (``_day_value``). A top-level ``Versioned`` target is pruned only with ``prune``, across its source's rows (the
    whole table when it has no source column); a child (a target with inherited columns) is always pruned, scoped to
    its parent's staged table through its inherited columns. When the
    table or the column is missing from the catalog, the type falls back to ``"date"``; that is reachable only for a
    table altered after ``_columns_checked`` cached its target, and the ``sql`` writer then fails. For
    ``ReplacePartition``, a ``date`` column is bounded by ``frame.dates`` and a timestamptz column by ``frame.start``
    and ``frame.end``, and no row's partition value may lie outside ``[start, end)``; both are checked before the
    frame is staged. A null partition value is not counted as outside. The column lookup always finds the column,
    because ``build_target`` (``_check_writer``) refuses a ``ReplacePartition`` column that is not a target column.

    Preconditions:
        ``conn`` is inside a transaction (``copy_in`` checks this). ``target`` has a table: it is not a container. For
        a child target (``ReplaceChildren``, or ``Versioned`` with inherited columns), ``parents`` has ``target``
        (``load`` ensures this when the parent, a container included, is in the frames before the child). For
        ``ReplacePartition``, the partition column has no nulls (``_check_writer`` makes it non-nullable and
        ``frame_of`` refuses ``None`` in a non-nullable column). For ``Versioned``, ``valid_from`` is ``integer``,
        ``bigint`` or ``date`` (``mismatches``, run by ``_columns_checked`` before the writes).

    Args:
        conn: Connection inside the load transaction.
        source_id: The task's source.
        target: The target whose table is written.
        data: The target's frame, columns in ``target.columns`` order.
        parents: Staged table of each child target's parent, keyed by the child target.
        frame: The task's time frame, or ``None``; read only for ``ReplacePartition``.
        prune: Passed to ``sql.snapshot``, ``sql.update`` and, for a top-level target, ``sql.versioned``.
        observed: The task's observation day in the job's zone, required for a ``Versioned`` target; ``None``, the
            default, otherwise.

    Returns:
        The staged table's name, the count the ``sql`` writer returned, and the stale versions a ``Versioned``
        writer left alone (0 for every other writer).

    Raises:
        ValueError: For ``ReplacePartition`` when ``frame`` is ``None``, when the column is a ``date`` and ``frame`` is
            not whole local days in ``frame.zone``, or when a row's partition value lies outside ``[start, end)``
            (naming the table, the column, the number of such rows and the bounds); for ``Versioned`` when
            ``observed`` is ``None``, before any catalog read (the frame is already staged); from the ``sql``
            writers for a malformed table name (unreachable: ``build_target`` checks ``target.table`` against
            ``TABLE_PATTERN``); for a container (``target.table`` None), before anything is staged.
        TypeError: From ``TimeFrame.dates``, for a date column when ``frame.zone`` is not a ``tzinfo``.
        OverflowError: From ``TimeFrame.dates``, for a date column when a bound in ``frame.zone`` leaves the
            ``datetime`` range.
        KeyError: For ``ReplaceChildren`` or a ``Versioned`` child when ``parents`` lacks ``target``.
        RuntimeError: From ``copy_in``, when ``conn`` is not inside a transaction.
        DatabaseConflict: From ``db_errors`` in the callees, on a serialization failure or deadlock.
        DatabaseUnavailable: From ``db_errors`` in the callees, on a lost or refused connection.
        KeyViolation: From ``db_errors`` in the callees, when a written row violates a unique key of its table.
        psycopg.Error: Any other database error from the callees (``table_info`` included), unchanged.

    """
    table, columns = target.table, [column.name for column in target.columns]
    if table is None:  # load stages a container's keys itself, so this is never reached from load
        raise ValueError("a container has no table to write; only its children are written")
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
    key = row_key(target) or ()  # () only for ReplacePartition, which has none
    match target.writer:
        case Upsert():
            changed = await sql.upsert(conn, table, staged, columns, key)
        case ReplaceChildren():
            changed = await sql.replace_children(conn, table, staged, columns, target.inherited, key, parents[target])
        case ReplacePartition(column=column) if bounds is not None:  # bounds is always set for ReplacePartition
            changed = await sql.replace_partition(
                conn, table, staged, columns, target.source_column, source, column, *bounds
            )
        case Versioned(valid_from=first, valid_to=last, opens_from=opens):
            if observed is None:
                raise ValueError(f"{table}: a Versioned target needs the job's zone to date its observation day")
            info = await table_info(conn, table)
            day_type = info.columns[first].type if info is not None and first in info.columns else "date"
            day, previous = _day_value(observed, day_type), _day_value(observed - timedelta(days=1), day_type)
            under = parents[target] if target.inherited else None  # a child prunes only under its staged parents
            changed, stale = await sql.versioned(
                conn,
                table,
                staged,
                columns,
                key,
                target.source_column,
                source,
                first,
                last,
                opens,
                day,
                previous,
                prune or under is not None,
                target.inherited,
                under,
            )
            return staged, changed, stale
        case Snapshot():
            changed = await sql.snapshot(conn, table, staged, columns, key, target.source_column, source, prune)
        case Update(missing=missing):
            changed = await sql.update(conn, table, staged, columns, key, target.source_column, source, missing, prune)
    return staged, changed, 0


def _day_value(day: date, column_type: str) -> date | int:
    """Return a day in the type of a ``Versioned`` table's ``valid_from`` column: a ``YYYYMMDD`` integer or the date.

    Any type other than ``"integer"`` or ``"bigint"`` returns the date unchanged; ``mismatches`` limits
    ``valid_from`` to those two and ``"date"``. Every ``YYYYMMDD`` value (at most 99991231) fits an ``integer``.

    Preconditions:
        None.

    Args:
        day: The day to convert.
        column_type: The column's type as ``table_info`` gives it (``format_type``), e.g. ``"integer"`` or ``"date"``.

    Returns:
        ``day.year * 10000 + day.month * 100 + day.day`` for ``"integer"`` or ``"bigint"``, else ``day`` itself.

    Raises:
        Nothing.

    """
    return day.year * 10000 + day.month * 100 + day.day if column_type in ("integer", "bigint") else day


async def _columns_checked(conn: Connection, target: Target) -> None:
    """Compare a target with its table once per worker process and refuse a target that does not match.

    A target already in ``_checked``, or a container (which has no table), returns at once. Otherwise its table's
    catalog entry is fetched with ``table_info`` and compared with ``mismatches``; a target that matches is added to
    ``_checked``, so later calls in this process skip the lookup. A mismatched target is not cached and is checked again
    on its next call.

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
    if target in _checked or target.table is None:  # a container has no table to compare
        return
    found = mismatches(target, await table_info(conn, target.table))
    if found:
        raise TableMismatch("; ".join(found))
    _checked.add(target)
