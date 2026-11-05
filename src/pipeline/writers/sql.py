"""Merge a staged table into a warehouse table with one of five SQL write strategies.

Public API:
    upsert: Insert staged rows and update existing rows by key where their other columns differ.
    replace_children: Delete the rows of listed parents missing from the stage, then upsert the staged rows by key.
    replace_partition: Delete the rows in one source's half-open range of a column, then insert the staged rows.
    snapshot: Upsert staged rows, clear their ``removed_at``, and optionally mark rows missing from the stage removed.
    versioned: Keep day-ranged versions per key: rewrite or drop today's, close older changed or missing ones.
    update: Update existing rows by key from the stage, and optionally zero rows missing from the stage.

"""

from collections.abc import Sequence
from datetime import date, datetime
from typing import Literal
from uuid import UUID

from psycopg import sql

from pipeline.db.pool import Connection
from pipeline.errors import db_errors


async def upsert(conn: Connection, table: str, staged: str, columns: Sequence[str], key: Sequence[str]) -> int:
    """Insert every staged row into the table, updating the existing row on a key conflict.

    On a conflict, the non-key columns are overwritten only when at least one of them differs (``IS DISTINCT FROM``);
    when every column is a key column, a conflicting row is left as it is.

    Preconditions:
        ``table`` is ``"schema.name"``. ``key`` is non-empty, a subset of ``columns``, and exactly the columns of a
        unique index or constraint on the table. ``staged`` has every name in ``columns`` and no two rows with the
        same key (else Postgres raises ``CardinalityViolation``).

    Args:
        conn: Connection the statement runs on; the caller owns the transaction.
        table: Target table as ``"schema.name"``.
        staged: Unqualified name of the staged table.
        columns: Columns copied from the stage to the table.
        key: Conflict columns.

    Returns:
        The number of rows inserted plus the number of existing rows updated.

    Raises:
        ValueError: When ``table`` does not contain exactly one ``.``.
        DatabaseConflict: From ``db_errors``, on a serialization failure or deadlock.
        DatabaseUnavailable: From ``db_errors``, on a lost or refused connection.
        KeyViolation: From ``db_errors``, when a written row violates a unique key of the table.
        psycopg.Error: Any other database error, unchanged.

    """
    return await _changed(conn, _upsert(table, staged, columns, key, clear_removed=False))


async def replace_children(
    conn: Connection,
    table: str,
    staged: str,
    columns: Sequence[str],
    inherited: Sequence[str],
    key: Sequence[str],
    parents: str,
) -> int:
    """Make each listed parent's rows in the table equal its staged rows, touching only rows that change.

    First every table row whose parent key (``inherited``) is listed in ``parents`` and whose row key (``key``) has
    no staged row is deleted; then every staged row is upserted on ``key`` as ``upsert`` does, so an existing row is
    updated only when a non-key column differs. Re-running unchanged data therefore deletes, inserts and updates
    nothing. The two statements run on the same connection; they are atomic only if the caller's transaction makes
    them so. The parent key is read from ``parents`` qualified, so a key column missing from ``parents`` raises
    ``psycopg.errors.UndefinedColumn`` rather than matching the table's own column.

    Preconditions:
        ``table`` is ``"schema.name"``. ``inherited`` is non-empty and names the parent-key columns of ``table``, and
        ``parents`` has a column of each of those names (Load passes the parent's staged table, and ``build_target``
        makes the child's inherited columns the parent's writer key, which the parent's columns include). ``key`` is
        ``inherited`` followed by the child's own key columns and is exactly the columns of a unique index or
        constraint on the table (``row_key``; ``load`` checks the table with ``mismatches``). ``staged`` has every name
        in ``columns``, no two rows with the same ``key`` (else Postgres raises ``CardinalityViolation``), and each
        staged row belongs to a parent in ``parents`` (rows of any other parent are upserted beside the existing ones).

    Args:
        conn: Connection the statements run on; the caller owns the transaction.
        table: Target table as ``"schema.name"``.
        staged: Unqualified name of the staged table.
        columns: Columns copied from the stage to the table.
        inherited: Parent-key columns, matched by name in ``table`` and ``parents``.
        key: Row-key columns: the conflict columns, also used to find the rows missing from the stage.
        parents: Unqualified name of the table listing the parents to replace.

    Returns:
        The number of rows deleted, plus the number of rows inserted or updated.

    Raises:
        ValueError: When ``table`` does not contain exactly one ``.``.
        DatabaseConflict: From ``db_errors``, on a serialization failure or deadlock.
        DatabaseUnavailable: From ``db_errors``, on a lost or refused connection.
        KeyViolation: From ``db_errors``, when an upserted row violates another unique key of the table.
        psycopg.Error: Any other database error, unchanged.

    """
    delete = sql.SQL("DELETE FROM {table} AS t WHERE {under} AND {missing}").format(
        table=_table(table), under=_under(inherited, parents), missing=_missing(staged, key)
    )
    deleted = await _changed(conn, delete)
    return deleted + await _changed(conn, _upsert(table, staged, columns, key, clear_removed=False))


async def replace_partition(
    conn: Connection,
    table: str,
    staged: str,
    columns: Sequence[str],
    source_column: str | None,
    source_id: UUID | None,
    column: str,
    start: datetime | date,  # a date for a date column, so the session's time zone plays no part
    end: datetime | date,
) -> int:
    """Delete the source's rows with ``start <= column < end``, then insert every staged row.

    The two statements run on the same connection; they are atomic only if the caller's transaction makes them so.

    Preconditions:
        ``table`` is ``"schema.name"``. ``source_id`` is not ``None`` when ``source_column`` is set (``load._write``
        passes the task's source exactly when the target has a source column, and a task's source is never None).
        ``staged`` has every name in ``columns``, and every staged row lies in the source and in ``[start, end)``
        (any other row is added beside the existing ones). ``start`` and ``end`` are dates for a date column and
        aware datetimes for a ``timestamptz`` column (Load passes ``TimeFrame`` bounds, which are aware UTC
        datetimes, or the frame's local dates for a date column).

    Args:
        conn: Connection the statements run on; the caller owns the transaction.
        table: Target table as ``"schema.name"``.
        staged: Unqualified name of the staged table.
        columns: Columns copied from the stage to the table.
        source_column: Column holding the source, or ``None`` for a table that is not per source.
        source_id: Source whose rows are replaced; ignored when ``source_column`` is ``None``.
        column: Column the range applies to.
        start: Inclusive lower bound.
        end: Exclusive upper bound.

    Returns:
        The number of rows deleted plus the number of rows inserted.

    Raises:
        ValueError: When ``table`` does not contain exactly one ``.``.
        DatabaseConflict: From ``db_errors``, on a serialization failure or deadlock.
        DatabaseUnavailable: From ``db_errors``, on a lost or refused connection.
        KeyViolation: From ``db_errors``, when a written row violates a unique key of the table.
        psycopg.Error: Any other database error, unchanged.

    """
    source, params = _source(source_column, source_id)
    delete = sql.SQL("DELETE FROM {table} AS t WHERE {source} AND {column} >= %s AND {column} < %s").format(
        table=_table(table), source=source, column=sql.Identifier("t", column)
    )
    deleted = await _changed(conn, delete, (*params, start, end))
    return deleted + await _changed(conn, _insert(table, staged, columns))


async def snapshot(
    conn: Connection,
    table: str,
    staged: str,
    columns: Sequence[str],
    key: Sequence[str],
    source_column: str | None,
    source_id: UUID | None,
    prune: bool,
) -> int:
    """Upsert every staged row with ``removed_at`` cleared, then optionally mark the source's missing rows removed.

    A conflicting row is updated when a non-key column differs or its ``removed_at`` is set. With ``prune``, every
    row of the source whose key is not in the stage and whose ``removed_at`` is null gets ``removed_at = now()``; an
    empty stage therefore marks every row of the source.

    Preconditions:
        As for ``upsert``. The table has a ``removed_at`` column and ``columns`` does not include it.
        ``source_column``, when set, is in ``key`` (``checks.targets`` refuses a writer key without it), so the prune
        compares keys within one source. ``source_id`` is not ``None`` when ``source_column`` is set (``load._write``
        passes the task's source exactly when the target has a source column, and a task's source is never None).

    Args:
        conn: Connection the statements run on; the caller owns the transaction.
        table: Target table as ``"schema.name"``.
        staged: Unqualified name of the staged table.
        columns: Columns copied from the stage to the table.
        key: Conflict columns, also used to find rows missing from the stage.
        source_column: Column holding the source, or ``None`` to prune across the whole table.
        source_id: Source to prune; ignored when ``source_column`` is ``None``.
        prune: Whether to mark rows missing from the stage removed.

    Returns:
        The number of rows inserted or updated, plus the number of rows marked removed.

    Raises:
        ValueError: When ``table`` does not contain exactly one ``.``.
        DatabaseConflict: From ``db_errors``, on a serialization failure or deadlock.
        DatabaseUnavailable: From ``db_errors``, on a lost or refused connection.
        KeyViolation: From ``db_errors``, when a written row violates a unique key of the table.
        psycopg.Error: Any other database error, unchanged.

    """
    changed = await _changed(conn, _upsert(table, staged, columns, key, clear_removed=True))
    if not prune:
        return changed
    source, params = _source(source_column, source_id)
    mark = sql.SQL(
        "UPDATE {table} AS t SET removed_at = now() WHERE {source} AND {missing} AND t.removed_at IS NULL"
    ).format(table=_table(table), source=source, missing=_missing(staged, key))
    return changed + await _changed(conn, mark, params)


async def versioned(
    conn: Connection,
    table: str,
    staged: str,
    columns: Sequence[str],
    key: Sequence[str],
    source_column: str | None,
    source_id: UUID | None,
    valid_from: str,
    valid_to: str,
    opens_from: str | None,
    day: date | int,
    previous: date | int,
    prune: bool,
    inherited: Sequence[str] = (),
    parents: str | None = None,
) -> tuple[int, int]:
    """Make each staged key's open version equal its staged row; only versions opened on the day change in place.

    ``day`` is the observation day D and ``previous`` the day before it, both in the type of ``valid_from`` (a
    ``YYYYMMDD`` integer or a ``date``). An open version is a row whose ``valid_to`` is null. Rows differ when any
    column of ``columns`` outside ``key`` is distinct (``IS DISTINCT FROM``, so nulls compare equal); a key-only
    table never differs and skips steps 2 and 3. Keys compare with ``ROW(...) = ROW(...)``, which the non-nullable
    writer key (``checks.targets``) keeps exact.

    A version that reaches past ``previous`` (starts after D, or was closed on D or later) can only have been
    written by a later day's task: a task on D closes on ``previous`` and drops, rather than closes, a version it
    opened on D. A staged key with such a version is stale. In order:

    1. Count the stale staged keys. Non-overlapping versions mean a stale key has no open version starting on or
       before D, so steps 2 and 3 never touch it, step 4 only touches unstaged keys and step 5 skips it: the count
       holds after them and these keys are left alone.
    2. Overwrite, in place, the non-key columns of the open versions that started on D and differ from their
       staged row (the day's last observation wins).
    3. Close on ``previous`` the open versions that started before D and differ from their staged row.
    4. With ``prune``, for the source's open versions whose key is not staged: delete those that started on D (the
       day's last observation is that the key is gone, so the day keeps no version of it) and close on ``previous``
       those that started before D. With ``parents`` (a child table), only the open versions whose ``inherited``
       columns match a row of ``parents`` are pruned, as ``replace_children`` scopes its delete: a key under a parent
       that is not staged is left as it is.
    5. Insert a version for every staged key that is not stale and has no open version left: from D when the key
       has any version at all, else from its ``opens_from`` value (D when ``opens_from`` is None or the value is
       null), but never after D. Every version of a non-stale key ends on or before ``previous``, so the new one
       neither shares ``valid_from`` with nor overlaps any of them.

    Closing runs before inserting, so a non-deferrable exclusion constraint on a key's version ranges never sees
    two open versions.

    Step 4 does not check for later days: an unstaged key's open version that started before D is closed on
    ``previous`` even when a later day's task has already run and seen the key (an unchanged key leaves no row dated
    after D), so a late or retried task for an earlier day ends a version that is still current, in a child table
    too (todo.txt 17).

    Preconditions:
        As for ``snapshot``, except that the table has no ``removed_at`` and several rows per key: ``valid_from`` and
        ``valid_to`` are its day columns and ``columns`` includes neither. ``staged`` has no two rows with one key
        (Load holds key violations first). ``day`` and ``previous`` match ``valid_from``'s type, and ``opens_from``,
        when set, is in ``columns`` and of that type (``checks.targets`` checks the table with ``mismatches``). With
        ``prune`` and no ``parents``, ``staged`` is not empty (Load raises ``NothingLoaded`` first, as ``Versioned`` is
        a ``Snapshot``); with ``parents``, ``staged`` may be empty, so every open version under a listed parent is
        closed or dropped. With ``parents``, ``inherited`` is non-empty and names columns of ``table`` and of
        ``parents``. A staged row need not belong to a parent in ``parents``: it is still versioned by steps 2, 3 and
        5, but the keys missing under its parent are not pruned.

    Args:
        conn: Connection the statements run on; the caller owns the transaction.
        table: Target table as ``"schema.name"``.
        staged: Unqualified name of the staged table.
        columns: Columns copied from the stage to the table.
        key: Columns that identify a key's versions, the source column included when the table has one (a child
            table's row key, ``checks.targets.row_key``).
        source_column: Column holding the source, or ``None`` to prune across the whole table (or, with ``parents``,
            across the listed parents' rows).
        source_id: Source to prune; ignored when ``source_column`` is ``None``.
        valid_from: Column holding a version's first day.
        valid_to: Column holding a version's last day, null while it is open.
        opens_from: Column dating a new key's first version, or ``None``.
        day: The observation day.
        previous: The day before ``day``.
        prune: Whether to close the open versions of keys missing from the stage.
        inherited: Parent-key columns, matched by name in ``table`` and ``parents``; read only with ``parents``.
            Defaults to none.
        parents: Unqualified name of the table listing the parents whose versions may be pruned, or ``None``, the
            default, to prune by source alone.

    Returns:
        The rows changed (overwritten, closed, deleted and inserted, so a changed key counts twice) and the number of
        stale staged keys.

    Raises:
        ValueError: When ``table`` does not contain exactly one ``.``.
        DatabaseConflict: From ``db_errors``, on a serialization failure or deadlock.
        DatabaseUnavailable: From ``db_errors``, on a lost or refused connection.
        KeyViolation: From ``db_errors``, when a written row violates a unique key of the table.
        psycopg.Error: Any other database error, unchanged.

    """
    t_table, first, last = _table(table), sql.Identifier(valid_from), sql.Identifier(valid_to)
    same = sql.SQL("ROW({s}) = ROW({t})").format(s=_names(key, "s"), t=_names(key, "t"))
    details = [column for column in columns if column not in key]
    differs = (
        sql.SQL("ROW({t}) IS DISTINCT FROM ROW({s})").format(t=_names(details, "t"), s=_names(details, "s"))
        if details
        else sql.SQL("FALSE")  # a key-only table has nothing that can change
    )
    # a version reaching past the day before D was written by a later day's task: the staged row is older than it
    later = sql.SQL("EXISTS (SELECT FROM {table} AS t WHERE {same} AND (t.{first} > %s OR t.{last} >= %s))").format(
        table=t_table, same=same, first=first, last=last
    )
    stale_query = sql.SQL("SELECT count(*) FROM {staged} AS s WHERE {later}").format(
        staged=sql.Identifier(staged), later=later
    )
    with db_errors():
        cursor = await conn.execute(stale_query, (day, day))
        row = await cursor.fetchone()
    stale = int(row[0]) if row is not None else 0
    changed = 0
    if details:
        overwrite = sql.SQL(
            "UPDATE {table} AS t SET ({cols}) = ROW({values}) FROM {staged} AS s "
            "WHERE {same} AND t.{last} IS NULL AND t.{first} = %s AND {differs}"
        ).format(
            table=t_table,
            cols=_names(details),
            values=_names(details, "s"),
            staged=sql.Identifier(staged),
            same=same,
            last=last,
            first=first,
            differs=differs,
        )
        changed += await _changed(conn, overwrite, (day,))
        close = sql.SQL(
            "UPDATE {table} AS t SET {last} = %s FROM {staged} AS s "
            "WHERE {same} AND t.{last} IS NULL AND t.{first} < %s AND {differs}"
        ).format(table=t_table, last=last, staged=sql.Identifier(staged), same=same, first=first, differs=differs)
        changed += await _changed(conn, close, (previous, day))
    if prune:
        source, params = _source(source_column, source_id)
        if parents is not None:  # a child: only under the parents staged with it
            source = sql.SQL("{source} AND {under}").format(source=source, under=_under(inherited, parents))
        missing = _missing(staged, key)
        drop_today = sql.SQL(
            "DELETE FROM {table} AS t WHERE {source} AND t.{last} IS NULL AND t.{first} = %s AND {missing}"
        ).format(table=t_table, source=source, last=last, first=first, missing=missing)
        changed += await _changed(conn, drop_today, (*params, day))
        close_missing = sql.SQL(
            "UPDATE {table} AS t SET {last} = %s WHERE {source} AND t.{last} IS NULL AND t.{first} < %s AND {missing}"
        ).format(table=t_table, last=last, source=source, first=first, missing=missing)
        changed += await _changed(conn, close_missing, (previous, *params, day))
    opening = (
        sql.SQL("COALESCE(s.{opens}, %s)").format(opens=sql.Identifier(opens_from)) if opens_from else sql.SQL("%s")
    )
    insert = sql.SQL(
        "INSERT INTO {table} ({cols}, {first}) SELECT {values}, "
        "CASE WHEN EXISTS (SELECT FROM {table} AS t WHERE {same}) THEN %s ELSE LEAST({opening}, %s) END "
        "FROM {staged} AS s WHERE NOT EXISTS (SELECT FROM {table} AS t WHERE {same} AND t.{last} IS NULL) "
        "AND NOT {later}"
    ).format(
        table=t_table,
        cols=_names(columns),
        first=first,
        values=_names(columns, "s"),
        same=same,
        opening=opening,
        staged=sql.Identifier(staged),
        last=last,
        later=later,
    )
    changed += await _changed(conn, insert, (day, day, day, day, day))
    return changed, stale


async def update(
    conn: Connection,
    table: str,
    staged: str,
    columns: Sequence[str],
    key: Sequence[str],
    source_column: str | None,
    source_id: UUID | None,
    missing: Literal["keep", "zero"],
    prune: bool,
) -> int:
    """Update existing rows from the stage by key, then optionally set the source's missing rows' columns to zero.

    Staged rows with no matching table row are ignored; nothing is inserted. A matching row is updated only when a
    non-key column differs. Missing rows are zeroed only when ``prune`` is true and ``missing`` is ``"zero"``.
    There are two arguments because ``missing`` is the table's declared policy and ``prune`` is this task's
    permission: Load passes ``prune=False`` when the task held a record, so ``missing="zero", prune=False`` is the
    held-records case and leaves missing rows as they are.

    Preconditions:
        ``table`` is ``"schema.name"``. ``key`` is non-empty and ``columns`` has at least one name not in ``key``
        (``checks.targets`` refuses an Update model with no column outside its key when the class is defined).
        ``staged`` has every name in ``columns`` and ``key`` and no two rows with the same key (else which staged
        row wins is undefined). When zeroing, every non-key column accepts ``0`` (``checks.targets`` refuses a
        ``missing="zero"`` column outside the key that is not ``bigint`` or ``numeric(p,s)``). ``source_column``, when
        set, is in ``key`` (``checks.targets`` refuses a writer key without it), so it is never written or zeroed.
        ``source_id`` is not ``None`` when ``source_column`` is set (``load._write`` passes the task's source exactly
        when the target has a source column, and a task's source is never None).

    Args:
        conn: Connection the statements run on; the caller owns the transaction.
        table: Target table as ``"schema.name"``.
        staged: Unqualified name of the staged table.
        columns: Columns the stage supplies; those not in ``key`` are the ones written.
        key: Columns that match a staged row to a table row.
        source_column: Column holding the source, or ``None`` to zero across the whole table.
        source_id: Source to zero; ignored when ``source_column`` is ``None``.
        missing: ``"zero"`` to zero rows missing from the stage, ``"keep"`` to leave them.
        prune: Whether rows missing from the stage are handled at all.

    Returns:
        The number of rows updated from the stage, plus the number of rows zeroed.

    Raises:
        ValueError: When ``table`` does not contain exactly one ``.``.
        DatabaseConflict: From ``db_errors``, on a serialization failure or deadlock.
        DatabaseUnavailable: From ``db_errors``, on a lost or refused connection.
        KeyViolation: From ``db_errors``, when a written row violates a unique key of the table.
        psycopg.Error: Any other database error, unchanged.

    """
    owned = [name for name in columns if name not in key]
    query = sql.SQL(
        "UPDATE {table} AS t SET {sets} FROM {staged} AS s WHERE ROW({t_key}) = ROW({s_key}) "
        "AND ROW({t_owned}) IS DISTINCT FROM ROW({s_owned})"
    ).format(
        table=_table(table),
        sets=sql.SQL(", ").join(sql.SQL("{} = {}").format(sql.Identifier(c), sql.Identifier("s", c)) for c in owned),
        staged=sql.Identifier(staged),
        t_key=_names(key, "t"),
        s_key=_names(key, "s"),
        t_owned=_names(owned, "t"),
        s_owned=_names(owned, "s"),
    )
    changed = await _changed(conn, query)
    if not prune or missing == "keep":
        return changed
    source, params = _source(source_column, source_id)
    zero = sql.SQL(
        "UPDATE {table} AS t SET ({owned}) = ROW({zeros}) "
        "WHERE {source} AND {missing} AND ROW({t_owned}) IS DISTINCT FROM ROW({zeros})"
    ).format(
        table=_table(table),
        owned=_names(owned),
        zeros=sql.SQL(", ").join(sql.SQL("0") for _ in owned),
        source=source,
        missing=_missing(staged, key),
        t_owned=_names(owned, "t"),
    )
    return changed + await _changed(conn, zero, params)


def _upsert(
    table: str, staged: str, columns: Sequence[str], key: Sequence[str], *, clear_removed: bool
) -> sql.Composed:
    """Build an ``INSERT ... SELECT ... ON CONFLICT`` statement from the stage into the table aliased ``t``.

    The update sets every non-key column from ``EXCLUDED`` and, with ``clear_removed``, sets ``removed_at`` to
    null; it applies only where a non-key column differs or, with ``clear_removed``, ``t.removed_at`` is set. With
    nothing to set, the action is ``DO NOTHING``.

    Preconditions:
        As for ``upsert``. With ``clear_removed``, the table has a ``removed_at`` column and ``columns`` does not
        include it (``checks.targets`` refuses a Snapshot model that loads into ``removed_at`` when the class is
        defined).

    Args:
        table: Target table as ``"schema.name"``.
        staged: Unqualified name of the staged table.
        columns: Columns copied from the stage to the table.
        key: Conflict columns.
        clear_removed: Whether a conflicting row also gets ``removed_at = NULL``.

    Returns:
        The composed statement, without parameters.

    Raises:
        ValueError: When ``table`` does not contain exactly one ``.``.

    """
    owned = [name for name in columns if name not in key]
    sets: list[sql.Composable] = [sql.SQL("{c} = EXCLUDED.{c}").format(c=sql.Identifier(c)) for c in owned]
    differs: list[sql.Composable] = []
    if owned:
        differs.append(
            sql.SQL("ROW({}) IS DISTINCT FROM ROW({})").format(_names(owned, "t"), _names(owned, "excluded"))
        )
    if clear_removed:  # a row the connector returns again is no longer removed
        sets.append(sql.SQL("removed_at = NULL"))
        differs.append(sql.SQL("t.removed_at IS NOT NULL"))
    action = (
        sql.SQL("DO UPDATE SET {} WHERE {}").format(sql.SQL(", ").join(sets), sql.SQL(" OR ").join(differs))
        if sets
        else sql.SQL("DO NOTHING")
    )
    return sql.SQL("{insert} ON CONFLICT ({key}) {action}").format(
        insert=_insert(table, staged, columns, alias="t"), key=_names(key), action=action
    )


def _insert(table: str, staged: str, columns: Sequence[str], alias: str | None = None) -> sql.Composed:
    """Build an ``INSERT INTO table (columns) SELECT columns FROM staged`` statement.

    Preconditions:
        ``table`` is ``"schema.name"`` and ``columns`` is non-empty.

    Args:
        table: Target table as ``"schema.name"``.
        staged: Unqualified name of the staged table.
        columns: Columns copied, in the same order on both sides.
        alias: Alias for the target table, or ``None`` (or empty) for none.

    Returns:
        The composed statement, without parameters.

    Raises:
        ValueError: When ``table`` does not contain exactly one ``.``.

    """
    target = sql.SQL("{} AS {}").format(_table(table), sql.Identifier(alias)) if alias else _table(table)
    return sql.SQL("INSERT INTO {target} ({columns}) SELECT {columns} FROM {staged}").format(
        target=target, columns=_names(columns), staged=sql.Identifier(staged)
    )


def _source(source_column: str | None, source_id: UUID | None) -> tuple[sql.Composable, tuple[object, ...]]:
    """Build the condition that limits a statement on the table aliased ``t`` to one source.

    Preconditions:
        ``source_id`` is not ``None`` when ``source_column`` is set (``load._write`` passes the task's source exactly
        when the target has a source column, and a task's source is never None).

    Args:
        source_column: Column holding the source, or ``None`` for a table that is not per source.
        source_id: Source to match; ignored when ``source_column`` is ``None``.

    Returns:
        ``TRUE`` with no parameters when ``source_column`` is ``None``; otherwise ``t.<source_column> = %s`` with
        ``(source_id,)`` as its parameters.

    Raises:
        Nothing.

    """
    if source_column is None:  # a table not per source: the whole table is the scope
        return sql.SQL("TRUE"), ()
    return sql.SQL("{} = %s").format(sql.Identifier("t", source_column)), (source_id,)


def _under(inherited: Sequence[str], parents: str) -> sql.Composed:
    """Build the condition that a row of the table aliased ``t`` belongs to a parent listed in ``parents``.

    The parent key is read from ``parents`` qualified, so a key column missing from ``parents`` raises
    ``psycopg.errors.UndefinedColumn`` when the statement runs, rather than matching the table's own column.

    Preconditions:
        ``inherited`` is non-empty and names columns of the table and of ``parents``.

    Args:
        inherited: Parent-key columns, matched by name.
        parents: Unqualified name of the table listing the parents.

    Returns:
        The composed ``ROW(...) IN (SELECT ...)`` condition, without parameters.

    Raises:
        Nothing.

    """
    return sql.SQL("ROW({t_key}) IN (SELECT {p_key} FROM {parents} AS p)").format(
        t_key=_names(inherited, "t"),
        p_key=_names(inherited, "p"),  # qualified, so a key column missing from parents is an error
        parents=sql.Identifier(parents),
    )


def _missing(staged: str, key: Sequence[str]) -> sql.Composed:
    """Build the condition that a row of the table aliased ``t`` has no staged row with the same key.

    A row with a null key column never matches a staged row, so it always counts as missing.

    Preconditions:
        ``key`` is non-empty and ``staged`` has every name in it.

    Args:
        staged: Unqualified name of the staged table.
        key: Columns compared between the table and the stage.

    Returns:
        The composed ``NOT EXISTS`` condition, without parameters.

    Raises:
        Nothing.

    """
    return sql.SQL("NOT EXISTS (SELECT FROM {staged} AS s WHERE ROW({s_key}) = ROW({t_key}))").format(
        staged=sql.Identifier(staged),
        s_key=_names(key, "s"),
        t_key=_names(key, "t"),
    )


def _table(table: str) -> sql.Identifier:
    """Split a ``"schema.name"`` string into a qualified identifier.

    Preconditions:
        None.

    Args:
        table: Table name with exactly one ``.``.

    Returns:
        The identifier ``"schema"."name"``.

    Raises:
        ValueError: When ``table`` does not contain exactly one ``.``.

    """
    schema, name = table.split(".")
    return sql.Identifier(schema, name)


def _names(columns: Sequence[str], qualifier: str | None = None) -> sql.Composed:
    """Join column names into a comma-separated list of identifiers.

    Preconditions:
        None.

    Args:
        columns: Column names, in order.
        qualifier: Table alias to prefix each name with, or ``None`` (or empty) for none.

    Returns:
        The composed list; empty when ``columns`` is empty.

    Raises:
        Nothing.

    """
    return sql.SQL(", ").join(sql.Identifier(qualifier, c) if qualifier else sql.Identifier(c) for c in columns)


async def _changed(conn: Connection, query: sql.Composed, params: tuple[object, ...] | None = None) -> int:
    """Execute one statement and return its row count.

    Preconditions:
        ``params`` has one value per ``%s`` placeholder in ``query``.

    Args:
        conn: Connection the statement runs on.
        query: Statement to execute.
        params: Placeholder values, or ``None`` when there are none.

    Returns:
        The cursor's ``rowcount``.

    Raises:
        DatabaseConflict: From ``db_errors``, on a serialization failure or deadlock.
        DatabaseUnavailable: From ``db_errors``, on a lost or refused connection.
        KeyViolation: From ``db_errors``, when a written row violates a unique key of the table.
        psycopg.Error: Any other database error, unchanged.

    """
    with db_errors():
        cursor = await conn.execute(query, params)
    return cursor.rowcount
