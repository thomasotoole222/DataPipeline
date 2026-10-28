"""Merge a staged table into a warehouse table with one of five SQL write strategies.

Public API:
    upsert: Insert staged rows and update existing rows by key where their other columns differ.
    replace_children: Delete the rows of every parent listed in a parents table, then insert the staged rows.
    replace_partition: Delete the rows in one source's half-open range of a column, then insert the staged rows.
    snapshot: Upsert staged rows, clear their ``removed_at``, and optionally mark rows missing from the stage removed.
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
        psycopg.Error: Any other database error, unchanged.

    """
    return await _changed(conn, _upsert(table, staged, columns, key, clear_removed=False))


async def replace_children(
    conn: Connection, table: str, staged: str, columns: Sequence[str], key: Sequence[str], parents: str
) -> int:
    """Delete every table row whose parent key is listed in ``parents``, then insert every staged row.

    The two statements run on the same connection; they are atomic only if the caller's transaction makes them so.
    The parent key is read from ``parents`` qualified, so a key column missing from ``parents`` raises
    ``psycopg.errors.UndefinedColumn`` rather than matching the table's own column.

    Preconditions:
        ``table`` is ``"schema.name"``. ``key`` is non-empty and names the parent-key columns of ``table``, and
        ``parents`` has a column of each of those names (Load passes the parent's staged table, and
        ``build_target`` makes the child's key the parent's writer key, which the parent's columns include).
        ``staged`` has every name in ``columns``, and each staged row belongs to a parent in ``parents`` (rows of
        any other parent are added beside the existing ones).

    Args:
        conn: Connection the statements run on; the caller owns the transaction.
        table: Target table as ``"schema.name"``.
        staged: Unqualified name of the staged table.
        columns: Columns copied from the stage to the table.
        key: Parent-key columns, matched by name in ``table`` and ``parents``.
        parents: Unqualified name of the table listing the parents to replace.

    Returns:
        The number of rows deleted plus the number of rows inserted.

    Raises:
        ValueError: When ``table`` does not contain exactly one ``.``.
        DatabaseConflict: From ``db_errors``, on a serialization failure or deadlock.
        DatabaseUnavailable: From ``db_errors``, on a lost or refused connection.
        psycopg.Error: Any other database error, unchanged.

    """
    delete = sql.SQL("DELETE FROM {table} WHERE ROW({key}) IN (SELECT {p_key} FROM {parents} AS p)").format(
        table=_table(table), key=_names(key), p_key=_names(key, "p"), parents=sql.Identifier(parents)
    )  # p_key is qualified, so a key column missing from parents is an error, never the outer table's column
    return await _changed(conn, delete) + await _changed(conn, _insert(table, staged, columns))


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
        psycopg.Error: Any other database error, unchanged.

    """
    with db_errors():
        cursor = await conn.execute(query, params)
    return cursor.rowcount
