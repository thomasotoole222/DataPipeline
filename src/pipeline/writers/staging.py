"""Copy a polars frame into a new temporary staging table inside the load transaction.

Public API:
    copy_in: Create a temporary table that is dropped on commit, COPY a frame's rows into it, and return its name.

"""

from collections.abc import Sequence
from uuid import uuid4

import polars as pl
from psycopg import pq, sql

from pipeline.db.pool import Connection
from pipeline.errors import db_errors


async def copy_in(conn: Connection, frame: pl.DataFrame, columns: Sequence[tuple[str, str]]) -> str:
    """Create a temporary table dropped on commit, COPY every row of the frame into it, and return its name.

    The table is named ``staged_`` plus a random 32-digit hex UUID, and has one column per ``columns`` entry, in that
    order. Column names are quoted as identifiers; each type string is pasted into the ``CREATE TEMP TABLE`` as raw
    SQL. Rows are written one at a time from ``frame.iter_rows()`` with a text ``COPY ... FROM STDIN`` into the named
    columns, so the frame's i-th column goes into the i-th entry of ``columns``; the frame's own column names are not
    read. The table exists only until the transaction ends (``ON COMMIT DROP``) and is visible only to ``conn``.

    Preconditions:
        ``conn`` is inside a transaction that has not failed (checked: ``RuntimeError`` otherwise). ``frame``'s
        columns are ``columns``' names in the same order: ``load._write`` passes both from the same ``Target`` --
        ``columns`` from ``target.columns`` and the frame from ``frame_of``, which builds it from ``target.columns``.
        Each type string comes from ``checks.targets._column_type`` (a ``_SCALARS`` type name or ``numeric(p,s)``
        built from the model's integer constraints), never from data, so it is inserted as SQL while names are quoted
        as identifiers. ``columns`` is non-empty: every ``Target`` has at least its writer's key or partition column,
        which ``_check_writer`` requires when the model is defined, or for ``ReplaceChildren`` the parent's key
        columns it inherits. Every frame value can be dumped by psycopg as text for its column's type; a
        ``timestamptz`` column holds aware datetimes (``build_target`` maps ``AwareDatetime`` to
        ``pl.Datetime("us", "UTC")``, and Transform checks values are aware).

    Args:
        conn: Connection inside the load transaction.
        frame: Rows to stage, columns in ``columns`` order.
        columns: ``(name, Postgres type)`` of each staged column, in frame column order.

    Returns:
        The staged table's name, valid until the transaction commits or rolls back.

    Raises:
        RuntimeError: When ``conn``'s transaction status is not ``INTRANS`` (idle, failed, or a query is active).
        DatabaseConflict: From ``db_errors``, on a serialization failure or deadlock.
        DatabaseUnavailable: From ``db_errors``, on a lost or refused connection.
        psycopg.Error: Any other database or adaptation error, unchanged, such as a value Postgres cannot parse.
            After a database error the transaction is failed and must be rolled back.

    """
    if conn.info.transaction_status != pq.TransactionStatus.INTRANS:
        raise RuntimeError("copy_in must run inside the Load transaction")
    staged = f"staged_{uuid4().hex}"  # unique in the session; the table is dropped at commit
    definition = sql.SQL(", ").join(
        sql.SQL("{} {}").format(sql.Identifier(name), sql.SQL(type_)) for name, type_ in columns
    )
    names = sql.SQL(", ").join(sql.Identifier(name) for name, _ in columns)
    create = sql.SQL("CREATE TEMP TABLE {} ({}) ON COMMIT DROP").format(sql.Identifier(staged), definition)
    copy_sql = sql.SQL("COPY {} ({}) FROM STDIN").format(sql.Identifier(staged), names)
    with db_errors():
        await conn.execute(create)
        async with conn.cursor() as cursor, cursor.copy(copy_sql) as copy:
            for row in frame.iter_rows():
                await copy.write_row(row)
    return staged
