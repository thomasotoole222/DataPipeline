"""Read values from warehouse tables: which of a set of values exist in a column, and a job's one-column query.

Public API:
    known_values: Return the subset of the given values that appear in ``schema.table.column``, optionally only in
        one source's rows.
    read_column: Run a one-column query in a read-only transaction and return its values.

"""

from collections.abc import Collection, Mapping
from typing import Any, Final, LiteralString
from uuid import UUID

from psycopg import sql

from pipeline.db.pool import Connection, storable_text
from pipeline.errors import db_errors

#: Types whose bare name means length 1, mapped to their unbounded forms, so a longer value is not refused. The
#: array columns of ``char(n)`` and ``bit(n)`` read as ``character[]`` and ``bit[]``, whose elements would otherwise
#: mean length 1, so those names are mapped too.
_UNBOUNDED: Final[Mapping[str, str]] = {
    "character": "bpchar",
    "bit": "varbit",
    "character[]": "bpchar[]",
    "bit[]": "varbit[]",
}

#: Column types compared through a cast of the column itself to the unbounded type: Postgres has no ``=`` between
#: ``bit[]`` and ``varbit[]``, while ``bit = varbit`` and ``character[] = bpchar[]`` need none.
_CAST_COLUMN: Final[frozenset[str]] = frozenset({"bit[]"})


async def known_values(
    conn: Connection,
    table: str,
    column: str,
    values: Collection[str],
    *,
    source: tuple[str, UUID] | None = None,  # (column, source id): only that source's rows count
) -> frozenset[str]:
    """Return the subset of ``values`` that appear in ``column`` of ``table``, compared in the column's own type.

    With ``source`` set to ``(source_column, source_id)``, a value is found only in a row whose ``source_column``
    equals ``source_id``: the second query adds ``AND t.<source_column> = <source_id>``, the id sent as untyped text,
    so Postgres reads it in the column's own type (``uuid`` or ``text``). A value present only in another source's
    rows is not found.

    With no values it returns an empty frozenset without touching the database. Otherwise it splits ``table`` on
    ``"."`` and, inside ``db_errors``, runs two queries. The first reads the column's type as
    ``pg_typeof(column)::text`` from one row of the table; that spelling carries no type modifier. When the table has
    no rows, nothing can be found and an empty frozenset is returned. The second joins the values, as ``text``, to
    the table on ``column = CASE WHEN pg_input_is_valid(value, <type>) THEN value::<type> END``, so each value is
    cast to the column's type and compared there: an upper-case ``UUID`` matches a ``uuid`` column, and a ``text``
    column stays case-sensitive. A value that is not a valid literal of the type is never cast (``CASE`` checks it
    first) and is simply not found. A value that Postgres ``text`` cannot hold at all (one that ``storable_text``
    would change: it contains NUL or a lone surrogate) is dropped before the queries, so it too is not found; when
    no value is left, an empty frozenset is returned without touching the database.

    Dropping the modifier is harmless for ``varchar``, ``numeric`` and ``timestamp`` (the bare name means "no
    limit"), but not for ``char(n)`` and ``bit(n)``: their bare names ``character`` and ``bit`` mean length 1. Those
    two names and their array names ``character[]`` and ``bit[]`` are replaced through ``_UNBOUNDED`` with the
    unbounded forms ``bpchar``, ``varbit``, ``bpchar[]`` and ``varbit[]``, so in a ``char(5)`` column holding
    ``"abc"`` the value ``"abc"`` is found, and in a ``char(5)[]`` column holding ``{abc}`` the value ``"{abc}"`` is.
    A ``bit[]`` column is itself cast to ``varbit[]`` for the comparison (``_CAST_COLUMN``), since Postgres has no
    ``=`` between ``bit[]`` and ``varbit[]``; the cast is exact, so ``{1010}`` is found in a ``bit(4)[]`` column.
    A ``timestamptz`` value with no offset is read in the session's ``TimeZone``, which ``open_pool`` pins to UTC.
    Every other spelling follows the type's own input rules: a ``timestamp`` column drops a value's offset, a
    ``date`` column drops its time of day, and a ``boolean`` column takes ``"yes"`` as ``true``.

    Preconditions:
        ``table`` is ``"schema.name"`` with exactly one dot, as ``Lookup.__post_init__`` guarantees through
        ``TABLE_PATTERN``; not checked here. The server is Postgres 16 or later, for ``pg_input_is_valid``. A
        ``source`` column is of a type whose input reads a UUID's text, such as ``uuid`` or ``text``.

    Args:
        conn: Open warehouse connection.
        table: Schema-qualified table name, ``"schema.name"``.
        column: Column to match the values against.
        values: Values to look for, as text; may be empty.
        source: The column of ``table`` that holds a row's source and the source whose rows count, or ``None`` to
            look across every row. Defaults to ``None``.

    Returns:
        A frozenset of the given values that match a row, spelled as given; empty when ``values`` holds no value
        Postgres ``text`` can store or the table has no rows.

    Raises:
        DatabaseConflict: From ``db_errors``, if a query raises an error in its conflict set.
        DatabaseUnavailable: From ``db_errors``, if a query raises an unavailable error or a bare
            ``psycopg.OperationalError``.
        psycopg.Error: Any other database error, unchanged (e.g. ``UndefinedTable`` when the table does not exist,
            ``UndefinedColumn`` when the column does not, ``UndefinedFunction`` when the column's type has no ``=``
            operator, such as ``json``; ``InvalidTextRepresentation`` when the ``source`` column's type cannot read the
            source id, such as ``integer``).

    """
    storable = [value for value in values if storable_text(value) == value]  # NUL or a surrogate is never stored
    if not storable:
        return frozenset()
    schema, name = table.split(".")
    names = {"column": sql.Identifier(column), "table": sql.Identifier(schema, name)}
    type_query = sql.SQL("SELECT pg_typeof({column})::text FROM {table} LIMIT 1").format(**names)
    with db_errors():
        cursor = await conn.execute(type_query)
        row = await cursor.fetchone()
        if row is None:  # an empty table holds none of the values
            return frozenset()
        type_name = _UNBOUNDED.get(row[0], row[0])
        column_type = sql.SQL(type_name)  # the catalog's own spelling of a type, quoted where it needs to be
        stored = sql.SQL("t.{column}").format(**names)
        if row[0] in _CAST_COLUMN:  # no = between the column's type and the unbounded one
            stored = sql.SQL("{stored}::{type}").format(stored=stored, type=column_type)
        query = sql.SQL(
            "SELECT DISTINCT w.v FROM unnest(%s::text[]) AS w(v) JOIN {table} AS t "
            "ON {stored} = CASE WHEN pg_input_is_valid(w.v, %s) THEN w.v::{type} END"  # an invalid value is not cast
        ).format(table=names["table"], stored=stored, type=column_type)
        params: list[object] = [storable, type_name]
        if source is not None:  # only the task's source's rows count
            query += sql.SQL(" AND t.{source} = %s").format(source=sql.Identifier(source[0]))
            params.append(str(source[1]))  # untyped text: Postgres reads it in the column's type
        cursor = await conn.execute(query, params)
        rows = await cursor.fetchall()
    return frozenset(str(found[0]) for found in rows)


async def read_column(conn: Connection, query: LiteralString, params: Mapping[str, Any]) -> list[Any]:
    """Run a query that returns one column in a read-only transaction on ``conn`` and return its values in row order.

    The transaction is opened with ``conn.transaction()`` and made ``READ ONLY`` with ``SET TRANSACTION READ ONLY``
    before the query runs, so a statement that writes fails; it is committed (nothing to commit) when the rows are
    read. The query is sent with ``prepare=True``, so it always goes over the extended protocol, where Postgres
    refuses more than one statement: without it, a query that names no parameter would go over the simple query
    protocol, which runs ``"SELECT id FROM t; COMMIT; DELETE FROM t"`` statement by statement, its ``DELETE``
    outside the read-only transaction. Everything runs inside ``db_errors``.

    Preconditions:
        ``conn`` is open, in autocommit mode with no transaction open, and not used by another coroutine. ``query``
        takes only named parameters (``%(name)s``) that are keys of ``params``, writes a literal ``%`` as ``%%``,
        and is one statement.

    Args:
        conn: Open warehouse connection.
        query: The query, from a job's declaration.
        params: The values of its named parameters; keys it does not name are ignored.

    Returns:
        The first and only column of every row, as psycopg reads it, in the order the query returns them.

    Raises:
        ValueError: When the query's result has no columns (a statement that returns no rows) or more than one
            column; the transaction rolls back.
        DatabaseConflict: From ``db_errors``, on a serialization failure or deadlock.
        DatabaseUnavailable: From ``db_errors``, when the connection is lost or the warehouse cannot be reached.
        psycopg.Error: Any other database error unchanged, such as ``ReadOnlySqlTransaction`` for a single statement
            that writes, ``SyntaxError`` for a query of more than one statement ("cannot insert multiple commands
            into a prepared statement"), or ``ProgrammingError`` for a parameter missing from ``params`` or a bare
            ``%`` in ``query``.

    """
    with db_errors():
        async with conn.transaction():
            await conn.execute("SET TRANSACTION READ ONLY")
            cursor = await conn.execute(query, params, prepare=True)  # one statement only, even with no parameter
            columns = 0 if cursor.description is None else len(cursor.description)
            if columns != 1:
                raise ValueError(f"the query must return one column, not {columns}")
            rows = await cursor.fetchall()
    return [row[0] for row in rows]
