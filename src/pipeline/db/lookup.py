"""Find which of a set of values already exist in a warehouse table column.

Public API:
    known_values: Return the subset of the given values that appear in ``schema.table.column``.

"""

from collections.abc import Collection, Mapping
from typing import Final

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


async def known_values(conn: Connection, table: str, column: str, values: Collection[str]) -> frozenset[str]:
    """Return the subset of ``values`` that appear in ``column`` of ``table``, compared in the column's own type.

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
        ``TABLE_PATTERN``; not checked here. The server is Postgres 16 or later, for ``pg_input_is_valid``.

    Args:
        conn: Open warehouse connection.
        table: Schema-qualified table name, ``"schema.name"``.
        column: Column to match the values against.
        values: Values to look for, as text; may be empty.

    Returns:
        A frozenset of the given values that match a row, spelled as given; empty when ``values`` holds no value
        Postgres ``text`` can store or the table has no rows.

    Raises:
        DatabaseConflict: From ``db_errors``, if a query raises an error in its conflict set.
        DatabaseUnavailable: From ``db_errors``, if a query raises an unavailable error or a bare
            ``psycopg.OperationalError``.
        psycopg.Error: Any other database error, unchanged (e.g. ``UndefinedTable`` when the table does not exist,
            ``UndefinedColumn`` when the column does not, ``UndefinedFunction`` when the column's type has no ``=``
            operator, such as ``json``).

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
        cursor = await conn.execute(query, (storable, type_name))
        rows = await cursor.fetchall()
    return frozenset(str(found[0]) for found in rows)
