"""Find which of a set of values already exist in a warehouse table column.

Public API:
    known_values: Return the subset of the given values that appear in ``schema.table.column``.

"""

from collections.abc import Collection

from psycopg import sql

from cipher_pipeline.db.pool import Connection
from cipher_pipeline.errors import db_errors


async def known_values(conn: Connection, table: str, column: str, values: Collection[str]) -> frozenset[str]:
    """Return the subset of ``values`` that appear in ``column`` of ``table``.

    With no values it returns an empty frozenset without touching the database. Otherwise it splits ``table`` on
    ``"."`` and runs ``SELECT DISTINCT column::text FROM table WHERE column::text = ANY (values::text[])`` inside
    ``db_errors``. The column is cast to text on both sides, so values are compared as text and a lookup works on a
    ``uuid`` or ``bigint`` column too. A value that is not a valid literal of the column's type is simply not found,
    and the matched values come back as the column's text.

    Preconditions:
        ``table`` is ``"schema.name"`` with exactly one dot, as ``Lookup.__post_init__`` guarantees through
        ``_TABLE``; not checked here.

    Args:
        conn: Open warehouse connection.
        table: Schema-qualified table name, ``"schema.name"``.
        column: Column to match the values against.
        values: Values to look for; may be empty.

    Returns:
        A frozenset of the matched values as the column's text; empty when ``values`` is empty.

    Raises:
        DatabaseConflict: From ``db_errors``, if the query raises an error in its conflict set.
        DatabaseUnavailable: From ``db_errors``, if the query raises an unavailable error or a bare
            ``psycopg.OperationalError``.
        psycopg.Error: Any other database error, unchanged (e.g. ``UndefinedTable`` when the table does not exist).

    """
    if not values:
        return frozenset()
    schema, name = table.split(".")
    query = sql.SQL("SELECT DISTINCT {column}::text FROM {table} WHERE {column}::text = ANY(%s::text[])").format(
        column=sql.Identifier(column), table=sql.Identifier(schema, name)
    )
    with db_errors():
        cursor = await conn.execute(query, (list(values),))
        rows = await cursor.fetchall()
    return frozenset(str(row[0]) for row in rows)
