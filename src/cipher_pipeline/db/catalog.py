"""Read a warehouse table's columns and unique keys from the Postgres system catalog.

Public API:
    TableColumn: One column of a table as the catalog describes it.
    TableInfo: A table's columns and the unique keys an ``ON CONFLICT`` clause can target.
    table_info: Fetch the ``TableInfo`` of a ``schema.table``, or None when no such table is found.

"""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Final

from cipher_pipeline.db.pool import Connection
from cipher_pipeline.errors import db_errors


@dataclass(frozen=True, slots=True)
class TableColumn:
    """One column of a table as the catalog describes it.

    ``__post_init__`` is not defined, so no field is validated; ``table_info`` fills every field from the catalog.

    TODO(review) 1: ``has_default`` is True for generated and identity columns, and no field tells them apart from a
    column with an ordinary ``DEFAULT``. A ``GENERATED ALWAYS`` column rejects a written value, so a caller cannot tell
    from this object whether writing the column will fail.

    Attributes:
        name: The column name as stored in the catalog.
        type: The type as ``format_type`` prints it, with any modifier, e.g. ``character varying(50)``,
            ``numeric(12,2)``, ``timestamp with time zone`` or ``integer[]``. A domain prints as its own name.
        nullable: True when the column has no ``NOT NULL`` constraint.
        has_default: True when the column has a ``DEFAULT``, is an identity column (``ALWAYS`` or ``BY DEFAULT``), or
            is a generated column.

    """

    name: str
    type: str
    nullable: bool
    has_default: bool


@dataclass(frozen=True, slots=True)
class TableInfo:
    """A table's columns and the unique keys an ``ON CONFLICT`` clause can target.

    ``__post_init__`` is not defined, so no field is validated; ``table_info`` builds every instance from the catalog.

    Attributes:
        table: The ``schema.table`` name exactly as passed to ``table_info``.
        columns: Each live column keyed by its name, in table column order (``attnum``).
        unique_keys: The column-name sets of the table's unique indexes, the primary key included. Only indexes that
            are valid, not deferrable, not partial and without expression columns count; ``INCLUDE`` columns are left
            out. Indexes with the same key columns appear once. A key may contain nullable columns, and under the
            default ``NULLS DISTINCT`` rows whose key holds a NULL never conflict.

    """

    table: str
    columns: Mapping[str, TableColumn]
    unique_keys: frozenset[frozenset[str]]


#: Query for the live columns of one ordinary or partitioned table, in column order; parameters are schema and name.
#: Each row is (name, formatted type, nullable, has default or is identity or generated), matching ``TableColumn``.
_COLUMNS: Final = """
SELECT a.attname, format_type(a.atttypid, a.atttypmod), NOT a.attnotnull,
       a.atthasdef OR a.attidentity <> '' OR a.attgenerated <> ''
FROM pg_attribute a
JOIN pg_class c ON c.oid = a.attrelid
JOIN pg_namespace n ON n.oid = c.relnamespace
WHERE n.nspname = %s AND c.relname = %s AND c.relkind IN ('r', 'p') AND a.attnum > 0 AND NOT a.attisdropped
ORDER BY a.attnum
"""

#: Query for the key columns of each valid, immediate, non-partial, expression-free unique index on one relation;
#: parameters are schema and name. Each row holds one index's key column names as an array, ``INCLUDE`` columns left
#: out. The relation kind is not filtered; ``table_info`` uses these rows only when ``_COLUMNS`` found a table.
_UNIQUE_KEYS: Final = """
SELECT array_agg(a.attname::text)
FROM pg_index i
JOIN pg_class c ON c.oid = i.indrelid
JOIN pg_namespace n ON n.oid = c.relnamespace
CROSS JOIN LATERAL unnest(i.indkey) WITH ORDINALITY AS k(attnum, position)
JOIN pg_attribute a ON a.attrelid = i.indrelid AND a.attnum = k.attnum
WHERE n.nspname = %s AND c.relname = %s AND i.indisunique AND i.indimmediate AND i.indisvalid
  AND i.indpred IS NULL AND i.indexprs IS NULL
  AND k.position <= i.indnkeyatts
GROUP BY i.indexrelid
"""


async def table_info(conn: Connection, table: str) -> TableInfo | None:
    """Fetch the columns and unique keys of a table from the catalog.

    The schema and table names are matched exactly as stored, with no case folding, so ``Public.Users`` does not find
    a table created as ``public.users``. Only ordinary and partitioned tables are found; a view, materialized view or
    foreign table gives None. The two catalog queries run one after the other on ``conn``, in whatever transaction
    state it is in.

    TODO(review) 2: a table that exists with zero columns (``CREATE TABLE s.t ()``) gives None, the same result as a
    missing table.

    Preconditions:
        ``table`` contains exactly one ``.``, which splits it into schema and table name. Nothing on this page checks
        this beyond the unpacking that raises ``ValueError``.

    Args:
        conn: The connection to run the catalog queries on.
        table: The table as ``schema.table``.

    Returns:
        The table's ``TableInfo``, or None when no ordinary or partitioned table with that schema and name has a live
        column.

    Raises:
        ValueError: When ``table`` does not contain exactly one ``.``; raised before any query, outside ``db_errors``.
        DatabaseConflict: From ``db_errors``, when a query raises an error in its conflict set.
        DatabaseUnavailable: From ``db_errors``, when a query raises an error in its unavailable set or a bare
            ``psycopg.OperationalError``.
        psycopg.Error: Any other error from a query, unchanged.

    """
    schema, name = table.split(".")
    with db_errors():
        cursor = await conn.execute(_COLUMNS, (schema, name))
        columns = await cursor.fetchall()
        cursor = await conn.execute(_UNIQUE_KEYS, (schema, name))
        keys = await cursor.fetchall()
    if not columns:
        return None
    return TableInfo(
        table=table,
        columns={row[0]: TableColumn(*row) for row in columns},
        unique_keys=frozenset(frozenset(row[0]) for row in keys),
    )
