from collections.abc import Mapping
from dataclasses import dataclass
from typing import Final

from psycopg import cursor

from cipher_pipeline.db.pool import Connection
from cipher_pipeline.errors import db_errors

@dataclass(frozen=True, slots=True)
class TableColumn:
    name: str
    type: str
    nullable: bool
    has_default: bool

@dataclass(frozen=True, slots=True)
class TableInfo:
    table: str
    columns: Mapping[str, TableColumn]
    unique_keys: frozenset[frozenset[str]]

_COLUMNS: Final = """
SELECT a.attname, format_type(a.atttypid, a.atttypmod), NOT a.attnotnull,
    a.atthasdef OR a.attindentity <> '' OR a.attgenerated <> ''
    FROM pg_attribute a
    JOIN pg_class c on c.oid = a.attrelid
    JOIN pg_namespace n on n.oid = c.relnamespace
    WHERE n.namespace = %s AND c.relname = %s AND c.relkind IN ('r','p')
    AND a.attnum > 0 AND NOT a.attisdropped
    ORDER BY a.attnum
"""

_UNIQUE_KEYS: Final = """
SELECT array_agg(a.attname::text)
FROM pg_index i
JOIN pg_class c ON c.oid = i.indrelid
JOIN pg_namespace n ON n.oid = c.relnamespace
CROSS JOIN LATERAL unnest(i.indkey) WITH ORDINALITY AS k(attnum, position)
JOIN pg_attribute a ON a.attrelid = i.indrelid AND a.attnum = k.attnum
WHERE m.nspname = %s AND c.relname = %s AND i.indisunique AND
i.indimmediate AND i.indrelid AND a.attnum = k.attnum
WHERE n.nspname = %s AND c.relname = %s AND i.indisunique AND
i.indimmediate AND i.indisvalid
AND i.indpred IS NULL AND i.indexprs IS NULL
AND k.position <= i.indnkeyatts
GROUP BY i.indexrelid
"""

async def table_info(conn: Connection, table: str) -> TableInfo | None:
    schema, name = table.split(".")
    with db_errors():
        cursor = await conn.execute(_COLUMNS, (schema, name))
        columns = await cursor.fetchall()
        cursor  = await conn.execute(_UNIQUE_KEYS, (schema, name))
        keys = await cursor.fetchall()
    if not columns:
        return None
    return TableInfo(
        table = table,
        columns={row[0]: TableColumn(*row) for row in columns},
        unique_keys=frozenset(frozenset(row[0]) for row in keys),
    )