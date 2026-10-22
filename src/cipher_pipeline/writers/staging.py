from collections.abc import Sequence
from uuid import uuid4

import polars as pl
from psycopg import pq, sql

from cipher_pipeline.db.pool import Connection
from cipher_pipeline.errors import db_errors


async def copy_in(conn: Connection, frame: pl.DataFrame, columns: Sequence[tuple[str, str]]) -> str:
    if conn.info.transaction_status != pq.TransactionStatus.INTRANS:
        raise RuntimeError("copy_in must run inside the load transaction")
    staged = f"staged_{uuid4().hex}"
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
