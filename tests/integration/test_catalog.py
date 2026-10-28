"""Integration tests for ``pipeline.db.catalog`` against tables in a scratch schema."""

import os
from collections.abc import AsyncIterator

import pytest
from psycopg_pool import AsyncConnectionPool

from pipeline.db.catalog import TableColumn, table_info
from pipeline.db.pool import Pool

DSN = os.environ.get("TEST_PIPELINE_DSN", "postgresql://postgres@localhost:5432/pipeline_test")


@pytest.fixture
async def pool() -> AsyncIterator[Pool]:
    """Yield an open one-connection pool on an empty ``t8_writers`` schema, dropped again afterwards."""
    p: Pool = AsyncConnectionPool(DSN, min_size=1, max_size=1, kwargs={"autocommit": True}, open=False)
    await p.open(wait=True)
    async with p.connection() as c:
        await c.execute("DROP SCHEMA IF EXISTS t8_writers CASCADE")
        await c.execute("CREATE SCHEMA t8_writers")
    try:
        yield p
    finally:
        async with p.connection() as c:
            await c.execute("DROP SCHEMA t8_writers CASCADE")
        await p.close()


async def test_table_info_reads_domain_constraints(pool: Pool) -> None:
    """Check a column of a NOT NULL domain reads as not nullable, and one of a domain with a default as having one."""
    async with pool.connection() as c:
        await c.execute("CREATE DOMAIN t8_writers.req_text AS text NOT NULL")
        await c.execute("CREATE DOMAIN t8_writers.def_text AS text DEFAULT 'x'")
        await c.execute("CREATE TABLE t8_writers.domained (a t8_writers.req_text, b t8_writers.def_text)")
        info = await table_info(c, "t8_writers.domained")
    assert info is not None
    assert list(info.columns.values()) == [
        TableColumn("a", "t8_writers.req_text", False, False, True),
        TableColumn("b", "t8_writers.def_text", True, True, True),
    ]
