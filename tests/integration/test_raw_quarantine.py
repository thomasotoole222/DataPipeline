"""Integration tests for ``cipher_pipeline.db.raw`` and ``cipher_pipeline.db.quarantine`` on ``raw.page``."""

import os
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta, timezone
from typing import Any
from uuid import uuid4

import psycopg
import pytest
from psycopg_pool import AsyncConnectionPool

from cipher_pipeline.db.pool import Pool
from cipher_pipeline.db.quarantine import Failure, HeldRow, hold
from cipher_pipeline.db.raw import RawPage, discard_pages, last_page, read_pages, save_page

DSN = os.environ.get("PIPELINE_TEST_DSN", "postgresql://postgres@localhost:5432/pipeline_test")
ENDPOINT = "tiktok_shop.orders_search"


@pytest.fixture
async def pool() -> AsyncIterator[Pool]:
    """Yield an open pool on empty raw and quarantine tables."""
    p: Pool = AsyncConnectionPool(DSN, min_size=1, max_size=2, kwargs={"autocommit": True}, open=False)
    await p.open(wait=True)
    async with p.connection() as c:
        await c.execute("TRUNCATE raw.page, quarantine.row")
    try:
        yield p
    finally:
        await p.close()


def page(n: int, nxt: str | None = "c", task: int = 1, payload: dict[str, Any] | None = None) -> RawPage:
    """Build page ``n`` of ``task``."""
    return RawPage(
        task_id=task,
        page_no=n,
        endpoint=ENDPOINT,
        cursor_in=None if n == 0 else f"c{n}",
        cursor_next=nxt,
        fetched_at=datetime(2026, 10, 12, 9, tzinfo=timezone(timedelta(hours=1))),
        payload=payload or {"n": n},
    )


def held(**kw: Any) -> HeldRow:
    """Build a held row, overriding any field."""
    base: dict[str, Any] = {
        "task_id": 1,
        "job": "orders",
        "shop_id": uuid4(),
        "partition_key": "2026-10-11",
        "raw_page_no": 0,
        "raw_index": 3,
        "group_key": "o1",
        "payload": {"id": "o1"},
        "failures": (Failure("order_id_present", 1, "id is null"),),
    }
    base.update(kw)
    return HeldRow(**base)


def test_rawpage_utc_and_checks() -> None:
    """Check fetched_at is converted to UTC and page_no and payload are checked."""
    p = page(0)
    assert p.fetched_at == datetime(2026, 10, 12, 8, tzinfo=UTC)
    assert p.fetched_at.tzinfo is UTC
    with pytest.raises(ValueError, match="page_no"):
        page(-1)
    with pytest.raises(ValueError, match="naive"):
        RawPage(
            task_id=1,
            page_no=0,
            endpoint=ENDPOINT,
            cursor_in=None,
            cursor_next=None,
            fetched_at=datetime(2026, 1, 1),  # noqa: DTZ001 - the naive value under test
            payload={},
        )
    with pytest.raises(TypeError, match="JSON object"):
        RawPage(
            task_id=1,
            page_no=0,
            endpoint=ENDPOINT,
            cursor_in=None,
            cursor_next=None,
            fetched_at=datetime(2026, 1, 1, tzinfo=UTC),
            payload=[1],  # type: ignore[arg-type]
        )


async def test_save_and_last(pool: Pool) -> None:
    """Check last_page returns the highest page saved, endpoint included, or None."""
    async with pool.connection() as c:
        assert await last_page(c, 1) is None
        await save_page(c, page(0, "c1"))
        await save_page(c, page(1, "c2"))
        last = await last_page(c, 1)
        assert last == page(1, "c2")
        assert last is not None
        assert last.endpoint == ENDPOINT
        assert last.fetched_at.tzinfo is UTC


async def test_save_is_idempotent_and_overwrites(pool: Pool) -> None:
    """Check saving the same task and page again overwrites it."""
    async with pool.connection() as c:
        await save_page(c, page(0, "c1"))
        await save_page(c, page(0, "zz", payload={"v": 2}))
        rows = [p async for p in read_pages(c, 1)]
        assert len(rows) == 1
        assert rows[0].cursor_next == "zz"
        assert rows[0].payload == {"v": 2}


async def test_read_pages_order_and_isolation(pool: Pool) -> None:
    """Check read_pages yields one task's pages in page order, whatever the endpoint of other tasks."""
    async with pool.connection() as c:
        for n in (2, 0, 1):
            await save_page(c, page(n))
        other = RawPage(
            task_id=2,
            page_no=0,
            endpoint="cruva.samples",
            cursor_in=None,
            cursor_next=None,
            fetched_at=datetime(2026, 10, 12, tzinfo=UTC),
            payload={},
        )
        await save_page(c, other)
        assert [p.page_no async for p in read_pages(c, 1)] == [0, 1, 2]
        assert [p async for p in read_pages(c, 2)] == [other]
        assert [p async for p in read_pages(c, 3)] == []


async def test_read_pages_early_break_leaves_connection_usable(pool: Pool) -> None:
    """Check leaving read_pages early does not leave a cursor open on the connection."""
    async with pool.connection() as c:
        for n in range(5):
            await save_page(c, page(n))
        async for _ in read_pages(c, 1):
            break
        last = await last_page(c, 1)
        assert last is not None
        assert last.page_no == 4


async def test_discard(pool: Pool) -> None:
    """Check discard_pages deletes one task's pages and returns how many."""
    async with pool.connection() as c:
        for n in range(3):
            await save_page(c, page(n))
        await save_page(c, page(0, task=2))
        assert await discard_pages(c, 1) == 3
        assert await last_page(c, 1) is None
        assert await discard_pages(c, 1) == 0
        assert await last_page(c, 2) is not None


async def test_endpoint_is_required(pool: Pool) -> None:
    """Check the table refuses a page without an endpoint."""
    async with pool.connection() as c:
        with pytest.raises(psycopg.errors.NotNullViolation):
            await c.execute("INSERT INTO raw.page (task_id, page_no, fetched_at, payload) VALUES (1, 0, now(), '{}')")


def test_heldrow_checks() -> None:
    """Check a held row needs a failure and non-negative positions."""
    with pytest.raises(ValueError, match="at least one failure"):
        held(failures=())
    with pytest.raises(ValueError, match="0 or more"):
        held(raw_page_no=-1)
    with pytest.raises(ValueError, match="0 or more"):
        held(raw_index=-1)


async def test_hold_outside_transaction(pool: Pool) -> None:
    """Check hold refuses to run outside a transaction."""
    async with pool.connection() as c:
        with pytest.raises(RuntimeError, match="inside the Load transaction"):
            await hold(c, [held()])


async def test_hold_writes_rows(pool: Pool) -> None:
    """Check hold writes each row with its failures and points back to the raw page by task and page."""
    rows = [held(), held(group_key=None, failures=(Failure("a", 1, "x"), Failure("b", 2, "y")))]
    async with pool.connection() as c:
        async with c.transaction():
            assert await hold(c, rows) == 2
            assert await hold(c, []) == 0
        cursor = await c.execute(
            "SELECT status, failures, group_key, resolved_at, task_id, raw_page_no, raw_index "
            "FROM quarantine.row ORDER BY id"
        )
        got = await cursor.fetchall()
        assert got[0] == (
            "open",
            [{"rule": "order_id_present", "version": 1, "message": "id is null"}],
            "o1",
            None,
            1,
            0,
            3,
        )
        assert got[1][1] == [{"rule": "a", "version": 1, "message": "x"}, {"rule": "b", "version": 2, "message": "y"}]


async def test_hold_rolls_back_with_load(pool: Pool) -> None:
    """Check held rows go when the Load transaction rolls back."""
    async with pool.connection() as c:
        with pytest.raises(ZeroDivisionError):
            async with c.transaction():
                await hold(c, [held()])
                1 / 0  # noqa: B018 - the failure that rolls the Load back
        cursor = await c.execute("SELECT count(*) FROM quarantine.row")
        assert await cursor.fetchone() == (0,)


async def test_quarantine_has_no_raw_table_column(pool: Pool) -> None:
    """Check quarantine.row no longer names a raw table."""
    async with pool.connection() as c:
        cursor = await c.execute(
            "SELECT count(*) FROM information_schema.columns "
            "WHERE table_schema = 'quarantine' AND table_name = 'row' AND column_name = 'raw_table'"
        )
        assert await cursor.fetchone() == (0,)
