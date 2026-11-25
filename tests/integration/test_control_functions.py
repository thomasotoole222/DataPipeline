"""Integration tests for the control functions and the published registry, against a database with control loaded."""

import asyncio
import os
from collections.abc import AsyncIterator
from typing import Any, cast
from uuid import UUID

import psycopg
import pytest
from psycopg_pool import AsyncConnectionPool

from pipeline.db.control import PublishedJob, PublishedSource, publish_registry
from pipeline.db.pool import Pool

DSN = os.environ.get("TEST_PIPELINE_DSN", "postgresql://postgres@localhost:5432/pipeline_test")
SHOP = UUID("00000000-0000-0000-0000-00000000000a")
SANDBOX = UUID("00000000-0000-0000-0000-0000000000ff")
ORDERS = PublishedJob("orders", "tiktok_shop", "day")
SHOP_A = PublishedSource("tiktok_shop", SHOP, "Shop A", in_all=True)
SANDBOX_SOURCE = PublishedSource("tiktok_shop", SANDBOX, "SANDBOX", in_all=False)
CREATE = "SELECT task_id, created FROM control.create_tasks(%s, 'tiktok_shop', 'orders', %s, %s, %s, 5, NULL)"
WAIT = 0.3  # seconds a blocked call is given to show that it waits


@pytest.fixture
async def pool() -> AsyncIterator[Pool]:
    """Yield an open pool on empty control tables, the published registry included."""
    p: Pool = AsyncConnectionPool(DSN, min_size=1, max_size=2, kwargs={"autocommit": True}, open=False)
    await p.open(wait=True)
    async with p.connection() as c:
        await c.execute("TRUNCATE control.lease_loss, control.task, control.job, control.source RESTART IDENTITY")
    try:
        yield p
    finally:
        await p.close()


async def _rows(pool: Pool, query: str) -> list[tuple[Any, ...]]:
    """Return every row ``query`` returns."""
    async with pool.connection() as c:
        return await (await c.execute(query)).fetchall()


@pytest.mark.integration
async def test_publish_registry_replaces_both_sets(pool: Pool) -> None:
    """Check a second publish replaces both published sets whole."""
    await publish_registry(pool, [ORDERS, PublishedJob("products", "tiktok_shop", "none")], [SHOP_A, SANDBOX_SOURCE])
    await publish_registry(pool, [ORDERS], [SHOP_A])
    assert await _rows(pool, "SELECT job, connector, partition FROM control.job") == [("orders", "tiktok_shop", "day")]
    assert await _rows(pool, "SELECT connector, source_id, label, in_all FROM control.source") == [
        ("tiktok_shop", SHOP, "Shop A", True)
    ]


@pytest.mark.integration
async def test_publish_registry_with_nothing_empties_both(pool: Pool) -> None:
    """Check an empty publish empties both tables."""
    await publish_registry(pool, [ORDERS], [SHOP_A])
    await publish_registry(pool, [], [])
    assert await _rows(pool, "SELECT (SELECT count(*) FROM control.job), (SELECT count(*) FROM control.source)") == [
        (0, 0)
    ]


@pytest.mark.integration
async def test_publish_registry_leaves_existing_tasks(pool: Pool) -> None:
    """Check a publish without a job leaves that job's tasks as they are."""
    await publish_registry(pool, [ORDERS], [SHOP_A])
    async with pool.connection() as c:
        await c.execute(CREATE, (1, [SHOP], ["2026-11-15"], [0]))
    await publish_registry(pool, [], [])
    assert await _rows(pool, "SELECT job, state FROM control.task") == [("orders", "pending")]


@pytest.mark.integration
async def test_a_second_publisher_waits_and_wins(pool: Pool) -> None:
    """Check a second publish waits for an open one to commit, then replaces its set."""
    async with await psycopg.AsyncConnection.connect(DSN) as first:
        await first.execute(
            "SELECT control.publish_registry(%s, %s, %s, %s, %s, %s, %s)",
            (["orders"], ["tiktok_shop"], ["day"], [], [], [], []),
        )
        second = asyncio.create_task(publish_registry(pool, [PublishedJob("products", "tiktok_shop", "none")], []))
        await asyncio.sleep(WAIT)
        assert not second.done()
        await first.commit()
        await asyncio.wait_for(second, 5)
    assert await _rows(pool, "SELECT job FROM control.job") == [("products",)]


@pytest.mark.integration
async def test_create_tasks_holds_a_publish_until_it_commits(pool: Pool) -> None:
    """Check a publish waits until an open ``create_tasks`` transaction commits."""
    await publish_registry(pool, [ORDERS], [SHOP_A])
    async with await psycopg.AsyncConnection.connect(DSN) as creator:
        await creator.execute(CREATE, (1, [SHOP], ["2026-11-15"], [0]))
        publish = asyncio.create_task(publish_registry(pool, [], []))
        await asyncio.sleep(WAIT)
        assert not publish.done()
        await creator.commit()
        await asyncio.wait_for(publish, 5)
    assert await _rows(pool, "SELECT count(*) FROM control.task") == [(1,)]


@pytest.mark.integration
async def test_two_runs_asking_for_one_key_share_its_task(pool: Pool) -> None:
    """Check a second run asking for a key another run is creating waits, then gets that run's task."""
    await publish_registry(pool, [ORDERS], [SHOP_A])
    async with await psycopg.AsyncConnection.connect(DSN) as first:
        ((first_id, first_created),) = await (await first.execute(CREATE, (1, [SHOP], ["2026-11-15"], [0]))).fetchall()
        async with pool.connection() as c:
            second = asyncio.create_task(c.execute(CREATE, (2, [SHOP], ["2026-11-15"], [0])))
            await asyncio.sleep(WAIT)
            assert not second.done()
            await first.commit()
            ((second_id, second_created),) = await (await asyncio.wait_for(second, 5)).fetchall()
    assert (first_created, second_created, second_id) == (True, False, first_id)


def test_published_job_refuses_an_unknown_partition() -> None:
    """Check ``PublishedJob`` refuses a partition other than day or none."""
    with pytest.raises(ValueError, match="unknown partition 'week'"):
        PublishedJob("orders", "tiktok_shop", cast(Any, "week"))


def test_published_source_refuses_an_empty_label() -> None:
    """Check ``PublishedSource`` refuses an empty label."""
    with pytest.raises(ValueError, match="must not be empty"):
        PublishedSource("tiktok_shop", SHOP, "", in_all=True)
