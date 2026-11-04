"""Integration tests for the ``Versioned`` writer: ``pipeline.writers.sql.versioned`` through ``pipeline.stages.load``.

The table mirrors a Cipher Warehouse versioned dimension: versions keyed by (id, first day), ``YYYYMMDD`` integer day
columns, a generated ``is_current``, and an exclusion constraint that refuses overlapping versions of one id.
"""

import dataclasses
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import date, datetime, time, timedelta
from typing import Annotated, Any, ClassVar
from uuid import UUID
from zoneinfo import ZoneInfo

import pytest
from opentelemetry.sdk.metrics import MeterProvider
from psycopg_pool import AsyncConnectionPool

from pipeline.checks.checks import Validation
from pipeline.checks.record import Record
from pipeline.checks.targets import To, Versioned
from pipeline.connectors.paging import NoPaging
from pipeline.connectors.spec import EndpointSpec
from pipeline.db import control, raw
from pipeline.db.control import ClaimedTask, claim
from pipeline.db.pool import Connection, Pool
from pipeline.db.raw import RawPage
from pipeline.errors import NothingLoaded
from pipeline.stages.load import Loaded, load
from pipeline.stages.transform import transform
from pipeline.stages.validate import validate
from pipeline.telemetry import Instruments
from pipeline.writers import sql

DSN = os.environ.get("TEST_PIPELINE_DSN", "postgresql://postgres@localhost:5432/pipeline_test")
SOURCE = UUID("00000000-0000-0000-0000-000000000002")
LONDON = ZoneInfo("Europe/London")
KEYS = frozenset({"data"})
SPEC = EndpointSpec(
    name="tiktok_shop.products_search",
    connector="tiktok_shop",
    method="POST",
    path="/product/202309/products/search",
    paginator=NoPaging(),
    records_at=("data", "products"),
)


class Product(
    Record,
    table="t_ver.product",
    writer=Versioned(
        ("source_id", "product_id"), valid_from="valid_from_key", valid_to="valid_to_key", opens_from="created_date_key"
    ),
):
    """A product whose title is versioned; its first version starts on its created day."""

    record_key: ClassVar[tuple[str, ...]] = ("id",)
    record_version: ClassVar[int] = 1

    id: Annotated[str, To("product_id")]
    title: Annotated[str, To("title")]
    created: Annotated[int | None, To("created_date_key")] = None


VALIDATION = Validation(model=Product)


@pytest.fixture
async def pool() -> AsyncIterator[Pool]:
    """Yield an open pool on empty control, raw and quarantine tables and a fresh versioned table."""
    p: Pool = AsyncConnectionPool(DSN, min_size=1, max_size=2, kwargs={"autocommit": True}, open=False)
    await p.open(wait=True)
    async with p.connection() as c:
        await c.execute("TRUNCATE control.lease_loss, control.task RESTART IDENTITY")
        await c.execute("TRUNCATE raw.page, raw.payload, quarantine.row")
        await c.execute("CREATE EXTENSION IF NOT EXISTS btree_gist")
        await c.execute("DROP SCHEMA IF EXISTS t_ver CASCADE")
        await c.execute("CREATE SCHEMA t_ver")
        await c.execute(
            "CREATE TABLE t_ver.product (source_id uuid NOT NULL, product_id text NOT NULL, title text NOT NULL, "
            "created_date_key bigint, valid_from_key integer NOT NULL, valid_to_key integer, "
            "is_current boolean GENERATED ALWAYS AS (valid_to_key IS NULL) STORED, "
            "PRIMARY KEY (product_id, valid_from_key), "
            "EXCLUDE USING gist (product_id WITH =, int4range(valid_from_key, valid_to_key, '[]') WITH &&))"
        )
    try:
        yield p
    finally:
        await p.close()


@pytest.fixture
def instruments() -> Instruments:
    """Build the pipeline's instruments on a throwaway meter provider."""
    meter = MeterProvider().get_meter("test")
    return Instruments(
        task_duration=meter.create_histogram("pipeline.task.duration", unit="s"),
        rows=meter.create_counter("pipeline.rows"),
        connector_requests=meter.create_counter("pipeline.connector.requests"),
        log_dropped=meter.create_counter("pipeline.log.dropped"),
        slots_busy=meter.create_up_down_counter("pipeline.worker.slots_busy"),
        queue_wait=meter.create_histogram("pipeline.task.queue_wait", unit="s"),
        heartbeat_lag=meter.create_histogram("pipeline.lease.heartbeat_lag", unit="s"),
        lease_losses=meter.create_counter("pipeline.lease.losses"),
        lease_unwind=meter.create_histogram("pipeline.lease.unwind", unit="s"),
        unknown_jobs=meter.create_counter("pipeline.task.unknown_job"),
        worker_paused=meter.create_counter("pipeline.worker.paused"),
        check_failures=meter.create_counter("pipeline.check.failures"),
        text_replaced=meter.create_counter("pipeline.connector.text_replaced"),
    )


class FakeFence:
    """A ``Fence`` that opens a transaction and runs ``control.fence`` in it, as the worker's ``Lease.fenced`` does."""

    def __init__(self, task: ClaimedTask) -> None:
        """Fence on ``task``."""
        self.task = task

    @asynccontextmanager
    async def fenced(self, conn: Connection) -> AsyncIterator[None]:
        """Open a transaction on ``conn``, fence it on the task's lease, and run the block."""
        async with conn.transaction():
            await control.fence(conn, self.task)
            yield


def product(product_id: str, title: str | None, created: int | None = None) -> dict[str, Any]:
    """Build a product payload; a ``None`` title makes the record fail the schema check, so it is held."""
    return {"id": product_id, "title": title, "created": created}


async def run(pool: Pool, instruments: Instruments, day: date, products: list[dict[str, Any]]) -> Loaded:
    """Land ``products`` as one page of a new task observed on ``day`` in London, then validate, transform and load."""
    async with pool.connection() as c:
        await c.execute(
            "INSERT INTO control.task (run_id, connector, job, source_id, partition_key) "
            "VALUES (7, 'tiktok_shop', 'products', %s, gen_random_uuid()::text)",
            (SOURCE,),
        )
        claimed = await claim(c, "test-worker")
    assert claimed is not None
    at = datetime.combine(day, time(12), LONDON)  # the observation day; the lease moves with it, as ClaimedTask checks
    task = dataclasses.replace(claimed, claimed_at=at, lease_expires_at=at + timedelta(minutes=5))
    body = {"code": 0, "data": {"products": products}}
    page = RawPage(
        task_id=task.id,
        page_no=0,
        connector="tiktok_shop",
        endpoint="products_search",
        cursor_in=None,
        cursor_next=None,
        fetched_at=task.claimed_at,
        payload=body,
    )
    async with pool.connection() as c:
        await raw.save_page(c, page, KEYS)
    validated = await validate(pool, task, SPEC, None, VALIDATION, instruments)
    transformed = transform(Product, validated.accepted, task.source_id)
    return await load(
        pool, task, FakeFence(task), validated, transformed, None, instruments, policy=VALIDATION.policy, zone=LONDON
    )


async def versions(pool: Pool) -> list[tuple[Any, ...]]:
    """Return every version as (product, title, first day, last day, current), oldest first."""
    async with pool.connection() as c:
        cursor = await c.execute(
            "SELECT product_id, title, valid_from_key, valid_to_key, is_current FROM t_ver.product "
            "ORDER BY product_id, valid_from_key"
        )
        return await cursor.fetchall()


async def test_a_new_id_opens_from_its_created_day_or_the_observation_day(pool: Pool, instruments: Instruments) -> None:
    """Check a new id's first version starts at opens_from, and at the observation day when that is null."""
    await run(pool, instruments, date(2026, 11, 4), [product("p1", "Lash", 20260301), product("p2", "Brow")])
    assert await versions(pool) == [("p1", "Lash", 20260301, None, True), ("p2", "Brow", 20261104, None, True)]


async def test_an_unchanged_id_writes_nothing(pool: Pool, instruments: Instruments) -> None:
    """Check observing the same details again changes no row."""
    await run(pool, instruments, date(2026, 11, 4), [product("p1", "Lash", 20260301)])
    loaded = await run(pool, instruments, date(2026, 11, 5), [product("p1", "Lash", 20260301)])
    assert loaded.changed == {"t_ver.product": 0}
    assert await versions(pool) == [("p1", "Lash", 20260301, None, True)]


async def test_a_change_closes_the_open_version_and_opens_one_from_the_day(
    pool: Pool, instruments: Instruments
) -> None:
    """Check a changed id's open version closes the day before and a new version opens on the day."""
    await run(pool, instruments, date(2026, 11, 4), [product("p1", "Lash", 20260301)])
    await run(pool, instruments, date(2026, 11, 10), [product("p1", "Lash Pro", 20260301)])
    assert await versions(pool) == [
        ("p1", "Lash", 20260301, 20261109, False),
        ("p1", "Lash Pro", 20261110, None, True),
    ]


async def test_a_second_change_on_the_same_day_overwrites_the_days_version(
    pool: Pool, instruments: Instruments
) -> None:
    """Check a version opened today is overwritten in place by a later change the same day."""
    await run(pool, instruments, date(2026, 11, 4), [product("p1", "Lash", 20260301)])
    await run(pool, instruments, date(2026, 11, 10), [product("p1", "Lash Pro", 20260301)])
    await run(pool, instruments, date(2026, 11, 10), [product("p1", "Lash Max", 20260301)])
    assert await versions(pool) == [
        ("p1", "Lash", 20260301, 20261109, False),
        ("p1", "Lash Max", 20261110, None, True),
    ]


async def test_a_missing_id_is_closed_and_comes_back_as_a_new_version(pool: Pool, instruments: Instruments) -> None:
    """Check an id missing from a response is closed, and opens a new version from the day it returns."""
    await run(pool, instruments, date(2026, 11, 4), [product("p1", "Lash", 20260301), product("p2", "Brow")])
    await run(pool, instruments, date(2026, 11, 6), [product("p2", "Brow")])
    await run(pool, instruments, date(2026, 11, 9), [product("p1", "Lash", 20260301), product("p2", "Brow")])
    assert await versions(pool) == [
        ("p1", "Lash", 20260301, 20261105, False),
        ("p1", "Lash", 20261109, None, True),
        ("p2", "Brow", 20261104, None, True),
    ]


async def test_a_held_record_stops_missing_ids_being_closed(pool: Pool, instruments: Instruments) -> None:
    """Check no missing id is closed while a record is held, since the held one may be the missing one."""
    await run(pool, instruments, date(2026, 11, 4), [product("p1", "Lash", 20260301), product("p2", "Brow")])
    loaded = await run(pool, instruments, date(2026, 11, 6), [product("p2", "Brow"), product("p3", None)])
    assert loaded.held == 1
    assert [row[:4] for row in await versions(pool)] == [
        ("p1", "Lash", 20260301, None),
        ("p2", "Brow", 20261104, None),
    ]


async def test_an_observation_older_than_the_open_version_is_left_alone(pool: Pool, instruments: Instruments) -> None:
    """Check a stale observation rewrites no history: the open version that started later stays as it is."""
    await run(pool, instruments, date(2026, 11, 10), [product("p1", "Lash")])
    await run(pool, instruments, date(2026, 11, 8), [product("p1", "Lash Old")])
    assert await versions(pool) == [("p1", "Lash", 20261110, None, True)]


async def test_an_empty_response_is_refused_when_pruning(pool: Pool, instruments: Instruments) -> None:
    """Check an empty response would close every id of the source, so it is refused and nothing is written."""
    await run(pool, instruments, date(2026, 11, 4), [product("p1", "Lash", 20260301)])
    with pytest.raises(NothingLoaded):
        await run(pool, instruments, date(2026, 11, 5), [])
    assert await versions(pool) == [("p1", "Lash", 20260301, None, True)]


async def test_versioned_returns_the_stale_count_and_handles_date_columns(pool: Pool) -> None:
    """Check ``sql.versioned`` on ``date`` day columns, and that it counts a stale key without changing it."""
    async with pool.connection() as c, c.transaction():
        await c.execute(
            "CREATE TABLE t_ver.price (sku_id text NOT NULL, price numeric(6,2) NOT NULL, "
            "valid_from date NOT NULL, valid_to date, PRIMARY KEY (sku_id, valid_from))"
        )
        await c.execute("INSERT INTO t_ver.price VALUES ('s1', 10, '2026-11-10', NULL), ('s2', 5, '2026-11-01', NULL)")
        await c.execute("CREATE TEMP TABLE staged (sku_id text, price numeric(6,2)) ON COMMIT DROP")
        await c.execute("INSERT INTO staged VALUES ('s1', 12), ('s2', 6)")
        changed, stale = await sql.versioned(
            c, "t_ver.price", "staged", ["sku_id", "price"], ["sku_id"], None, None,
            "valid_from", "valid_to", None, date(2026, 11, 5), date(2026, 11, 4), prune=True,
        )  # fmt: skip
        cursor = await c.execute("SELECT sku_id, price, valid_from, valid_to FROM t_ver.price ORDER BY 1, 3")
        found = await cursor.fetchall()
    assert stale == 1  # s1's open version starts after the observation day
    assert changed == 2  # s2: one version closed, one opened
    assert [(s, int(p), f, t) for s, p, f, t in found] == [
        ("s1", 10, date(2026, 11, 10), None),
        ("s2", 5, date(2026, 11, 1), date(2026, 11, 4)),
        ("s2", 6, date(2026, 11, 5), None),
    ]
