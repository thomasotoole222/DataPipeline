"""Integration tests for a container record: one product-day record fills two fact tables and no table of its own.

The container has no table; its key (shop, product, day) is staged for its children only. Sales by traffic source are
replaced under each product-day, and the rating is a one-to-one ``Upsert`` keyed by the product-day.
"""

import dataclasses
import os
from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager
from datetime import date, datetime, time, timedelta
from typing import Annotated, Any, ClassVar
from uuid import UUID
from zoneinfo import ZoneInfo

import pytest
from opentelemetry.sdk.metrics import MeterProvider
from psycopg_pool import AsyncConnectionPool
from pydantic import Field

from pipeline.checks.checks import Lookup, Severity, Validation
from pipeline.checks.record import DateKey, Int32, Part, Record, TaskDay
from pipeline.checks.targets import ReplaceChildren, Rows, To, Upsert
from pipeline.connectors.paging import NoPaging
from pipeline.connectors.spec import EndpointSpec
from pipeline.db import control, raw
from pipeline.db.control import ClaimedTask, claim
from pipeline.db.pool import Connection, Pool
from pipeline.db.raw import RawPage
from pipeline.stages.load import Loaded, load
from pipeline.stages.transform import transform
from pipeline.stages.validate import validate
from pipeline.telemetry import Instruments

DSN = os.environ.get("TEST_PIPELINE_DSN", "postgresql://postgres@localhost:5432/pipeline_test")
SOURCE = UUID("00000000-0000-0000-0000-000000000004")
OTHER_SOURCE = UUID("00000000-0000-0000-0000-000000000005")
LONDON = ZoneInfo("Europe/London")
DAY = date(2026, 11, 4)
KEYS = frozenset({"data"})
SPEC = EndpointSpec(
    name="tiktok_shop.product_performance",
    connector="tiktok_shop",
    method="GET",
    path="/analytics/202405/shop_products/performance",
    paginator=NoPaging(),
    records_at=("data", "intervals"),
)
TABLES = ("t_box.sales", "t_box.rating")


class Sales(Part, table="t_box.sales", writer=ReplaceChildren(("source_key",))):
    """A product-day's sales from one traffic source, replaced under its product-day."""

    source_key: Annotated[int, To("source_key")]
    gmv: Annotated[int, To("gmv")]


class Rating(Part, table="t_box.rating", writer=Upsert(())):
    """A product-day's rating: one-to-one, keyed by the product-day it inherits."""

    stars: Annotated[int, To("stars")]


class ProductDay(Record, container=True):
    """One product's day of performance, with no table of its own."""

    record_key: ClassVar[tuple[str, ...]] = ("product", "day")
    record_version: ClassVar[int] = 1

    product: Annotated[str, To("tiktok_product_id")]
    day: Annotated[Int32, To("date_key")]
    sales: Annotated[list[Sales], Rows()] = Field(default_factory=list)
    rating: Annotated[Rating | None, Rows()] = None


VALIDATION = Validation(model=ProductDay)


class TaskProductDay(Record, container=True):
    """One product's day of performance, dated by the task's day: the response carries no date."""

    record_key: ClassVar[tuple[str, ...]] = ("product", "day")
    record_version: ClassVar[int] = 1

    product: Annotated[str, To("tiktok_product_id")]
    day: Annotated[Int32, TaskDay(), To("date_key")]
    sales: Annotated[list[Sales], Rows()] = Field(default_factory=list)


TASK_DAY = Validation(model=TaskProductDay)


class ReportProductDay(Record, container=True):
    """One product's day of performance from a report, dated by its ISO ``stat_time_day``."""

    record_key: ClassVar[tuple[str, ...]] = ("product", "stat_time_day")
    record_version: ClassVar[int] = 1

    product: Annotated[str, To("tiktok_product_id")]
    stat_time_day: Annotated[DateKey, To("date_key")]
    sales: Annotated[list[Sales], Rows()] = Field(default_factory=list)


REPORT = Validation(model=ReportProductDay)

#: The traffic source of each named block of a product-day response.
BLOCKS = {"total": 1, "live": 2}


def blocks_to_sales(record: Mapping[str, Any]) -> Mapping[str, Any]:
    """Build a product-day from named blocks: ``id`` as the product, one sales row per block, the TaskDay kept.

    Raises ``KeyError`` for a record with no ``blocks``, and reads ``day`` only as the TaskDay filled it.
    """
    sales = [{"source_key": BLOCKS[name], "gmv": block["gmv"]} for name, block in record["blocks"].items()]
    return {"product": record["id"], "day": record["day"], "sales": sales}


class ShapedProductDay(Record, container=True):
    """One product's day of performance, its sales rows built from the response's named blocks."""

    record_key: ClassVar[tuple[str, ...]] = ("product", "day")
    record_version: ClassVar[int] = 2
    reshape = blocks_to_sales

    product: Annotated[str, To("tiktok_product_id")]
    day: Annotated[Int32, TaskDay(), To("date_key")]
    sales: Annotated[list[Sales], Rows()] = Field(default_factory=list)


SHAPED = Validation(model=ShapedProductDay)

#: A sales row whose traffic source is not in t_box.source is dropped; the product-day loads its other rows.
KNOWN_SOURCES = Validation(
    model=ProductDay,
    checks=(
        Lookup[Sales](
            "known_source",
            lambda row: [str(row.source_key)],
            table="t_box.source",
            column="source_key",
            severity=Severity.DROP,
            rows="sales",
        ),
    ),
)

#: A product-day waits until its product is known for the task's own source; one known for another source is not.
OWN_PRODUCTS = Validation(
    model=ProductDay,
    checks=(
        Lookup[ProductDay](
            "own_product",
            lambda record: [record.product],
            table="t_box.product",
            column="tiktok_product_id",
            until="products",
            source_column="source_id",
        ),
    ),
)
#: The same lookup across the whole table, without ``source_column``.
ANY_PRODUCTS = Validation(
    model=ProductDay,
    checks=(
        Lookup[ProductDay](
            "any_product", lambda record: [record.product], table="t_box.product", column="tiktok_product_id", until="p"
        ),
    ),
)


@pytest.fixture
async def pool() -> AsyncIterator[Pool]:
    """Yield an open pool on empty control, raw and quarantine tables and fresh fact tables."""
    p: Pool = AsyncConnectionPool(DSN, min_size=1, max_size=2, kwargs={"autocommit": True}, open=False)
    await p.open(wait=True)
    key = "source_id uuid NOT NULL, tiktok_product_id text NOT NULL, date_key integer NOT NULL"
    async with p.connection() as c:
        await c.execute("TRUNCATE control.lease_loss, control.task RESTART IDENTITY")
        await c.execute("TRUNCATE raw.page, raw.payload, quarantine.row")
        await c.execute("DROP SCHEMA IF EXISTS t_box CASCADE")
        await c.execute("CREATE SCHEMA t_box")
        await c.execute(
            f"CREATE TABLE t_box.sales ({key}, source_key bigint NOT NULL, gmv bigint NOT NULL, "
            "PRIMARY KEY (source_id, tiktok_product_id, date_key, source_key))"
        )
        await c.execute(
            f"CREATE TABLE t_box.rating ({key}, stars bigint NOT NULL, "
            "PRIMARY KEY (source_id, tiktok_product_id, date_key))"
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


def interval(product: str, sales: dict[int, int], stars: int | None = None) -> dict[str, Any]:
    """Build a product-day payload on ``DAY``: GMV by traffic source, and an optional rating."""
    return {
        "product": product,
        "day": DAY.year * 10000 + DAY.month * 100 + DAY.day,
        "sales": [{"source_key": source, "gmv": gmv} for source, gmv in sales.items()],
        "rating": None if stars is None else {"stars": stars},
    }


async def run(
    pool: Pool,
    instruments: Instruments,
    intervals: list[dict[str, Any]],
    validation: Validation[Any] = VALIDATION,
    partition_key: str | None = None,
) -> Loaded:
    """Land ``intervals`` as one page of a new task, then validate, transform and load.

    The task's partition key is ``partition_key``, or a fresh uuid when it is None, so tasks never share one.
    """
    async with pool.connection() as c:
        await c.execute(
            "INSERT INTO control.task (run_id, connector, job, source_id, partition_key) "
            "VALUES (7, 'tiktok_shop', 'product_days', %s, coalesce(%s, gen_random_uuid()::text))",
            (SOURCE, partition_key),
        )
        claimed = await claim(c, "test-worker")
    assert claimed is not None
    at = datetime.combine(DAY, time(12), LONDON)
    task = dataclasses.replace(claimed, claimed_at=at, lease_expires_at=at + timedelta(minutes=5))
    page = RawPage(
        task_id=task.id,
        page_no=0,
        connector="tiktok_shop",
        endpoint="product_performance",
        cursor_in=None,
        cursor_next=None,
        fetched_at=task.claimed_at,
        payload={"code": 0, "data": {"intervals": intervals}},
    )
    async with pool.connection() as c:
        await raw.save_page(c, page, KEYS)
    validated = await validate(pool, task, SPEC, None, validation, instruments)
    transformed = transform(validation.model, validated.accepted, task.source_id)
    return await load(
        pool, task, FakeFence(task), validated, transformed, None, instruments, policy=validation.policy, zone=LONDON
    )


async def rows(pool: Pool, table: str) -> list[tuple[Any, ...]]:
    """Return every row of a ``t_box`` table without its source column, in key order."""
    columns = {"t_box.sales": "tiktok_product_id, source_key, gmv", "t_box.rating": "tiktok_product_id, stars"}
    async with pool.connection() as c:
        cursor = await c.execute(f"SELECT {columns[table]} FROM {table} ORDER BY 1, 2")  # noqa: S608 - test names
        return await cursor.fetchall()


async def test_one_container_record_fills_two_tables_and_writes_none_of_its_own(
    pool: Pool, instruments: Instruments
) -> None:
    """Check a product-day writes its sales and rating with its key, and nothing is sent for the container."""
    loaded = await run(pool, instruments, [interval("p1", {1: 100, 2: 50}, stars=4), interval("p2", {1: 7})])
    assert loaded.sent == {"t_box.sales": 3, "t_box.rating": 1}
    assert await rows(pool, "t_box.sales") == [("p1", 1, 100), ("p1", 2, 50), ("p2", 1, 7)]
    assert await rows(pool, "t_box.rating") == [("p1", 4)]


async def test_rerunning_unchanged_data_writes_nothing(pool: Pool, instruments: Instruments) -> None:
    """Check loading the same product-days again changes no row in either table."""
    intervals = [interval("p1", {1: 100, 2: 50}, stars=4)]
    await run(pool, instruments, intervals)
    loaded = await run(pool, instruments, intervals)
    assert loaded.changed == dict.fromkeys(TABLES, 0)
    assert await rows(pool, "t_box.sales") == [("p1", 1, 100), ("p1", 2, 50)]


async def test_an_empty_child_list_is_valid_and_replaces_only_its_own_product_day(
    pool: Pool, instruments: Instruments
) -> None:
    """Check a product-day with no sales loads, clears its own sales, and leaves product-days not in the batch."""
    await run(pool, instruments, [interval("p1", {1: 100}), interval("p2", {1: 7})])
    loaded = await run(pool, instruments, [interval("p1", {})])
    assert loaded.sent == {"t_box.sales": 0, "t_box.rating": 0}
    assert await rows(pool, "t_box.sales") == [("p2", 1, 7)]  # p1's replaced by none; p2 is not staged


async def test_an_empty_batch_is_valid(pool: Pool, instruments: Instruments) -> None:
    """Check a batch with no product-days writes nothing and removes nothing: no container key is staged."""
    await run(pool, instruments, [interval("p1", {1: 100}, stars=3)])
    loaded = await run(pool, instruments, [])
    assert loaded.changed == dict.fromkeys(TABLES, 0)
    assert await rows(pool, "t_box.sales") == [("p1", 1, 100)]


async def test_a_task_day_key_is_filled_from_the_partition_over_any_value_landed(
    pool: Pool, instruments: Instruments
) -> None:
    """Check a TaskDay key field gets the task's day as YYYYMMDD, whether the record left it out or carried another."""
    intervals = [{"product": "p1", "sales": [{"source_key": 1, "gmv": 5}]}, {**interval("p2", {1: 7}), "day": 1}]
    loaded = await run(pool, instruments, intervals, TASK_DAY, "2026-11-03")
    assert loaded.sent == {"t_box.sales": 2}
    async with pool.connection() as c:
        cursor = await c.execute("SELECT tiktok_product_id, date_key, gmv FROM t_box.sales ORDER BY 1")
        assert await cursor.fetchall() == [("p1", 20261103, 5), ("p2", 20261103, 7)]


async def test_an_iso_date_key_loads_as_yyyymmdd_and_a_day_that_does_not_exist_is_held(
    pool: Pool, instruments: Instruments
) -> None:
    """Check a DateKey key field loads "2026-11-04" as 20261104, and a record dated 2026-02-30 is held, not loaded."""
    days: list[dict[str, Any]] = [
        {"product": "p1", "stat_time_day": "2026-11-04", "sales": [{"source_key": 1, "gmv": 5}]},
        {"product": "p2", "stat_time_day": "2026-02-30", "sales": [{"source_key": 1, "gmv": 7}]},
        {"product": "p3", "stat_time_day": 20261105, "sales": [{"source_key": 2, "gmv": 9}]},
    ]
    loaded = await run(pool, instruments, days, REPORT)
    assert loaded.held == 1
    async with pool.connection() as c:
        cursor = await c.execute("SELECT tiktok_product_id, date_key, gmv FROM t_box.sales ORDER BY 1")
        assert await cursor.fetchall() == [("p1", 20261104, 5), ("p3", 20261105, 9)]


async def test_a_reshape_builds_child_rows_from_named_blocks_after_the_task_day_is_filled(
    pool: Pool, instruments: Instruments
) -> None:
    """Check reshape runs on the record with its TaskDay filled, and a record it fails on is held as reshape_failed.

    The held row keeps the record as landed, not as reshaped, and has no group key.
    """
    records = [{"id": "p1", "blocks": {"total": {"gmv": 10}, "live": {"gmv": 3}}}, {"id": "p2", "no_blocks": {}}]
    loaded = await run(pool, instruments, records, SHAPED, "2026-11-03")
    assert loaded.held == 1
    async with pool.connection() as c:
        cursor = await c.execute("SELECT tiktok_product_id, date_key, source_key, gmv FROM t_box.sales ORDER BY 3")
        assert await cursor.fetchall() == [("p1", 20261103, 1, 10), ("p1", 20261103, 2, 3)]
        cursor = await c.execute("SELECT group_key, payload, failures FROM quarantine.row")
        assert await cursor.fetchall() == [
            (
                None,
                {"id": "p2", "no_blocks": {}, "day": 20261103},
                [{"rule": "reshape_failed", "version": 2, "message": "KeyError: 'blocks'", "until": None}],
            )
        ]


async def test_a_drop_lookup_removes_only_the_child_rows_it_does_not_find(pool: Pool, instruments: Instruments) -> None:
    """Check a child row whose source is unknown is dropped, its record's other rows load, and nothing is held.

    The product-day already had a row for the unknown source, which the replace then deletes: Transform saw only the
    kept rows.
    """
    async with pool.connection() as c:
        await c.execute("CREATE TABLE t_box.source (source_key bigint PRIMARY KEY)")
        await c.execute("INSERT INTO t_box.source VALUES (1), (2)")
    await run(pool, instruments, [interval("p1", {1: 100, 9: 50})])  # loaded with no checks: 9 is in
    loaded = await run(pool, instruments, [interval("p1", {1: 100, 9: 50}), interval("p2", {2: 7})], KNOWN_SOURCES)
    assert loaded.held == 0
    assert loaded.sent == {"t_box.sales": 2, "t_box.rating": 0}
    assert await rows(pool, "t_box.sales") == [("p1", 1, 100), ("p2", 2, 7)]


async def test_a_source_scoped_lookup_waits_for_a_value_known_only_for_another_source(
    pool: Pool, instruments: Instruments
) -> None:
    """Check a lookup with source_column finds a product only in the task's source's rows, and holds the other.

    p1 is known only for another source, p2 for the task's own. The same lookup across the whole table finds both.
    """
    async with pool.connection() as c:
        await c.execute("CREATE TABLE t_box.product (source_id uuid NOT NULL, tiktok_product_id text NOT NULL)")
        await c.execute("INSERT INTO t_box.product VALUES (%s, 'p1'), (%s, 'p2')", (OTHER_SOURCE, SOURCE))
    intervals = [interval("p1", {1: 100}), interval("p2", {1: 7})]
    loaded = await run(pool, instruments, intervals, OWN_PRODUCTS)
    assert loaded.held == 1
    assert await rows(pool, "t_box.sales") == [("p2", 1, 7)]
    async with pool.connection() as c:
        cursor = await c.execute("SELECT group_key, failures FROM quarantine.row")
        assert await cursor.fetchall() == [
            (
                '["p1","20261104"]',
                [
                    {
                        "rule": "own_product",
                        "version": 1,
                        "message": "not in t_box.product.tiktok_product_id for this source_id: p1",
                        "until": "products",
                    }
                ],
            )
        ]
    loaded = await run(pool, instruments, intervals, ANY_PRODUCTS)
    assert loaded.held == 0
