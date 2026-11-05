"""Integration tests for children under a keyed writer: a ``Versioned`` product, its SKUs, their prices and bundles.

The tables mirror the Cipher Warehouse product dimensions: a versioned product per shop (the source column), a
versioned SKU keyed by its own id that carries its product's id, a one-to-one versioned price keyed by the SKU, and
bundle components replaced under each SKU. Every versioned table has ``YYYYMMDD`` integer day columns and an exclusion
constraint that refuses overlapping versions of one id.
"""

import dataclasses
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import date, datetime, time, timedelta
from decimal import Decimal
from typing import Annotated, Any, ClassVar
from uuid import UUID
from zoneinfo import ZoneInfo

import pytest
from opentelemetry.sdk.metrics import MeterProvider
from psycopg_pool import AsyncConnectionPool
from pydantic import Field

from pipeline.checks.checks import Validation
from pipeline.checks.record import Part, Record
from pipeline.checks.targets import ReplaceChildren, Rows, To, Versioned
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
SOURCE = UUID("00000000-0000-0000-0000-000000000003")
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
DAYS = {"valid_from": "valid_from_key", "valid_to": "valid_to_key"}
TABLES = ("t_kid.product", "t_kid.sku", "t_kid.sku_price", "t_kid.sku_bundle")

type Amount = Annotated[Decimal, Field(max_digits=6, decimal_places=2)]


class Component(Part, table="t_kid.sku_bundle", writer=ReplaceChildren(("component_id",))):
    """A bundle component, replaced under its SKU."""

    id: Annotated[str, To("component_id")]
    quantity: Annotated[int, To("quantity")]


class Price(Part, table="t_kid.sku_price", writer=Versioned((), **DAYS)):
    """A SKU's price: one-to-one, keyed by the SKU id it inherits."""

    amount: Annotated[Amount, To("price")]


class Sku(Part, table="t_kid.sku", writer=Versioned(("sku_id",), **DAYS)):
    """A SKU keyed by its own id, carrying its product's id as a plain column."""

    id: Annotated[str, To("sku_id")]
    seller_sku: Annotated[str | None, To("seller_sku")] = None
    price: Annotated[Price | None, Rows()] = None
    bundle: Annotated[list[Component], Rows()] = Field(default_factory=list)


class Product(
    Record,
    table="t_kid.product",
    writer=Versioned(("shop_id", "product_id"), **DAYS),
    source_column="shop_id",
):
    """A versioned product of a shop, with its SKUs."""

    record_key: ClassVar[tuple[str, ...]] = ("id",)
    record_version: ClassVar[int] = 1

    id: Annotated[str, To("product_id")]
    title: Annotated[str, To("title")]
    skus: Annotated[list[Sku], Rows()] = Field(default_factory=list)


VALIDATION = Validation(model=Product)


@pytest.fixture
async def pool() -> AsyncIterator[Pool]:
    """Yield an open pool on empty control, raw and quarantine tables and fresh product tables."""
    p: Pool = AsyncConnectionPool(DSN, min_size=1, max_size=2, kwargs={"autocommit": True}, open=False)
    await p.open(wait=True)
    days = "valid_from_key integer NOT NULL, valid_to_key integer"
    overlap = "int4range(valid_from_key, valid_to_key, '[]') WITH &&"
    async with p.connection() as c:
        await c.execute("TRUNCATE control.lease_loss, control.task RESTART IDENTITY")
        await c.execute("TRUNCATE raw.page, raw.payload, quarantine.row")
        await c.execute("CREATE EXTENSION IF NOT EXISTS btree_gist")
        await c.execute("DROP SCHEMA IF EXISTS t_kid CASCADE")
        await c.execute("CREATE SCHEMA t_kid")
        await c.execute(
            "CREATE TABLE t_kid.product (shop_id uuid NOT NULL, product_id text NOT NULL, title text NOT NULL, "
            f"{days}, "
            f"PRIMARY KEY (shop_id, product_id, valid_from_key), EXCLUDE USING gist (product_id WITH =, {overlap}))"
        )
        await c.execute(
            f"CREATE TABLE t_kid.sku (product_id text NOT NULL, sku_id text NOT NULL, seller_sku text, {days}, "
            f"PRIMARY KEY (sku_id, valid_from_key), EXCLUDE USING gist (sku_id WITH =, {overlap}))"
        )
        await c.execute(
            f"CREATE TABLE t_kid.sku_price (sku_id text NOT NULL, price numeric(6,2) NOT NULL, {days}, "
            f"PRIMARY KEY (sku_id, valid_from_key), EXCLUDE USING gist (sku_id WITH =, {overlap}))"
        )
        await c.execute(
            "CREATE TABLE t_kid.sku_bundle (sku_id text NOT NULL, component_id text NOT NULL, "
            "quantity bigint NOT NULL, PRIMARY KEY (sku_id, component_id))"
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


def sku(sku_id: str, price: str | None = None, bundle: dict[str, int] | None = None) -> dict[str, Any]:
    """Build a SKU payload with an optional price and bundle components (component id to quantity)."""
    components = [{"id": component, "quantity": quantity} for component, quantity in (bundle or {}).items()]
    return {"id": sku_id, "price": None if price is None else {"amount": price}, "bundle": components}


def product(product_id: str, title: str | None, skus: list[dict[str, Any]]) -> dict[str, Any]:
    """Build a product payload; a ``None`` title makes the record fail the schema check, so it is held."""
    return {"id": product_id, "title": title, "skus": skus}


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


async def rows(pool: Pool, table: str) -> list[tuple[Any, ...]]:
    """Return every row of a ``t_kid`` table but the product's shop, versions as (..., first day, last day)."""
    columns = {
        "product": "product_id, title, valid_from_key, valid_to_key",
        "sku": "product_id, sku_id, valid_from_key, valid_to_key",
        "sku_price": "sku_id, price::text, valid_from_key, valid_to_key",
        "sku_bundle": "sku_id, component_id, quantity",
    }[table]
    async with pool.connection() as c:
        cursor = await c.execute(f"SELECT {columns} FROM t_kid.{table} ORDER BY 1, 2, 3")  # noqa: S608 - test names
        return await cursor.fetchall()


async def test_a_product_its_skus_their_prices_and_bundles_load_in_one_transaction(
    pool: Pool, instruments: Instruments
) -> None:
    """Check one load writes the product, each SKU with its product's id, the one-to-one prices and the bundles."""
    loaded = await run(
        pool,
        instruments,
        date(2026, 11, 4),
        [product("p1", "Lash Kit", [sku("s1", "10.00", {"c1": 2, "c2": 1}), sku("s2")])],
    )
    assert loaded.sent == {"t_kid.product": 1, "t_kid.sku": 2, "t_kid.sku_price": 1, "t_kid.sku_bundle": 2}
    assert await rows(pool, "product") == [("p1", "Lash Kit", 20261104, None)]
    assert await rows(pool, "sku") == [("p1", "s1", 20261104, None), ("p1", "s2", 20261104, None)]
    assert await rows(pool, "sku_price") == [("s1", "10.00", 20261104, None)]  # s2 has no price
    assert await rows(pool, "sku_bundle") == [("s1", "c1", 2), ("s1", "c2", 1)]


async def test_a_sku_dropped_from_its_product_is_closed_and_its_children_are_left(
    pool: Pool, instruments: Instruments
) -> None:
    """Check a SKU no longer listed is closed the day before, a changed price is versioned, and nothing cascades."""
    await run(
        pool, instruments, date(2026, 11, 4), [product("p1", "Lash Kit", [sku("s1", "10.00"), sku("s2", "5.00")])]
    )
    await run(pool, instruments, date(2026, 11, 10), [product("p1", "Lash Kit", [sku("s1", "12.00")])])
    assert await rows(pool, "sku") == [("p1", "s1", 20261104, None), ("p1", "s2", 20261104, 20261109)]
    assert await rows(pool, "sku_price") == [
        ("s1", "10.00", 20261104, 20261109),
        ("s1", "12.00", 20261110, None),
        ("s2", "5.00", 20261104, None),  # closing s2 does not close its price: s2 is not staged, so not its parent
    ]


async def test_a_held_record_does_not_stop_the_other_products_children_being_pruned(
    pool: Pool, instruments: Instruments
) -> None:
    """Check a held product stops the root prune only: another product's dropped SKU is still closed."""
    await run(
        pool,
        instruments,
        date(2026, 11, 4),
        [product("p1", "Lash Kit", [sku("s1"), sku("s2")]), product("p2", "Brow Kit", [sku("s3")])],
    )
    loaded = await run(
        pool, instruments, date(2026, 11, 6), [product("p1", "Lash Kit", [sku("s1")]), product("p3", None, [])]
    )
    assert loaded.held == 1
    assert await rows(pool, "product") == [("p1", "Lash Kit", 20261104, None), ("p2", "Brow Kit", 20261104, None)]
    assert await rows(pool, "sku") == [
        ("p1", "s1", 20261104, None),
        ("p1", "s2", 20261104, 20261105),  # pruned under the staged p1, though a record is held
        ("p2", "s3", 20261104, None),  # p2 is not staged, so its SKUs are left
    ]


async def test_an_empty_child_list_is_valid_and_closes_the_products_skus(pool: Pool, instruments: Instruments) -> None:
    """Check a product with no SKUs loads, without NothingLoaded, and closes the SKUs it listed before."""
    await run(pool, instruments, date(2026, 11, 4), [product("p1", "Lash Kit", [sku("s1", "10.00", {"c1": 1})])])
    loaded = await run(pool, instruments, date(2026, 11, 6), [product("p1", "Lash Kit", [])])
    assert loaded.sent == {"t_kid.product": 1, "t_kid.sku": 0, "t_kid.sku_price": 0, "t_kid.sku_bundle": 0}
    assert await rows(pool, "sku") == [("p1", "s1", 20261104, 20261105)]
    assert await rows(pool, "sku_price") == [("s1", "10.00", 20261104, None)]
    assert await rows(pool, "sku_bundle") == [("s1", "c1", 1)]


async def test_a_price_removed_from_a_listed_sku_is_closed(pool: Pool, instruments: Instruments) -> None:
    """Check a one-to-one child that becomes None is closed under its still-listed parent."""
    await run(pool, instruments, date(2026, 11, 4), [product("p1", "Lash Kit", [sku("s1", "10.00")])])
    await run(pool, instruments, date(2026, 11, 6), [product("p1", "Lash Kit", [sku("s1")])])
    assert await rows(pool, "sku_price") == [("s1", "10.00", 20261104, 20261105)]


async def test_rerunning_unchanged_data_writes_nothing(pool: Pool, instruments: Instruments) -> None:
    """Check observing the same product tree again, the same day or a later one, changes no row in any table."""
    products = [product("p1", "Lash Kit", [sku("s1", "10.00", {"c1": 2}), sku("s2")])]
    await run(pool, instruments, date(2026, 11, 4), products)
    before = [await rows(pool, table.split(".")[1]) for table in TABLES]
    for day in (date(2026, 11, 4), date(2026, 11, 5)):
        loaded = await run(pool, instruments, day, products)
        assert loaded.changed == dict.fromkeys(TABLES, 0)
    assert [await rows(pool, table.split(".")[1]) for table in TABLES] == before
