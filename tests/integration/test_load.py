"""Integration tests for ``pipeline.stages.load`` and ``pipeline.jobs.base`` on test tables, control and raw."""

import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, date, datetime
from decimal import Decimal
from typing import Annotated, Any, ClassVar
from uuid import UUID
from zoneinfo import ZoneInfo

import pytest
from opentelemetry.sdk.metrics import MeterProvider
from psycopg_pool import AsyncConnectionPool
from pydantic import Field, computed_field

from pipeline.checks.checks import Validation
from pipeline.checks.record import Part, Record
from pipeline.checks.targets import ReplaceChildren, Rows, To, Upsert
from pipeline.connectors.paging import CursorPaginator
from pipeline.connectors.spec import ConnectorRequest, EndpointSpec
from pipeline.db import control, raw
from pipeline.db.control import ClaimedTask, TaskReport, claim
from pipeline.db.pool import Connection, Pool
from pipeline.db.raw import RawPage
from pipeline.errors import QualityFailed, TableMismatch
from pipeline.jobs.base import BatchJob
from pipeline.stages.load import Loaded, load
from pipeline.stages.transform import transform
from pipeline.stages.validate import validate
from pipeline.telemetry import Instruments
from pipeline.timeutil import TimeFrame

DSN = os.environ.get("TEST_PIPELINE_DSN", "postgresql://postgres@localhost:5432/pipeline_test")
SOURCE = UUID("00000000-0000-0000-0000-000000000001")
FRAME = TimeFrame.days(date(2026, 10, 19), date(2026, 10, 20), ZoneInfo("Europe/London"))
KEYS = frozenset({"data"})
SPEC = EndpointSpec(
    name="tiktok_shop.orders_search",
    connector="tiktok_shop",
    method="POST",
    path="/order/202309/orders/search",
    paginator=CursorPaginator(
        token_param="page_token",  # noqa: S106 - a parameter name, not a password
        size_param="page_size",
        size=100,
        next_at=("data", "next_page_token"),
    ),
    records_at=("data", "orders"),
)

type Amount = Annotated[Decimal, Field(max_digits=6, decimal_places=2)]


class Line(Part, table="t_load.line", writer=ReplaceChildren(key=("line_id",))):
    """An order line, replaced under its order."""

    id: Annotated[str, To("line_id")]
    amount: Annotated[Amount, To("amount")]


class Order(Record, table="t_load.order", writer=Upsert(("source_id", "order_id"))):
    """An order whose total is the sum of its lines."""

    record_key: ClassVar[tuple[str, ...]] = ("id",)
    record_version: ClassVar[int] = 3

    id: Annotated[str, To("order_id")]
    lines: Annotated[list[Line], Rows()] = Field(default_factory=list)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def total(self) -> Annotated[Amount, To("total")]:
        """Return the sum of the line amounts.

        Preconditions:
            None.

        Returns:
            The sum, ``0`` when there are no lines.

        Raises:
            Nothing.

        """
        return sum((line.amount for line in self.lines), Decimal(0))


VALIDATION = Validation(model=Order)


@pytest.fixture
async def pool() -> AsyncIterator[Pool]:
    """Yield an open pool on empty control, raw and quarantine tables and fresh test tables."""
    p: Pool = AsyncConnectionPool(DSN, min_size=1, max_size=2, kwargs={"autocommit": True}, open=False)
    await p.open(wait=True)
    async with p.connection() as c:
        await c.execute("TRUNCATE control.lease_loss, control.task RESTART IDENTITY")
        await c.execute("TRUNCATE raw.page, raw.payload, quarantine.row")
        await c.execute("DROP SCHEMA IF EXISTS t_load CASCADE")
        await c.execute("CREATE SCHEMA t_load")
        await c.execute(
            'CREATE TABLE t_load."order" (source_id uuid NOT NULL, order_id text NOT NULL, '
            "total numeric(6,2) NOT NULL, PRIMARY KEY (source_id, order_id))"
        )
        await c.execute(
            "CREATE TABLE t_load.line (source_id uuid NOT NULL, order_id text NOT NULL, line_id text NOT NULL, "
            "amount numeric(6,2) NOT NULL, PRIMARY KEY (source_id, order_id, line_id))"
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


def order(order_id: str, *lines: tuple[str, str]) -> dict[str, Any]:
    """Build an order payload with ``(line id, amount)`` lines."""
    return {"id": order_id, "lines": [{"id": line, "amount": amount} for line, amount in lines]}


async def claimed(pool: Pool) -> ClaimedTask:
    """Insert a pending orders task and claim it."""
    async with pool.connection() as c:
        await c.execute(
            "INSERT INTO control.task (run_id, connector, job, source_id, partition_key) "
            "VALUES (7, 'tiktok_shop', 'orders', %s, gen_random_uuid()::text)",  # a new partition, so tasks never clash
            (SOURCE,),
        )
        task = await claim(c, "test-worker")
    assert task is not None
    return task


async def run(pool: Pool, instruments: Instruments, orders: list[dict[str, Any]]) -> Loaded:
    """Land ``orders`` as one page of a new task, then validate, transform and load them."""
    task = await claimed(pool)
    body = {"code": 0, "request_id": f"r{task.id}", "data": {"orders": orders, "next_page_token": ""}}
    page = RawPage(
        task_id=task.id,
        page_no=0,
        connector="tiktok_shop",
        endpoint="orders_search",
        cursor_in=None,
        cursor_next=None,
        fetched_at=datetime(2026, 10, 19, 12, tzinfo=UTC),
        payload=body,
    )
    async with pool.connection() as c:
        await raw.save_page(c, page, KEYS)
    validated = await validate(pool, task, SPEC, FRAME, VALIDATION, instruments)
    transformed = transform(Order, validated.accepted, task.source_id)
    return await load(pool, task, FakeFence(task), validated, transformed, FRAME, instruments, policy=VALIDATION.policy)


async def rows(pool: Pool, query: str) -> list[tuple[Any, ...]]:
    """Return every row ``query`` selects."""
    async with pool.connection() as c:
        cursor = await c.execute(query)
        return await cursor.fetchall()


async def test_a_record_with_a_repeated_child_key_is_held_and_the_rest_loads(
    pool: Pool, instruments: Instruments
) -> None:
    """Check a record whose lines share a key is held as KeyViolation and its neighbour loads."""
    loaded = await run(pool, instruments, [order("o1", ("l1", "1.00"), ("l1", "2.00")), order("o2", ("l2", "3.00"))])
    assert loaded.held == 1
    assert await rows(pool, 'SELECT order_id FROM t_load."order"') == [("o2",)]
    assert await rows(pool, "SELECT order_id, line_id FROM t_load.line") == [("o2", "l2")]
    ((group_key, failures),) = await rows(pool, "SELECT group_key, failures FROM quarantine.row")
    assert group_key == '["o1"]'
    assert failures == [
        {"rule": "KeyViolation", "version": 3, "message": "t_load.line: another row has the same key", "until": None}
    ]


async def test_a_derived_value_that_overflows_holds_only_its_record(pool: Pool, instruments: Instruments) -> None:
    """Check a summed total too large for its column holds that record as Overflow, and its neighbour loads."""
    loaded = await run(pool, instruments, [order("o1", ("a", "9999.99"), ("b", "9999.99")), order("o2", ("c", "1.00"))])
    assert loaded.held == 1
    assert await rows(pool, 'SELECT order_id, total FROM t_load."order"') == [("o2", Decimal("1.00"))]
    ((failures,),) = await rows(pool, "SELECT failures FROM quarantine.row")
    assert failures == [
        {
            "rule": "Overflow",
            "version": 3,
            "message": "t_load.order.total: a value does not fit numeric(6,2) NOT NULL",
            "until": None,
        }
    ]


async def test_unchanged_children_are_not_rewritten(pool: Pool, instruments: Instruments) -> None:
    """Check a re-run of the same orders changes no row, and a changed run touches only the rows that changed."""
    first = await run(pool, instruments, [order("o1", ("a", "1.00"), ("b", "2.00"))])
    assert first.changed == {"t_load.order": 1, "t_load.line": 2}
    again = await run(pool, instruments, [order("o1", ("a", "1.00"), ("b", "2.00"))])
    assert again.changed == {"t_load.order": 0, "t_load.line": 0}
    edited = await run(pool, instruments, [order("o1", ("a", "1.50"), ("c", "2.00"))])
    assert edited.changed == {"t_load.order": 1, "t_load.line": 3}  # a updated, b deleted, c inserted
    assert await rows(pool, "SELECT line_id, amount FROM t_load.line ORDER BY line_id") == [
        ("a", Decimal("1.50")),
        ("c", Decimal("2.00")),
    ]


async def test_held_records_at_load_count_towards_the_batch_policy(pool: Pool, instruments: Instruments) -> None:
    """Check Load's held records join Validate's in the batch policy, and a batch that then fails writes nothing."""
    orders = [
        {"id": "o1", "lines": "not a list"},  # held by Validate: 1 of 5 passes the policy
        order("o2", ("l", "1.00"), ("l", "2.00")),
        order("o3", ("l", "1.00"), ("l", "2.00")),
        order("o4", ("l", "1.00")),
        order("o5", ("l", "1.00")),
    ]
    with pytest.raises(QualityFailed) as info:
        await run(pool, instruments, orders)
    assert (info.value.rows_total, info.value.rows_quarantined) == (5, 3)
    assert info.value.by_rule == {"KeyViolation": 2, "schema": 1}
    assert await rows(pool, 'SELECT order_id FROM t_load."order"') == []
    assert await rows(pool, "SELECT line_id FROM t_load.line") == []
    assert await rows(pool, "SELECT id FROM quarantine.row") == []


class FakeClient:
    """A connector client that returns one page of queued orders."""

    connector = "tiktok_shop"
    hashed_keys = KEYS

    def __init__(self, orders: list[dict[str, Any]]) -> None:
        """Queue the orders."""
        self.orders = orders

    async def send(self, request: ConnectorRequest, source_id: UUID) -> dict[str, Any]:
        """Return the orders as one last page."""
        return {"code": 0, "data": {"orders": self.orders, "next_page_token": ""}}


class Missing(Record, table="t_load.missing", writer=Upsert(("source_id", "order_id"))):
    """An order loaded into a table that does not exist."""

    record_key: ClassVar[tuple[str, ...]] = ("id",)
    record_version: ClassVar[int] = 1

    id: Annotated[str, To("order_id")]


class MissingJob(BatchJob[Missing]):
    """A job whose Load always fails, because its table does not exist."""

    name = "orders"
    spec = SPEC
    window = None
    partition = "none"
    validation = Validation(model=Missing)


class FakeLease(FakeFence):
    """A ``TaskLease`` on a pool that fences like ``FakeFence`` and reports through ``control.report_outcome``."""

    def __init__(self, task: ClaimedTask, pool: Pool) -> None:
        """Lease ``task`` on ``pool``."""
        super().__init__(task)
        self._pool = pool

    @property
    def pool(self) -> Pool:
        """Return the pool."""
        return self._pool

    async def complete(self, conn: Connection, report: TaskReport) -> str:
        """Report the outcome."""
        return await control.report_outcome(conn, self.task, report)


async def test_a_failure_after_extract_carries_the_rows_fetched(pool: Pool, instruments: Instruments) -> None:
    """Check a Load failure leaves BatchJob with rows_fetched set to the records Extract fetched."""
    task = await claimed(pool)
    job = MissingJob(client=FakeClient([{"id": "o1"}, {"id": "o2"}]), zone=ZoneInfo("UTC"), instruments=instruments)
    with pytest.raises(TableMismatch) as info:
        await job(task, FakeLease(task, pool))
    assert getattr(info.value, "rows_fetched", None) == 2
