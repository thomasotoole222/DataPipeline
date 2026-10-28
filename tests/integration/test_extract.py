"""Integration tests for ``pipeline.stages.extract`` against a database with control and raw loaded."""

import os
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from datetime import UTC, date, datetime
from typing import Any
from uuid import UUID
from zoneinfo import ZoneInfo

import pytest
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import InMemoryMetricReader, NumberDataPoint
from psycopg_pool import AsyncConnectionPool

from pipeline.connectors.paging import CursorPaginator
from pipeline.connectors.spec import ConnectorRequest, EndpointSpec
from pipeline.connectors.windows import UnixSecondsWindow
from pipeline.db import control, raw
from pipeline.db.control import ClaimedTask, claim
from pipeline.db.pool import Connection, Pool
from pipeline.db.raw import RawPage
from pipeline.errors import LeaseConflict, TransientError
from pipeline.stages.extract import extract
from pipeline.telemetry import Instruments
from pipeline.timeutil import TimeFrame

DSN = os.environ.get("TEST_PIPELINE_DSN", "postgresql://postgres@localhost:5432/pipeline_test")
SOURCE = UUID("00000000-0000-0000-0000-000000000001")
FRAME = TimeFrame.days(date(2026, 10, 19), date(2026, 10, 20), ZoneInfo("Europe/London"))
SPEC = EndpointSpec(
    name="tiktok_shop.orders_search",
    connector="tiktok_shop",
    method="POST",
    path="/order/202309/orders/search",
    paginator=CursorPaginator(
        token_param="page_token",  # noqa: S106 - a parameter name, not a password
        size_param="page_size",
        size=2,
        next_at=("data", "next_page_token"),
    ),
    records_at=("data", "orders"),
    windows={"update": UnixSecondsWindow(start_param="update_time_ge", end_param="update_time_lt")},
)


def body(orders: list[str], next_token: str = "") -> dict[str, Any]:
    """Build a TikTok-style search response."""
    return {"code": 0, "data": {"orders": [{"id": o} for o in orders], "next_page_token": next_token}}


type Hook = Callable[[int], Awaitable[None]]


class FakeClient:
    """A connector client that returns queued bodies and runs a hook before each send."""

    connector = "tiktok_shop"

    def __init__(self, bodies: list[dict[str, Any] | Exception], hook: Hook | None = None) -> None:
        """Queue the responses, in order; an exception is raised instead of returned."""
        self.bodies = bodies
        self.hook = hook
        self.requests: list[ConnectorRequest] = []
        self.source_ids: list[UUID] = []

    async def send(self, request: ConnectorRequest, source_id: UUID) -> dict[str, Any]:
        """Return the next queued body."""
        n = len(self.requests)
        self.requests.append(request)
        self.source_ids.append(source_id)
        if self.hook is not None:
            await self.hook(n)
        reply = self.bodies[n]
        if isinstance(reply, Exception):
            raise reply
        return reply


@pytest.fixture
async def pool() -> AsyncIterator[Pool]:
    """Yield an open one-connection pool on empty control and raw tables."""
    p: Pool = AsyncConnectionPool(DSN, min_size=1, max_size=1, kwargs={"autocommit": True}, open=False)
    await p.open(wait=True)
    async with p.connection() as c:
        await c.execute("TRUNCATE control.lease_loss, control.task RESTART IDENTITY")
        await c.execute("TRUNCATE raw.page")
    try:
        yield p
    finally:
        await p.close()


@pytest.fixture
def reader() -> InMemoryMetricReader:
    """Return an in-memory metric reader."""
    return InMemoryMetricReader()


@pytest.fixture
def instruments(reader: InMemoryMetricReader) -> Instruments:
    """Build the pipeline's instruments on a meter provider read by ``reader``."""
    meter = MeterProvider(metric_readers=[reader]).get_meter("test")
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


def fetched_points(reader: InMemoryMetricReader) -> list[tuple[int, dict[str, Any]]]:
    """Return ``pipeline.rows`` data points as (value, attributes)."""
    data = reader.get_metrics_data()
    if data is None:
        return []
    return [
        (int(point.value), dict(point.attributes or {}))
        for rm in data.resource_metrics
        for sm in rm.scope_metrics
        for metric in sm.metrics
        if metric.name == "pipeline.rows"
        for point in metric.data.data_points
        if isinstance(point, NumberDataPoint)
    ]


async def claimed(pool: Pool) -> ClaimedTask:
    """Insert a pending orders task and claim it."""
    async with pool.connection() as c:
        await c.execute(
            "INSERT INTO control.task (run_id, connector, job, source_id, partition_key) "
            "VALUES (7, 'tiktok_shop', 'orders', %s, '2026-10-19')",
            (SOURCE,),
        )
        task = await claim(c, "test-worker")
    assert task is not None
    return task


async def steal(pool: Pool, task: ClaimedTask) -> None:
    """Move the task to another lease, as a re-claim by another worker would."""
    async with pool.connection() as c:
        await c.execute("UPDATE control.task SET lease_token = gen_random_uuid() WHERE id = %s", (task.id,))


async def pages(pool: Pool, task_id: int) -> list[RawPage]:
    """Read a task's landed pages."""
    async with pool.connection() as c:
        return [p async for p in raw.read_pages(c, task_id)]


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


async def run(pool: Pool, task: ClaimedTask, client: FakeClient, instruments: Instruments) -> int:
    """Run extract for the orders spec over ``FRAME``."""
    return await extract(pool, task, FakeFence(task), client, SPEC, "update", FRAME, instruments)


async def test_lands_every_page_with_its_cursors(pool: Pool, instruments: Instruments) -> None:
    """Check each page is saved under its task, connector and endpoint with its cursors, and the rows are counted."""
    task = await claimed(pool)
    client = FakeClient([body(["a", "b"], "t1"), body(["c", "d"], "t2"), body(["e"])])
    assert await run(pool, task, client, instruments) == 5
    landed = await pages(pool, task.id)
    assert [p.page_no for p in landed] == [0, 1, 2]
    assert landed[0].cursor_in is None
    assert landed[0].cursor_next == landed[1].cursor_in
    assert landed[1].cursor_next == landed[2].cursor_in
    assert landed[2].cursor_next is None
    assert [p.payload for p in landed] == [body(["a", "b"], "t1"), body(["c", "d"], "t2"), body(["e"])]
    assert all(p.task_id == task.id for p in landed)
    assert all(p.connector == "tiktok_shop" and p.endpoint == "orders_search" for p in landed)


async def test_the_window_and_source_reach_the_connector(pool: Pool, instruments: Instruments) -> None:
    """Check the request carries the frame through the named window and is sent for the task's source."""
    task = await claimed(pool)
    client = FakeClient([body(["a"])])
    await run(pool, task, client, instruments)
    assert client.requests[0].body is not None
    assert client.requests[0].body["update_time_ge"] == int(FRAME.start.timestamp())
    assert client.requests[0].body["update_time_lt"] == int(FRAME.end.timestamp())
    assert client.source_ids == [SOURCE]


async def test_an_empty_page_is_landed(pool: Pool, instruments: Instruments) -> None:
    """Check a page with no records is still saved, and counts no rows."""
    task = await claimed(pool)
    assert await run(pool, task, FakeClient([body([])]), instruments) == 0
    landed = await pages(pool, task.id)
    assert [p.payload for p in landed] == [body([])]


async def test_a_retry_starts_over(pool: Pool, instruments: Instruments) -> None:
    """Check pages left by an earlier attempt are discarded, even when the new fetch is shorter."""
    task = await claimed(pool)
    await run(pool, task, FakeClient([body(["a", "b"], "t1"), body(["c", "d"], "t2"), body(["e"])]), instruments)
    assert await run(pool, task, FakeClient([body(["x"])]), instruments) == 1
    landed = await pages(pool, task.id)
    assert [(p.page_no, p.payload) for p in landed] == [(0, body(["x"]))]


async def test_other_tasks_pages_are_kept(pool: Pool, instruments: Instruments) -> None:
    """Check the discard touches only this task's pages."""
    other = RawPage(
        task_id=999,
        page_no=0,
        connector="cruva",
        endpoint="samples",
        cursor_in=None,
        cursor_next=None,
        fetched_at=datetime.now(UTC),
        payload={},
    )
    async with pool.connection() as c:
        await raw.save_page(c, other)
    task = await claimed(pool)
    await run(pool, task, FakeClient([body(["a"])]), instruments)
    assert len(await pages(pool, 999)) == 1


async def test_a_lost_lease_discards_nothing(pool: Pool, instruments: Instruments) -> None:
    """Check a worker that no longer holds the lease cannot delete the new owner's pages."""
    task = await claimed(pool)
    await run(pool, task, FakeClient([body(["a"])]), instruments)
    await steal(pool, task)
    client = FakeClient([body(["x"])])
    with pytest.raises(LeaseConflict) as caught:
        await run(pool, task, client, instruments)
    assert caught.value.rows_fetched == 0  # type: ignore[attr-defined]
    assert client.requests == []
    assert [p.payload for p in await pages(pool, task.id)] == [body(["a"])]


async def test_a_lease_lost_mid_fetch_stops_the_writes(pool: Pool, instruments: Instruments) -> None:
    """Check no page is saved once the lease has moved, and the rows landed so far ride on the error."""
    task = await claimed(pool)

    async def steal_before_page_1(n: int) -> None:
        if n == 1:
            await steal(pool, task)

    client = FakeClient([body(["a", "b"], "t1"), body(["c"])], steal_before_page_1)
    with pytest.raises(LeaseConflict) as caught:
        await run(pool, task, client, instruments)
    assert caught.value.rows_fetched == 2  # type: ignore[attr-defined]
    assert [p.page_no for p in await pages(pool, task.id)] == [0]


async def test_a_failure_carries_the_rows_fetched(pool: Pool, instruments: Instruments) -> None:
    """Check the connector's own exception is re-raised with the attempt's count and the landed pages kept."""
    task = await claimed(pool)
    failure = TransientError("connector timed out")
    client = FakeClient([body(["a", "b"], "t1"), body(["c", "d"], "t2"), failure])
    with pytest.raises(TransientError) as caught:
        await run(pool, task, client, instruments)
    assert caught.value is failure
    assert caught.value.rows_fetched == 4  # type: ignore[attr-defined]
    assert [p.page_no for p in await pages(pool, task.id)] == [0, 1]


async def test_no_connection_is_held_across_a_connector_call(pool: Pool, instruments: Instruments) -> None:
    """Check the one-connection pool is free whenever the connector is called."""
    task = await claimed(pool)

    async def borrow(n: int) -> None:
        async with pool.connection(timeout=0.5):
            pass

    await run(pool, task, FakeClient([body(["a"], "t1"), body(["b"])], borrow), instruments)


async def test_rows_fetched_are_counted_per_page(
    pool: Pool, instruments: Instruments, reader: InMemoryMetricReader
) -> None:
    """Check pipeline.rows counts the fetched rows with the job and the fetched stage."""
    task = await claimed(pool)
    await run(pool, task, FakeClient([body(["a", "b"], "t1"), body(["c"])]), instruments)
    assert fetched_points(reader) == [(3, {"job": "orders", "stage": "fetched"})]
